// Copyright 2025 achetronic
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package contextguard

import (
	"fmt"
	"log/slog"
	"sync"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/model"
)

// slidingWindowStrategy implements turn-count-based compaction. When the
// number of new Content entries since the last compaction exceeds maxTurns,
// all but a small recent window (30% of maxTurns, minimum 3) are summarized
// and replaced with a single summary message.
type slidingWindowStrategy struct {
	registry ModelRegistry
	llm      model.LLM
	maxTurns int
	mu       sync.Mutex
}

const recentKeepRatio = 0.30

// newSlidingWindowStrategy creates a sliding window strategy for a single agent.
func newSlidingWindowStrategy(registry ModelRegistry, llm model.LLM, maxTurns int) *slidingWindowStrategy {
	return &slidingWindowStrategy{
		registry: registry,
		llm:      llm,
		maxTurns: maxTurns,
	}
}

// Name returns the strategy identifier for logging.
func (s *slidingWindowStrategy) Name() string {
	return StrategySlidingWindow
}

// Compact counts Content entries that arrived after the last compaction.
// If that count exceeds maxTurns, it summarizes all old entries and keeps
// only a small recent window. Otherwise it injects the existing summary
// (if any) and returns without touching the conversation.
func (s *slidingWindowStrategy) Compact(ctx agent.CallbackContext, req *model.LLMRequest) error {
	existingSummary := loadSummary(ctx)
	contentsAtLastCompaction := loadContentsAtCompaction(ctx)

	totalContents := len(req.Contents)
	turnsSinceCompaction := totalContents - contentsAtLastCompaction

	if turnsSinceCompaction <= s.maxTurns {
		if existingSummary != "" {
			injectSummary(req, existingSummary)
		}
		return nil
	}

	slog.Info("ContextGuard [sliding_window]: turn limit exceeded, summarizing",
		"agent", ctx.AgentName(),
		"session", ctx.SessionID(),
		"totalContents", totalContents,
		"contentsAtLastCompaction", contentsAtLastCompaction,
		"turnsSinceCompaction", turnsSinceCompaction,
		"maxTurns", s.maxTurns,
	)

	s.mu.Lock()
	defer s.mu.Unlock()

	recentKeep := max(3, s.maxTurns*30/100)
	splitIdx := safeSplitIndex(req.Contents, len(req.Contents)-recentKeep)
	oldContents := req.Contents[:splitIdx]
	recentContents := req.Contents[splitIdx:]

	contextWindow := s.registry.ContextWindow(req.Model)
	buffer := computeBuffer(contextWindow)

	summary, err := summarize(ctx, s.llm, oldContents, existingSummary, buffer)
	if err != nil {
		slog.Error("ContextGuard [sliding_window]: summarization FAILED",
			"agent", ctx.AgentName(),
			"session", ctx.SessionID(),
			"error", err,
		)
		return fmt.Errorf("summarization failed: %w", err)
	}

	slog.Info("ContextGuard [sliding_window]: summarization succeeded, compressing conversation",
		"agent", ctx.AgentName(),
		"session", ctx.SessionID(),
		"summaryLen", len(summary),
	)

	tokenEstimate := estimateContentTokens(oldContents)
	persistSummary(ctx, summary, tokenEstimate)
	persistContentsAtCompaction(ctx, totalContents)

	replaceSummary(req, summary, recentContents)

	slog.Info("ContextGuard [sliding_window]: conversation compressed",
		"agent", ctx.AgentName(),
		"session", ctx.SessionID(),
		"oldMessages", len(oldContents),
		"recentMessages", len(recentContents),
		"newTokenEstimate", estimateTokens(req),
		"watermarkWritten", totalContents,
	)

	return nil
}
