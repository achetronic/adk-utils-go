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

// thresholdStrategy implements token-based compaction. It estimates total
// tokens before every LLM call and summarizes the older portion of the
// conversation when remaining capacity drops below a safety buffer.
type thresholdStrategy struct {
	registry  ModelRegistry
	llm       model.LLM
	maxTokens int
	mu        sync.Mutex
}

// newThresholdStrategy creates a threshold strategy. If maxTokens > 0 it
// overrides the registry lookup for the context window size.
func newThresholdStrategy(registry ModelRegistry, llm model.LLM, maxTokens int) *thresholdStrategy {
	return &thresholdStrategy{
		registry:  registry,
		llm:       llm,
		maxTokens: maxTokens,
	}
}

// Name returns the strategy identifier for logging.
func (s *thresholdStrategy) Name() string {
	return StrategyThreshold
}

// Compact checks the token estimate against the model's context window and,
// if the threshold is exceeded, splits the conversation into old + recent,
// summarizes the old portion, and rewrites req.Contents in place.
func (s *thresholdStrategy) Compact(ctx agent.CallbackContext, req *model.LLMRequest) error {
	var contextWindow int
	if s.maxTokens > 0 {
		contextWindow = s.maxTokens
	} else {
		contextWindow = s.registry.ContextWindow(req.Model)
	}
	buffer := computeBuffer(contextWindow)
	threshold := contextWindow - buffer

	existingSummary := loadSummary(ctx)
	if existingSummary != "" {
		injectSummary(req, existingSummary)
	}

	totalTokens := estimateTokens(req)
	if totalTokens < threshold {
		return nil
	}

	slog.Info("ContextGuard [threshold]: threshold exceeded, summarizing",
		"agent", ctx.AgentName(),
		"session", ctx.SessionID(),
		"tokens", totalTokens,
		"threshold", threshold,
		"contextWindow", contextWindow,
		"buffer", buffer,
		"maxSummaryWords", int(float64(buffer)*0.50*0.75),
	)

	s.mu.Lock()
	defer s.mu.Unlock()

	recentBudget := int(float64(contextWindow) * recentWindowRatio)
	splitIdx := findSplitIndex(req.Contents, recentBudget)

	oldContents := req.Contents[:splitIdx]
	recentContents := req.Contents[splitIdx:]

	summary, err := summarize(ctx, s.llm, oldContents, existingSummary, buffer)
	if err != nil {
		return fmt.Errorf("summarization failed: %w", err)
	}

	persistSummary(ctx, summary, totalTokens)
	replaceSummary(req, summary, recentContents)

	slog.Info("ContextGuard [threshold]: conversation compressed",
		"agent", ctx.AgentName(),
		"session", ctx.SessionID(),
		"oldMessages", len(oldContents),
		"recentMessages", len(recentContents),
		"newTokenEstimate", estimateTokens(req),
	)

	return nil
}
