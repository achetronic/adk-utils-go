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
	"log/slog"
	"sync"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/model"
)

// thresholdStrategy implements token-based compaction. It estimates total
// tokens before every LLM call and summarizes the entire conversation
// when remaining capacity drops below a safety buffer.
//
// When real token counts are available from the provider (persisted by the
// AfterModelCallback), they are preferred over the len/4 heuristic. A
// calibrated heuristic bridges the timing gap between callbacks so that
// tool results added after the last LLM call are accounted for.
//
// Compaction always produces a full summary (no recent tail preserved),
// matching Crush CLI behaviour. The result is [summary] + [continuation].
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
// if the threshold is exceeded, summarizes the entire conversation and
// rewrites req.Contents to [summary] + [continuation instruction].
//
// Token source priority: calibrated heuristic > stale real tokens > raw heuristic.
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
	contentsAtLastCompaction := loadContentsAtCompaction(ctx)
	totalSessionContents := len(req.Contents)
	if existingSummary != "" {
		injectSummary(req, existingSummary, contentsAtLastCompaction)
	}

	totalTokens := tokenCount(ctx, req)
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

	userContent := ctx.UserContent()
	todos := loadTodos(ctx)

	contentsForSummary := truncateForSummarizer(req.Contents, contextWindow)

	summary, err := summarize(ctx, s.llm, contentsForSummary, existingSummary, buffer, todos)
	if err != nil {
		slog.Warn("ContextGuard [threshold]: summarization failed, using fallback",
			"agent", ctx.AgentName(),
			"session", ctx.SessionID(),
			"error", err,
		)
		summary = buildFallbackSummary(contentsForSummary, existingSummary)
	}

	persistSummary(ctx, summary, totalTokens)
	persistContentsAtCompaction(ctx, totalSessionContents)
	replaceSummary(req, summary, nil)
	injectContinuation(req, userContent)

	resetCalibration(ctx)

	newTokens := estimateTokens(req)

	slog.Info("ContextGuard [threshold]: compaction completed",
		"agent", ctx.AgentName(),
		"session", ctx.SessionID(),
		"oldMessages", len(req.Contents),
		"newTokenEstimate", newTokens,
		"threshold", threshold,
	)

	return nil
}
