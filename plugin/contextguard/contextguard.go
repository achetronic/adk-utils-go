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

// Package contextguard implements an ADK plugin that prevents conversations
// from exceeding the LLM's context window. Before every model call it
// delegates to a configurable Strategy that decides whether and how to
// compact the conversation history.
//
// Two strategies are provided out of the box:
//
//   - ThresholdStrategy: estimates token count and summarizes when the
//     remaining capacity drops below a safety buffer (two-tier: fixed 20k
//     for large windows, 20% for small ones). This is a reactive guard.
//
//   - SlidingWindowStrategy: compacts when the number of Content entries
//     exceeds a configured maximum, regardless of token count. This is a
//     preventive, periodic compaction based on turn count.
//
// Both strategies use the agent's own LLM for summarization and share the
// same structured system prompt, state keys, and helper functions.
//
// Usage:
//
//	guard := contextguard.New(registry)
//	guard.Add("assistant", llmModel)
//	guard.Add("researcher", llmResearcher, contextguard.WithSlidingWindow(30))
//
//	runnr, _ := runner.New(runner.Config{
//	    Agent:        myAgent,
//	    PluginConfig: guard.PluginConfig(),
//	})
package contextguard

import (
	"log/slog"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/plugin"
	"google.golang.org/adk/runner"
)

const (
	// StrategyThreshold selects the token-threshold strategy: summarization
	// fires when estimated token usage approaches the model's context window.
	StrategyThreshold = "threshold"

	// StrategySlidingWindow selects the sliding-window strategy: summarization
	// fires when the number of Content entries exceeds a configured limit.
	StrategySlidingWindow = "sliding_window"
)

const (
	stateKeyPrefixSummary              = "__context_guard_summary_"
	stateKeyPrefixSummarizedAt         = "__context_guard_summarized_at_"
	stateKeyPrefixContentsAtCompaction = "__context_guard_contents_at_compaction_"

	largeContextWindowThreshold = 200_000
	largeContextWindowBuffer    = 20_000
	smallContextWindowRatio     = 0.20
	recentWindowRatio           = 0.20

	maxCompactionAttempts = 3
)

const defaultMaxTurns = 20

// Strategy defines how a compaction algorithm decides whether and how to
// compact conversation history before an LLM call.
type Strategy interface {
	Name() string
	Compact(ctx agent.CallbackContext, req *model.LLMRequest) error
}

// AgentOption configures per-agent behavior when calling Add.
type AgentOption func(*agentConfig)

type agentConfig struct {
	strategy  string
	maxTurns  int
	maxTokens int
}

// WithSlidingWindow selects the sliding-window strategy with the given
// maximum number of Content entries before compaction.
func WithSlidingWindow(maxTurns int) AgentOption {
	return func(c *agentConfig) {
		c.strategy = StrategySlidingWindow
		c.maxTurns = maxTurns
	}
}

// WithMaxTokens sets a manual context window size override (in tokens).
// Only used by the threshold strategy. When set, the ModelRegistry is
// bypassed for this agent.
func WithMaxTokens(maxTokens int) AgentOption {
	return func(c *agentConfig) {
		c.maxTokens = maxTokens
	}
}

// ContextGuard accumulates per-agent strategies and produces a single
// runner.PluginConfig. Use New to create one, Add to register agents,
// and PluginConfig to get the final configuration.
type ContextGuard struct {
	registry   ModelRegistry
	strategies map[string]Strategy
}

// New creates a ContextGuard backed by the given ModelRegistry.
func New(registry ModelRegistry) *ContextGuard {
	return &ContextGuard{
		registry:   registry,
		strategies: make(map[string]Strategy),
	}
}

// Add registers an agent with its LLM for summarization. Without options,
// the threshold strategy is used with limits from the ModelRegistry.
func (g *ContextGuard) Add(agentID string, llm model.LLM, opts ...AgentOption) {
	cfg := &agentConfig{
		strategy: StrategyThreshold,
	}
	for _, opt := range opts {
		opt(cfg)
	}

	switch cfg.strategy {
	case StrategySlidingWindow:
		maxTurns := cfg.maxTurns
		if maxTurns <= 0 {
			maxTurns = defaultMaxTurns
		}
		g.strategies[agentID] = newSlidingWindowStrategy(g.registry, llm, maxTurns)
	default:
		g.strategies[agentID] = newThresholdStrategy(g.registry, llm, cfg.maxTokens)
	}

	slog.Info("ContextGuard: strategy configured",
		"agent", agentID,
		"strategy", g.strategies[agentID].Name(),
	)
}

// PluginConfig returns a runner.PluginConfig ready to pass to the ADK
// launcher or runner.
func (g *ContextGuard) PluginConfig() runner.PluginConfig {
	guard := &contextGuard{strategies: g.strategies}

	p, _ := plugin.New(plugin.Config{
		Name:                "context_guard",
		BeforeModelCallback: llmagent.BeforeModelCallback(guard.beforeModel),
	})

	return runner.PluginConfig{
		Plugins: []*plugin.Plugin{p},
	}
}

// contextGuard is the internal state of the plugin, holding per-agent
// strategies keyed by agent ID.
type contextGuard struct {
	strategies map[string]Strategy
}

// beforeModel is the BeforeModelCallback invoked by ADK before every LLM
// call. It looks up the agent's strategy and delegates compaction to it.
func (g *contextGuard) beforeModel(ctx agent.CallbackContext, req *model.LLMRequest) (*model.LLMResponse, error) {
	if req == nil || len(req.Contents) == 0 {
		return nil, nil
	}

	strategy, ok := g.strategies[ctx.AgentName()]
	if !ok {
		return nil, nil
	}

	if err := strategy.Compact(ctx, req); err != nil {
		slog.Warn("ContextGuard: compaction failed, passing through",
			"agent", ctx.AgentName(),
			"strategy", strategy.Name(),
			"error", err,
		)
	}

	return nil, nil
}
