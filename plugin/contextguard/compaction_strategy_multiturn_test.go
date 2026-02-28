package contextguard

import (
	"fmt"
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/model"
)

// simulateSession runs a full multi-turn session through the threshold
// compaction pipeline, mimicking the real ADK flow:
//
//	For each turn:
//	  1. BeforeModelCallback: Compact() checks tokens, may summarize
//	  2. persistLastHeuristic (what beforeModel does after Compact)
//	  3. LLM responds (simulated) → AfterModelCallback persists real tokens
//	  4. New user message arrives → next turn
//
// This catches bugs that single-shot Compact() tests miss: stale calibration
// data, correction factor drift, compaction loops, etc.
type sessionConfig struct {
	contextWindow    int
	systemPromptSize int     // chars of system instruction
	modelName        string  // for registry lookup
	hasUsageMetadata bool    // whether the provider returns PromptTokenCount
	tokenRatio       float64 // real_tokens / heuristic_tokens (simulates tokenizer accuracy)
}

type turnConfig struct {
	userMessage string
	toolCalls   []toolCall // tools the model calls this turn
}

type toolCall struct {
	name         string
	responseSize int // chars in the response
}

type sessionResult struct {
	turns            int
	compactions      int
	finalTokens      int
	maxTokensSeen    int
	overflowed       bool // tokens ever exceeded contextWindow (not threshold — the actual window)
	compactionFailed bool
	loopDetected     bool // compacted but tokens didn't decrease
}

// simulateSession models the real ADK execution loop faithfully:
//
//	User sends message
//	LOOP {
//	    preprocess (loads contents into req)
//	    BeforeModelCallback ← ContextGuard checks tokens, may compact
//	    LLM call (simulated)
//	    AfterModelCallback ← persists real token count
//	    if model returned tool calls:
//	        execute tools → append [FunctionCall, FunctionResponse] to contents
//	        CONTINUE LOOP (another beforeModel + LLM call with tool results)
//	    else:
//	        append model text response to contents
//	        BREAK (wait for next user message)
//	}
//
// This means BeforeModelCallback fires before EVERY LLM call, including
// the call that processes tool results. Tool responses are always visible
// to ContextGuard before the next LLM call — matching real ADK behavior.
func simulateSession(t *testing.T, cfg sessionConfig, turns []turnConfig) sessionResult {
	t.Helper()

	registry := &mockRegistry{
		contextWindows: map[string]int{cfg.modelName: cfg.contextWindow},
		maxTokens:      map[string]int{cfg.modelName: 4096},
	}
	llm := &mockLLM{
		name:     cfg.modelName,
		response: "Summary: conversation involved investigating issues with tools. Key decisions were made. Specific next steps identified.",
	}
	strategy := newThresholdStrategy(registry, llm, 0)

	guard := &contextGuard{
		strategies: map[string]Strategy{
			"test-agent": strategy,
		},
	}

	ctx := newMockCallbackContext("test-agent")
	ctx.sessionID = "stress-session"

	var systemInstruction *genai.Content
	if cfg.systemPromptSize > 0 {
		systemInstruction = &genai.Content{
			Parts: []*genai.Part{{Text: strings.Repeat("You are a helpful assistant. ", cfg.systemPromptSize/28+1)[:cfg.systemPromptSize]}},
		}
	}

	var contents []*genai.Content
	result := sessionResult{}

	if cfg.tokenRatio == 0 {
		cfg.tokenRatio = 2.0
	}

	// runLLMStep simulates one iteration of the ADK inner loop:
	// preprocess → BeforeModelCallback → LLM → AfterModelCallback
	runLLMStep := func(turnIdx int, label string) {
		req := &model.LLMRequest{
			Model:    cfg.modelName,
			Contents: copyContents(contents),
			Config:   &genai.GenerateContentConfig{},
		}
		if systemInstruction != nil {
			req.Config.SystemInstruction = systemInstruction
		}

		tokensBefore := estimateTokens(req)
		_, err := guard.beforeModel(ctx, req)
		if err != nil {
			t.Logf("Turn %d [%s]: beforeModel error: %v", turnIdx, label, err)
			result.compactionFailed = true
		}

		tokensAfter := estimateTokens(req)
		if tokensAfter < tokensBefore && loadSummary(ctx) != "" {
			result.compactions++
			if tokensAfter >= tokensBefore {
				result.loopDetected = true
			}
		}

		contents = copyContents(req.Contents)

		estimatedReal := int(float64(tokensAfter) * cfg.tokenRatio)
		if estimatedReal > result.maxTokensSeen {
			result.maxTokensSeen = estimatedReal
		}
		if estimatedReal > cfg.contextWindow {
			result.overflowed = true
			t.Logf("Turn %d [%s]: OVERFLOW — estimated real tokens %d > context window %d",
				turnIdx, label, estimatedReal, cfg.contextWindow)
		}

		if cfg.hasUsageMetadata {
			realPromptTokens := int(float64(estimateTokens(req)) * cfg.tokenRatio)
			resp := &model.LLMResponse{
				Content: textContent("model", "Model response"),
				UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
					PromptTokenCount: int32(realPromptTokens),
				},
			}
			guard.afterModel(ctx, resp, nil)
		}
	}

	for i, turn := range turns {
		// User sends a message
		contents = append(contents, textContent("user", turn.userMessage))

		// --- ADK loop iteration 1: process user message ---
		runLLMStep(i, "user-msg")

		if len(turn.toolCalls) > 0 {
			// Model decides to call tools (simulated LLM response with function calls).
			// ADK executes each tool and appends results to the session.
			for _, tc := range turn.toolCalls {
				contents = append(contents,
					&genai.Content{
						Role: "model",
						Parts: []*genai.Part{{
							FunctionCall: &genai.FunctionCall{
								Name: tc.name,
								Args: map[string]any{"param": "value"},
							},
						}},
					},
					&genai.Content{
						Role: "user",
						Parts: []*genai.Part{{
							FunctionResponse: &genai.FunctionResponse{
								Name:     tc.name,
								Response: map[string]any{"result": strings.Repeat("x", tc.responseSize)},
							},
						}},
					},
				)
			}

			// --- ADK loop iteration 2: process tool results ---
			// preprocess loads tool results into req → BeforeModelCallback fires again
			runLLMStep(i, "tool-results")
		}

		// Model produces a final text response (no more tool calls) → BREAK inner loop
		modelResp := fmt.Sprintf("Analysis for turn %d complete. Here is my detailed response about the topic you asked. I will proceed with the investigation.", i)
		contents = append(contents, textContent("model", modelResp))
	}

	result.turns = len(turns)
	result.finalTokens = estimateContentTokens(contents)
	return result
}

// longMessage generates a realistic user message of approximately n characters.
func longMessage(turn, length int) string {
	base := fmt.Sprintf("Turn %d: I need a detailed explanation of how the Kubernetes pod lifecycle works, "+
		"including init containers, readiness probes, liveness probes, and the termination grace period. "+
		"Also explain how resource limits and requests interact with the scheduler, how the kubelet handles "+
		"OOM kills, what happens during node pressure eviction, and how pod disruption budgets affect rolling "+
		"updates. Additionally, describe how horizontal pod autoscaler metrics are collected and how custom "+
		"metrics from Prometheus can drive autoscaling decisions. Finally, explain service mesh sidecar "+
		"injection and how Istio's envoy proxy handles traffic routing between services. ", turn)
	if len(base) >= length {
		return base[:length]
	}
	return base + strings.Repeat("Please elaborate on these concepts in great detail. ", (length-len(base))/52+1)[:length-len(base)]
}

// ---------------------------------------------------------------------------
// Test: Multi-turn session with 200k context window (Anthropic-like)
// ---------------------------------------------------------------------------

func TestStress_200k_NormalConversation(t *testing.T) {
	turns := make([]turnConfig, 30)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: fmt.Sprintf("Turn %d: Can you help me debug the issue with the API endpoint returning 500 errors? I have checked the logs and the stack trace points to the database connection pool.", i),
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    200_000,
		systemPromptSize: 2000,
		modelName:        "claude-sonnet",
		hasUsageMetadata: true,
		tokenRatio:       2.0,
	}, turns)

	t.Logf("200k/normal: turns=%d compactions=%d maxTokens=%d overflowed=%v", r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("200k normal conversation should never overflow")
	}
	if r.loopDetected {
		t.Error("compaction loop detected")
	}
}

func TestStress_200k_ToolHeavy(t *testing.T) {
	turns := make([]turnConfig, 20)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: fmt.Sprintf("Turn %d: Check the pod status and retrieve the logs", i),
			toolCalls: []toolCall{
				{name: "kubectl_get_pods", responseSize: 10_000},
				{name: "kubectl_describe", responseSize: 5_000},
				{name: "kubectl_logs", responseSize: 20_000},
			},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    200_000,
		systemPromptSize: 3000,
		modelName:        "claude-sonnet",
		hasUsageMetadata: true,
		tokenRatio:       2.0,
	}, turns)

	t.Logf("200k/tool-heavy: turns=%d compactions=%d maxTokens=%d overflowed=%v", r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("200k tool-heavy should not overflow")
	}
	if r.compactions == 0 {
		t.Error("expected at least one compaction with heavy tool usage")
	}
}

func TestStress_200k_SingleGiantToolResponse(t *testing.T) {
	turns := []turnConfig{
		{userMessage: "Get all pods in JSON format for analysis", toolCalls: []toolCall{
			{name: "kubectl_get_pods", responseSize: 300_000},
		}},
		{userMessage: "Now analyze the pod statuses and identify failures"},
		{userMessage: "What about the failing ones? Show me the details"},
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    200_000,
		systemPromptSize: 2000,
		modelName:        "claude-sonnet",
		hasUsageMetadata: true,
		tokenRatio:       2.0,
	}, turns)

	t.Logf("200k/giant-response: turns=%d compactions=%d maxTokens=%d overflowed=%v", r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("200k session with giant tool response should not overflow")
	}
}

func TestStress_200k_ToolBurst(t *testing.T) {
	turns := []turnConfig{
		{userMessage: "Debug the full stack with all available tools", toolCalls: []toolCall{
			{name: "tool_1", responseSize: 5_000},
			{name: "tool_2", responseSize: 5_000},
			{name: "tool_3", responseSize: 5_000},
			{name: "tool_4", responseSize: 5_000},
			{name: "tool_5", responseSize: 5_000},
			{name: "tool_6", responseSize: 5_000},
			{name: "tool_7", responseSize: 5_000},
			{name: "tool_8", responseSize: 5_000},
			{name: "tool_9", responseSize: 5_000},
			{name: "tool_10", responseSize: 5_000},
		}},
		{userMessage: "What did you find in all those tool results?"},
		{userMessage: "Can you fix the issues?", toolCalls: []toolCall{
			{name: "edit_file", responseSize: 2_000},
			{name: "run_tests", responseSize: 10_000},
		}},
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    200_000,
		systemPromptSize: 2000,
		modelName:        "claude-sonnet",
		hasUsageMetadata: true,
		tokenRatio:       2.0,
	}, turns)

	t.Logf("200k/tool-burst: turns=%d compactions=%d maxTokens=%d overflowed=%v", r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("tool burst within 200k should not overflow")
	}
}

// ---------------------------------------------------------------------------
// Test: Multi-turn session with 8k context window (small model)
// Each user message is ~800 chars → ~200 heuristic tokens per turn.
// With model response, tool calls, etc., accumulation fills 8k quickly.
// ---------------------------------------------------------------------------

func TestStress_8k_NormalConversation(t *testing.T) {
	turns := make([]turnConfig, 20)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: longMessage(i, 800),
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    8_000,
		systemPromptSize: 500,
		modelName:        "small-model",
		hasUsageMetadata: true,
		tokenRatio:       1.8,
	}, turns)

	t.Logf("8k/normal: turns=%d compactions=%d maxTokens=%d overflowed=%v", r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("8k normal conversation should not overflow")
	}
	if r.compactions == 0 {
		t.Error("expected compactions in 8k window with 20 turns of long messages")
	}
}

func TestStress_8k_SmallToolCalls(t *testing.T) {
	turns := make([]turnConfig, 15)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: longMessage(i, 500),
			toolCalls: []toolCall{
				{name: "web_search", responseSize: 1_000},
			},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    8_000,
		systemPromptSize: 500,
		modelName:        "small-model",
		hasUsageMetadata: true,
		tokenRatio:       1.8,
	}, turns)

	t.Logf("8k/small-tools: turns=%d compactions=%d maxTokens=%d overflowed=%v", r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("8k with small tools should not overflow")
	}
	if r.compactions == 0 {
		t.Error("expected compactions in 8k window with tool usage")
	}
}

func TestStress_8k_LargeToolResponse(t *testing.T) {
	turns := []turnConfig{
		{userMessage: "Get the full log file from the production server for analysis", toolCalls: []toolCall{
			{name: "read_file", responseSize: 20_000},
		}},
		{userMessage: "What errors are in the log file? List them all"},
		{userMessage: "Fix the first error you found"},
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    8_000,
		systemPromptSize: 500,
		modelName:        "small-model",
		hasUsageMetadata: true,
		tokenRatio:       1.8,
	}, turns)

	t.Logf("8k/large-tool: turns=%d compactions=%d maxTokens=%d overflowed=%v", r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.compactions == 0 {
		t.Error("20k tool response in 8k window must trigger compaction")
	}
}

// ---------------------------------------------------------------------------
// Test: Provider without UsageMetadata (pure heuristic mode)
// ---------------------------------------------------------------------------

func TestStress_200k_NoUsageMetadata(t *testing.T) {
	turns := make([]turnConfig, 25)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: fmt.Sprintf("Turn %d: Help me with the deployment of this service to Kubernetes", i),
			toolCalls: []toolCall{
				{name: "kubectl", responseSize: 8_000},
			},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    200_000,
		systemPromptSize: 2000,
		modelName:        "custom-model",
		hasUsageMetadata: false,
		tokenRatio:       2.5,
	}, turns)

	t.Logf("200k/no-usage: turns=%d compactions=%d maxTokens=%d overflowed=%v", r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("should not overflow even without usage metadata")
	}
}

func TestStress_8k_NoUsageMetadata(t *testing.T) {
	turns := make([]turnConfig, 25)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: longMessage(i, 800),
			toolCalls: []toolCall{
				{name: "tool", responseSize: 1_500},
			},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    8_000,
		systemPromptSize: 300,
		modelName:        "custom-model",
		hasUsageMetadata: false,
		tokenRatio:       2.0,
	}, turns)

	t.Logf("8k/no-usage: turns=%d compactions=%d maxTokens=%d overflowed=%v", r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("8k without usage metadata should not overflow")
	}
	if r.compactions == 0 {
		t.Error("expected compactions in 8k window with 25 turns even without usage metadata")
	}
}

// ---------------------------------------------------------------------------
// Test: Repeated compactions (long-running session)
// ---------------------------------------------------------------------------

func TestStress_200k_LongRunning_50Turns(t *testing.T) {
	turns := make([]turnConfig, 50)
	for i := range turns {
		tools := []toolCall{}
		if i%3 == 0 {
			tools = []toolCall{
				{name: "search", responseSize: 5_000},
				{name: "fetch", responseSize: 15_000},
			}
		}
		turns[i] = turnConfig{
			userMessage: fmt.Sprintf("Turn %d: Continue investigating the performance issue with the database connection pool and the slow queries", i),
			toolCalls:   tools,
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    200_000,
		systemPromptSize: 3000,
		modelName:        "claude-sonnet",
		hasUsageMetadata: true,
		tokenRatio:       2.2,
	}, turns)

	t.Logf("200k/50turns: turns=%d compactions=%d maxTokens=%d overflowed=%v looped=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed, r.loopDetected)
	if r.overflowed {
		t.Error("50-turn session should not overflow 200k window")
	}
	if r.loopDetected {
		t.Error("compaction loop detected in long session")
	}
}

func TestStress_8k_LongRunning_40Turns(t *testing.T) {
	turns := make([]turnConfig, 40)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: longMessage(i, 600),
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    8_000,
		systemPromptSize: 500,
		modelName:        "small-model",
		hasUsageMetadata: true,
		tokenRatio:       1.8,
	}, turns)

	t.Logf("8k/40turns: turns=%d compactions=%d maxTokens=%d overflowed=%v looped=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed, r.loopDetected)
	if r.overflowed {
		t.Error("40-turn session should not overflow 8k window")
	}
	if r.compactions < 2 {
		t.Errorf("expected at least 2 compactions in 40 turns with 8k window, got %d", r.compactions)
	}
}

// ---------------------------------------------------------------------------
// Test: Worst case — tool burst with massive responses
// ---------------------------------------------------------------------------

func TestStress_200k_MassiveToolBurst(t *testing.T) {
	turns := []turnConfig{
		{userMessage: "Analyze all services in the cluster", toolCalls: func() []toolCall {
			calls := make([]toolCall, 15)
			for i := range calls {
				calls[i] = toolCall{name: fmt.Sprintf("service_%d", i), responseSize: 50_000}
			}
			return calls
		}()},
		{userMessage: "Summarize the findings from all services"},
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    200_000,
		systemPromptSize: 3000,
		modelName:        "claude-sonnet",
		hasUsageMetadata: true,
		tokenRatio:       2.0,
	}, turns)

	t.Logf("200k/massive-burst: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("massive tool burst should not overflow")
	}
	if r.compactions == 0 {
		t.Error("massive tool burst should trigger compaction")
	}
}

func TestStress_8k_ToolBurst(t *testing.T) {
	turns := []turnConfig{
		{userMessage: "Run all diagnostic tools on the system", toolCalls: []toolCall{
			{name: "diag_1", responseSize: 3_000},
			{name: "diag_2", responseSize: 3_000},
			{name: "diag_3", responseSize: 3_000},
		}},
		{userMessage: "What diagnostic issues did you find?"},
		{userMessage: "Fix the first issue", toolCalls: []toolCall{
			{name: "fix", responseSize: 1_000},
			{name: "verify", responseSize: 2_000},
		}},
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    8_000,
		systemPromptSize: 500,
		modelName:        "small-model",
		hasUsageMetadata: true,
		tokenRatio:       1.8,
	}, turns)

	t.Logf("8k/tool-burst: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("8k tool burst should not overflow")
	}
}

// ---------------------------------------------------------------------------
// Test: High tokenizer ratio (worst case for heuristic underestimation)
// ---------------------------------------------------------------------------

func TestStress_200k_HighTokenRatio(t *testing.T) {
	turns := make([]turnConfig, 15)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: fmt.Sprintf("Turn %d: Check system status for all nodes", i),
			toolCalls: []toolCall{
				{name: "status", responseSize: 10_000},
			},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    200_000,
		systemPromptSize: 5000,
		modelName:        "claude-sonnet",
		hasUsageMetadata: true,
		tokenRatio:       3.0,
	}, turns)

	t.Logf("200k/high-ratio: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("should handle high token ratio without overflow")
	}
}

func TestStress_8k_HighTokenRatio(t *testing.T) {
	turns := make([]turnConfig, 20)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: longMessage(i, 600),
			toolCalls: []toolCall{
				{name: "tool", responseSize: 1_000},
			},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    8_000,
		systemPromptSize: 300,
		modelName:        "small-model",
		hasUsageMetadata: true,
		tokenRatio:       3.0,
	}, turns)

	t.Logf("8k/high-ratio: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("8k with high token ratio should not overflow")
	}
	if r.compactions == 0 {
		t.Error("expected compactions with 3.0 token ratio in 8k window")
	}
}

// ---------------------------------------------------------------------------
// Test: Large system prompt relative to context window
// ---------------------------------------------------------------------------

func TestStress_200k_LargeSystemPrompt(t *testing.T) {
	turns := make([]turnConfig, 20)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: fmt.Sprintf("Turn %d: Continue the analysis", i),
			toolCalls: []toolCall{
				{name: "tool", responseSize: 5_000},
			},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    200_000,
		systemPromptSize: 50_000,
		modelName:        "claude-sonnet",
		hasUsageMetadata: true,
		tokenRatio:       2.0,
	}, turns)

	t.Logf("200k/large-system: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("large system prompt should not cause overflow")
	}
}

func TestStress_8k_LargeSystemPrompt(t *testing.T) {
	turns := make([]turnConfig, 15)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: longMessage(i, 600),
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    8_000,
		systemPromptSize: 8_000,
		modelName:        "small-model",
		hasUsageMetadata: true,
		tokenRatio:       1.8,
	}, turns)

	t.Logf("8k/large-system: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("large system prompt in 8k window should not cause overflow")
	}
	if r.compactions == 0 {
		t.Error("expected compactions with large system prompt in 8k window")
	}
}

// ---------------------------------------------------------------------------
// Test: Compaction must not enter infinite loop
// ---------------------------------------------------------------------------

func TestStress_CompactionNoInfiniteLoop(t *testing.T) {
	turns := make([]turnConfig, 5)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: longMessage(i, 500),
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    8_000,
		systemPromptSize: 12_000,
		modelName:        "small-model",
		hasUsageMetadata: true,
		tokenRatio:       2.5,
	}, turns)

	t.Logf("no-loop: turns=%d compactions=%d maxTokens=%d overflowed=%v looped=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed, r.loopDetected)
	if r.loopDetected {
		t.Error("compaction should not loop")
	}
}

// ---------------------------------------------------------------------------
// Test: Mixed provider scenario (first few turns no usage, then usage arrives)
// ---------------------------------------------------------------------------

func TestStress_200k_LateUsageMetadata(t *testing.T) {
	turnsPhase1 := make([]turnConfig, 5)
	for i := range turnsPhase1 {
		turnsPhase1[i] = turnConfig{
			userMessage: fmt.Sprintf("Phase1 turn %d: initial exploration", i),
			toolCalls:   []toolCall{{name: "tool", responseSize: 5_000}},
		}
	}

	r1 := simulateSession(t, sessionConfig{
		contextWindow:    200_000,
		systemPromptSize: 2000,
		modelName:        "custom-model",
		hasUsageMetadata: false,
		tokenRatio:       2.0,
	}, turnsPhase1)

	t.Logf("200k/late-usage-phase1: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r1.turns, r1.compactions, r1.maxTokensSeen, r1.overflowed)
	if r1.overflowed {
		t.Error("phase 1 should not overflow")
	}

	turnsPhase2 := make([]turnConfig, 20)
	for i := range turnsPhase2 {
		turnsPhase2[i] = turnConfig{
			userMessage: fmt.Sprintf("Phase2 turn %d: deeper investigation", i),
			toolCalls:   []toolCall{{name: "tool", responseSize: 10_000}},
		}
	}

	r2 := simulateSession(t, sessionConfig{
		contextWindow:    200_000,
		systemPromptSize: 2000,
		modelName:        "custom-model",
		hasUsageMetadata: true,
		tokenRatio:       2.5,
	}, turnsPhase2)

	t.Logf("200k/late-usage-phase2: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r2.turns, r2.compactions, r2.maxTokensSeen, r2.overflowed)
	if r2.overflowed {
		t.Error("phase 2 should not overflow")
	}
}

// ---------------------------------------------------------------------------
// Test: Repeated compaction stress (compaction fires multiple times)
// ---------------------------------------------------------------------------

func TestStress_8k_RepeatedCompactions(t *testing.T) {
	turns := make([]turnConfig, 40)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: longMessage(i, 400),
			toolCalls: []toolCall{
				{name: "query", responseSize: 2_000},
			},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    8_000,
		systemPromptSize: 500,
		modelName:        "small-model",
		hasUsageMetadata: true,
		tokenRatio:       1.8,
	}, turns)

	t.Logf("8k/repeated: turns=%d compactions=%d maxTokens=%d overflowed=%v looped=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed, r.loopDetected)
	if r.overflowed {
		t.Error("repeated compactions in 8k should not overflow")
	}
	if r.compactions < 3 {
		t.Errorf("expected at least 3 compactions in 40 turns with 8k window, got %d", r.compactions)
	}
}

func TestStress_200k_RepeatedCompactions(t *testing.T) {
	turns := make([]turnConfig, 60)
	for i := range turns {
		tools := []toolCall{}
		if i%2 == 0 {
			tools = []toolCall{
				{name: "fetch_data", responseSize: 30_000},
				{name: "process", responseSize: 10_000},
			}
		}
		turns[i] = turnConfig{
			userMessage: fmt.Sprintf("Turn %d: Analyze the next batch of data from the pipeline", i),
			toolCalls:   tools,
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    200_000,
		systemPromptSize: 5000,
		modelName:        "claude-sonnet",
		hasUsageMetadata: true,
		tokenRatio:       2.0,
	}, turns)

	t.Logf("200k/repeated: turns=%d compactions=%d maxTokens=%d overflowed=%v looped=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed, r.loopDetected)
	if r.overflowed {
		t.Error("repeated compactions in 200k should not overflow")
	}
	if r.compactions < 2 {
		t.Errorf("expected at least 2 compactions in 60 turns with heavy tools, got %d", r.compactions)
	}
}

// ---------------------------------------------------------------------------
// Test: Extreme edge cases
// ---------------------------------------------------------------------------

func TestStress_8k_OnlyToolResponses(t *testing.T) {
	turns := make([]turnConfig, 10)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: fmt.Sprintf("Turn %d: run", i),
			toolCalls: []toolCall{
				{name: "execute", responseSize: 5_000},
			},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    8_000,
		systemPromptSize: 200,
		modelName:        "small-model",
		hasUsageMetadata: true,
		tokenRatio:       2.0,
	}, turns)

	t.Logf("8k/tool-only: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("8k tool-only should not overflow")
	}
	if r.compactions == 0 {
		t.Error("expected compactions with 5k tool responses in 8k window")
	}
}

func TestStress_200k_VeryHighTokenRatio_4x(t *testing.T) {
	turns := make([]turnConfig, 10)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: fmt.Sprintf("Turn %d: analyze the JSON schema definition", i),
			toolCalls: []toolCall{
				{name: "read_schema", responseSize: 20_000},
			},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    200_000,
		systemPromptSize: 3000,
		modelName:        "claude-sonnet",
		hasUsageMetadata: true,
		tokenRatio:       4.0,
	}, turns)

	t.Logf("200k/4x-ratio: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("200k with 4x token ratio should not overflow")
	}
}

func TestStress_8k_RapidFireShortMessages(t *testing.T) {
	turns := make([]turnConfig, 80)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: longMessage(i, 400),
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    8_000,
		systemPromptSize: 200,
		modelName:        "small-model",
		hasUsageMetadata: true,
		tokenRatio:       1.8,
	}, turns)

	t.Logf("8k/rapid-fire: turns=%d compactions=%d maxTokens=%d overflowed=%v looped=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed, r.loopDetected)
	if r.overflowed {
		t.Error("80-turn rapid-fire should not overflow 8k")
	}
	if r.compactions < 2 {
		t.Errorf("expected at least 2 compactions in 80 rapid-fire turns, got %d", r.compactions)
	}
}

func TestStress_200k_100Turns_MixedWorkload(t *testing.T) {
	turns := make([]turnConfig, 100)
	for i := range turns {
		var tools []toolCall
		switch {
		case i%10 == 0:
			tools = []toolCall{
				{name: "big_fetch", responseSize: 50_000},
			}
		case i%5 == 0:
			tools = []toolCall{
				{name: "small_tool", responseSize: 2_000},
				{name: "medium_tool", responseSize: 5_000},
			}
		case i%3 == 0:
			tools = []toolCall{
				{name: "query", responseSize: 1_000},
			}
		}
		turns[i] = turnConfig{
			userMessage: fmt.Sprintf("Turn %d: Continue the multi-step investigation of the distributed system failure across clusters", i),
			toolCalls:   tools,
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    200_000,
		systemPromptSize: 5000,
		modelName:        "claude-sonnet",
		hasUsageMetadata: true,
		tokenRatio:       2.3,
	}, turns)

	t.Logf("200k/100turns-mixed: turns=%d compactions=%d maxTokens=%d overflowed=%v looped=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed, r.loopDetected)
	if r.overflowed {
		t.Error("100-turn mixed workload should not overflow 200k")
	}
	if r.loopDetected {
		t.Error("compaction loop detected")
	}
}

func TestStress_8k_AlternatingToolAndText(t *testing.T) {
	turns := make([]turnConfig, 30)
	for i := range turns {
		tc := turnConfig{
			userMessage: longMessage(i, 500),
		}
		if i%2 == 0 {
			tc.toolCalls = []toolCall{
				{name: "tool", responseSize: 3_000},
			}
		}
		turns[i] = tc
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    8_000,
		systemPromptSize: 400,
		modelName:        "small-model",
		hasUsageMetadata: true,
		tokenRatio:       2.0,
	}, turns)

	t.Logf("8k/alternating: turns=%d compactions=%d maxTokens=%d overflowed=%v looped=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed, r.loopDetected)
	if r.overflowed {
		t.Error("alternating tool/text should not overflow 8k")
	}
	if r.compactions < 2 {
		t.Errorf("expected at least 2 compactions, got %d", r.compactions)
	}
}
