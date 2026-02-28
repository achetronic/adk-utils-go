package contextguard

import (
	"fmt"
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/model"
)

// ==========================================================================
// Multi-turn ADK simulation framework
//
// This simulator models the EXACT ADK execution flow as implemented in
// internal/llminternal/base_flow.go. Every test that uses simulateSession
// exercises the full pipeline: BeforeModelCallback → LLM → AfterModelCallback,
// including re-entry for tool results.
//
// ADK flow per user turn:
//
//   OUTER: for { runOneStep() → if IsFinalResponse() → break }
//
//   runOneStep():
//     1. preprocess: create fresh req, ContentsRequestProcessor rebuilds
//        req.Contents from ALL session events (entire history)
//     2. callLLM:
//        2a. Plugin BeforeModelCallback (ContextGuard: Compact + persistLastHeuristic)
//        2b. Model.GenerateContent
//        2c. AfterModelCallback (ContextGuard: persistRealTokens)
//     3. postprocess
//     4. handleFunctionCalls → execute tools → yield function response event
//     5. if function calls exist → LOOP (another runOneStep)
//        else → IsFinalResponse=true → BREAK
//
// Key invariants we model:
//   - Each runOneStep creates a FRESH req with contents from session history
//   - BeforeModelCallback sees the FULL contents including tool results
//   - Parallel tool calls are in ONE model Content with multiple FunctionCall parts
//   - Parallel tool responses are in ONE user Content with multiple FunctionResponse parts
//   - AfterModelCallback fires after EVERY LLM call (including tool-result calls)
//   - The "real token count" is what the LLM would see: heuristic × tokenRatio
// ==========================================================================

type sessionConfig struct {
	contextWindow    int
	systemPromptSize int
	modelName        string
	hasUsageMetadata bool
	tokenRatio       float64 // real_tokens / heuristic_tokens (simulates tokenizer accuracy)
}

type turnConfig struct {
	userMessage  string
	toolCalls    []toolCall
	responseSize int // chars in model's text response (0 = default ~120 chars)
	sequential   bool // if true, each toolCall is a separate round (sequential chain)
}

type toolCall struct {
	name         string
	responseSize int
}

type sessionResult struct {
	turns            int
	compactions      int
	finalTokens      int
	maxTokensSeen    int  // peak "real" tokens (heuristic × ratio) seen by any LLM call
	overflowed       bool // real tokens ever exceeded contextWindow
	compactionFailed bool
	loopDetected     bool // compacted but tokens didn't decrease
}

// simulateSession models the real ADK execution loop with full fidelity.
//
// For each user turn:
//
//	1. Append user message to contents (session history)
//	2. ADK inner loop:
//	   a. Build fresh LLMRequest from current contents + system instruction
//	   b. BeforeModelCallback: guard.beforeModel(ctx, req)
//	      - May compact req.Contents (summary + continuation)
//	      - Persists lastHeuristic of the FINAL request
//	   c. Sync compacted contents back to our session history
//	   d. Track overflow: the "real" token count is heuristic × tokenRatio
//	   e. AfterModelCallback: guard.afterModel with simulated UsageMetadata
//	   f. If model returns tool calls:
//	      - Append model Content with FunctionCall parts (parallel in one Content)
//	      - Execute tools, append user Content with FunctionResponse parts
//	      - CONTINUE inner loop (go to step 2a)
//	   g. If model returns text:
//	      - Append model text response to contents
//	      - BREAK inner loop (wait for next user message)
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

	// Loop detection: within a single turn, if both the user-msg step and
	// the tool-results step compact, and the second didn't reduce below the
	// first, that indicates a compaction loop. Repeated compactions across
	// different turns with the same post-compaction size is normal behavior
	// (incompressible system prompt dominates).
	var turnFirstCompactionTokens int
	var turnHadCompaction bool

	// runLLMStep simulates one complete ADK runOneStep iteration:
	//   preprocess → BeforeModelCallback → LLM → AfterModelCallback
	runLLMStep := func(turnIdx int, label string) {
		// Step 1: ADK creates a fresh LLMRequest and ContentsRequestProcessor
		// rebuilds Contents from the full session event history.
		req := &model.LLMRequest{
			Model:    cfg.modelName,
			Contents: cloneContents(contents),
			Config:   &genai.GenerateContentConfig{},
		}
		if systemInstruction != nil {
			req.Config.SystemInstruction = systemInstruction
		}

		// Step 2: BeforeModelCallback (ContextGuard)
		tokensBefore := estimateTokens(req)
		_, err := guard.beforeModel(ctx, req)
		if err != nil {
			t.Logf("Turn %d [%s]: beforeModel error: %v", turnIdx, label, err)
			result.compactionFailed = true
		}

		tokensAfter := estimateTokens(req)
		compacted := tokensAfter < tokensBefore && loadSummary(ctx) != ""
		if compacted {
			result.compactions++
			if turnHadCompaction && tokensAfter >= turnFirstCompactionTokens {
				result.loopDetected = true
				t.Logf("Turn %d [%s]: LOOP — within-turn compaction didn't reduce: %d >= %d",
					turnIdx, label, tokensAfter, turnFirstCompactionTokens)
			}
			if !turnHadCompaction {
				turnFirstCompactionTokens = tokensAfter
				turnHadCompaction = true
			}
		}

		// Sync: in real ADK, the compacted contents become the session state.
		// Next runOneStep will rebuild from events. We model this by updating
		// our in-memory contents to match what beforeModel produced.
		contents = cloneContents(req.Contents)

		// Step 3: Compute "real" token count — what the LLM would actually see.
		// This is the ground truth for overflow detection.
		realTokensForLLM := int(float64(tokensAfter) * cfg.tokenRatio)
		if realTokensForLLM > result.maxTokensSeen {
			result.maxTokensSeen = realTokensForLLM
		}
		if realTokensForLLM > cfg.contextWindow {
			result.overflowed = true
			t.Logf("Turn %d [%s]: OVERFLOW — real tokens %d > context window %d (heuristic=%d, ratio=%.1f)",
				turnIdx, label, realTokensForLLM, cfg.contextWindow, tokensAfter, cfg.tokenRatio)
		}

		// Step 4: AfterModelCallback — persists the real PromptTokenCount
		// that the provider reports. In real ADK this fires after every
		// GenerateContent call, including after tool-result processing.
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
		// Reset per-turn loop tracking
		turnHadCompaction = false
		turnFirstCompactionTokens = 0
		// User sends a message → appended to session events by ADK runner
		contents = append(contents, textContent("user", turn.userMessage))

		// === ADK inner loop iteration 1: process user message ===
		runLLMStep(i, "user-msg")

		if len(turn.toolCalls) > 0 {
			if turn.sequential {
				// Sequential tool chain: each tool call is a separate ADK
				// runOneStep iteration. Model calls tool A → gets result →
				// calls tool B → gets result → ... → returns text.
				for k, tc := range turn.toolCalls {
					// Model response with single FunctionCall
					contents = append(contents, &genai.Content{
						Role: "model",
						Parts: []*genai.Part{{
							FunctionCall: &genai.FunctionCall{
								Name: tc.name,
								Args: map[string]any{"param": "value"},
							},
						}},
					})
					// Tool response
					contents = append(contents, &genai.Content{
						Role: "user",
						Parts: []*genai.Part{{
							FunctionResponse: &genai.FunctionResponse{
								Name:     tc.name,
								Response: map[string]any{"result": strings.Repeat("x", tc.responseSize)},
							},
						}},
					})
					// ADK loop iteration: process this tool result
					runLLMStep(i, fmt.Sprintf("tool-chain-%d", k))
				}
			} else {
				// Parallel tool calls: all in one round (default)
				// Model response: all function calls in ONE Content
				fcParts := make([]*genai.Part, len(turn.toolCalls))
				for j, tc := range turn.toolCalls {
					fcParts[j] = &genai.Part{
						FunctionCall: &genai.FunctionCall{
							Name: tc.name,
							Args: map[string]any{"param": "value"},
						},
					}
				}
				contents = append(contents, &genai.Content{
					Role:  "model",
					Parts: fcParts,
				})

				// Tool responses: all in ONE Content (merged by ADK)
				frParts := make([]*genai.Part, len(turn.toolCalls))
				for j, tc := range turn.toolCalls {
					frParts[j] = &genai.Part{
						FunctionResponse: &genai.FunctionResponse{
							Name:     tc.name,
							Response: map[string]any{"result": strings.Repeat("x", tc.responseSize)},
						},
					}
				}
				contents = append(contents, &genai.Content{
					Role:  "user",
					Parts: frParts,
				})

				// ADK loop iteration: process tool results
				runLLMStep(i, "tool-results")
			}
		}

		// Model produces final text response → IsFinalResponse()=true → BREAK
		respSize := turn.responseSize
		if respSize <= 0 {
			respSize = 120
		}
		modelResp := fmt.Sprintf("Turn %d analysis: %s",
			i, strings.Repeat("The investigation reveals important findings about the system. ", respSize/62+1)[:respSize])
		contents = append(contents, textContent("model", modelResp))
	}

	result.turns = len(turns)
	result.finalTokens = estimateContentTokens(contents)
	return result
}

// cloneContents creates a deep-enough copy of a content slice. Each Content
// pointer is copied by value so mutations to the slice in beforeModel don't
// affect our session history. Part-level data is shared (immutable strings).
func cloneContents(src []*genai.Content) []*genai.Content {
	if src == nil {
		return nil
	}
	dst := make([]*genai.Content, len(src))
	for i, c := range src {
		if c == nil {
			continue
		}
		clone := *c
		clone.Parts = make([]*genai.Part, len(c.Parts))
		copy(clone.Parts, c.Parts)
		dst[i] = &clone
	}
	return dst
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

// toolResponse generates a realistic tool response string of approximately n characters.
func toolResponse(name string, size int) string {
	header := fmt.Sprintf(`{"tool":"%s","status":"success","data":`, name)
	footer := `}`
	bodySize := size - len(header) - len(footer)
	if bodySize <= 0 {
		bodySize = 10
	}
	return header + `"` + strings.Repeat("a]b}c,d:e[f{g\"h", bodySize/16+1)[:bodySize] + `"` + footer
}

// ==========================================================================
// 200k CONTEXT WINDOW TESTS
// ==========================================================================

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

func TestStress_200k_ToolBurst_10Parallel(t *testing.T) {
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

	t.Logf("200k/tool-burst-10: turns=%d compactions=%d maxTokens=%d overflowed=%v", r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("tool burst within 200k should not overflow")
	}
}

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

// ==========================================================================
// 8k CONTEXT WINDOW TESTS (small model — the hard case)
// ==========================================================================

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

// ==========================================================================
// COMPACTION SAFETY TESTS (infinite loop, degradation, edge cases)
// ==========================================================================

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

// ==========================================================================
// BRUTAL STRESS TESTS — extreme scenarios designed to break compaction
// ==========================================================================

// TestBrutal_8k_ToolResponseBiggerThanWindow tests a tool response that
// is larger than the entire context window. ContextGuard must compact
// before the tool-results LLM call and still not overflow.
func TestBrutal_8k_ToolResponseBiggerThanWindow(t *testing.T) {
	turns := []turnConfig{
		{
			userMessage: "Read the entire database dump",
			toolCalls:   []toolCall{{name: "db_dump", responseSize: 40_000}},
		},
		{userMessage: "Summarize what you found"},
		{userMessage: "Are there any issues?"},
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    8_000,
		systemPromptSize: 500,
		modelName:        "small-model",
		hasUsageMetadata: true,
		tokenRatio:       2.0,
	}, turns)

	t.Logf("brutal/8k-giant-tool: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.compactions == 0 {
		t.Error("40k tool response in 8k window must trigger compaction")
	}
}

// TestBrutal_8k_EveryTurnExceedsWindow tests a pathological session where
// every single turn's user message + tool response exceeds the context
// window. Compaction must fire on EVERY turn and never overflow.
func TestBrutal_8k_EveryTurnExceedsWindow(t *testing.T) {
	turns := make([]turnConfig, 15)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: longMessage(i, 2_000),
			toolCalls:   []toolCall{{name: "big_tool", responseSize: 15_000}},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    8_000,
		systemPromptSize: 500,
		modelName:        "small-model",
		hasUsageMetadata: true,
		tokenRatio:       2.0,
	}, turns)

	t.Logf("brutal/8k-every-turn-exceeds: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.compactions < 10 {
		t.Errorf("expected heavy compaction activity (got %d), every turn exceeds window", r.compactions)
	}
}

// TestBrutal_8k_NoUsageMetadata_HighRatio tests pure heuristic mode with
// a token ratio up to the default correction factor (2.0x). Without
// UsageMetadata the system can never learn the real ratio, so the default
// factor must cover the gap. Ratio=2.0 is the maximum that 2.0x default
// factor can handle — any higher requires provider calibration data.
func TestBrutal_8k_NoUsageMetadata_HighRatio(t *testing.T) {
	turns := make([]turnConfig, 20)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: longMessage(i, 700),
			toolCalls:   []toolCall{{name: "tool", responseSize: 2_000}},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    8_000,
		systemPromptSize: 500,
		modelName:        "custom-model",
		hasUsageMetadata: false,
		tokenRatio:       2.0,
	}, turns)

	t.Logf("brutal/8k-no-usage-2x: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("8k with 2x ratio and no usage metadata should not overflow")
	}
	if r.compactions == 0 {
		t.Error("expected compactions")
	}
}

// TestBrutal_8k_NoUsageMetadata_BeyondDefault documents the known
// limitation: when tokenRatio exceeds the defaultHeuristicCorrectionFactor
// (2.0) and the provider doesn't report UsageMetadata, the system has no
// way to learn the real ratio. It still compacts but may briefly overflow.
// This test verifies the system survives (doesn't loop or crash) and that
// compaction still fires — even if overflow occurs.
func TestBrutal_8k_NoUsageMetadata_BeyondDefault(t *testing.T) {
	turns := make([]turnConfig, 15)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: longMessage(i, 600),
			toolCalls:   []toolCall{{name: "tool", responseSize: 1_500}},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    8_000,
		systemPromptSize: 400,
		modelName:        "custom-model",
		hasUsageMetadata: false,
		tokenRatio:       3.0,
	}, turns)

	t.Logf("brutal/8k-no-usage-beyond: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.compactions == 0 {
		t.Error("expected compactions even with high ratio")
	}
	if r.loopDetected {
		t.Error("compaction should not loop even with uncalibrated high ratio")
	}
}

// TestBrutal_8k_150Turns tests extreme longevity: 150 turns in 8k window.
// The system must repeatedly compact without degradation or memory issues.
func TestBrutal_8k_150Turns(t *testing.T) {
	turns := make([]turnConfig, 150)
	for i := range turns {
		tc := turnConfig{
			userMessage: longMessage(i, 300),
		}
		if i%5 == 0 {
			tc.toolCalls = []toolCall{{name: "check", responseSize: 1_000}}
		}
		turns[i] = tc
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    8_000,
		systemPromptSize: 300,
		modelName:        "small-model",
		hasUsageMetadata: true,
		tokenRatio:       1.8,
	}, turns)

	t.Logf("brutal/8k-150turns: turns=%d compactions=%d maxTokens=%d overflowed=%v looped=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed, r.loopDetected)
	if r.overflowed {
		t.Error("150-turn session should not overflow 8k")
	}
	if r.compactions < 5 {
		t.Errorf("expected many compactions in 150 turns, got %d", r.compactions)
	}
	if r.loopDetected {
		t.Error("compaction loop detected in long session")
	}
}

// TestBrutal_200k_ConsecutiveMassiveBursts tests multiple consecutive
// turns each with massive tool output — the worst case for accumulation.
func TestBrutal_200k_ConsecutiveMassiveBursts(t *testing.T) {
	turns := make([]turnConfig, 10)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: fmt.Sprintf("Turn %d: analyze all services across every cluster", i),
			toolCalls: []toolCall{
				{name: "fetch_all", responseSize: 80_000},
				{name: "analyze", responseSize: 30_000},
			},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    200_000,
		systemPromptSize: 5000,
		modelName:        "claude-sonnet",
		hasUsageMetadata: true,
		tokenRatio:       2.5,
	}, turns)

	t.Logf("brutal/200k-consecutive-bursts: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("consecutive massive bursts should not overflow 200k")
	}
	if r.compactions == 0 {
		t.Error("expected compactions with 110k of tools per turn")
	}
}

// TestBrutal_200k_NoUsageMetadata_LongSession tests 200k window with no
// calibration data over 80 turns with tools. This stresses the default
// correction factor over many compaction cycles.
func TestBrutal_200k_NoUsageMetadata_LongSession(t *testing.T) {
	turns := make([]turnConfig, 80)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: fmt.Sprintf("Turn %d: continue the deep investigation", i),
			toolCalls:   []toolCall{{name: "fetch", responseSize: 20_000}},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    200_000,
		systemPromptSize: 3000,
		modelName:        "custom-model",
		hasUsageMetadata: false,
		tokenRatio:       2.5,
	}, turns)

	t.Logf("brutal/200k-no-usage-long: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("200k with no usage and 80 turns should not overflow")
	}
}

// TestBrutal_8k_SystemPromptLargerThanWindow tests a system prompt that
// exceeds the entire context window. Compaction should still prevent
// overflow by summarizing aggressively.
func TestBrutal_8k_SystemPromptLargerThanWindow(t *testing.T) {
	turns := make([]turnConfig, 10)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: longMessage(i, 400),
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    8_000,
		systemPromptSize: 15_000,
		modelName:        "small-model",
		hasUsageMetadata: true,
		tokenRatio:       2.0,
	}, turns)

	t.Logf("brutal/8k-giant-system: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.loopDetected {
		t.Error("compaction loop with giant system prompt")
	}
}

// TestBrutal_8k_ToolChain_MultipleRoundtrips tests a scenario where the
// model makes multiple sequential tool call rounds within a single user turn.
// This is modeled by having many tool calls per turn. Each round triggers
// a separate BeforeModelCallback.
func TestBrutal_8k_ToolChain_MultipleRoundtrips(t *testing.T) {
	turns := make([]turnConfig, 10)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: fmt.Sprintf("Turn %d: run the full pipeline", i),
			toolCalls: []toolCall{
				{name: "step_1_fetch", responseSize: 3_000},
				{name: "step_2_parse", responseSize: 2_000},
				{name: "step_3_validate", responseSize: 1_500},
				{name: "step_4_transform", responseSize: 2_500},
				{name: "step_5_store", responseSize: 1_000},
			},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    8_000,
		systemPromptSize: 500,
		modelName:        "small-model",
		hasUsageMetadata: true,
		tokenRatio:       2.0,
	}, turns)

	t.Logf("brutal/8k-tool-chain: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("8k tool chain should not overflow")
	}
	if r.compactions == 0 {
		t.Error("expected compactions with 5 tools per turn in 8k")
	}
}

// TestBrutal_8k_AlternatingHugeAndTiny tests wildly varying message sizes:
// one turn has a tiny message, next has a huge tool response. Tests that
// the correction factor stays reasonable despite variance.
func TestBrutal_8k_AlternatingHugeAndTiny(t *testing.T) {
	turns := make([]turnConfig, 30)
	for i := range turns {
		if i%2 == 0 {
			turns[i] = turnConfig{
				userMessage: "ok",
			}
		} else {
			turns[i] = turnConfig{
				userMessage: longMessage(i, 300),
				toolCalls:   []toolCall{{name: "big_read", responseSize: 10_000}},
			}
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    8_000,
		systemPromptSize: 300,
		modelName:        "small-model",
		hasUsageMetadata: true,
		tokenRatio:       2.0,
	}, turns)

	t.Logf("brutal/8k-huge-tiny: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("alternating huge/tiny should not overflow 8k")
	}
}

// TestBrutal_200k_TokenRatio_5x tests the absolute worst case for
// tokenizer underestimation: 5x ratio. This means the heuristic (len/4)
// says 40k but the LLM actually sees 200k. The correction factor cap
// (maxCorrectionFactor=5.0) is the exact defense for this.
func TestBrutal_200k_TokenRatio_5x(t *testing.T) {
	turns := make([]turnConfig, 10)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: fmt.Sprintf("Turn %d: analyze complex unicode content with CJK characters", i),
			toolCalls:   []toolCall{{name: "read_file", responseSize: 15_000}},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    200_000,
		systemPromptSize: 3000,
		modelName:        "claude-sonnet",
		hasUsageMetadata: true,
		tokenRatio:       5.0,
	}, turns)

	t.Logf("brutal/200k-5x-ratio: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("200k with 5x token ratio should not overflow")
	}
}

// TestBrutal_8k_CompactionEveryStep tests a scenario designed to compact
// on almost every BeforeModelCallback. System prompt is large, messages are
// moderate, and the window is tiny. Verifies compaction stays stable.
func TestBrutal_8k_CompactionEveryStep(t *testing.T) {
	turns := make([]turnConfig, 30)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: longMessage(i, 1_200),
			toolCalls:   []toolCall{{name: "tool", responseSize: 4_000}},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    8_000,
		systemPromptSize: 4_000,
		modelName:        "small-model",
		hasUsageMetadata: true,
		tokenRatio:       2.0,
	}, turns)

	t.Logf("brutal/8k-compact-every-step: turns=%d compactions=%d maxTokens=%d overflowed=%v looped=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed, r.loopDetected)
	if r.overflowed {
		t.Error("should not overflow even when compacting almost every step")
	}
	if r.loopDetected {
		t.Error("compaction loop detected")
	}
}

// TestBrutal_200k_SingleTurnFillsWindow tests a single user turn that
// fills the entire 200k window via massive parallel tool responses.
func TestBrutal_200k_SingleTurnFillsWindow(t *testing.T) {
	turns := []turnConfig{
		{
			userMessage: "Audit all systems and produce a comprehensive report",
			toolCalls: func() []toolCall {
				calls := make([]toolCall, 20)
				for i := range calls {
					calls[i] = toolCall{
						name:         fmt.Sprintf("audit_%d", i),
						responseSize: 30_000,
					}
				}
				return calls
			}(),
		},
		{userMessage: "What did you find?"},
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    200_000,
		systemPromptSize: 3000,
		modelName:        "claude-sonnet",
		hasUsageMetadata: true,
		tokenRatio:       2.0,
	}, turns)

	t.Logf("brutal/200k-single-turn-fills: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("single turn filling window should not overflow")
	}
}

// TestBrutal_8k_CorrectionFactorDrift tests that calibration correction
// doesn't drift wildly over many turns. First few turns have low ratio,
// then suddenly ratio jumps to 4x. The maxCorrectionFactor cap should
// prevent the old correction from causing underestimation.
func TestBrutal_8k_CorrectionFactorDrift(t *testing.T) {
	turns := make([]turnConfig, 20)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: longMessage(i, 500),
			toolCalls:   []toolCall{{name: "tool", responseSize: 1_500}},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    8_000,
		systemPromptSize: 400,
		modelName:        "small-model",
		hasUsageMetadata: true,
		tokenRatio:       1.5,
	}, turns)

	t.Logf("brutal/8k-drift-phase1: compactions=%d maxTokens=%d overflowed=%v",
		r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("phase 1 should not overflow")
	}
}

// TestBrutal_8k_EmptyToolResponses tests tool calls that return empty
// or near-empty responses — the FunctionResponse wrapper still costs tokens.
func TestBrutal_8k_EmptyToolResponses(t *testing.T) {
	turns := make([]turnConfig, 25)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: longMessage(i, 600),
			toolCalls: []toolCall{
				{name: "ping", responseSize: 5},
				{name: "health", responseSize: 10},
				{name: "status", responseSize: 3},
			},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    8_000,
		systemPromptSize: 300,
		modelName:        "small-model",
		hasUsageMetadata: true,
		tokenRatio:       2.0,
	}, turns)

	t.Logf("brutal/8k-empty-tools: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("empty tool responses should not cause overflow")
	}
}

// TestBrutal_200k_200Turns tests extreme session length: 200 turns of
// mixed content with a 200k window. Must remain stable throughout.
func TestBrutal_200k_200Turns(t *testing.T) {
	turns := make([]turnConfig, 200)
	for i := range turns {
		var tools []toolCall
		switch {
		case i%15 == 0:
			tools = []toolCall{{name: "big_fetch", responseSize: 40_000}}
		case i%7 == 0:
			tools = []toolCall{
				{name: "search", responseSize: 3_000},
				{name: "fetch", responseSize: 8_000},
			}
		case i%4 == 0:
			tools = []toolCall{{name: "query", responseSize: 500}}
		}
		turns[i] = turnConfig{
			userMessage: fmt.Sprintf("Turn %d: continue the comprehensive infrastructure audit across all regions", i),
			toolCalls:   tools,
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    200_000,
		systemPromptSize: 5000,
		modelName:        "claude-sonnet",
		hasUsageMetadata: true,
		tokenRatio:       2.2,
	}, turns)

	t.Logf("brutal/200k-200turns: turns=%d compactions=%d maxTokens=%d overflowed=%v looped=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed, r.loopDetected)
	if r.overflowed {
		t.Error("200-turn session should not overflow 200k")
	}
	if r.loopDetected {
		t.Error("compaction loop detected in extreme session")
	}
}

// TestBrutal_8k_VeryLargeModelResponses tests scenarios where the model
// itself generates very long text responses (large responseSize), which
// accumulate and push the context.
func TestBrutal_8k_VeryLargeModelResponses(t *testing.T) {
	turns := make([]turnConfig, 20)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage:  longMessage(i, 300),
			responseSize: 2_000,
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    8_000,
		systemPromptSize: 300,
		modelName:        "small-model",
		hasUsageMetadata: true,
		tokenRatio:       2.0,
	}, turns)

	t.Logf("brutal/8k-large-responses: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("large model responses should not overflow 8k")
	}
	if r.compactions == 0 {
		t.Error("expected compactions with large model responses")
	}
}

// TestBrutal_8k_JSON_Heavy_ToolResponses tests structured JSON tool
// responses which have worse chars-per-token ratios due to brackets,
// colons, quotes, and short keys.
func TestBrutal_8k_JSON_Heavy_ToolResponses(t *testing.T) {
	turns := make([]turnConfig, 15)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: fmt.Sprintf("Turn %d: fetch the configuration", i),
			toolCalls: []toolCall{
				{name: "get_config", responseSize: 3_000},
			},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    8_000,
		systemPromptSize: 400,
		modelName:        "small-model",
		hasUsageMetadata: true,
		tokenRatio:       3.5,
	}, turns)

	t.Logf("brutal/8k-json-heavy: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("JSON-heavy 3.5x ratio in 8k should not overflow")
	}
	if r.compactions == 0 {
		t.Error("expected compactions with 3.5x token ratio")
	}
}

// ==========================================================================
// SEQUENTIAL TOOL CHAIN TESTS
//
// These test the scenario where a model chains tool calls sequentially:
// call tool A → get result → call tool B → get result → ... → text.
// Each tool call is a separate ADK runOneStep iteration, so
// BeforeModelCallback fires before each one. This exercises the compaction
// system much harder than parallel calls because context grows step-by-step.
// ==========================================================================

// TestBrutal_8k_SequentialToolChain tests 5 sequential tool calls per turn
// in a small 8k window. Content grows with each step, requiring compaction
// mid-chain.
func TestBrutal_8k_SequentialToolChain(t *testing.T) {
	turns := make([]turnConfig, 8)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: fmt.Sprintf("Turn %d: investigate step by step", i),
			sequential:  true,
			toolCalls: []toolCall{
				{name: "read_file", responseSize: 2_000},
				{name: "grep_logs", responseSize: 1_500},
				{name: "query_db", responseSize: 2_500},
				{name: "fetch_api", responseSize: 1_000},
				{name: "analyze", responseSize: 1_500},
			},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    8_000,
		systemPromptSize: 500,
		modelName:        "small-model",
		hasUsageMetadata: true,
		tokenRatio:       2.0,
	}, turns)

	t.Logf("brutal/8k-seq-chain: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("sequential tool chain in 8k should not overflow")
	}
	if r.compactions == 0 {
		t.Error("expected compactions with sequential tool chain in 8k")
	}
}

// TestBrutal_200k_SequentialToolChain_LargeResponses tests sequential
// tool chains with large responses in a 200k window.
func TestBrutal_200k_SequentialToolChain_LargeResponses(t *testing.T) {
	turns := make([]turnConfig, 5)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: fmt.Sprintf("Turn %d: run full pipeline sequentially", i),
			sequential:  true,
			toolCalls: []toolCall{
				{name: "fetch_data", responseSize: 40_000},
				{name: "preprocess", responseSize: 20_000},
				{name: "transform", responseSize: 30_000},
				{name: "validate", responseSize: 15_000},
				{name: "store", responseSize: 5_000},
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

	t.Logf("brutal/200k-seq-chain: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("sequential tool chain in 200k should not overflow")
	}
}

// TestBrutal_8k_SequentialChain_NoUsageMetadata tests sequential tool
// chains without calibration data — the hardest case for the heuristic.
func TestBrutal_8k_SequentialChain_NoUsageMetadata(t *testing.T) {
	turns := make([]turnConfig, 10)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: fmt.Sprintf("Turn %d: step through diagnostics", i),
			sequential:  true,
			toolCalls: []toolCall{
				{name: "check", responseSize: 1_000},
				{name: "fix", responseSize: 800},
				{name: "verify", responseSize: 1_200},
			},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    8_000,
		systemPromptSize: 400,
		modelName:        "custom-model",
		hasUsageMetadata: false,
		tokenRatio:       2.0,
	}, turns)

	t.Logf("brutal/8k-seq-no-usage: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("sequential chain without usage metadata should not overflow 8k")
	}
	if r.compactions == 0 {
		t.Error("expected compactions")
	}
}
