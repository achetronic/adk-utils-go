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
	tokenRatio       float64      // real_tokens / heuristic_tokens (simulates tokenizer accuracy)
	tools            []*genai.Tool // tool definitions attached to every LLM request
}

type turnConfig struct {
	userMessage  string
	toolCalls    []toolCall
	responseSize int  // chars in model's text response (0 = default ~120 chars)
	sequential   bool // if true, each toolCall is a separate round (sequential chain)
	inlineData   []inlineAttachment // inline blobs attached to the user message
}

type inlineAttachment struct {
	mimeType string
	size     int // bytes of data
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

	// Loop detection: a compaction loop is when compaction fires but produces
	// a result that is NOT smaller than the input (compaction had zero effect).
	// This would happen if the summary is as large as the original conversation.
	// We track this per-step, not per-turn, since each step independently
	// rebuilds from session events and applies injectSummary.

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
		if len(cfg.tools) > 0 {
			req.Config.Tools = cfg.tools
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
			if tokensAfter >= tokensBefore {
				result.loopDetected = true
				t.Logf("Turn %d [%s]: LOOP — compaction had no effect: %d >= %d",
					turnIdx, label, tokensAfter, tokensBefore)
			}
		}

		// NOTE: In real ADK, session events are append-only. The compacted
		// req.Contents only affects this LLM call. ContentsRequestProcessor
		// will rebuild from ALL events next time, and injectSummary strips
		// the already-summarized events using the watermark. We do NOT
		// update `contents` here — it represents the immutable event history.

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
		// User sends a message → appended to session events by ADK runner
		userParts := []*genai.Part{{Text: turn.userMessage}}
		for _, att := range turn.inlineData {
			userParts = append(userParts, &genai.Part{
				InlineData: &genai.Blob{
					MIMEType: att.mimeType,
					Data:     make([]byte, att.size),
				},
			})
		}
		contents = append(contents, &genai.Content{Role: "user", Parts: userParts})

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

// makeMCPTools generates a set of realistic tool definitions simulating an MCP
// server that exposes n tools, each with a JSON schema of approximately
// schemaSize characters.
func makeMCPTools(n, schemaSize int) []*genai.Tool {
	var decls []*genai.FunctionDeclaration
	for i := range n {
		props := make(map[string]any)
		propCount := schemaSize / 120
		if propCount < 1 {
			propCount = 1
		}
		for j := range propCount {
			props[fmt.Sprintf("param_%d", j)] = map[string]any{
				"type":        "string",
				"description": fmt.Sprintf("Parameter %d for tool %d. %s", j, i, strings.Repeat("Detailed description. ", schemaSize/(propCount*60)+1)[:60]),
			}
		}
		decls = append(decls, &genai.FunctionDeclaration{
			Name:        fmt.Sprintf("mcp_tool_%d", i),
			Description: fmt.Sprintf("Tool %d from MCP server. %s", i, strings.Repeat("Performs complex operations on the system. ", schemaSize/100+1)[:100]),
			ParametersJsonSchema: map[string]any{
				"type":       "object",
				"properties": props,
				"required":   []string{"param_0"},
			},
		})
	}
	return []*genai.Tool{{FunctionDeclarations: decls}}
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

// ==========================================================================
// TOOL DEFINITIONS TESTS — verify that tool schemas are counted in the
// heuristic and compaction fires before overflow.
// ==========================================================================

// TestStress_200k_HeavyToolDefinitions tests a 200k window where 20 MCP tools
// with large schemas (~2k each) are attached to every request. The tool
// definitions alone consume ~10k heuristic tokens, and with the tokenRatio
// the real cost is ~20k. Without counting tools, the heuristic would miss
// this overhead entirely.
func TestStress_200k_HeavyToolDefinitions(t *testing.T) {
	tools := makeMCPTools(20, 2_000)
	turns := make([]turnConfig, 30)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: fmt.Sprintf("Turn %d: Use the MCP tools to inspect the infrastructure", i),
			toolCalls: []toolCall{
				{name: "mcp_tool_0", responseSize: 5_000},
				{name: "mcp_tool_5", responseSize: 8_000},
			},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    200_000,
		systemPromptSize: 3_000,
		modelName:        "claude-sonnet-4-6",
		hasUsageMetadata: true,
		tokenRatio:       2.0,
		tools:            tools,
	}, turns)

	t.Logf("200k/heavy-tool-defs: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("200k with heavy tool definitions should not overflow")
	}
	if r.compactions == 0 {
		t.Error("expected at least one compaction with heavy tools + responses")
	}
}

// TestStress_8k_HeavyToolDefinitions tests a tiny 8k window where 10 MCP
// tools with ~1k schemas each are attached. The tools alone eat ~2.5k
// heuristic tokens (~5k real), leaving very little room for conversation.
func TestStress_8k_HeavyToolDefinitions(t *testing.T) {
	tools := makeMCPTools(10, 1_000)
	turns := make([]turnConfig, 20)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: fmt.Sprintf("Turn %d: run diagnostics", i),
			toolCalls:   []toolCall{{name: "mcp_tool_0", responseSize: 1_500}},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    8_000,
		systemPromptSize: 400,
		modelName:        "custom-model",
		hasUsageMetadata: true,
		tokenRatio:       2.0,
		tools:            tools,
	}, turns)

	t.Logf("8k/heavy-tool-defs: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("8k with heavy tool definitions should not overflow")
	}
	if r.compactions == 0 {
		t.Error("expected compactions in 8k with large tool schemas")
	}
}

// TestBrutal_200k_ToolDefinitionsDominateWindow tests the extreme case where
// tool definitions are so large they consume a significant portion of the
// context window. 50 tools × 4k schema each ≈ 50k heuristic tokens from
// tools alone. The heuristic must account for these to avoid overflow.
func TestBrutal_200k_ToolDefinitionsDominateWindow(t *testing.T) {
	tools := makeMCPTools(50, 4_000)
	turns := make([]turnConfig, 20)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: fmt.Sprintf("Turn %d: analyze the cluster state comprehensively", i),
			toolCalls: []toolCall{
				{name: "mcp_tool_0", responseSize: 10_000},
				{name: "mcp_tool_10", responseSize: 15_000},
				{name: "mcp_tool_20", responseSize: 5_000},
			},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    200_000,
		systemPromptSize: 5_000,
		modelName:        "claude-sonnet-4-6",
		hasUsageMetadata: true,
		tokenRatio:       2.5,
		tools:            tools,
	}, turns)

	t.Logf("brutal/200k-tool-defs-dominate: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("200k with dominating tool definitions should not overflow")
	}
}

// TestBrutal_8k_ToolDefinitionsNoUsageMetadata tests the worst case: large
// tool schemas on a small window with no calibration data. The default
// correction factor must compensate for both content underestimation and
// the tool overhead.
func TestBrutal_8k_ToolDefinitionsNoUsageMetadata(t *testing.T) {
	tools := makeMCPTools(8, 800)
	turns := make([]turnConfig, 15)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: fmt.Sprintf("Turn %d: check system status", i),
			toolCalls:   []toolCall{{name: "mcp_tool_0", responseSize: 1_000}},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    8_000,
		systemPromptSize: 300,
		modelName:        "custom-model",
		hasUsageMetadata: false,
		tokenRatio:       2.0,
		tools:            tools,
	}, turns)

	t.Logf("brutal/8k-tool-defs-no-usage: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("8k with tool definitions and no usage metadata should not overflow")
	}
}

// TestBrutal_200k_ToolDefinitionsHighRatio tests high token ratio (3.5x) with
// large tool schemas. Simulates JSON-heavy tool definitions where the
// tokenizer is particularly aggressive (many structural tokens for brackets,
// colons, quotes).
func TestBrutal_200k_ToolDefinitionsHighRatio(t *testing.T) {
	tools := makeMCPTools(30, 3_000)
	turns := make([]turnConfig, 15)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: fmt.Sprintf("Turn %d: deep analysis with all tools", i),
			toolCalls: []toolCall{
				{name: "mcp_tool_0", responseSize: 20_000},
				{name: "mcp_tool_15", responseSize: 10_000},
			},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    200_000,
		systemPromptSize: 4_000,
		modelName:        "claude-sonnet-4-6",
		hasUsageMetadata: true,
		tokenRatio:       3.5,
		tools:            tools,
	}, turns)

	t.Logf("brutal/200k-tool-defs-high-ratio: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("200k with tool definitions and 3.5x ratio should not overflow")
	}
}

// ==========================================================================
// INLINE DATA TESTS — verify that InlineData (images, PDFs, audio, video)
// is counted in the heuristic.
// ==========================================================================

// TestStress_200k_InlineImages tests a conversation where the user sends
// images (e.g. screenshots for visual debugging). Each image is ~100KB
// base64, which is ~25k heuristic tokens per image.
func TestStress_200k_InlineImages(t *testing.T) {
	turns := make([]turnConfig, 15)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: fmt.Sprintf("Turn %d: Here's a screenshot of the error, what do you think?", i),
			inlineData:  []inlineAttachment{{mimeType: "image/png", size: 100_000}},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    200_000,
		systemPromptSize: 2_000,
		modelName:        "claude-sonnet-4-6",
		hasUsageMetadata: true,
		tokenRatio:       2.0,
	}, turns)

	t.Logf("200k/inline-images: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("200k with inline images should not overflow")
	}
	if r.compactions == 0 {
		t.Error("expected compactions with 100KB images per turn")
	}
}

// TestStress_8k_InlineSmallImages tests inline images on a small context
// window. Even a 10KB image (~2.5k heuristic tokens) fills a significant
// portion of an 8k window.
func TestStress_8k_InlineSmallImages(t *testing.T) {
	turns := make([]turnConfig, 10)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: fmt.Sprintf("Turn %d: check this small diagram", i),
			inlineData:  []inlineAttachment{{mimeType: "image/png", size: 10_000}},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    8_000,
		systemPromptSize: 200,
		modelName:        "custom-model",
		hasUsageMetadata: true,
		tokenRatio:       2.0,
	}, turns)

	t.Logf("8k/inline-small-images: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("8k with small inline images should not overflow")
	}
}

// TestBrutal_200k_LargeInlineDocuments tests large inline documents (e.g.
// PDFs converted to base64). A 500KB PDF is ~125k heuristic tokens — a
// single attachment can nearly fill a 200k window.
func TestBrutal_200k_LargeInlineDocuments(t *testing.T) {
	turns := []turnConfig{
		{
			userMessage: "Analyze this PDF report",
			inlineData:  []inlineAttachment{{mimeType: "application/pdf", size: 500_000}},
		},
		{
			userMessage: "Now compare with this second document",
			inlineData:  []inlineAttachment{{mimeType: "application/pdf", size: 300_000}},
		},
		{
			userMessage: "Summarize the key differences",
		},
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    200_000,
		systemPromptSize: 2_000,
		modelName:        "claude-sonnet-4-6",
		hasUsageMetadata: true,
		tokenRatio:       2.0,
	}, turns)

	t.Logf("brutal/200k-large-inline-docs: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("200k with large inline PDFs should not overflow")
	}
}

// TestBrutal_200k_MultipleInlinePerTurn tests turns with multiple inline
// attachments — simulating a user pasting several screenshots at once.
func TestBrutal_200k_MultipleInlinePerTurn(t *testing.T) {
	turns := make([]turnConfig, 8)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: fmt.Sprintf("Turn %d: Here are 3 screenshots from different pages", i),
			inlineData: []inlineAttachment{
				{mimeType: "image/png", size: 80_000},
				{mimeType: "image/jpeg", size: 60_000},
				{mimeType: "image/png", size: 90_000},
			},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    200_000,
		systemPromptSize: 2_000,
		modelName:        "claude-sonnet-4-6",
		hasUsageMetadata: true,
		tokenRatio:       2.0,
	}, turns)

	t.Logf("brutal/200k-multi-inline: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("200k with multiple inline attachments per turn should not overflow")
	}
	if r.compactions == 0 {
		t.Error("expected compactions with ~230KB of inline data per turn")
	}
}

// ==========================================================================
// COMBINED: TOOL DEFINITIONS + INLINE DATA + TOOL RESPONSES
// ==========================================================================

// TestBrutal_200k_ToolDefsAndInlineImages tests the production scenario that
// caused the original bug: an agent with tool definitions sending inline
// images (e.g. a Telegram bot with MCP tools where users paste screenshots).
// All three blind spots are exercised simultaneously.
func TestBrutal_200k_ToolDefsAndInlineImages(t *testing.T) {
	tools := makeMCPTools(15, 1_500)
	turns := make([]turnConfig, 20)
	for i := range turns {
		tc := turnConfig{
			userMessage: fmt.Sprintf("Turn %d: look at this and use the tools to fix it", i),
			toolCalls: []toolCall{
				{name: "mcp_tool_0", responseSize: 5_000},
			},
		}
		if i%3 == 0 {
			tc.inlineData = []inlineAttachment{{mimeType: "image/png", size: 50_000}}
		}
		turns[i] = tc
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    200_000,
		systemPromptSize: 3_000,
		modelName:        "claude-sonnet-4-6",
		hasUsageMetadata: true,
		tokenRatio:       2.5,
		tools:            tools,
	}, turns)

	t.Logf("brutal/200k-tools-and-images: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("200k with tool definitions + inline images should not overflow")
	}
}

// TestBrutal_200k_ToolDefsAndInlineNoUsageMetadata is the hardest variant:
// tool definitions + inline data + no UsageMetadata. The heuristic must
// handle everything without calibration.
func TestBrutal_200k_ToolDefsAndInlineNoUsageMetadata(t *testing.T) {
	tools := makeMCPTools(10, 1_200)
	turns := make([]turnConfig, 15)
	for i := range turns {
		tc := turnConfig{
			userMessage: fmt.Sprintf("Turn %d: analyze this image with the tools", i),
			toolCalls:   []toolCall{{name: "mcp_tool_0", responseSize: 3_000}},
		}
		if i%2 == 0 {
			tc.inlineData = []inlineAttachment{{mimeType: "image/jpeg", size: 30_000}}
		}
		turns[i] = tc
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    200_000,
		systemPromptSize: 2_000,
		modelName:        "claude-sonnet-4-6",
		hasUsageMetadata: false,
		tokenRatio:       2.0,
		tools:            tools,
	}, turns)

	t.Logf("brutal/200k-tools-inline-no-usage: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("200k with tool definitions + inline + no usage should not overflow")
	}
}

// TestBrutal_8k_ToolDefsAndInlineCombined tests the absolute worst case on
// a small window: tool schemas + inline data + tool responses + no usage
// metadata. Every blind spot exercised at once on an 8k window.
func TestBrutal_8k_ToolDefsAndInlineCombined(t *testing.T) {
	tools := makeMCPTools(5, 600)
	turns := make([]turnConfig, 20)
	for i := range turns {
		tc := turnConfig{
			userMessage: fmt.Sprintf("Turn %d: fix this", i),
			toolCalls:   []toolCall{{name: "mcp_tool_0", responseSize: 800}},
		}
		if i%4 == 0 {
			tc.inlineData = []inlineAttachment{{mimeType: "image/png", size: 5_000}}
		}
		turns[i] = tc
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    8_000,
		systemPromptSize: 300,
		modelName:        "custom-model",
		hasUsageMetadata: false,
		tokenRatio:       2.0,
		tools:            tools,
	}, turns)

	t.Logf("brutal/8k-all-blind-spots: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("8k with all blind spots combined should not overflow")
	}
}

// ==========================================================================
// REAL-WORLD USE CASE TESTS — production patterns from singleshot tests
// ported to the multi-turn simulator for full ADK flow validation.
//
// These replicate the exact tool patterns from kubeAgentConversation,
// mixedConversation, buildCodingAgentConversation, and pureToolStorm
// but exercise them across many turns with compaction, calibration,
// and re-compaction.
// ==========================================================================

// --- Kubernetes Agent Pattern ---
// Each turn: user asks → model calls kubectl_get_pods (huge JSON) →
// kubectl_describe_pod (huge text) → kubectl_get_logs (huge text) → model responds.
// Matches kubeAgentConversation from singleshot tests.

func TestStress_200k_KubeAgent(t *testing.T) {
	turns := make([]turnConfig, 20)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: fmt.Sprintf("Turn %d: Check the status of pods in the production namespace and investigate any failures", i),
			toolCalls: []toolCall{
				{name: "kubectl_get_pods", responseSize: 15_000},
				{name: "kubectl_describe_pod", responseSize: 10_000},
				{name: "kubectl_get_logs", responseSize: 25_000},
			},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    200_000,
		systemPromptSize: 5_000,
		modelName:        "claude-sonnet-4-6",
		hasUsageMetadata: true,
		tokenRatio:       2.5,
	}, turns)

	t.Logf("200k/kube-agent: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("200k kube agent should not overflow")
	}
	if r.compactions == 0 {
		t.Error("expected compactions with 50k of kubectl output per turn")
	}
}

func TestStress_8k_KubeAgent(t *testing.T) {
	turns := make([]turnConfig, 10)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: fmt.Sprintf("Turn %d: Get pod status", i),
			toolCalls: []toolCall{
				{name: "kubectl_get_pods", responseSize: 5_000},
				{name: "kubectl_describe_pod", responseSize: 3_000},
				{name: "kubectl_get_logs", responseSize: 8_000},
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

	t.Logf("8k/kube-agent: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("8k kube agent should not overflow")
	}
	if r.compactions == 0 {
		t.Error("expected compactions with 16k of kubectl output per turn in 8k window")
	}
}

func TestBrutal_200k_KubeAgent_30Rounds(t *testing.T) {
	turns := make([]turnConfig, 30)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: fmt.Sprintf("Turn %d: Deep investigation - check pods, describe failures, get full logs from all containers", i),
			toolCalls: []toolCall{
				{name: "kubectl_get_pods", responseSize: 20_000},
				{name: "kubectl_describe_pod", responseSize: 15_000},
				{name: "kubectl_get_logs", responseSize: 40_000},
			},
			responseSize: 500,
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    200_000,
		systemPromptSize: 5_000,
		modelName:        "claude-sonnet-4-6",
		hasUsageMetadata: true,
		tokenRatio:       2.5,
	}, turns)

	t.Logf("brutal/200k-kube-30rounds: turns=%d compactions=%d maxTokens=%d overflowed=%v looped=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed, r.loopDetected)
	if r.overflowed {
		t.Error("200k kube agent 30 rounds should not overflow")
	}
	if r.loopDetected {
		t.Error("compaction loop detected")
	}
}

// --- Mixed Debugging Session Pattern ---
// Interleaved text, small HTTP health checks, big Prometheus queries, SQL queries.
// Matches mixedConversation from singleshot tests.

func TestStress_200k_MixedDebugSession(t *testing.T) {
	turns := make([]turnConfig, 25)
	for i := range turns {
		var tools []toolCall
		switch {
		case i%7 == 0:
			tools = []toolCall{
				{name: "prometheus_query", responseSize: 20_000},
				{name: "sql_query", responseSize: 15_000},
			}
		case i%4 == 0:
			tools = []toolCall{
				{name: "http_get_health", responseSize: 500},
			}
		case i%3 == 0:
			tools = []toolCall{
				{name: "grep_logs", responseSize: 8_000},
			}
		}
		turns[i] = turnConfig{
			userMessage: fmt.Sprintf("Turn %d: The API is returning 500 errors intermittently. Let me check the metrics and database state.", i),
			toolCalls:   tools,
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    200_000,
		systemPromptSize: 3_000,
		modelName:        "claude-sonnet-4-6",
		hasUsageMetadata: true,
		tokenRatio:       2.2,
	}, turns)

	t.Logf("200k/mixed-debug: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("200k mixed debug session should not overflow")
	}
}

func TestStress_8k_MixedDebugSession(t *testing.T) {
	turns := make([]turnConfig, 20)
	for i := range turns {
		var tools []toolCall
		switch {
		case i%5 == 0:
			tools = []toolCall{
				{name: "prometheus_query", responseSize: 3_000},
				{name: "sql_query", responseSize: 2_000},
			}
		case i%3 == 0:
			tools = []toolCall{
				{name: "http_get_health", responseSize: 200},
			}
		}
		turns[i] = turnConfig{
			userMessage: fmt.Sprintf("Turn %d: Check the error rate from Prometheus and verify the DB connection pool", i),
			toolCalls:   tools,
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    8_000,
		systemPromptSize: 500,
		modelName:        "small-model",
		hasUsageMetadata: true,
		tokenRatio:       1.8,
	}, turns)

	t.Logf("8k/mixed-debug: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("8k mixed debug session should not overflow")
	}
}

func TestBrutal_200k_MixedDebugSession_LongInvestigation(t *testing.T) {
	turns := make([]turnConfig, 50)
	for i := range turns {
		var tools []toolCall
		switch {
		case i%10 == 0:
			tools = []toolCall{
				{name: "prometheus_query", responseSize: 30_000},
				{name: "grafana_dashboard", responseSize: 10_000},
			}
		case i%6 == 0:
			tools = []toolCall{
				{name: "sql_query", responseSize: 20_000},
			}
		case i%4 == 0:
			tools = []toolCall{
				{name: "http_get", responseSize: 2_000},
				{name: "curl_endpoint", responseSize: 1_500},
			}
		case i%3 == 0:
			tools = []toolCall{
				{name: "grep_logs", responseSize: 5_000},
			}
		}
		turns[i] = turnConfig{
			userMessage:  fmt.Sprintf("Turn %d: Continue investigating the cascading failure across services", i),
			toolCalls:    tools,
			responseSize: 300,
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    200_000,
		systemPromptSize: 5_000,
		modelName:        "claude-sonnet-4-6",
		hasUsageMetadata: true,
		tokenRatio:       2.3,
	}, turns)

	t.Logf("brutal/200k-mixed-debug-long: turns=%d compactions=%d maxTokens=%d overflowed=%v looped=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed, r.loopDetected)
	if r.overflowed {
		t.Error("200k mixed debug long investigation should not overflow")
	}
	if r.loopDetected {
		t.Error("compaction loop detected")
	}
}

// --- Pure Tool Storm Pattern ---
// Nothing but tool calls with minimal/no user text between them.
// Matches pureToolStorm from singleshot tests.

func TestStress_200k_PureToolStorm(t *testing.T) {
	turns := make([]turnConfig, 30)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: fmt.Sprintf("Turn %d: go", i),
			toolCalls: []toolCall{
				{name: fmt.Sprintf("tool_%d_a", i), responseSize: 5_000},
				{name: fmt.Sprintf("tool_%d_b", i), responseSize: 5_000},
			},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    200_000,
		systemPromptSize: 1_000,
		modelName:        "claude-sonnet",
		hasUsageMetadata: true,
		tokenRatio:       2.0,
	}, turns)

	t.Logf("200k/pure-tool-storm: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("200k pure tool storm should not overflow")
	}
}

func TestStress_8k_PureToolStorm(t *testing.T) {
	turns := make([]turnConfig, 20)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: "go",
			toolCalls: []toolCall{
				{name: "execute", responseSize: 2_000},
			},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    8_000,
		systemPromptSize: 200,
		modelName:        "small-model",
		hasUsageMetadata: true,
		tokenRatio:       1.8,
	}, turns)

	t.Logf("8k/pure-tool-storm: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("8k pure tool storm should not overflow")
	}
	if r.compactions == 0 {
		t.Error("expected compactions with 2k tool responses per turn in 8k window")
	}
}

func TestBrutal_200k_PureToolStorm_HugeResponses(t *testing.T) {
	turns := make([]turnConfig, 20)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: "next",
			toolCalls: []toolCall{
				{name: "fetch_all", responseSize: 50_000},
				{name: "process", responseSize: 20_000},
			},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    200_000,
		systemPromptSize: 1_000,
		modelName:        "claude-sonnet",
		hasUsageMetadata: true,
		tokenRatio:       2.0,
	}, turns)

	t.Logf("brutal/200k-pure-storm-huge: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("200k pure tool storm with huge responses should not overflow")
	}
	if r.compactions == 0 {
		t.Error("expected compactions with 70k of tool output per turn")
	}
}

func TestBrutal_8k_PureToolStorm_50Turns(t *testing.T) {
	turns := make([]turnConfig, 50)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: "run",
			toolCalls: []toolCall{
				{name: "exec", responseSize: 3_000},
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

	t.Logf("brutal/8k-pure-storm-50: turns=%d compactions=%d maxTokens=%d overflowed=%v looped=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed, r.loopDetected)
	if r.overflowed {
		t.Error("8k pure tool storm 50 turns should not overflow")
	}
	if r.compactions < 3 {
		t.Errorf("expected at least 3 compactions in 50-turn pure tool storm, got %d", r.compactions)
	}
}

// --- Coding Agent Pattern ---
// Each turn: user asks → model thinks (text response) → reads file → analyzes →
// edits file → runs tests → model summarizes. Sequential tool chains.
// Matches buildCodingAgentConversation from singleshot tests.

func TestStress_200k_CodingAgent(t *testing.T) {
	turns := make([]turnConfig, 15)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage:  fmt.Sprintf("Turn %d: Fix the bug in the authentication middleware and make sure tests pass", i),
			sequential:   true,
			responseSize: 400,
			toolCalls: []toolCall{
				{name: "read_file", responseSize: 8_000},
				{name: "edit_file", responseSize: 2_000},
				{name: "run_tests", responseSize: 12_000},
			},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    200_000,
		systemPromptSize: 8_000,
		modelName:        "claude-sonnet-4-6",
		hasUsageMetadata: true,
		tokenRatio:       2.5,
	}, turns)

	t.Logf("200k/coding-agent: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("200k coding agent should not overflow")
	}
}

func TestStress_8k_CodingAgent(t *testing.T) {
	turns := make([]turnConfig, 10)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage:  fmt.Sprintf("Turn %d: Fix the next issue", i),
			sequential:   true,
			responseSize: 200,
			toolCalls: []toolCall{
				{name: "read_file", responseSize: 3_000},
				{name: "edit_file", responseSize: 1_000},
				{name: "run_tests", responseSize: 4_000},
			},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    8_000,
		systemPromptSize: 800,
		modelName:        "small-model",
		hasUsageMetadata: true,
		tokenRatio:       2.0,
	}, turns)

	t.Logf("8k/coding-agent: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("8k coding agent should not overflow")
	}
	if r.compactions == 0 {
		t.Error("expected compactions with sequential tool chains in 8k window")
	}
}

func TestBrutal_200k_CodingAgent_DeepRefactor(t *testing.T) {
	turns := make([]turnConfig, 25)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage:  fmt.Sprintf("Turn %d: Refactor the service layer to use dependency injection and update all tests", i),
			sequential:   true,
			responseSize: 600,
			toolCalls: []toolCall{
				{name: "read_file", responseSize: 15_000},
				{name: "grep_codebase", responseSize: 5_000},
				{name: "edit_file", responseSize: 3_000},
				{name: "read_file", responseSize: 10_000},
				{name: "edit_file", responseSize: 4_000},
				{name: "run_tests", responseSize: 20_000},
			},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    200_000,
		systemPromptSize: 10_000,
		modelName:        "claude-sonnet-4-6",
		hasUsageMetadata: true,
		tokenRatio:       2.5,
	}, turns)

	t.Logf("brutal/200k-coding-deep-refactor: turns=%d compactions=%d maxTokens=%d overflowed=%v looped=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed, r.loopDetected)
	if r.overflowed {
		t.Error("200k coding agent deep refactor should not overflow")
	}
	if r.loopDetected {
		t.Error("compaction loop detected")
	}
}

func TestBrutal_8k_CodingAgent_NoUsageMetadata(t *testing.T) {
	turns := make([]turnConfig, 15)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: fmt.Sprintf("Turn %d: fix the next test failure", i),
			sequential:  true,
			toolCalls: []toolCall{
				{name: "read_file", responseSize: 2_000},
				{name: "edit_file", responseSize: 800},
				{name: "run_tests", responseSize: 3_000},
			},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    8_000,
		systemPromptSize: 500,
		modelName:        "custom-model",
		hasUsageMetadata: false,
		tokenRatio:       2.0,
	}, turns)

	t.Logf("brutal/8k-coding-no-usage: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("8k coding agent without usage metadata should not overflow")
	}
	if r.compactions == 0 {
		t.Error("expected compactions with sequential chains in 8k")
	}
}

// --- Tiny Context Window (4k) ---
// Tests the absolute smallest realistic context windows where even a single
// tool response may exceed the threshold.

func TestStress_4k_NormalConversation(t *testing.T) {
	turns := make([]turnConfig, 20)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: longMessage(i, 600),
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    4_000,
		systemPromptSize: 300,
		modelName:        "tiny-model",
		hasUsageMetadata: true,
		tokenRatio:       1.8,
	}, turns)

	t.Logf("4k/normal: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("4k normal conversation should not overflow")
	}
	if r.compactions == 0 {
		t.Error("expected compactions in 4k window with 20 turns of long messages")
	}
}

func TestStress_4k_WithSmallTools(t *testing.T) {
	turns := make([]turnConfig, 10)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: longMessage(i, 200),
			toolCalls:   []toolCall{{name: "search", responseSize: 500}},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    4_000,
		systemPromptSize: 200,
		modelName:        "tiny-model",
		hasUsageMetadata: true,
		tokenRatio:       1.8,
	}, turns)

	t.Logf("4k/small-tools: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("4k with small tools should not overflow")
	}
	if r.compactions == 0 {
		t.Error("expected compactions in 4k window")
	}
}

func TestBrutal_4k_ToolResponseExceedsWindow(t *testing.T) {
	turns := []turnConfig{
		{
			userMessage: "Get the full output",
			toolCalls:   []toolCall{{name: "read_all", responseSize: 20_000}},
		},
		{userMessage: "Summarize what you found"},
		{userMessage: "Any issues?"},
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    4_000,
		systemPromptSize: 200,
		modelName:        "tiny-model",
		hasUsageMetadata: true,
		tokenRatio:       2.0,
	}, turns)

	t.Logf("brutal/4k-tool-exceeds: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.compactions == 0 {
		t.Error("20k tool response in 4k window must trigger compaction")
	}
}

func TestBrutal_4k_EveryTurnExceedsWindow(t *testing.T) {
	turns := make([]turnConfig, 20)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: longMessage(i, 1_000),
			toolCalls:   []toolCall{{name: "tool", responseSize: 8_000}},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    4_000,
		systemPromptSize: 200,
		modelName:        "tiny-model",
		hasUsageMetadata: true,
		tokenRatio:       2.0,
	}, turns)

	t.Logf("brutal/4k-every-turn-exceeds: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.compactions < 10 {
		t.Errorf("expected heavy compaction activity in 4k with oversized turns, got %d", r.compactions)
	}
}

func TestBrutal_4k_KubeAgent(t *testing.T) {
	turns := make([]turnConfig, 10)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: fmt.Sprintf("Turn %d: check pods", i),
			toolCalls: []toolCall{
				{name: "kubectl_get_pods", responseSize: 3_000},
				{name: "kubectl_describe", responseSize: 2_000},
				{name: "kubectl_logs", responseSize: 5_000},
			},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    4_000,
		systemPromptSize: 300,
		modelName:        "tiny-model",
		hasUsageMetadata: true,
		tokenRatio:       2.0,
	}, turns)

	t.Logf("brutal/4k-kube: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.compactions == 0 {
		t.Error("expected heavy compaction with kube pattern in 4k window")
	}
}

func TestBrutal_4k_CodingAgent(t *testing.T) {
	turns := make([]turnConfig, 10)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: fmt.Sprintf("Turn %d: fix it", i),
			sequential:  true,
			toolCalls: []toolCall{
				{name: "read_file", responseSize: 2_000},
				{name: "edit_file", responseSize: 500},
				{name: "run_tests", responseSize: 3_000},
			},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    4_000,
		systemPromptSize: 300,
		modelName:        "tiny-model",
		hasUsageMetadata: true,
		tokenRatio:       2.0,
	}, turns)

	t.Logf("brutal/4k-coding: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.compactions == 0 {
		t.Error("expected compactions with coding pattern in 4k window")
	}
}

func TestBrutal_4k_MixedDebug(t *testing.T) {
	turns := make([]turnConfig, 15)
	for i := range turns {
		var tools []toolCall
		switch {
		case i%5 == 0:
			tools = []toolCall{
				{name: "prometheus_query", responseSize: 3_000},
				{name: "sql_query", responseSize: 2_000},
			}
		case i%3 == 0:
			tools = []toolCall{
				{name: "http_get", responseSize: 200},
			}
		}
		turns[i] = turnConfig{
			userMessage: fmt.Sprintf("Turn %d: debug the issue", i),
			toolCalls:   tools,
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    4_000,
		systemPromptSize: 200,
		modelName:        "tiny-model",
		hasUsageMetadata: true,
		tokenRatio:       2.0,
	}, turns)

	t.Logf("brutal/4k-mixed-debug: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("4k mixed debug should not overflow")
	}
}

func TestBrutal_4k_PureToolStorm(t *testing.T) {
	turns := make([]turnConfig, 20)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: "go",
			toolCalls:   []toolCall{{name: "exec", responseSize: 2_000}},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    4_000,
		systemPromptSize: 100,
		modelName:        "tiny-model",
		hasUsageMetadata: true,
		tokenRatio:       2.0,
	}, turns)

	t.Logf("brutal/4k-pure-storm: turns=%d compactions=%d maxTokens=%d overflowed=%v looped=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed, r.loopDetected)
	if r.compactions < 3 {
		t.Errorf("expected at least 3 compactions in 4k pure tool storm, got %d", r.compactions)
	}
}

// --- 1M Context Window ---
// Tests for very large context windows (e.g., Gemini 1.5 Pro).

func TestStress_1M_NormalConversation(t *testing.T) {
	turns := make([]turnConfig, 50)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: fmt.Sprintf("Turn %d: Continue the comprehensive analysis of the distributed system architecture across all microservices and their interactions", i),
			toolCalls: []toolCall{
				{name: "fetch", responseSize: 10_000},
			},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    1_000_000,
		systemPromptSize: 10_000,
		modelName:        "gemini-pro",
		hasUsageMetadata: true,
		tokenRatio:       2.0,
	}, turns)

	t.Logf("1M/normal: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("1M normal conversation should not overflow")
	}
}

func TestStress_1M_HeavyTools(t *testing.T) {
	turns := make([]turnConfig, 30)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: fmt.Sprintf("Turn %d: Analyze the full codebase", i),
			toolCalls: []toolCall{
				{name: "read_file", responseSize: 50_000},
				{name: "grep_codebase", responseSize: 20_000},
				{name: "run_tests", responseSize: 30_000},
			},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    1_000_000,
		systemPromptSize: 10_000,
		modelName:        "gemini-pro",
		hasUsageMetadata: true,
		tokenRatio:       2.0,
	}, turns)

	t.Logf("1M/heavy-tools: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("1M with heavy tools should not overflow")
	}
}

func TestBrutal_1M_KubeAgent_ExtremeLongevity(t *testing.T) {
	turns := make([]turnConfig, 100)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: fmt.Sprintf("Turn %d: Check the full cluster state including all namespaces", i),
			toolCalls: []toolCall{
				{name: "kubectl_get_pods", responseSize: 30_000},
				{name: "kubectl_describe", responseSize: 20_000},
				{name: "kubectl_logs", responseSize: 50_000},
			},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    1_000_000,
		systemPromptSize: 10_000,
		modelName:        "gemini-pro",
		hasUsageMetadata: true,
		tokenRatio:       2.0,
	}, turns)

	t.Logf("brutal/1M-kube-100turns: turns=%d compactions=%d maxTokens=%d overflowed=%v looped=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed, r.loopDetected)
	if r.overflowed {
		t.Error("1M kube agent 100 turns should not overflow")
	}
	if r.loopDetected {
		t.Error("compaction loop detected in 1M session")
	}
}

func TestBrutal_1M_PureToolStorm_MonsterResponses(t *testing.T) {
	turns := make([]turnConfig, 50)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: "next batch",
			toolCalls: []toolCall{
				{name: "fetch_dataset", responseSize: 100_000},
			},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    1_000_000,
		systemPromptSize: 5_000,
		modelName:        "gemini-pro",
		hasUsageMetadata: true,
		tokenRatio:       2.0,
	}, turns)

	t.Logf("brutal/1M-monster-storm: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("1M with 100k tool responses should not overflow")
	}
}

func TestBrutal_1M_NoUsageMetadata(t *testing.T) {
	turns := make([]turnConfig, 40)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: fmt.Sprintf("Turn %d: Continue the deep investigation across all systems", i),
			toolCalls: []toolCall{
				{name: "fetch", responseSize: 30_000},
				{name: "analyze", responseSize: 15_000},
			},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    1_000_000,
		systemPromptSize: 5_000,
		modelName:        "custom-1m-model",
		hasUsageMetadata: false,
		tokenRatio:       2.5,
	}, turns)

	t.Logf("brutal/1M-no-usage: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("1M with no usage metadata should not overflow")
	}
}

// --- Production Magec Scenario ---
// Simulates the exact production setup that triggered the original bug:
// Telegram bot with MCP tools, users sending screenshots, tool definitions
// attached, structured JSON responses, high token ratio.

func TestBrutal_200k_MagecProductionScenario(t *testing.T) {
	tools := makeMCPTools(25, 2_500)
	turns := make([]turnConfig, 30)
	for i := range turns {
		tc := turnConfig{
			userMessage: fmt.Sprintf("Turn %d: The deployment pipeline is failing. Check the CI/CD status and fix the configuration.", i),
			toolCalls: []toolCall{
				{name: "mcp_tool_0", responseSize: 8_000},
				{name: "mcp_tool_5", responseSize: 5_000},
			},
			responseSize: 400,
		}
		if i%4 == 0 {
			tc.inlineData = []inlineAttachment{{mimeType: "image/png", size: 80_000}}
		}
		if i%6 == 0 {
			tc.toolCalls = append(tc.toolCalls,
				toolCall{name: "mcp_tool_10", responseSize: 15_000},
				toolCall{name: "mcp_tool_15", responseSize: 10_000},
			)
		}
		turns[i] = tc
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    200_000,
		systemPromptSize: 8_000,
		modelName:        "claude-sonnet-4-6",
		hasUsageMetadata: true,
		tokenRatio:       2.5,
		tools:            tools,
	}, turns)

	t.Logf("brutal/200k-magec-production: turns=%d compactions=%d maxTokens=%d overflowed=%v looped=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed, r.loopDetected)
	if r.overflowed {
		t.Error("magec production scenario should not overflow")
	}
	if r.loopDetected {
		t.Error("compaction loop detected in magec production scenario")
	}
}

func TestBrutal_200k_MagecProductionScenario_NoUsageMetadata(t *testing.T) {
	tools := makeMCPTools(20, 2_000)
	turns := make([]turnConfig, 20)
	for i := range turns {
		tc := turnConfig{
			userMessage: fmt.Sprintf("Turn %d: Investigate and fix the issue", i),
			toolCalls: []toolCall{
				{name: "mcp_tool_0", responseSize: 5_000},
			},
		}
		if i%3 == 0 {
			tc.inlineData = []inlineAttachment{{mimeType: "image/png", size: 50_000}}
		}
		turns[i] = tc
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    200_000,
		systemPromptSize: 5_000,
		modelName:        "claude-sonnet-4-6",
		hasUsageMetadata: false,
		tokenRatio:       2.5,
		tools:            tools,
	}, turns)

	t.Logf("brutal/200k-magec-no-usage: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("magec production scenario without usage metadata should not overflow")
	}
}

// --- Sequential Tool Chains with Escalating Sizes ---
// Each turn has sequential tool calls where each step produces progressively
// larger output. Tests that compaction handles growing context within a single turn.

func TestBrutal_200k_SequentialEscalatingSizes(t *testing.T) {
	turns := make([]turnConfig, 10)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: fmt.Sprintf("Turn %d: run the full analysis pipeline", i),
			sequential:  true,
			toolCalls: []toolCall{
				{name: "list_files", responseSize: 2_000},
				{name: "read_small_file", responseSize: 5_000},
				{name: "read_large_file", responseSize: 20_000},
				{name: "analyze_data", responseSize: 40_000},
				{name: "generate_report", responseSize: 60_000},
			},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    200_000,
		systemPromptSize: 3_000,
		modelName:        "claude-sonnet-4-6",
		hasUsageMetadata: true,
		tokenRatio:       2.0,
	}, turns)

	t.Logf("brutal/200k-seq-escalating: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("200k sequential escalating sizes should not overflow")
	}
}

func TestBrutal_8k_SequentialEscalatingSizes(t *testing.T) {
	turns := make([]turnConfig, 10)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: fmt.Sprintf("Turn %d: pipeline", i),
			sequential:  true,
			toolCalls: []toolCall{
				{name: "step1", responseSize: 500},
				{name: "step2", responseSize: 1_500},
				{name: "step3", responseSize: 3_000},
				{name: "step4", responseSize: 5_000},
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

	t.Logf("brutal/8k-seq-escalating: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.overflowed {
		t.Error("8k sequential escalating sizes should not overflow")
	}
	if r.compactions == 0 {
		t.Error("expected compactions with escalating tool responses in 8k")
	}
}

func TestBrutal_4k_SequentialEscalatingSizes(t *testing.T) {
	turns := make([]turnConfig, 8)
	for i := range turns {
		turns[i] = turnConfig{
			userMessage: fmt.Sprintf("Turn %d: go", i),
			sequential:  true,
			toolCalls: []toolCall{
				{name: "step1", responseSize: 300},
				{name: "step2", responseSize: 800},
				{name: "step3", responseSize: 2_000},
			},
		}
	}

	r := simulateSession(t, sessionConfig{
		contextWindow:    4_000,
		systemPromptSize: 200,
		modelName:        "tiny-model",
		hasUsageMetadata: true,
		tokenRatio:       2.0,
	}, turns)

	t.Logf("brutal/4k-seq-escalating: turns=%d compactions=%d maxTokens=%d overflowed=%v",
		r.turns, r.compactions, r.maxTokensSeen, r.overflowed)
	if r.compactions == 0 {
		t.Error("expected compactions with escalating sequential tools in 4k")
	}
}
