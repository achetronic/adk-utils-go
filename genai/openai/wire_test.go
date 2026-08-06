// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package openai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"
)

// captureBody points a Model at a local fake endpoint via BaseURL, fires one
// non-streaming request, and returns the JSON body the openai-go SDK actually
// put on the wire. Asserting on this (not on the pre-SDK params) is what proves
// the bytes a real OpenAI-compatible server receives are correct.
func captureBody(t *testing.T, req *model.LLMRequest) map[string]any {
	t.Helper()
	return captureBodyWithConfig(t, Config{}, req)
}

// captureBodyWithConfig is captureBody for a caller-provided Config, so
// options that only show up on the wire (the reasoning egress shape, for
// instance) can be asserted per mode. BaseURL, APIKey and ModelName are
// filled in for the fixture.
func captureBodyWithConfig(t *testing.T, cfg Config, req *model.LLMRequest) map[string]any {
	t.Helper()

	var captured []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		// Minimal valid ChatCompletion so convertResponse succeeds.
		io.WriteString(w, `{"id":"x","object":"chat.completion","created":0,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer srv.Close()

	cfg.BaseURL = srv.URL
	cfg.APIKey = "test-key"
	cfg.ModelName = "gpt-test"
	m := New(cfg)

	for _, err := range m.GenerateContent(context.Background(), req, false) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
	}
	if len(captured) == 0 {
		t.Fatalf("server captured no request body")
	}
	var body map[string]any
	if err := json.Unmarshal(captured, &body); err != nil {
		t.Fatalf("unmarshal captured body: %v", err)
	}
	return body
}

// messageOfRole returns the first message in the request body with the given role.
func messageOfRole(t *testing.T, body map[string]any, role string) map[string]any {
	t.Helper()
	msgs, _ := body["messages"].([]any)
	for _, raw := range msgs {
		msg, _ := raw.(map[string]any)
		if msg["role"] == role {
			return msg
		}
	}
	t.Fatalf("no %q message in body: %v", role, body["messages"])
	return nil
}

// On the wire, a tool call with nil Args must send arguments:"{}", not "null".
func TestWireBody_NilFunctionCallArgs(t *testing.T) {
	body := captureBody(t, &model.LLMRequest{
		Config: &genai.GenerateContentConfig{},
		Contents: []*genai.Content{
			{Role: "model", Parts: []*genai.Part{
				{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "exit_loop"}},
			}},
		},
	})

	assistant := messageOfRole(t, body, "assistant")
	calls, _ := assistant["tool_calls"].([]any)
	if len(calls) != 1 {
		t.Fatalf("expected 1 tool_call, got %v", assistant["tool_calls"])
	}
	fn, _ := calls[0].(map[string]any)["function"].(map[string]any)
	if fn["arguments"] != "{}" {
		t.Errorf("arguments = %q, want \"{}\"", fn["arguments"])
	}
}

// On the wire, a tool result with nil Response must send content:"{}", not "null".
func TestWireBody_NilFunctionResponse(t *testing.T) {
	body := captureBody(t, &model.LLMRequest{
		Config: &genai.GenerateContentConfig{},
		Contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{
				{FunctionResponse: &genai.FunctionResponse{ID: "call_1"}},
			}},
		},
	})

	tool := messageOfRole(t, body, "tool")
	if tool["content"] != "{}" {
		t.Errorf("tool content = %q, want \"{}\"", tool["content"])
	}
}

// On the wire, a replayed thought part must reach the server as its own
// reasoning_content key next to content, not folded into content. DeepSeek in
// thinking mode and Kimi K2 thinking check for the literal key and reject the
// request without it.
func TestWireBody_ReasoningContentOnAssistantMessage(t *testing.T) {
	body := captureBody(t, &model.LLMRequest{
		Config: &genai.GenerateContentConfig{},
		Contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: "What's the weather?"}}},
			{Role: "model", Parts: []*genai.Part{
				{Text: "The user wants weather info, I should check my tools...", Thought: true},
				{Text: "It's sunny today."},
			}},
			{Role: "user", Parts: []*genai.Part{{Text: "Thanks, and tomorrow?"}}},
		},
	})

	assistant := messageOfRole(t, body, "assistant")
	if assistant["content"] != "It's sunny today." {
		t.Errorf("content = %q, want the reply only", assistant["content"])
	}
	if assistant["reasoning_content"] != "The user wants weather info, I should check my tools..." {
		t.Errorf("reasoning_content = %q, want the reasoning", assistant["reasoning_content"])
	}
}

// The tool-loop case: reasoning_content must sit next to tool_calls on the
// assistant message, which is what the strict thinking providers require on
// every intermediate step.
func TestWireBody_ReasoningContentWithToolCall(t *testing.T) {
	body := captureBody(t, &model.LLMRequest{
		Config: &genai.GenerateContentConfig{},
		Contents: []*genai.Content{
			{Role: "model", Parts: []*genai.Part{
				{Text: "I need the status tool first.", Thought: true},
				{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "check_status"}},
			}},
		},
	})

	assistant := messageOfRole(t, body, "assistant")
	if calls, _ := assistant["tool_calls"].([]any); len(calls) != 1 {
		t.Fatalf("expected 1 tool_call, got %v", assistant["tool_calls"])
	}
	if assistant["reasoning_content"] != "I need the status tool first." {
		t.Errorf("reasoning_content = %q, want the reasoning", assistant["reasoning_content"])
	}
}

// Think-tag mode must put the reasoning inside content and send no extra key,
// so backends that validate messages against a schema forbidding extra fields
// accept the request.
func TestWireBody_ReasoningAsThinkTags(t *testing.T) {
	body := captureBodyWithConfig(t, Config{ReasoningEgress: ReasoningEgressThinkTags}, &model.LLMRequest{
		Config: &genai.GenerateContentConfig{},
		Contents: []*genai.Content{
			{Role: "model", Parts: []*genai.Part{
				{Text: "thinking", Thought: true},
				{Text: "reply"},
			}},
		},
	})

	assistant := messageOfRole(t, body, "assistant")
	if assistant["content"] != "<think>\nthinking\n</think>\nreply" {
		t.Errorf("content = %q, want the reasoning in a think block ahead of the reply", assistant["content"])
	}
	if _, ok := assistant["reasoning_content"]; ok {
		t.Errorf("reasoning_content must be absent in think-tag mode: %v", assistant)
	}
}

// A configured field name is what lands on the wire, for gateways that only
// accept "reasoning".
func TestWireBody_ReasoningCustomFieldName(t *testing.T) {
	body := captureBodyWithConfig(t, Config{ReasoningField: "reasoning"}, &model.LLMRequest{
		Config: &genai.GenerateContentConfig{},
		Contents: []*genai.Content{
			{Role: "model", Parts: []*genai.Part{
				{Text: "thinking", Thought: true},
				{Text: "reply"},
			}},
		},
	})

	assistant := messageOfRole(t, body, "assistant")
	if assistant["reasoning"] != "thinking" {
		t.Errorf("reasoning = %q, want the reasoning", assistant["reasoning"])
	}
	if _, ok := assistant["reasoning_content"]; ok {
		t.Errorf("reasoning_content must be absent when another field name is configured: %v", assistant)
	}
}

// On the wire, OpenRouter's reasoning_details array must sit on the assistant
// message next to tool_calls, with each block intact. That combination is the
// case OpenRouter calls out as the reason to preserve blocks: a model pausing
// mid-reasoning to call a tool resumes from these blocks once the result
// comes back.
func TestWireBody_ReasoningDetailsWithToolCall(t *testing.T) {
	body := captureBodyWithConfig(t, Config{
		ReasoningField:           "reasoning",
		SupportsReasoningDetails: true,
	}, &model.LLMRequest{
		Config: &genai.GenerateContentConfig{},
		Contents: []*genai.Content{
			{Role: "model", Parts: []*genai.Part{
				{
					Text:    "I need the status tool first.",
					Thought: true,
					PartMetadata: map[string]any{ReasoningDetailMetadataKey: map[string]any{
						"type":      "reasoning.text",
						"text":      "I need the status tool first.",
						"signature": "sig-1",
						"index":     float64(0),
					}},
				},
				{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "check_status"}},
			}},
		},
	})

	assistant := messageOfRole(t, body, "assistant")
	if calls, _ := assistant["tool_calls"].([]any); len(calls) != 1 {
		t.Fatalf("expected 1 tool_call, got %v", assistant["tool_calls"])
	}
	details, ok := assistant["reasoning_details"].([]any)
	if !ok || len(details) != 1 {
		t.Fatalf("reasoning_details = %v, want 1 block beside tool_calls", assistant["reasoning_details"])
	}
	block, _ := details[0].(map[string]any)
	if block["type"] != "reasoning.text" || block["signature"] != "sig-1" {
		t.Errorf("block = %#v, want the signature preserved", block)
	}
	// The block already carries the reasoning; repeating it in the plain-text
	// field would send the same thing twice.
	if _, ok := assistant["reasoning"]; ok {
		t.Errorf("plain-text reasoning must not duplicate the block: %v", assistant)
	}
}

// captureStreamBody is the streaming twin of captureBody: it points the model at
// a fake SSE endpoint, fires one streaming request, and returns the JSON body
// that hit the wire. Serves a minimal valid SSE stream so the accumulator drains
// cleanly and generateStream yields its terminal LLMResponse.
func captureStreamBody(t *testing.T, req *model.LLMRequest) map[string]any {
	t.Helper()
	return captureStreamBodyWithConfig(t, Config{}, req)
}

// captureStreamBodyWithConfig is captureStreamBody for a caller-provided
// Config, so options that only show up on the wire can be asserted on the
// streaming path too.
func captureStreamBodyWithConfig(t *testing.T, cfg Config, req *model.LLMRequest) map[string]any {
	t.Helper()

	var captured []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.WriteHeader(http.StatusOK)
		io.WriteString(w,
			"data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"ok\"},\"finish_reason\":null}]}\n\n"+
				"data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"+
				"data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2,\"total_tokens\":3}}\n\n"+
				"data: [DONE]\n\n")
	}))
	defer srv.Close()

	cfg.BaseURL = srv.URL
	cfg.APIKey = "test-key"
	cfg.ModelName = "gpt-test"
	m := New(cfg)

	for _, err := range m.GenerateContent(context.Background(), req, true) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
	}
	if len(captured) == 0 {
		t.Fatalf("server captured no request body")
	}
	var body map[string]any
	if err := json.Unmarshal(captured, &body); err != nil {
		t.Fatalf("unmarshal captured body: %v", err)
	}
	return body
}

// On the wire, a streaming request must set stream_options.include_usage=true.
// Without this opt-in the OpenAI server never emits the terminal usage chunk,
// the ChatCompletionAccumulator's Usage stays zero, and buildStreamFinalResponse
// yields empty UsageMetadata - leaving consumers no way to price the turn.
func TestWireBody_StreamRequestsUsage(t *testing.T) {
	body := captureStreamBody(t, &model.LLMRequest{
		Config: &genai.GenerateContentConfig{},
		Contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: "hi"}}},
		},
	})

	if body["stream"] != true {
		t.Fatalf("stream = %v, want true", body["stream"])
	}
	opts, ok := body["stream_options"].(map[string]any)
	if !ok {
		t.Fatalf("stream_options missing or not an object: %v", body["stream_options"])
	}
	if opts["include_usage"] != true {
		t.Errorf("stream_options.include_usage = %v, want true", opts["include_usage"])
	}
}
