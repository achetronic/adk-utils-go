// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package completions

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
func captureBody(t *testing.T, req *model.LLMRequest, dialect Dialect) map[string]any {
	t.Helper()

	var captured []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		// Minimal valid ChatCompletion so convertResponse succeeds.
		io.WriteString(w, `{"id":"x","object":"chat.completion","created":0,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer srv.Close()

	cfg := Config{BaseURL: srv.URL, APIKey: "test-key", ModelName: "gpt-test", Dialect: dialect}
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

// captureBodyWithNoDialect is captureBody with no dialect, for tests that
// pin the OpenAI-pure default.
func captureBodyWithNoDialect(t *testing.T, req *model.LLMRequest) map[string]any {
	return captureBody(t, req, nil)
}

// captureStreamBodyWithNoDialect is the streaming twin.
func captureStreamBodyWithNoDialect(t *testing.T, req *model.LLMRequest) map[string]any {
	return captureStreamBody(t, req, nil)
}

// On the wire, a tool call with nil Args must send arguments:"{}", not "null".
func TestWireBody_NilFunctionCallArgs(t *testing.T) {
	body := captureBodyWithNoDialect(t, &model.LLMRequest{
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
	body := captureBodyWithNoDialect(t, &model.LLMRequest{
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

// On the wire, a thought Part must travel as its own reasoning field on the
// assistant message, not merged into content. This is the byte-level contract
// the strict thinking providers check for.
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
	}, NewTextDialect())

	assistant := messageOfRole(t, body, "assistant")
	if assistant["content"] != "It's sunny today." {
		t.Errorf("content = %q, want the reply only", assistant["content"])
	}
	if assistant["reasoning_content"] != "The user wants weather info, I should check my tools..." {
		t.Errorf("reasoning_content = %q, want the reasoning", assistant["reasoning_content"])
	}
}

// ExtraBody keys land at the root of the request body, which is how providers
// like OpenRouter take their reasoning controls.
func TestWireBody_ExtraBodyMergesAtRoot(t *testing.T) {
	var captured []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"id":"x","object":"chat.completion","created":0,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer srv.Close()

	m := New(Config{
		BaseURL:   srv.URL,
		APIKey:    "test-key",
		ModelName: "gpt-test",
		ExtraBody: map[string]any{"reasoning": map[string]any{"effort": "high"}},
	})
	for _, err := range m.GenerateContent(context.Background(), &model.LLMRequest{
		Config:   &genai.GenerateContentConfig{},
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "hi"}}}},
	}, false) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
	}

	var body map[string]any
	if err := json.Unmarshal(captured, &body); err != nil {
		t.Fatalf("unmarshal captured body: %v", err)
	}
	reasoning, ok := body["reasoning"].(map[string]any)
	if !ok {
		t.Fatalf("reasoning missing at body root: %v", body)
	}
	if reasoning["effort"] != "high" {
		t.Errorf("reasoning.effort = %v, want high", reasoning["effort"])
	}
}

// captureStreamBody is the streaming twin of captureBody: it points the model at
// a fake SSE endpoint, fires one streaming request, and returns the JSON body
// that hit the wire. Serves a minimal valid SSE stream so the accumulator drains
// cleanly and generateStream yields its terminal LLMResponse.
func captureStreamBody(t *testing.T, req *model.LLMRequest, dialect Dialect) map[string]any {
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

	cfg := Config{BaseURL: srv.URL, APIKey: "test-key", ModelName: "gpt-test", Dialect: dialect}
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
	body := captureStreamBodyWithNoDialect(t, &model.LLMRequest{
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

// A streamed turn's reasoning must reach the terminal response. The SDK's
// accumulator keeps no raw JSON, so the reasoning can only come from the
// adapter's own accumulator; driving real chunks through the public streaming
// path is the only way to prove that, since constructing an accumulator by
// hand would populate raw JSON a live stream never has.
func TestGenerateStream_ReasoningReachesFinalResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		io.WriteString(w,
			"data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"reasoning_content\":\"think\"},\"finish_reason\":null}]}\n\n"+
				"data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\" more\"},\"finish_reason\":null}]}\n\n"+
				"data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"answer\"},\"finish_reason\":null}]}\n\n"+
				"data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"+
				"data: [DONE]\n\n")
	}))
	defer srv.Close()

	m := New(Config{BaseURL: srv.URL, APIKey: "test-key", ModelName: "gpt-test", Dialect: NewTextDialect()})

	var final *model.LLMResponse
	for resp, err := range m.GenerateContent(context.Background(), &model.LLMRequest{
		Config:   &genai.GenerateContentConfig{},
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "hi"}}}},
	}, true) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
		if !resp.Partial {
			final = resp
		}
	}

	if final == nil {
		t.Fatalf("no terminal response")
	}
	parts := final.Content.Parts
	if len(parts) != 2 {
		t.Fatalf("expected reasoning + answer parts, got %d: %#v", len(parts), parts)
	}
	if !parts[0].Thought || parts[0].Text != "think more" {
		t.Errorf("thought part = %#v, want the concatenated reasoning", parts[0])
	}
	if parts[1].Thought || parts[1].Text != "answer" {
		t.Errorf("answer part = %#v, want the final answer", parts[1])
	}
}

// Without a dialect the assistant message carries no provider field on the
// wire: native mode degrades to think tags, so a stray thought folds into
// content instead of being dropped, and nothing untyped is sent. This is
// the OpenAI-native default; its reasoning models never expose the
// reasoning text on this API.
func TestWireBody_NilDialectFoldsThoughtsIntoThinkTags(t *testing.T) {
	body := captureBodyWithNoDialect(t, &model.LLMRequest{
		Config: &genai.GenerateContentConfig{},
		Contents: []*genai.Content{
			{Role: "model", Parts: []*genai.Part{
				{Text: "a thought", Thought: true},
				{Text: "the answer"},
			}},
		},
	})

	assistant := messageOfRole(t, body, "assistant")
	want := "<think>\na thought\n</think>\nthe answer"
	if assistant["content"] != want {
		t.Errorf("content = %q, want %q", assistant["content"], want)
	}
	if _, has := assistant["reasoning_content"]; has {
		t.Errorf("reasoning_content present without a dialect: %v", assistant)
	}
	if _, has := assistant["reasoning_details"]; has {
		t.Errorf("reasoning_details present without a dialect: %v", assistant)
	}
}

// With no dialect a streamed reasoning field must not surface as thought
// Parts: the default adapter reads nothing off the raw JSON. The answer
// still arrives untouched.
func TestGenerateStream_NilDialectIgnoresReasoningChunks(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		io.WriteString(w,
			"data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"reasoning_content\":\"hidden\"},\"finish_reason\":null}]}\n\n"+
				"data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"answer\"},\"finish_reason\":null}]}\n\n"+
				"data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"+
				"data: [DONE]\n\n")
	}))
	defer srv.Close()

	m := New(Config{BaseURL: srv.URL, APIKey: "test-key", ModelName: "gpt-test"})

	var final *model.LLMResponse
	for resp, err := range m.GenerateContent(context.Background(), &model.LLMRequest{
		Config:   &genai.GenerateContentConfig{},
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "hi"}}}},
	}, true) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
		if !resp.Partial {
			final = resp
		}
	}

	if final == nil {
		t.Fatalf("no terminal response")
	}
	parts := final.Content.Parts
	if len(parts) != 1 {
		t.Fatalf("expected only the answer part, got %d: %#v", len(parts), parts)
	}
	if parts[0].Thought || parts[0].Text != "answer" {
		t.Errorf("part = %#v, want the plain answer", parts[0])
	}
}
