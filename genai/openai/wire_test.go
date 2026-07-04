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

	var captured []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		// Minimal valid ChatCompletion so convertResponse succeeds.
		io.WriteString(w, `{"id":"x","object":"chat.completion","created":0,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer srv.Close()

	m := New(Config{BaseURL: srv.URL, APIKey: "test-key", ModelName: "gpt-test"})

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

func TestStreamReasoning(t *testing.T) {
	// Streaming final response must carry a Thought part when deltas stream reasoning.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		_, _ = w.Write([]byte("data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning\":\"thinking \"}}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning\":\"step by step\"}}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"}}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer srv.Close()

	m := New(Config{BaseURL: srv.URL, APIKey: "test-key", ModelName: "gpt-test"})
	ctx := context.Background()
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: "hi"}}},
		},
	}

	var finalResp *model.LLMResponse
	for resp, err := range m.GenerateContent(ctx, req, true) {
		if err != nil {
			t.Fatalf("unexpected streaming error: %v", err)
		}
		if !resp.Partial {
			finalResp = resp
		}
	}

	if finalResp == nil {
		t.Fatalf("expected non-nil final response")
	}

	parts := finalResp.Content.Parts
	if len(parts) != 2 {
		t.Fatalf("expected 2 parts, got %d: %#v", len(parts), parts)
	}

	if !parts[0].Thought {
		t.Errorf("expected first part to be a thought")
	}
	if parts[0].Text != "thinking step by step" {
		t.Errorf("got thought text %q, want %q", parts[0].Text, "thinking step by step")
	}

	if parts[1].Thought {
		t.Errorf("expected second part to not be a thought")
	}
	if parts[1].Text != "hello" {
		t.Errorf("got content text %q, want %q", parts[1].Text, "hello")
	}
}

func TestStreamReasoningOnlyDeltas(t *testing.T) {
	// Streaming deltas with reasoning only must yield partial responses with a Thought part.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		_, _ = w.Write([]byte("data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning\":\"only thought\"}}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer srv.Close()

	m := New(Config{BaseURL: srv.URL, APIKey: "test-key", ModelName: "gpt-test"})
	ctx := context.Background()
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: "hi"}}},
		},
	}

	var partialResps []*model.LLMResponse
	for resp, err := range m.GenerateContent(ctx, req, true) {
		if err != nil {
			t.Fatalf("unexpected streaming error: %v", err)
		}
		if resp.Partial {
			partialResps = append(partialResps, resp)
		}
	}

	if len(partialResps) == 0 {
		t.Fatalf("expected at least one partial response")
	}

	foundThought := false
	for _, partial := range partialResps {
		for _, part := range partial.Content.Parts {
			if part.Thought {
				if part.Text == "only thought" {
					foundThought = true
				}
			}
		}
	}

	if !foundThought {
		t.Errorf("failed to find partial response containing the Thought part")
	}
}
