// Copyright 2025 achetronic
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

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
