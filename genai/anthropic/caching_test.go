// Copyright 2026 Achetronic
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

package anthropic

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// captureBodyFor mirrors wire_test.go's captureBody but lets the test
// supply the full LLMRequest, so we can exercise caching against
// system prompts, tools and multi-turn histories.
func captureBodyFor(t *testing.T, cfg Config, req *model.LLMRequest) map[string]any {
	t.Helper()

	var captured []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	defer srv.Close()

	cfg.BaseURL = srv.URL
	cfg.APIKey = "test-key"
	if cfg.ModelName == "" {
		cfg.ModelName = "claude-test"
	}
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

// fullRequest builds an LLMRequest exercising all three cacheable
// sections: system instruction, one tool, and a two-turn history.
func fullRequest() *model.LLMRequest {
	return &model.LLMRequest{
		Config: &genai.GenerateContentConfig{
			SystemInstruction: &genai.Content{
				Parts: []*genai.Part{{Text: "You are a test agent."}},
			},
			Tools: []*genai.Tool{{
				FunctionDeclarations: []*genai.FunctionDeclaration{
					{Name: "tool_a", Description: "first tool"},
					{Name: "tool_b", Description: "second tool"},
				},
			}},
		},
		Contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: "first question"}}},
			{Role: "model", Parts: []*genai.Part{{Text: "first answer"}}},
			{Role: "user", Parts: []*genai.Part{{Text: "second question"}}},
		},
	}
}

// hasEphemeralCacheControl reports whether the JSON object carries
// cache_control: {type: "ephemeral"}.
func hasEphemeralCacheControl(obj map[string]any) bool {
	cc, ok := obj["cache_control"].(map[string]any)
	return ok && cc["type"] == "ephemeral"
}

// lastOf returns the last element of a JSON array as an object.
func lastOf(t *testing.T, v any, what string) map[string]any {
	t.Helper()
	arr, ok := v.([]any)
	if !ok || len(arr) == 0 {
		t.Fatalf("expected a non-empty %s array, got %v", what, v)
	}
	obj, ok := arr[len(arr)-1].(map[string]any)
	if !ok {
		t.Fatalf("last %s entry is not an object: %v", what, arr[len(arr)-1])
	}
	return obj
}

// TestWireBody_CacheControlBreakpoints pins the three breakpoints on
// the wire: last system block, last tool, and the last content block
// of the last message. Earlier entries must NOT carry the marker
// (each section spends exactly one of Anthropic's four slots).
func TestWireBody_CacheControlBreakpoints(t *testing.T) {
	body := captureBodyFor(t, Config{}, fullRequest())

	// System: last (only) block marked.
	if sys := lastOf(t, body["system"], "system"); !hasEphemeralCacheControl(sys) {
		t.Errorf("last system block must carry cache_control, got %v", sys)
	}

	// Tools: last marked, first not.
	tools, _ := body["tools"].([]any)
	if len(tools) != 2 {
		t.Fatalf("expected 2 tools, got %v", body["tools"])
	}
	first, _ := tools[0].(map[string]any)
	if hasEphemeralCacheControl(first) {
		t.Errorf("only the LAST tool may carry cache_control, first has it too")
	}
	if last := lastOf(t, body["tools"], "tools"); !hasEphemeralCacheControl(last) {
		t.Errorf("last tool must carry cache_control, got %v", last)
	}

	// Messages: only the last block of the last message marked.
	messages, _ := body["messages"].([]any)
	if len(messages) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(messages))
	}
	for i, raw := range messages {
		msg, _ := raw.(map[string]any)
		blocks, _ := msg["content"].([]any)
		for j, b := range blocks {
			block, _ := b.(map[string]any)
			marked := hasEphemeralCacheControl(block)
			isFinal := i == len(messages)-1 && j == len(blocks)-1
			if marked != isFinal {
				t.Errorf("message %d block %d: cache_control=%v, want %v", i, j, marked, isFinal)
			}
		}
	}
}

// TestWireBody_CacheControlSkipsThinkingBlocks verifies the history
// marker lands on the nearest eligible block when the last message
// ends in thinking blocks (which cannot carry cache_control).
func TestWireBody_CacheControlSkipsThinkingBlocks(t *testing.T) {
	req := fullRequest()
	// Assistant turn whose parts arrive as [thinking, text]: the
	// adapter reorders thinking first, so text is last and eligible.
	// Then add a thinking-ONLY trailing turn: the marker must fall
	// back to the previous message entirely.
	req.Contents = append(req.Contents, &genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{Text: "internal reasoning", Thought: true, ThoughtSignature: []byte("sig")},
		},
	})
	body := captureBodyFor(t, Config{}, req)

	messages, _ := body["messages"].([]any)
	lastMsg, _ := messages[len(messages)-1].(map[string]any)
	lastBlocks, _ := lastMsg["content"].([]any)
	for _, b := range lastBlocks {
		block, _ := b.(map[string]any)
		if block["type"] == "thinking" && hasEphemeralCacheControl(block) {
			t.Fatal("thinking blocks must never carry cache_control")
		}
	}
	// The marker must exist SOMEWHERE in the history (on the previous
	// message's text block).
	found := false
	for _, raw := range messages {
		msg, _ := raw.(map[string]any)
		blocks, _ := msg["content"].([]any)
		for _, b := range blocks {
			if block, _ := b.(map[string]any); hasEphemeralCacheControl(block) {
				found = true
			}
		}
	}
	if !found {
		t.Fatal("no history block carries cache_control; the marker was lost")
	}
}

// TestWireBody_CacheControlDisabled verifies the opt-out: with
// DisablePromptCaching no cache_control appears anywhere in the body.
func TestWireBody_CacheControlDisabled(t *testing.T) {
	body := captureBodyFor(t, Config{DisablePromptCaching: true}, fullRequest())
	raw, _ := json.Marshal(body)
	// String-level check is fine here: the key name cannot appear as
	// a value in these fixtures.
	if strings.Contains(string(raw), `"cache_control"`) {
		t.Fatalf("DisablePromptCaching must strip every cache_control, body: %s", raw)
	}
}
