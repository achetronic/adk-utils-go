//go:build integration

// Copyright 2025 achetronic
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

// These tests cover the cross-message semantic rules the Anthropic API enforces
// but no JSON schema expresses, discovered by probing the live API. Each test
// has two halves:
//
//	Canary: send the raw, unrepaired shape and require the API to reject it with
//	  4xx. If this stops failing, the rule changed and the test below is moot.
//	Adapter: feed the equivalent history through the adapter and require the API
//	  to ACCEPT the result, proving the adapter's repair/drop logic works against
//	  the real contract.
//
// Run: ANTHROPIC_API_KEY=... go test -tags=integration -run Semantic ./genai/anthropic/...
package anthropic

import (
	"bytes"
	"io"
	"net/http"
	"os"
	"testing"

	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// rawStatus sends a hand-built body to the real API and returns its status.
// Skipped unless ANTHROPIC_API_KEY is set.
func rawStatus(t *testing.T, body string) int {
	t.Helper()
	key := os.Getenv("ANTHROPIC_API_KEY")
	if key == "" {
		t.Skip("ANTHROPIC_API_KEY not set; skipping real-API step")
	}
	req, _ := http.NewRequest("POST", "https://api.anthropic.com/v1/messages", bytes.NewReader([]byte(body)))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", key)
	req.Header.Set("anthropic-version", "2023-06-01")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("call: %v", err)
	}
	defer resp.Body.Close()
	io.Copy(io.Discard, resp.Body)
	return resp.StatusCode
}

// Anthropic rejects an assistant tool_use with no tool_result in the next
// message. repairMessageHistory drops the orphan; the kept text must still
// reach the API and be accepted.
func TestIntegration_Anthropic_Semantic_OrphanedToolUse(t *testing.T) {
	v := newSchemaValidator(t)

	// Canary: an assistant tool_use whose following user message lacks the
	// matching tool_result is rejected.
	bad := `{"model":"` + testModel() + `","max_tokens":64,"messages":[` +
		`{"role":"user","content":"q"},` +
		`{"role":"assistant","content":[{"type":"text","text":"ok"},{"type":"tool_use","id":"toolu_1","name":"f","input":{}}]},` +
		`{"role":"user","content":"continue"}]}`
	if code := rawStatus(t, bad); code != http.StatusBadRequest {
		t.Fatalf("orphaned tool_use: want 400 from API, got %d (rule may have changed)", code)
	}

	// Adapter: the matching tool_result never arrived (lost/trimmed); the next
	// turn is plain user text. repairMessageHistory drops the orphan tool_use,
	// keeps the assistant text, and the conversation still ends with a user turn.
	req := &model.LLMRequest{
		Config: &genai.GenerateContentConfig{MaxOutputTokens: 64},
		Contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: "q"}}},
			{Role: "model", Parts: []*genai.Part{
				{Text: "ok"},
				{FunctionCall: &genai.FunctionCall{ID: "toolu_1", Name: "f"}},
			}},
			{Role: "user", Parts: []*genai.Part{{Text: "continue"}}},
		},
	}
	body := captureWireBody(t, Config{ModelName: testModel()}, req)
	validateBody(t, v, body)
	sendReal(t, body)
}

// Anthropic rejects thinking blocks outside assistant messages. The adapter
// drops thought parts that arrive under a non-assistant role (ADK rewrites
// foreign-agent events as user-role context), so the request stays valid.
func TestIntegration_Anthropic_Semantic_ThinkingUnderUserRole(t *testing.T) {
	v := newSchemaValidator(t)

	// Canary: a thinking block in a user message is rejected.
	bad := `{"model":"` + testModel() + `","max_tokens":64,"messages":[` +
		`{"role":"user","content":[{"type":"thinking","thinking":"x","signature":"y"}]}]}`
	if code := rawStatus(t, bad); code != http.StatusBadRequest {
		t.Fatalf("thinking under user: want 400 from API, got %d (rule may have changed)", code)
	}

	// Adapter: a user turn carrying a thought part plus real text. The thought
	// is dropped, the text survives.
	req := &model.LLMRequest{
		Config: &genai.GenerateContentConfig{MaxOutputTokens: 64},
		Contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{
				{Text: "real reasoning from another agent", Thought: true, ThoughtSignature: []byte("sig")},
				{Text: "hi"},
			}},
		},
	}
	body := captureWireBody(t, Config{ModelName: testModel()}, req)
	validateBody(t, v, body)
	sendReal(t, body)
}
