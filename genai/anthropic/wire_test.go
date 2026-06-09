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
	"testing"

	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// captureBody spins up a fake Anthropic endpoint, points a Model at it via
// BaseURL, fires one non-streaming request and returns the raw JSON body
// the SDK actually put on the wire. This exercises reasoningRequestOptions
// end to end (the option mutates the serialized body via sjson, so the only
// faithful way to assert on it is to inspect the real outgoing request).
func captureBody(t *testing.T, cfg Config) map[string]any {
	t.Helper()

	var captured []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		// Minimal valid Messages response so convertResponse succeeds.
		io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	defer srv.Close()

	cfg.BaseURL = srv.URL
	cfg.APIKey = "test-key"
	if cfg.ModelName == "" {
		cfg.ModelName = "claude-test"
	}
	m := New(cfg)

	req := &model.LLMRequest{
		Config:   &genai.GenerateContentConfig{},
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "hi"}}}},
	}
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

// The adaptive path is the one that fixed the Opus 4.5+ 400. The wire body
// must carry thinking.type=adaptive AND output_config.effort=<level>.
func TestWireBody_Adaptive(t *testing.T) {
	body := captureBody(t, Config{ThinkingEffort: "high"})

	thinking, ok := body["thinking"].(map[string]any)
	if !ok {
		t.Fatalf("expected a thinking object, got %v", body["thinking"])
	}
	if thinking["type"] != "adaptive" {
		t.Errorf("thinking.type = %v, want \"adaptive\"", thinking["type"])
	}
	oc, ok := body["output_config"].(map[string]any)
	if !ok {
		t.Fatalf("expected an output_config object, got %v", body["output_config"])
	}
	if oc["effort"] != "high" {
		t.Errorf("output_config.effort = %v, want \"high\"", oc["effort"])
	}
}

// The classic path must send thinking.type=enabled with the budget, and must
// NOT carry output_config.effort.
func TestWireBody_Enabled(t *testing.T) {
	body := captureBody(t, Config{ThinkingBudgetTokens: 4096, MaxOutputTokens: 8192})

	thinking, ok := body["thinking"].(map[string]any)
	if !ok {
		t.Fatalf("expected a thinking object, got %v", body["thinking"])
	}
	if thinking["type"] != "enabled" {
		t.Errorf("thinking.type = %v, want \"enabled\"", thinking["type"])
	}
	if got, want := thinking["budget_tokens"], float64(4096); got != want {
		t.Errorf("thinking.budget_tokens = %v, want %v", got, want)
	}
	if _, present := body["output_config"]; present {
		t.Errorf("classic mode must not send output_config, got %v", body["output_config"])
	}
}

// With no reasoning configured the body must carry no thinking and no
// output_config at all.
func TestWireBody_NoReasoning(t *testing.T) {
	body := captureBody(t, Config{})

	if v, present := body["thinking"]; present {
		t.Errorf("no reasoning must not send thinking, got %v", v)
	}
	if v, present := body["output_config"]; present {
		t.Errorf("no reasoning must not send output_config, got %v", v)
	}
}
