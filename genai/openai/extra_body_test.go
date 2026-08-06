// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package openai

import (
	"reflect"
	"testing"

	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"
)

// simpleTurn is the smallest request that still produces a body.
func simpleTurn() *model.LLMRequest {
	return &model.LLMRequest{
		Config:   &genai.GenerateContentConfig{},
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "hi"}}}},
	}
}

// ExtraBody lands at the root of the request body, next to model and
// messages, with nested objects intact. This is how a caller reaches provider
// extensions the Chat Completions schema does not define, such as
// OpenRouter's reasoning controls.
func TestWireBody_ExtraBody(t *testing.T) {
	body := captureBodyWithConfig(t, Config{
		ExtraBody: map[string]any{
			"reasoning":  map[string]any{"effort": "high", "exclude": false},
			"provider":   map[string]any{"order": []any{"anthropic", "openai"}},
			"transforms": []any{"middle-out"},
		},
	}, simpleTurn())

	reasoning, ok := body["reasoning"].(map[string]any)
	if !ok {
		t.Fatalf("reasoning missing from the request root: %v", body)
	}
	if reasoning["effort"] != "high" || reasoning["exclude"] != false {
		t.Errorf("reasoning = %#v", reasoning)
	}

	provider, ok := body["provider"].(map[string]any)
	if !ok {
		t.Fatalf("provider missing: %v", body)
	}
	if !reflect.DeepEqual(provider["order"], []any{"anthropic", "openai"}) {
		t.Errorf("provider.order = %#v", provider["order"])
	}

	if !reflect.DeepEqual(body["transforms"], []any{"middle-out"}) {
		t.Errorf("transforms = %#v", body["transforms"])
	}

	// The fields the adapter owns must still be there.
	if body["model"] != "gpt-test" {
		t.Errorf("model = %v, want the configured model", body["model"])
	}
	if msgs, _ := body["messages"].([]any); len(msgs) != 1 {
		t.Errorf("messages = %v, want the single user turn", body["messages"])
	}
}

// The same extensions have to reach the streaming endpoint, and must not
// disturb the usage opt-in the adapter sets there.
func TestWireBody_ExtraBodyOnStreamingRequest(t *testing.T) {
	body := captureStreamBodyWithConfig(t, Config{
		ExtraBody: map[string]any{"reasoning": map[string]any{"max_tokens": 2000}},
	}, simpleTurn())

	reasoning, ok := body["reasoning"].(map[string]any)
	if !ok {
		t.Fatalf("reasoning missing from the streaming request: %v", body)
	}
	if reasoning["max_tokens"] != float64(2000) {
		t.Errorf("reasoning.max_tokens = %v", reasoning["max_tokens"])
	}
	if body["stream"] != true {
		t.Errorf("stream = %v, want true", body["stream"])
	}
	opts, ok := body["stream_options"].(map[string]any)
	if !ok || opts["include_usage"] != true {
		t.Errorf("stream_options.include_usage = %v, want true", body["stream_options"])
	}
}

// An empty or absent ExtraBody changes nothing about the request.
func TestWireBody_ExtraBodyEmpty(t *testing.T) {
	plain := captureBody(t, simpleTurn())
	empty := captureBodyWithConfig(t, Config{ExtraBody: map[string]any{}}, simpleTurn())

	if !reflect.DeepEqual(plain, empty) {
		t.Errorf("an empty ExtraBody changed the request:\nwithout %#v\nwith    %#v", plain, empty)
	}
}

// The map is copied at construction, so a caller reusing or mutating its own
// map cannot alter what an already-built model sends.
func TestExtraBody_IsCopiedAtConstruction(t *testing.T) {
	caller := map[string]any{"reasoning": map[string]any{"effort": "low"}}

	m := New(Config{ModelName: "gpt-test", ExtraBody: caller})

	caller["reasoning"] = map[string]any{"effort": "high"}
	caller["injected"] = true

	params, err := m.buildChatCompletionParams(simpleTurn())
	if err != nil {
		t.Fatalf("buildChatCompletionParams: %v", err)
	}
	body := extraFields(t, params)

	reasoning, ok := body["reasoning"].(map[string]any)
	if !ok {
		t.Fatalf("reasoning missing: %v", body)
	}
	if reasoning["effort"] != "low" {
		t.Errorf("effort = %v, want the value present at construction", reasoning["effort"])
	}
	if _, ok := body["injected"]; ok {
		t.Errorf("a key added to the caller's map after construction leaked into the request: %v", body)
	}
}
