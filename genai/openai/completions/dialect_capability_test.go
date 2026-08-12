// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package completions

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"testing"

	"github.com/openai/openai-go/v3"
	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"
)

// mistralDialect is the test double for providers that reject the OpenAI
// tool_call_id shape: Mistral requires exactly 9 alphanumeric characters.
type mistralDialect struct{}

func (mistralDialect) Name() string { return "mistral-test" }

func (mistralDialect) NormalizeToolID(id string) string {
	sum := sha256.Sum256([]byte(id))
	return hex.EncodeToString(sum[:])[:9]
}

// A dialect's ToolIDNormalizer owns the wire shape entirely; the adapter
// keeps the reverse mapping so ADK keeps seeing its original IDs.
func TestNormalizeToolCallID_Dialect(t *testing.T) {
	m := New(Config{ModelName: "gpt-test", Dialect: mistralDialect{}})

	original := "call_abc-123_xyz"
	wire := m.normalizeToolCallID(original)
	if len(wire) != 9 {
		t.Fatalf("wire ID = %q, want 9 characters", wire)
	}
	if wire == original {
		t.Fatalf("wire ID unchanged, want the dialect's shape")
	}
	if got := m.denormalizeToolCallID(wire); got != original {
		t.Errorf("denormalize = %q, want the original ID back", got)
	}
}

// On the wire, the dialect's ID shape must reach both the assistant
// message's tool_calls and the tool message that refers back to it, so the
// provider can correlate the pair.
func TestWireBody_DialectToolIDsCorrelate(t *testing.T) {
	body := captureBody(t, &model.LLMRequest{
		Config: &genai.GenerateContentConfig{},
		Contents: []*genai.Content{
			{Role: "model", Parts: []*genai.Part{
				{FunctionCall: &genai.FunctionCall{ID: "call_abc-123_xyz", Name: "get_weather", Args: map[string]any{"city": "Madrid"}}},
			}},
			{Role: "user", Parts: []*genai.Part{
				{FunctionResponse: &genai.FunctionResponse{ID: "call_abc-123_xyz", Response: map[string]any{"temp": 21}}},
			}},
		},
	}, mistralDialect{})

	assistant := messageOfRole(t, body, "assistant")
	calls, _ := assistant["tool_calls"].([]any)
	if len(calls) != 1 {
		t.Fatalf("expected 1 tool_call, got %v", assistant["tool_calls"])
	}
	fn, _ := calls[0].(map[string]any)
	wireID, _ := fn["id"].(string)
	if len(wireID) != 9 {
		t.Fatalf("tool_call id = %q, want the dialect's 9-char shape", wireID)
	}

	tool := messageOfRole(t, body, "tool")
	if tool["tool_call_id"] != wireID {
		t.Errorf("tool_call_id = %v, want %q so the pair correlates", tool["tool_call_id"], wireID)
	}
}

// xaiDialect is the test double for xAI's constraint: reasoning models
// reject stop sequences (and the penalty knobs) even though the schema
// defines them. The adjuster strips them when a thinking config is present.
type xaiDialect struct{}

func (xaiDialect) Name() string { return "xai-test" }

func (xaiDialect) AdjustParams(params *openai.ChatCompletionNewParams, req *model.LLMRequest, stream bool) {
	if req.Config != nil && req.Config.ThinkingConfig != nil {
		params.Stop = openai.ChatCompletionNewParamsStopUnion{}
	}
}

// The ParamsAdjuster sees the final params, after ExtraBody, and mutates
// them in place. With a thinking config the stop sequences must not reach
// the wire; without one they pass through untouched.
func TestWireBody_ParamsAdjusterStripsStop(t *testing.T) {
	reqWithThinking := func() *model.LLMRequest {
		return &model.LLMRequest{
			Config: &genai.GenerateContentConfig{
				StopSequences: []string{"END"},
				ThinkingConfig: &genai.ThinkingConfig{
					ThinkingLevel: genai.ThinkingLevelHigh,
				},
			},
			Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "hi"}}}},
		}
	}

	withDialect := captureBody(t, reqWithThinking(), xaiDialect{})
	if _, has := withDialect["stop"]; has {
		t.Errorf("stop present with the xAI dialect: %v", withDialect["stop"])
	}

	withoutDialect := captureBody(t, reqWithThinking(), nil)
	if got := withoutDialect["stop"]; got != "END" {
		t.Errorf("stop = %v, want END without a dialect", got)
	}
}

// deepseekDialect is the test double for providers that report cache usage
// outside the standard object: DeepSeek puts prompt_cache_hit_tokens at
// the usage root instead of prompt_tokens_details.
type deepseekDialect struct{}

func (deepseekDialect) Name() string { return "deepseek-test" }

func (deepseekDialect) DecodeUsage(rawJSON string, usage *genai.GenerateContentResponseUsageMetadata) {
	probe := probeRawJSON(rawJSON)
	if probe == nil {
		return
	}
	var hit int64
	if raw, ok := probe["prompt_cache_hit_tokens"]; ok {
		if err := json.Unmarshal(raw, &hit); err == nil {
			usage.CachedContentTokenCount = int32(hit)
		}
	}
}

// The UsageDecoder folds the provider's buckets into the metadata the
// adapter already built, so cost-aware consumers see cache hits priced at
// the discounted rate even when the provider does not use the standard
// prompt_tokens_details position.
func TestConvertResponse_DialectUsage(t *testing.T) {
	raw := []byte(`{
		"id": "chatcmpl-x",
		"object": "chat.completion",
		"created": 0,
		"model": "deepseek-chat",
		"choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
		"usage": {
			"prompt_tokens": 100,
			"completion_tokens": 10,
			"total_tokens": 110,
			"prompt_cache_hit_tokens": 80,
			"prompt_cache_miss_tokens": 20
		}
	}`)

	var resp openai.ChatCompletion
	if err := json.Unmarshal(raw, &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	m := New(Config{ModelName: "gpt-test", Dialect: deepseekDialect{}})
	got, err := m.convertResponse(&resp)
	if err != nil {
		t.Fatalf("convertResponse: %v", err)
	}
	if got.UsageMetadata == nil {
		t.Fatalf("UsageMetadata = nil")
	}
	if got.UsageMetadata.CachedContentTokenCount != 80 {
		t.Errorf("CachedContentTokenCount = %d, want 80 from prompt_cache_hit_tokens", got.UsageMetadata.CachedContentTokenCount)
	}

	// Without the dialect the non-standard position is invisible and the
	// standard mapping stands.
	plain := New(Config{ModelName: "gpt-test"})
	got, err = plain.convertResponse(&resp)
	if err != nil {
		t.Fatalf("convertResponse: %v", err)
	}
	if got.UsageMetadata.CachedContentTokenCount != 0 {
		t.Errorf("CachedContentTokenCount = %d, want 0 without a dialect", got.UsageMetadata.CachedContentTokenCount)
	}
}

// The reasoning-effort knob varies by provider: OpenAI uses the typed
// reasoning_effort field, OpenRouter a reasoning object at the request
// root. A dialect implementing ThinkingMapper owns the mapping; without
// one the typed OpenAI field stands. Both cases pinned at the bytes.
func TestWireBody_ThinkingLevel(t *testing.T) {
	req := func() *model.LLMRequest {
		return &model.LLMRequest{
			Config: &genai.GenerateContentConfig{
				ThinkingConfig: &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelHigh},
			},
			Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "hi"}}}},
		}
	}

	t.Run("openrouter dialect writes the reasoning object, not the typed field", func(t *testing.T) {
		body := captureBody(t, req(), OpenRouter)
		reasoning, ok := body["reasoning"].(map[string]any)
		if !ok {
			t.Fatalf("reasoning object missing at body root: %v", body)
		}
		if reasoning["effort"] != "high" {
			t.Errorf("reasoning.effort = %v, want high", reasoning["effort"])
		}
		if _, has := body["reasoning_effort"]; has {
			t.Errorf("reasoning_effort set alongside the dialect's knob: %v", body["reasoning_effort"])
		}
	})

	t.Run("no dialect uses the typed OpenAI field", func(t *testing.T) {
		body := captureBody(t, req(), nil)
		if body["reasoning_effort"] != "high" {
			t.Errorf("reasoning_effort = %v, want high", body["reasoning_effort"])
		}
		if _, has := body["reasoning"]; has {
			t.Errorf("reasoning object present without a dialect: %v", body)
		}
	})
}
