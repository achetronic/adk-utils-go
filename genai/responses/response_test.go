// Copyright 2025 achetronic
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

package responses

import (
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	"github.com/openai/openai-go/v3/responses"
	"google.golang.org/genai"
)

// convertResponse rebuilds a genai.LLMResponse from the Responses API
// Response. Tests build the response via json.Unmarshal because the SDK
// populates internal JSON metadata through its generated UnmarshalJSON,
// which typed union dispatch (AsAny) depends on.

func TestConvertResponse_EmptyOutput(t *testing.T) {
	resp := &responses.Response{}
	_, err := convertResponse(resp)
	if !errors.Is(err, ErrNoOutputInResponse) {
		t.Errorf("err = %v, want %v", err, ErrNoOutputInResponse)
	}
}

func TestConvertResponse_TextOnly(t *testing.T) {
	raw := []byte(`{
		"id": "resp-1",
		"status": "completed",
		"output": [{
			"type": "message",
			"id": "msg-1",
			"role": "assistant",
			"status": "completed",
			"content": [{"type": "output_text", "text": "hello world"}]
		}],
		"usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
			"input_tokens_details": {"cached_tokens": 0},
			"output_tokens_details": {"reasoning_tokens": 0}}
	}`)

	var resp responses.Response
	if err := json.Unmarshal(raw, &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	got, err := convertResponse(&resp)
	if err != nil {
		t.Fatalf("convertResponse: %v", err)
	}
	if !got.TurnComplete {
		t.Errorf("TurnComplete = false, want true")
	}
	if got.FinishReason != genai.FinishReasonStop {
		t.Errorf("FinishReason = %v, want Stop", got.FinishReason)
	}
	if len(got.Content.Parts) != 1 || got.Content.Parts[0].Text != "hello world" {
		t.Errorf("Content parts = %#v, want single text", got.Content)
	}
	if got.UsageMetadata == nil || got.UsageMetadata.TotalTokenCount != 15 {
		t.Errorf("UsageMetadata = %#v, want TotalTokenCount=15", got.UsageMetadata)
	}
}

func TestConvertResponse_ToolCall(t *testing.T) {
	raw := []byte(`{
		"id": "resp-2",
		"status": "completed",
		"output": [{
			"type": "function_call",
			"id": "fc-1",
			"call_id": "call_42",
			"name": "search",
			"arguments": "{\"q\":\"weather\"}"
		}],
		"usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
			"input_tokens_details": {"cached_tokens": 0},
			"output_tokens_details": {"reasoning_tokens": 0}}
	}`)

	var resp responses.Response
	if err := json.Unmarshal(raw, &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	got, err := convertResponse(&resp)
	if err != nil {
		t.Fatalf("convertResponse: %v", err)
	}
	if len(got.Content.Parts) != 1 {
		t.Fatalf("expected 1 part, got %d", len(got.Content.Parts))
	}
	fc := got.Content.Parts[0].FunctionCall
	if fc == nil {
		t.Fatalf("expected FunctionCall")
	}
	if fc.ID != "call_42" || fc.Name != "search" {
		t.Errorf("FunctionCall = %#v, want id=call_42 name=search", fc)
	}
	if want := map[string]any{"q": "weather"}; !reflect.DeepEqual(fc.Args, want) {
		t.Errorf("Args = %#v, want %#v", fc.Args, want)
	}
}

func TestConvertResponse_TextPlusToolCall(t *testing.T) {
	raw := []byte(`{
		"id": "resp-3",
		"status": "completed",
		"output": [
			{
				"type": "message",
				"id": "msg-1",
				"role": "assistant",
				"status": "completed",
				"content": [{"type": "output_text", "text": "looking up"}]
			},
			{
				"type": "function_call",
				"id": "fc-1",
				"call_id": "call_1",
				"name": "get_weather",
				"arguments": "{\"city\":\"Madrid\"}"
			}
		],
		"usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
			"input_tokens_details": {"cached_tokens": 0},
			"output_tokens_details": {"reasoning_tokens": 0}}
	}`)

	var resp responses.Response
	if err := json.Unmarshal(raw, &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	got, err := convertResponse(&resp)
	if err != nil {
		t.Fatalf("convertResponse: %v", err)
	}
	if len(got.Content.Parts) != 2 {
		t.Fatalf("expected 2 parts, got %d", len(got.Content.Parts))
	}
	if got.Content.Parts[0].Text != "looking up" {
		t.Errorf("first part text = %q", got.Content.Parts[0].Text)
	}
	if got.Content.Parts[1].FunctionCall == nil {
		t.Errorf("second part should be a FunctionCall")
	}
}

// Reasoning items (ResponseReasoningItem) appear before messages in the
// output array. Their summary texts become Parts with Thought=true,
// preserving the temporal order.
func TestConvertResponse_WithReasoning(t *testing.T) {
	raw := []byte(`{
		"id": "resp-4",
		"status": "completed",
		"output": [
			{
				"type": "reasoning",
				"id": "rs-1",
				"summary": [{"type": "summary_text", "text": "Let me think about this."}]
			},
			{
				"type": "message",
				"id": "msg-1",
				"role": "assistant",
				"status": "completed",
				"content": [{"type": "output_text", "text": "The answer is 42."}]
			}
		],
		"usage": {"input_tokens": 5, "output_tokens": 10, "total_tokens": 15,
			"input_tokens_details": {"cached_tokens": 0},
			"output_tokens_details": {"reasoning_tokens": 7}}
	}`)

	var resp responses.Response
	if err := json.Unmarshal(raw, &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	got, err := convertResponse(&resp)
	if err != nil {
		t.Fatalf("convertResponse: %v", err)
	}
	if len(got.Content.Parts) != 2 {
		t.Fatalf("expected 2 parts, got %d", len(got.Content.Parts))
	}
	if !got.Content.Parts[0].Thought || got.Content.Parts[0].Text != "Let me think about this." {
		t.Errorf("first part should be a thought: %#v", got.Content.Parts[0])
	}
	if got.Content.Parts[1].Thought || got.Content.Parts[1].Text != "The answer is 42." {
		t.Errorf("second part should be plain text: %#v", got.Content.Parts[1])
	}
	if got.UsageMetadata == nil || got.UsageMetadata.ThoughtsTokenCount != 7 {
		t.Errorf("ThoughtsTokenCount = %v, want 7", got.UsageMetadata)
	}
	// Reasoning parts must carry reasoning_id for stateless round-tripping
	pm := got.Content.Parts[0].PartMetadata
	if pm == nil || pm["reasoning_id"] != "rs-1" {
		t.Errorf("PartMetadata = %v, want reasoning_id=rs-1", pm)
	}
}

// Phase metadata on assistant messages must be preserved in PartMetadata
// so that subsequent requests can echo it back to the model.
func TestConvertResponse_PhaseMetadata(t *testing.T) {
	raw := []byte(`{
		"id": "resp-5",
		"status": "completed",
		"output": [{
			"type": "message",
			"id": "msg-1",
			"role": "assistant",
			"status": "completed",
			"phase": "commentary",
			"content": [{"type": "output_text", "text": "intermediate"}]
		}],
		"usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
			"input_tokens_details": {"cached_tokens": 0},
			"output_tokens_details": {"reasoning_tokens": 0}}
	}`)

	var resp responses.Response
	if err := json.Unmarshal(raw, &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	got, err := convertResponse(&resp)
	if err != nil {
		t.Fatalf("convertResponse: %v", err)
	}
	if len(got.Content.Parts) != 1 {
		t.Fatalf("expected 1 part, got %d", len(got.Content.Parts))
	}
	pm := got.Content.Parts[0].PartMetadata
	if pm == nil {
		t.Fatalf("PartMetadata is nil, want phase and message_id")
	}
	if pm["phase"] != "commentary" {
		t.Errorf("PartMetadata[\"phase\"] = %v, want commentary", pm["phase"])
	}
	if pm["message_id"] != "msg-1" {
		t.Errorf("PartMetadata[\"message_id\"] = %v, want msg-1", pm["message_id"])
	}
}

// Messages without a phase still carry message_id in PartMetadata for
// round-tripping, but must not have a "phase" key.
func TestConvertResponse_NoPhase(t *testing.T) {
	raw := []byte(`{
		"id": "resp-6",
		"status": "completed",
		"output": [{
			"type": "message",
			"id": "msg-1",
			"role": "assistant",
			"status": "completed",
			"content": [{"type": "output_text", "text": "plain"}]
		}],
		"usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
			"input_tokens_details": {"cached_tokens": 0},
			"output_tokens_details": {"reasoning_tokens": 0}}
	}`)

	var resp responses.Response
	if err := json.Unmarshal(raw, &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	got, err := convertResponse(&resp)
	if err != nil {
		t.Fatalf("convertResponse: %v", err)
	}
	pm := got.Content.Parts[0].PartMetadata
	if pm != nil {
		if _, hasPhase := pm["phase"]; hasPhase {
			t.Errorf("PartMetadata has phase key, want absent for messages without phase")
		}
		if pm["message_id"] != "msg-1" {
			t.Errorf("message_id = %v, want msg-1", pm["message_id"])
		}
	}
}

func TestConvertUsageMetadata(t *testing.T) {
	t.Run("populated usage maps correctly", func(t *testing.T) {
		raw := []byte(`{
			"input_tokens": 11, "output_tokens": 22, "total_tokens": 33,
			"input_tokens_details": {"cached_tokens": 5},
			"output_tokens_details": {"reasoning_tokens": 8}
		}`)
		var usage responses.ResponseUsage
		if err := json.Unmarshal(raw, &usage); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		got := convertUsageMetadata(usage)
		if got == nil {
			t.Fatalf("got nil")
		}
		want := &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:        11,
			CandidatesTokenCount:    22,
			TotalTokenCount:         33,
			ThoughtsTokenCount:      8,
			CachedContentTokenCount: 5,
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("got = %#v, want %#v", got, want)
		}
	})

	t.Run("zero total tokens returns nil", func(t *testing.T) {
		usage := responses.ResponseUsage{}
		if got := convertUsageMetadata(usage); got != nil {
			t.Errorf("got = %#v, want nil", got)
		}
	})
}
