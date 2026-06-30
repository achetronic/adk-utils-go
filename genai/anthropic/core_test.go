// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package anthropic

import (
	"encoding/json"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"google.golang.org/genai"
)

// convertContentToMessage is the entry point for translating ADK Content into
// Anthropic message blocks. It must:
//   - return (nil, nil) when the content has no translatable parts, so that
//     callers can drop empty turns instead of sending Anthropic a 400 for
//     "messages must contain at least one block"
//   - emit one Anthropic block per genai Part that carries data, mapping text,
//     function calls and function responses each to their own block type
//   - propagate the Content.Role through convertRoleToAnthropic
func TestConvertContentToMessage(t *testing.T) {
	m := &Model{}

	cases := []struct {
		name      string
		content   *genai.Content
		wantNil   bool
		wantRole  anthropic.MessageParamRole
		wantTypes []string // ordered block discriminators: "text", "tool_use", "tool_result"
	}{
		{
			name:    "no parts yields nil message",
			content: &genai.Content{Role: "user", Parts: []*genai.Part{}},
			wantNil: true,
		},
		{
			name: "single text part",
			content: &genai.Content{
				Role:  "user",
				Parts: []*genai.Part{{Text: "hello world"}},
			},
			wantRole:  anthropic.MessageParamRoleUser,
			wantTypes: []string{"text"},
		},
		{
			name: "function call from model role",
			content: &genai.Content{
				Role: "model",
				Parts: []*genai.Part{
					{FunctionCall: &genai.FunctionCall{
						ID:   "call_1",
						Name: "test_func",
						Args: map[string]any{"a": 1},
					}},
				},
			},
			wantRole:  anthropic.MessageParamRoleAssistant,
			wantTypes: []string{"tool_use"},
		},
		{
			name: "function response from user role",
			content: &genai.Content{
				Role: "user",
				Parts: []*genai.Part{
					{FunctionResponse: &genai.FunctionResponse{
						ID:       "call_1",
						Response: map[string]any{"res": "ok"},
					}},
				},
			},
			wantRole:  anthropic.MessageParamRoleUser,
			wantTypes: []string{"tool_result"},
		},
		{
			name: "mixed parts preserve order",
			content: &genai.Content{
				Role: "model",
				Parts: []*genai.Part{
					{Text: "calling tool"},
					{FunctionCall: &genai.FunctionCall{ID: "call_2", Name: "test_func"}},
				},
			},
			wantRole:  anthropic.MessageParamRoleAssistant,
			wantTypes: []string{"text", "tool_use"},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := m.convertContentToMessage(c.content)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if c.wantNil {
				if got != nil {
					t.Errorf("expected nil message, got %#v", got)
				}
				return
			}

			if got == nil {
				t.Fatalf("expected message, got nil")
			}
			if got.Role != c.wantRole {
				t.Errorf("Role = %q, want %q", got.Role, c.wantRole)
			}

			gotTypes := blockTypes(got.Content)
			if !equalStringSlices(gotTypes, c.wantTypes) {
				t.Errorf("block types = %v, want %v", gotTypes, c.wantTypes)
			}
		})
	}
}

// convertResponse rebuilds a genai.LLMResponse from the SDK's Anthropic
// Message. We feed it a JSON-decoded Message rather than constructing one in
// memory because the SDK's content blocks use an opaque discriminated union
// whose fields are populated during unmarshal — building it manually would
// require reaching into unexported state and create a brittle dependency on
// the SDK's internals.
func TestConvertResponse(t *testing.T) {
	m := &Model{}

	raw := []byte(`{
		"id": "msg_1",
		"role": "assistant",
		"content": [
			{"type": "text", "text": "Hello"},
			{"type": "tool_use", "id": "toolu_1", "name": "my_tool", "input": {"arg": 1}}
		],
		"usage": {"input_tokens": 10, "output_tokens": 20},
		"stop_reason": "end_turn"
	}`)

	var resp anthropic.Message
	if err := json.Unmarshal(raw, &resp); err != nil {
		t.Fatalf("failed to unmarshal Anthropic message fixture: %v", err)
	}

	got, err := m.convertResponse(&resp)
	if err != nil {
		t.Fatalf("convertResponse error: %v", err)
	}

	if got.FinishReason != genai.FinishReasonStop {
		t.Errorf("FinishReason = %v, want %v", got.FinishReason, genai.FinishReasonStop)
	}
	if !got.TurnComplete {
		t.Errorf("TurnComplete = false, want true")
	}
	if got.UsageMetadata == nil {
		t.Fatalf("expected usage metadata to be populated")
	}
	if got.UsageMetadata.PromptTokenCount != 10 {
		t.Errorf("PromptTokenCount = %d, want 10", got.UsageMetadata.PromptTokenCount)
	}
	if got.UsageMetadata.CandidatesTokenCount != 20 {
		t.Errorf("CandidatesTokenCount = %d, want 20", got.UsageMetadata.CandidatesTokenCount)
	}
	if got.UsageMetadata.TotalTokenCount != 30 {
		t.Errorf("TotalTokenCount = %d, want 30", got.UsageMetadata.TotalTokenCount)
	}

	if got.Content == nil {
		t.Fatalf("expected Content to be populated")
	}
	if got.Content.Role != genai.RoleModel {
		t.Errorf("Content.Role = %q, want %q", got.Content.Role, genai.RoleModel)
	}
	if len(got.Content.Parts) != 2 {
		t.Fatalf("expected 2 parts, got %d", len(got.Content.Parts))
	}
	if got.Content.Parts[0].Text != "Hello" {
		t.Errorf("Parts[0].Text = %q, want %q", got.Content.Parts[0].Text, "Hello")
	}
	if got.Content.Parts[1].FunctionCall == nil {
		t.Fatalf("Parts[1].FunctionCall = nil, want populated")
	}
	if got.Content.Parts[1].FunctionCall.Name != "my_tool" {
		t.Errorf("Parts[1].FunctionCall.Name = %q, want %q", got.Content.Parts[1].FunctionCall.Name, "my_tool")
	}
}

// TestConvertResponseCachedUsage pins the prompt-token accounting when
// prompt caching is active: PromptTokenCount is input_tokens +
// cache_read_input_tokens + cache_creation_input_tokens, with the
// read-hit portion also surfaced in CachedContentTokenCount. The
// cache-less case is pinned by TestConvertResponse above.
func TestConvertResponseCachedUsage(t *testing.T) {
	m := &Model{}

	raw := []byte(`{
		"id": "msg_1",
		"role": "assistant",
		"content": [{"type": "text", "text": "Hello"}],
		"usage": {
			"input_tokens": 2,
			"output_tokens": 20,
			"cache_read_input_tokens": 130000,
			"cache_creation_input_tokens": 1000
		},
		"stop_reason": "end_turn"
	}`)

	var resp anthropic.Message
	if err := json.Unmarshal(raw, &resp); err != nil {
		t.Fatalf("failed to unmarshal Anthropic message fixture: %v", err)
	}

	got, err := m.convertResponse(&resp)
	if err != nil {
		t.Fatalf("convertResponse error: %v", err)
	}
	if got.UsageMetadata == nil {
		t.Fatalf("expected usage metadata to be populated")
	}
	if want := int32(2 + 130000 + 1000); got.UsageMetadata.PromptTokenCount != want {
		t.Errorf("PromptTokenCount = %d, want %d (input + cache_read + cache_creation)",
			got.UsageMetadata.PromptTokenCount, want)
	}
	if got.UsageMetadata.CachedContentTokenCount != 130000 {
		t.Errorf("CachedContentTokenCount = %d, want 130000", got.UsageMetadata.CachedContentTokenCount)
	}
	if got.UsageMetadata.CandidatesTokenCount != 20 {
		t.Errorf("CandidatesTokenCount = %d, want 20", got.UsageMetadata.CandidatesTokenCount)
	}
	if want := int32(131002 + 20); got.UsageMetadata.TotalTokenCount != want {
		t.Errorf("TotalTokenCount = %d, want %d", got.UsageMetadata.TotalTokenCount, want)
	}
}

// blockTypes returns a discriminator string per block in the order they appear,
// so tests can assert on shape without reaching into the SDK's union internals.
func blockTypes(blocks []anthropic.ContentBlockParamUnion) []string {
	out := make([]string, 0, len(blocks))
	for _, b := range blocks {
		switch {
		case b.OfText != nil:
			out = append(out, "text")
		case b.OfToolUse != nil:
			out = append(out, "tool_use")
		case b.OfToolResult != nil:
			out = append(out, "tool_result")
		case b.OfImage != nil:
			out = append(out, "image")
		case b.OfDocument != nil:
			out = append(out, "document")
		default:
			out = append(out, "unknown")
		}
	}
	return out
}

func equalStringSlices(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
