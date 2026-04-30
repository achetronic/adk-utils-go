// Copyright 2025 achetronic
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

package openai

import (
	"strings"
	"testing"

	"github.com/openai/openai-go/v3"
	"google.golang.org/genai"
)

// convertContentToMessages bridges ADK Content into OpenAI's per-role message
// shapes. The non-trivial bit is that a single Content can produce multiple
// OpenAI messages: function responses must become tool messages of their own,
// while text/media/tool calls coalesce into a single role message. This test
// pins down the resulting message shape (count, roles, payload) without
// reaching into the SDK's union internals — we use a small kind() helper that
// reads which OfXxx pointer is set.
func TestConvertContentToMessages(t *testing.T) {
	cases := []struct {
		name      string
		content   *genai.Content
		wantKinds []string // ordered "role" discriminators
		assert    func(t *testing.T, msgs []openai.ChatCompletionMessageParamUnion)
	}{
		{
			name: "user text becomes a single user message",
			content: &genai.Content{
				Role:  "user",
				Parts: []*genai.Part{{Text: "hello"}},
			},
			wantKinds: []string{"user"},
		},
		{
			name: "model text becomes an assistant message",
			content: &genai.Content{
				Role:  "model",
				Parts: []*genai.Part{{Text: "hi back"}},
			},
			wantKinds: []string{"assistant"},
		},
		{
			name: "function response becomes a standalone tool message preserving the (normalised) ID",
			content: &genai.Content{
				Role: "user",
				Parts: []*genai.Part{
					{FunctionResponse: &genai.FunctionResponse{
						ID:       "call_short",
						Response: map[string]any{"ok": true},
					}},
				},
			},
			wantKinds: []string{"tool"},
			assert: func(t *testing.T, msgs []openai.ChatCompletionMessageParamUnion) {
				if got := msgs[0].OfTool.ToolCallID; got != "call_short" {
					t.Errorf("ToolCallID = %q, want %q", got, "call_short")
				}
				content := msgs[0].OfTool.Content.OfString.Value
				if !strings.Contains(content, `"ok":true`) {
					t.Errorf("tool content = %q, expected to contain serialised response", content)
				}
			},
		},
		{
			name: "model text plus function call coalesce into a single assistant message",
			content: &genai.Content{
				Role: "model",
				Parts: []*genai.Part{
					{Text: "calling tool"},
					{FunctionCall: &genai.FunctionCall{
						ID:   "call_xyz",
						Name: "do_thing",
						Args: map[string]any{"foo": "bar"},
					}},
				},
			},
			wantKinds: []string{"assistant"},
			assert: func(t *testing.T, msgs []openai.ChatCompletionMessageParamUnion) {
				assistant := msgs[0].OfAssistant
				if assistant == nil {
					t.Fatalf("expected assistant message")
				}
				if got := assistant.Content.OfString.Value; got != "calling tool" {
					t.Errorf("assistant text = %q, want \"calling tool\"", got)
				}
				if len(assistant.ToolCalls) != 1 {
					t.Fatalf("expected 1 tool call, got %d", len(assistant.ToolCalls))
				}
				tc := assistant.ToolCalls[0].OfFunction
				if tc == nil {
					t.Fatalf("expected function tool call")
				}
				if tc.ID != "call_xyz" {
					t.Errorf("tool call ID = %q, want %q", tc.ID, "call_xyz")
				}
				if tc.Function.Name != "do_thing" {
					t.Errorf("function name = %q, want %q", tc.Function.Name, "do_thing")
				}
				if !strings.Contains(tc.Function.Arguments, `"foo":"bar"`) {
					t.Errorf("arguments = %q, want serialised map", tc.Function.Arguments)
				}
			},
		},
		{
			name: "user text plus inline image yields a multi-part user message",
			content: &genai.Content{
				Role: "user",
				Parts: []*genai.Part{
					{Text: "describe this"},
					{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte("fakepng")}},
				},
			},
			wantKinds: []string{"user"},
			assert: func(t *testing.T, msgs []openai.ChatCompletionMessageParamUnion) {
				user := msgs[0].OfUser
				if user == nil {
					t.Fatalf("expected user message")
				}
				parts := user.Content.OfArrayOfContentParts
				if len(parts) != 2 {
					t.Fatalf("expected 2 content parts (text + image), got %d", len(parts))
				}
				if parts[0].OfText == nil || parts[0].OfText.Text != "describe this" {
					t.Errorf("first part should be the text, got %#v", parts[0])
				}
				if parts[1].OfImageURL == nil {
					t.Errorf("second part should be the image, got %#v", parts[1])
				}
			},
		},
		{
			name: "function response and assistant call in the same content produce two messages",
			content: &genai.Content{
				Role: "model",
				Parts: []*genai.Part{
					{FunctionResponse: &genai.FunctionResponse{ID: "tool_1", Response: map[string]any{"k": "v"}}},
					{Text: "all done"},
				},
			},
			wantKinds: []string{"tool", "assistant"},
		},
		{
			name: "content with no usable parts produces no messages",
			content: &genai.Content{
				Role:  "user",
				Parts: []*genai.Part{{}}, // empty Part
			},
			wantKinds: nil,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			m := newModelForTest()
			got, err := m.convertContentToMessages(c.content)
			if err != nil {
				t.Fatalf("convertContentToMessages error: %v", err)
			}

			gotKinds := messageRoles(got)
			if !equalStringSlices(gotKinds, c.wantKinds) {
				t.Fatalf("message kinds = %v, want %v", gotKinds, c.wantKinds)
			}

			if c.assert != nil {
				c.assert(t, got)
			}
		})
	}
}

// convertContentToMessages must propagate function call IDs through the
// normalisation layer so OpenAI sees a valid tool_call_id (≤40 chars). The
// inverse mapping must be discoverable through denormalizeToolCallID, which
// callers use when correlating a tool result back to the original ADK ID.
func TestConvertContentToMessages_NormalisesLongToolIDs(t *testing.T) {
	m := newModelForTest()
	long := "call_" + strings.Repeat("z", 100)

	content := &genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{FunctionCall: &genai.FunctionCall{ID: long, Name: "tool", Args: map[string]any{}}},
		},
	}

	msgs, err := m.convertContentToMessages(content)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %d", len(msgs))
	}

	tc := msgs[0].OfAssistant.ToolCalls[0].OfFunction
	if len(tc.ID) > maxToolCallIDLength {
		t.Errorf("tool_call_id length = %d, want <= %d", len(tc.ID), maxToolCallIDLength)
	}
	if got := m.denormalizeToolCallID(tc.ID); got != long {
		t.Errorf("denormalize round-trip failed: got %q, want %q", got, long)
	}
}

// buildRoleMessage routes texts/media/tool calls to the right role-specific
// builder. Anything outside {user, model, system} (after convertRole) must
// return nil so the caller can decide to drop the turn rather than send a
// message with an undefined role.
func TestBuildRoleMessage(t *testing.T) {
	cases := []struct {
		name      string
		role      string
		texts     []string
		toolCalls int
		wantKind  string
	}{
		{"user role builds user message", "user", []string{"hi"}, 0, "user"},
		{"model role builds assistant message", "model", []string{"hello"}, 0, "assistant"},
		{"system role builds system message", "system", []string{"prompt"}, 0, "system"},
		{"unknown role returns nil", "tool", []string{"x"}, 0, ""},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			m := newModelForTest()
			toolCalls := make([]openai.ChatCompletionMessageToolCallUnionParam, c.toolCalls)
			got := m.buildRoleMessage(c.role, c.texts, nil, toolCalls)

			if c.wantKind == "" {
				if got != nil {
					t.Errorf("expected nil for unknown role %q, got %#v", c.role, got)
				}
				return
			}
			if got == nil {
				t.Fatalf("expected %s message, got nil", c.wantKind)
			}
			if kind := messageKind(*got); kind != c.wantKind {
				t.Errorf("kind = %q, want %q", kind, c.wantKind)
			}
		})
	}
}

// buildAssistantMessage never returns nil even when both texts and toolCalls
// are empty: callers depend on a non-nil pointer to inspect the union
// discriminator. The text content slot must remain unset when no texts are
// passed, otherwise OpenAI will see an empty string instead of "no content"
// and may refuse to dispatch a tool-only turn.
func TestBuildAssistantMessage(t *testing.T) {
	cases := []struct {
		name      string
		texts     []string
		toolCalls int
		wantText  string
		wantTools int
	}{
		{"text only", []string{"hi"}, 0, "hi", 0},
		{"tool calls only", nil, 2, "", 2},
		{"text + tool calls", []string{"a", "b"}, 1, "a\nb", 1},
		{"empty produces empty assistant", nil, 0, "", 0},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			toolCalls := make([]openai.ChatCompletionMessageToolCallUnionParam, c.toolCalls)
			got := buildAssistantMessage(c.texts, toolCalls)
			if got == nil {
				t.Fatalf("buildAssistantMessage returned nil")
			}
			a := got.OfAssistant
			if a == nil {
				t.Fatalf("OfAssistant = nil")
			}
			if got := a.Content.OfString.Value; got != c.wantText {
				t.Errorf("text = %q, want %q", got, c.wantText)
			}
			if len(a.ToolCalls) != c.wantTools {
				t.Errorf("tool calls = %d, want %d", len(a.ToolCalls), c.wantTools)
			}
		})
	}
}

// buildUserMessage takes the simple path (string content) when there are no
// media parts and switches to the multi-part array shape only when media is
// present. This split matters: the simple path is required for any provider
// that doesn't support multi-modal input (Ollama, older OpenAI-compatible
// servers).
func TestBuildUserMessage(t *testing.T) {
	t.Run("text only takes the simple string path", func(t *testing.T) {
		got := buildUserMessage([]string{"hello", "world"}, nil)
		if got.OfUser == nil {
			t.Fatalf("OfUser = nil")
		}
		if v := got.OfUser.Content.OfString.Value; v != "hello\nworld" {
			t.Errorf("string content = %q, want %q", v, "hello\nworld")
		}
		if got.OfUser.Content.OfArrayOfContentParts != nil {
			t.Errorf("expected array path to be unused when no media")
		}
	})

	t.Run("text plus media uses the array-of-parts path", func(t *testing.T) {
		media := []openai.ChatCompletionContentPartUnionParam{
			{OfImageURL: &openai.ChatCompletionContentPartImageParam{}},
		}
		got := buildUserMessage([]string{"caption"}, media)
		parts := got.OfUser.Content.OfArrayOfContentParts
		if len(parts) != 2 {
			t.Fatalf("parts = %d, want 2", len(parts))
		}
		if parts[0].OfText == nil || parts[0].OfText.Text != "caption" {
			t.Errorf("first part = %#v, want text \"caption\"", parts[0])
		}
		if parts[1].OfImageURL == nil {
			t.Errorf("second part should be the image")
		}
	})
}

// messageRoles inspects the union to return per-message role discriminators
// in order. Tests use this instead of fishing into the SDK's internals so
// they remain stable across SDK refactors.
func messageRoles(msgs []openai.ChatCompletionMessageParamUnion) []string {
	if len(msgs) == 0 {
		return nil
	}
	out := make([]string, 0, len(msgs))
	for _, m := range msgs {
		out = append(out, messageKind(m))
	}
	return out
}

func messageKind(m openai.ChatCompletionMessageParamUnion) string {
	switch {
	case m.OfUser != nil:
		return "user"
	case m.OfAssistant != nil:
		return "assistant"
	case m.OfSystem != nil:
		return "system"
	case m.OfTool != nil:
		return "tool"
	case m.OfDeveloper != nil:
		return "developer"
	case m.OfFunction != nil:
		return "function"
	default:
		return "unknown"
	}
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
