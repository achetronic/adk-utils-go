// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package openai

import (
	"encoding/json"
	"strings"
	"testing"

	"google.golang.org/genai"
)

// A thought Part replayed as history must leave the assistant message's
// content holding only the reply, with the reasoning under its own field.
// Merging the two would hide the chain of thought inside the answer and would
// omit the field DeepSeek in thinking mode and Kimi K2 thinking require.
func TestConvertContentToMessages_NativeReasoning(t *testing.T) {
	m := newModelForTest()

	msgs, err := m.convertContentToMessages(&genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{Text: "The user wants weather info, I should check my tools...", Thought: true},
			{Text: "It's sunny today."},
		},
	})
	if err != nil {
		t.Fatalf("convertContentToMessages: %v", err)
	}
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %d", len(msgs))
	}

	assistant := msgs[0].OfAssistant
	if assistant == nil {
		t.Fatalf("expected an assistant message")
	}
	if got := assistant.Content.OfString.Value; got != "It's sunny today." {
		t.Errorf("content = %q, want the reply only", got)
	}
	if got := extraField(t, assistant, defaultReasoningField); got != "The user wants weather info, I should check my tools..." {
		t.Errorf("%s = %q, want the reasoning", defaultReasoningField, got)
	}
}

// Several thought Parts (one per streamed chunk, once the caller persists
// partials) join into a single reasoning value, the same way plain text Parts
// join into a single content.
func TestConvertContentToMessages_JoinsSeveralThoughtParts(t *testing.T) {
	m := newModelForTest()

	msgs, err := m.convertContentToMessages(&genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{Text: "first", Thought: true},
			{Text: "second", Thought: true},
			{Text: "answer"},
		},
	})
	if err != nil {
		t.Fatalf("convertContentToMessages: %v", err)
	}

	assistant := msgs[0].OfAssistant
	if got := extraField(t, assistant, defaultReasoningField); got != "first\nsecond" {
		t.Errorf("%s = %q, want the joined reasoning", defaultReasoningField, got)
	}
}

// A thought Part with a tool call must keep the reasoning as a sibling of
// tool_calls: that is the shape DeepSeek in thinking mode and Kimi K2
// thinking require on every intermediate step of a tool loop.
func TestConvertContentToMessages_ReasoningWithToolCall(t *testing.T) {
	m := newModelForTest()

	msgs, err := m.convertContentToMessages(&genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{Text: "I need the status tool first.", Thought: true},
			{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "check_status"}},
		},
	})
	if err != nil {
		t.Fatalf("convertContentToMessages: %v", err)
	}

	assistant := msgs[0].OfAssistant
	if assistant == nil {
		t.Fatalf("expected an assistant message")
	}
	if len(assistant.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(assistant.ToolCalls))
	}
	if got := extraField(t, assistant, defaultReasoningField); got != "I need the status tool first." {
		t.Errorf("%s = %q, want the reasoning", defaultReasoningField, got)
	}
}

// The reasoning field name is configurable for gateways that only accept
// "reasoning", and it is the same name the adapter reads on ingest.
func TestConvertContentToMessages_CustomReasoningField(t *testing.T) {
	m := New(Config{ModelName: "gpt-test", ReasoningField: "reasoning"})

	msgs, err := m.convertContentToMessages(&genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{Text: "thinking", Thought: true},
			{Text: "reply"},
		},
	})
	if err != nil {
		t.Fatalf("convertContentToMessages: %v", err)
	}

	assistant := msgs[0].OfAssistant
	if got := extraField(t, assistant, "reasoning"); got != "thinking" {
		t.Errorf("reasoning = %q, want the reasoning", got)
	}
	if _, ok := extraFields(t, assistant)[defaultReasoningField]; ok {
		t.Errorf("%s must not be set when another field name is configured", defaultReasoningField)
	}
}

// Think-tag mode is for backends that validate messages against a schema
// forbidding extra fields: the reasoning travels inside content, wrapped in a
// <think> block ahead of the reply, and no extra field is emitted.
func TestConvertContentToMessages_ThinkTagsReasoning(t *testing.T) {
	m := New(Config{ModelName: "gpt-test", ReasoningEgress: ReasoningEgressThinkTags})

	msgs, err := m.convertContentToMessages(&genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{Text: "thinking", Thought: true},
			{Text: "reply"},
		},
	})
	if err != nil {
		t.Fatalf("convertContentToMessages: %v", err)
	}

	assistant := msgs[0].OfAssistant
	want := "<think>\nthinking\n</think>\nreply"
	if got := assistant.Content.OfString.Value; got != want {
		t.Errorf("content = %q, want %q", got, want)
	}
	if _, ok := extraFields(t, assistant)[defaultReasoningField]; ok {
		t.Errorf("%s must not be set in think-tag mode", defaultReasoningField)
	}
}

// Omit mode sends no reasoning in any shape: no reasoning field, no think
// block in content, and no reasoning_details. It exists for backends that
// discard reasoning history anyway (Qwen3) or callers who would rather not
// pay for the tokens.
func TestConvertContentToMessages_OmitReasoning(t *testing.T) {
	m := New(Config{
		ModelName:                "gpt-test",
		ReasoningEgress:          ReasoningEgressOmit,
		ReasoningField:           "reasoning",
		SupportsReasoningDetails: true,
	})

	msgs, err := m.convertContentToMessages(&genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{Text: "plain reasoning", Thought: true},
			detailPart("reasoning.text", map[string]any{"text": "block reasoning"}),
			detailPart("reasoning.encrypted", map[string]any{"data": "ZW5j"}),
			{Text: "reply"},
		},
	})
	if err != nil {
		t.Fatalf("convertContentToMessages: %v", err)
	}
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %d", len(msgs))
	}

	fields := extraFields(t, msgs[0].OfAssistant)
	if fields["content"] != "reply" {
		t.Errorf("content = %v, want the reply only", fields["content"])
	}
	for _, key := range []string{"reasoning", defaultReasoningField, reasoningDetailsField} {
		if _, ok := fields[key]; ok {
			t.Errorf("%s must be absent in omit mode: %v", key, fields)
		}
	}
	if strings.Contains(fields["content"].(string), "<think>") {
		t.Errorf("omit mode must not inline a think block: %v", fields["content"])
	}
}

// Omit mode drops reasoning on the way out only. A turn carrying nothing but
// reasoning therefore sends no message at all.
func TestConvertContentToMessages_OmitReasoningOnlyTurn(t *testing.T) {
	m := New(Config{ModelName: "gpt-test", ReasoningEgress: ReasoningEgressOmit})

	msgs, err := m.convertContentToMessages(&genai.Content{
		Role:  "model",
		Parts: []*genai.Part{{Text: "thinking", Thought: true}},
	})
	if err != nil {
		t.Fatalf("convertContentToMessages: %v", err)
	}
	if len(msgs) != 0 {
		t.Errorf("expected no message, got %d", len(msgs))
	}
}

// An unrecognised mode must fall back to native rather than silently
// discarding reasoning: the strict thinking providers reject a history that
// has lost it.
func TestNew_UnknownReasoningEgressModeFallsBackToNative(t *testing.T) {
	m := New(Config{ModelName: "gpt-test", ReasoningEgress: ReasoningEgressMode("nonsense")})
	if m.reasoningEgress != ReasoningEgressNative {
		t.Errorf("reasoningEgress = %q, want %q", m.reasoningEgress, ReasoningEgressNative)
	}
}

// In think-tag mode the reasoning IS the content, so a turn carrying nothing
// but reasoning still produces a message. In native mode it cannot: an
// assistant message with neither content nor tool_calls is not a valid Chat
// Completions message, and there is no other turn to attach the field to.
func TestConvertContentToMessages_ReasoningOnlyTurn(t *testing.T) {
	t.Run("native mode produces no message", func(t *testing.T) {
		m := newModelForTest()
		msgs, err := m.convertContentToMessages(&genai.Content{
			Role:  "model",
			Parts: []*genai.Part{{Text: "thinking", Thought: true}},
		})
		if err != nil {
			t.Fatalf("convertContentToMessages: %v", err)
		}
		if len(msgs) != 0 {
			t.Errorf("expected no message, got %d", len(msgs))
		}
	})

	t.Run("think-tag mode produces the message", func(t *testing.T) {
		m := New(Config{ModelName: "gpt-test", ReasoningEgress: ReasoningEgressThinkTags})
		msgs, err := m.convertContentToMessages(&genai.Content{
			Role:  "model",
			Parts: []*genai.Part{{Text: "thinking", Thought: true}},
		})
		if err != nil {
			t.Fatalf("convertContentToMessages: %v", err)
		}
		if len(msgs) != 1 {
			t.Fatalf("expected 1 message, got %d", len(msgs))
		}
		if got := msgs[0].OfAssistant.Content.OfString.Value; !strings.Contains(got, "<think>") {
			t.Errorf("content = %q, want a think block", got)
		}
	})
}

// Thought Parts under a non-assistant role are dropped. ADK's contents
// processor rewrites events authored by a different agent as user-role "For
// context:" content and passes non-text parts through verbatim; that
// reasoning belongs to another conversation and no provider accepts it on a
// user message.
func TestConvertContentToMessages_DropsReasoningOutsideAssistant(t *testing.T) {
	for _, mode := range []ReasoningEgressMode{ReasoningEgressNative, ReasoningEgressThinkTags, ReasoningEgressOmit} {
		t.Run(string(mode), func(t *testing.T) {
			m := New(Config{ModelName: "gpt-test", ReasoningEgress: mode})

			msgs, err := m.convertContentToMessages(&genai.Content{
				Role: "user",
				Parts: []*genai.Part{
					{Text: "foreign reasoning", Thought: true},
					{Text: "For context: something happened"},
				},
			})
			if err != nil {
				t.Fatalf("convertContentToMessages: %v", err)
			}
			if len(msgs) != 1 {
				t.Fatalf("expected 1 message, got %d", len(msgs))
			}
			user := msgs[0].OfUser
			if user == nil {
				t.Fatalf("expected a user message")
			}
			if got := user.Content.OfString.Value; got != "For context: something happened" {
				t.Errorf("content = %q, want the text part only", got)
			}
		})
	}
}

// A thought Part with no text (the shape the Anthropic adapter uses for a
// redacted thinking block) has nothing to send, and must not turn into an
// empty reasoning field.
func TestConvertContentToMessages_EmptyThoughtPart(t *testing.T) {
	m := newModelForTest()

	msgs, err := m.convertContentToMessages(&genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{Thought: true, ThoughtSignature: []byte("opaque")},
			{Text: "reply"},
		},
	})
	if err != nil {
		t.Fatalf("convertContentToMessages: %v", err)
	}

	assistant := msgs[0].OfAssistant
	if got := assistant.Content.OfString.Value; got != "reply" {
		t.Errorf("content = %q, want the reply", got)
	}
	if _, ok := extraFields(t, assistant)[defaultReasoningField]; ok {
		t.Errorf("%s must not be set for a text-less thought part", defaultReasoningField)
	}
}

// extraFields marshals the message and returns the resulting JSON object, so
// the assertions read the bytes the SDK produces rather than the SDK's
// internal extra-fields bookkeeping.
func extraFields(t *testing.T, msg any) map[string]any {
	t.Helper()
	raw, err := json.Marshal(msg)
	if err != nil {
		t.Fatalf("marshal message: %v", err)
	}
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("unmarshal message: %v", err)
	}
	return out
}

// extraField returns one field of the marshalled message as a string.
func extraField(t *testing.T, msg any, field string) string {
	t.Helper()
	value, ok := extraFields(t, msg)[field]
	if !ok {
		t.Fatalf("field %q missing from message: %v", field, extraFields(t, msg))
	}
	str, ok := value.(string)
	if !ok {
		t.Fatalf("field %q = %v, want a string", field, value)
	}
	return str
}
