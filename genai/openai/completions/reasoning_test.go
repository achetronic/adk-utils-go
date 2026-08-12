// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package completions

import (
	"testing"

	"google.golang.org/genai"
)

// A thought Part must never be merged into the assistant message's content:
// the strict thinking providers (DeepSeek in thinking mode, Kimi's
// preserved-thinking models) check for the reasoning key itself, and reasoning
// folded into content is indistinguishable from the reply. With the text
// dialect in the default native mode the joined reasoning travels as its own
// field.
func TestConvertContentToMessages_NativeReasoning(t *testing.T) {
	m := newTextModelForTest()

	msgs, err := m.convertContentToMessages(&genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{Text: "let me think", Thought: true},
			{Text: "the answer"},
		},
	})
	if err != nil {
		t.Fatalf("convertContentToMessages: %v", err)
	}
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %d", len(msgs))
	}

	a := msgs[0].OfAssistant
	if a == nil {
		t.Fatalf("OfAssistant = nil")
	}
	if got := a.Content.OfString.Value; got != "the answer" {
		t.Errorf("content = %q, want the reply only", got)
	}
	extra := a.ExtraFields()
	if extra[reasoningWriteField] != "let me think" {
		t.Errorf("extra[%q] = %q, want the reasoning", reasoningWriteField, extra[reasoningWriteField])
	}
}

// Several thought Parts join in order, mirroring how the model produced them.
func TestConvertContentToMessages_JoinsSeveralThoughtParts(t *testing.T) {
	m := newTextModelForTest()

	msgs, err := m.convertContentToMessages(&genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{Text: "step one", Thought: true},
			{Text: "step two", Thought: true},
			{Text: "reply"},
		},
	})
	if err != nil {
		t.Fatalf("convertContentToMessages: %v", err)
	}

	extra := msgs[0].OfAssistant.ExtraFields()
	if extra[reasoningWriteField] != "step one\nstep two" {
		t.Errorf("reasoning = %q, want the two steps joined in order", extra[reasoningWriteField])
	}
}

// On a tool-call turn the reasoning must sit next to tool_calls, which is the
// case the strict thinking providers call out as the reason to preserve it.
func TestConvertContentToMessages_ReasoningWithToolCall(t *testing.T) {
	m := newTextModelForTest()

	msgs, err := m.convertContentToMessages(&genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{Text: "need a tool", Thought: true},
			{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "get_weather", Args: map[string]any{"city": "Madrid"}}},
		},
	})
	if err != nil {
		t.Fatalf("convertContentToMessages: %v", err)
	}

	a := msgs[0].OfAssistant
	if len(a.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(a.ToolCalls))
	}
	if extra := a.ExtraFields(); extra[reasoningWriteField] != "need a tool" {
		t.Errorf("reasoning = %q, want it next to tool_calls", extra[reasoningWriteField])
	}
}

// Think-tag mode folds the reasoning into content ahead of the reply and sends
// no extra field, for backends that reject an unknown message field.
func TestConvertContentToMessages_ThinkTagsReasoning(t *testing.T) {
	m := New(Config{ModelName: "gpt-test", Dialect: NewTextDialect(), ReasoningEgress: ReasoningEgressThinkTags})

	msgs, err := m.convertContentToMessages(&genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{Text: "let me think", Thought: true},
			{Text: "the answer"},
		},
	})
	if err != nil {
		t.Fatalf("convertContentToMessages: %v", err)
	}

	a := msgs[0].OfAssistant
	want := "<think>\nlet me think\n</think>\nthe answer"
	if got := a.Content.OfString.Value; got != want {
		t.Errorf("content = %q, want %q", got, want)
	}
	if extra := a.ExtraFields(); len(extra) != 0 {
		t.Errorf("extra fields = %v, want none in think-tag mode", extra)
	}
}

// Omit mode sends no reasoning in any shape.
func TestConvertContentToMessages_OmitReasoning(t *testing.T) {
	m := New(Config{ModelName: "gpt-test", Dialect: NewTextDialect(), ReasoningEgress: ReasoningEgressOmit})

	msgs, err := m.convertContentToMessages(&genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{Text: "let me think", Thought: true},
			{Text: "the answer"},
		},
	})
	if err != nil {
		t.Fatalf("convertContentToMessages: %v", err)
	}

	a := msgs[0].OfAssistant
	if got := a.Content.OfString.Value; got != "the answer" {
		t.Errorf("content = %q, want the reply only", got)
	}
	if extra := a.ExtraFields(); len(extra) != 0 {
		t.Errorf("extra fields = %v, want none in omit mode", extra)
	}
}

// An unrecognised egress mode degrades to native rather than to silent data
// loss.
func TestNew_UnknownReasoningEgressModeFallsBackToNative(t *testing.T) {
	m := New(Config{ModelName: "gpt-test", Dialect: NewTextDialect(), ReasoningEgress: ReasoningEgressMode("bogus")})
	if m.reasoningEgress != ReasoningEgressNative {
		t.Errorf("reasoningEgress = %q, want native fallback", m.reasoningEgress)
	}
}

// A turn carrying nothing but reasoning sends no message in native mode: an
// assistant message with neither content nor tool_calls is not valid.
func TestConvertContentToMessages_ReasoningOnlyTurn(t *testing.T) {
	m := newTextModelForTest()

	msgs, err := m.convertContentToMessages(&genai.Content{
		Role:  "model",
		Parts: []*genai.Part{{Text: "only reasoning", Thought: true}},
	})
	if err != nil {
		t.Fatalf("convertContentToMessages: %v", err)
	}
	if len(msgs) != 0 {
		t.Errorf("expected no message for a reasoning-only turn, got %d", len(msgs))
	}
}

// Reasoning under a non-assistant role is dropped: ADK's contents processor
// rewrites foreign-agent events as user-role content, and no provider accepts
// reasoning on a user message.
func TestConvertContentToMessages_DropsReasoningOutsideAssistant(t *testing.T) {
	m := newTextModelForTest()

	msgs, err := m.convertContentToMessages(&genai.Content{
		Role: "user",
		Parts: []*genai.Part{
			{Text: "foreign reasoning", Thought: true},
			{Text: "user text"},
		},
	})
	if err != nil {
		t.Fatalf("convertContentToMessages: %v", err)
	}
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %d", len(msgs))
	}
	u := msgs[0].OfUser
	if u == nil {
		t.Fatalf("OfUser = nil")
	}
	if got := u.Content.OfString.Value; got != "user text" {
		t.Errorf("user content = %q, want the reasoning dropped", got)
	}
}

// A thought Part with no text (an encrypted block's readable form) contributes
// nothing in plain-text mode and must not panic.
func TestConvertContentToMessages_EmptyThoughtPart(t *testing.T) {
	m := newTextModelForTest()

	msgs, err := m.convertContentToMessages(&genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{Text: "", Thought: true},
			{Text: "reply"},
		},
	})
	if err != nil {
		t.Fatalf("convertContentToMessages: %v", err)
	}
	a := msgs[0].OfAssistant
	if got := a.Content.OfString.Value; got != "reply" {
		t.Errorf("content = %q, want the reply only", got)
	}
	if extra := a.ExtraFields(); len(extra) != 0 {
		t.Errorf("extra fields = %v, want none for a text-less thought", extra)
	}
}

// Without a dialect the adapter folds stray thought Parts into think tags
// rather than dropping them, and sends no provider field at all: there is
// no encoder to choose one, which is the OpenAI-native shape. The reply still
// travels as usual.
func TestConvertContentToMessages_NilDialectFoldsThoughtsIntoThinkTags(t *testing.T) {
	m := newModelForTest()
	if m.dialect != nil {
		t.Fatalf("test model must not have a dialect")
	}

	msgs, err := m.convertContentToMessages(&genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{Text: "some thought", Thought: true},
			{Text: "the answer"},
		},
	})
	if err != nil {
		t.Fatalf("convertContentToMessages: %v", err)
	}
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %d", len(msgs))
	}
	a := msgs[0].OfAssistant
	// Native mode with no encoder degrades to think tags: the trace
	// survives in content, and no provider field is sent at all.
	want := "<think>\nsome thought\n</think>\nthe answer"
	if got := a.Content.OfString.Value; got != want {
		t.Errorf("content = %q, want %q", got, want)
	}
	if extra := a.ExtraFields(); len(extra) != 0 {
		t.Errorf("extra fields = %v, want none without a dialect", extra)
	}
}

// An OpenRouter reasoning block round-trips byte-identical in native mode:
// ingest stores it verbatim in PartMetadata and egress replays it unchanged,
// which is the property OpenRouter requires.
func TestConvertContentToMessages_ReasoningDetailsRoundTrip(t *testing.T) {
	block := map[string]any{
		"type":      "reasoning.text",
		"text":      "verbatim",
		"signature": "sig-value",
		"id":        "t",
		"format":    "anthropic-claude-v1",
		"index":     float64(0),
	}
	m := New(Config{ModelName: "gpt-test", Dialect: OpenRouter})

	msgs, err := m.convertContentToMessages(&genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{Text: "verbatim", Thought: true, PartMetadata: map[string]any{ReasoningDetailMetadataKey: block}},
			{Text: "reply"},
		},
	})
	if err != nil {
		t.Fatalf("convertContentToMessages: %v", err)
	}

	extra := msgs[0].OfAssistant.ExtraFields()
	details, ok := extra[reasoningDetailsField].([]map[string]any)
	if !ok || len(details) != 1 {
		t.Fatalf("reasoning_details = %#v, want one block", extra[reasoningDetailsField])
	}
	if details[0]["signature"] != "sig-value" || details[0]["format"] != "anthropic-claude-v1" {
		t.Errorf("block lost its verbatim fields: %#v", details[0])
	}
	// The block fed the array, so the plain-text field must stay empty rather
	// than duplicate the reasoning.
	if _, hasText := extra[reasoningWriteField]; hasText {
		t.Errorf("reasoning field set alongside the block array: reasoning sent twice")
	}
}

// With think-tag or omit egress a block cannot be sent as an array, so its
// readable text degrades into the plain-text shape (think tags) or drops
// (omit). The block itself is never emitted outside native mode.
func TestConvertContentToMessages_ReasoningDetailsDegradeOutsideNative(t *testing.T) {
	block := map[string]any{"type": "reasoning.text", "text": "readable", "index": float64(0)}
	part := &genai.Part{Text: "readable", Thought: true, PartMetadata: map[string]any{ReasoningDetailMetadataKey: block}}

	think := New(Config{ModelName: "gpt-test", Dialect: OpenRouter, ReasoningEgress: ReasoningEgressThinkTags})
	msgs, err := think.convertContentToMessages(&genai.Content{Role: "model", Parts: []*genai.Part{part, {Text: "reply"}}})
	if err != nil {
		t.Fatalf("think-tags convert: %v", err)
	}
	if got := msgs[0].OfAssistant.Content.OfString.Value; got != "<think>\nreadable\n</think>\nreply" {
		t.Errorf("think-tag content = %q, want the block's readable text", got)
	}
	if extra := msgs[0].OfAssistant.ExtraFields(); len(extra) != 0 {
		t.Errorf("think-tag extra fields = %v, want none", extra)
	}
}

// A vetoed mode never reaches conversion: DeepSeek pins the replay to its
// own field even when the caller asks to omit it, because thinking mode
// rejects an assistant turn that lacks the key.
func TestConvertContentToMessages_DeepSeekOverridesVetoedEgress(t *testing.T) {
	for _, asked := range []ReasoningEgressMode{ReasoningEgressThinkTags, ReasoningEgressOmit} {
		m := New(Config{ModelName: "deepseek-reasoner", Dialect: DeepSeek, ReasoningEgress: asked})
		if m.reasoningEgress != ReasoningEgressNative {
			t.Errorf("egress = %q, want native for asked %q", m.reasoningEgress, asked)
		}

		msgs, err := m.convertContentToMessages(&genai.Content{
			Role: "model",
			Parts: []*genai.Part{
				{Text: "let me think", Thought: true},
				{Text: "the answer"},
			},
		})
		if err != nil {
			t.Fatalf("convertContentToMessages: %v", err)
		}
		extra := msgs[0].OfAssistant.ExtraFields()
		if extra[reasoningWriteField] != "let me think" {
			t.Errorf("extra[%q] = %v, want the reasoning even though %q was asked", reasoningWriteField, extra[reasoningWriteField], asked)
		}
	}
}

// A dialect without a veto keeps the caller's choice: omit against
// OpenRouter stays omit, since the provider tolerates a missing replay.
func TestConvertContentToMessages_OmitHonouredWithoutPolicy(t *testing.T) {
	m := New(Config{ModelName: "gpt-test", Dialect: OpenRouter, ReasoningEgress: ReasoningEgressOmit})
	if m.reasoningEgress != ReasoningEgressOmit {
		t.Fatalf("egress = %q, want omit", m.reasoningEgress)
	}

	msgs, err := m.convertContentToMessages(&genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{Text: "let me think", Thought: true},
			{Text: "the answer"},
		},
	})
	if err != nil {
		t.Fatalf("convertContentToMessages: %v", err)
	}
	if extra := msgs[0].OfAssistant.ExtraFields(); len(extra) != 0 {
		t.Errorf("extra fields = %v, want none in omit mode", extra)
	}
}
