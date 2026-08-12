// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package completions

import (
	"testing"

	"google.golang.org/genai"
)

// TextCodec reads the first configured plain-text reasoning field that holds
// a non-empty string from the raw JSON envelope that openai-go preserves on
// every typed response struct. The function exists because OpenAI-compatible
// providers (DeepSeek, Kimi, OpenRouter, vLLM) extend the Chat Completions
// schema with a reasoning field under names that vary, and openai-go does
// NOT type any of them - they live only in JSON.raw.
//
// The contract is: one thought Part carrying the field value when present and
// non-empty, nil otherwise. Malformed JSON must yield nil rather than an
// error because callers cannot meaningfully react to it - at the
// response-conversion layer, dropping a thought Part is the safe
// degradation.
func TestTextDialect_DecodeMessage(t *testing.T) {
	codec := NewTextDialect()
	cases := []struct {
		name string
		raw  string
		want string
	}{
		{
			name: "empty raw returns no parts",
			raw:  "",
			want: "",
		},
		{
			name: "raw without a reasoning field returns no parts",
			raw:  `{"role":"assistant","content":"hello"}`,
			want: "",
		},
		{
			name: "raw with reasoning_content returns the value",
			raw:  `{"role":"assistant","content":"hello","reasoning_content":"thinking step by step"}`,
			want: "thinking step by step",
		},
		{
			name: "raw with reasoning returns the value",
			raw:  `{"role":"assistant","content":"hello","reasoning":"openrouter trace"}`,
			want: "openrouter trace",
		},
		{
			name: "first configured field wins over a later one",
			raw:  `{"reasoning_content":"from content","reasoning":"from reasoning"}`,
			want: "from content",
		},
		{
			name: "raw with empty reasoning_content returns no parts",
			raw:  `{"role":"assistant","content":"hello","reasoning_content":""}`,
			want: "",
		},
		{
			name: "malformed JSON returns no parts rather than panicking",
			raw:  `{"role":"assistant"`,
			want: "",
		},
		{
			name: "reasoning field of wrong type (non-string) returns no parts",
			raw:  `{"reasoning_content":123}`,
			want: "",
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			parts := codec.DecodeMessage(c.raw)
			if c.want == "" {
				if len(parts) != 0 {
					t.Errorf("DecodeMessage(%q) = %#v, want no parts", c.raw, parts)
				}
				return
			}
			if len(parts) != 1 || !parts[0].Thought || parts[0].Text != c.want {
				t.Errorf("DecodeMessage(%q) = %#v, want one thought part %q", c.raw, parts, c.want)
			}
		})
	}
}

// ReadFields and WriteField override the defaults, which is how a provider
// that uses its own field names plugs in without any new code.
func TestTextDialect_CustomFields(t *testing.T) {
	codec := &TextDialect{WriteField: "thought", ReadFields: []string{"think"}}

	parts := codec.DecodeMessage(`{"think":"the thought","reasoning_content":"ignored"}`)
	if len(parts) != 1 || parts[0].Text != "the thought" {
		t.Fatalf("DecodeMessage = %#v, want the think field", parts)
	}

	extra := codec.EncodeReasoning([]*genai.Part{thoughtPart("back it goes")})
	if extra["thought"] != "back it goes" {
		t.Errorf("encode = %#v, want the thought field", extra)
	}
	if _, has := extra[reasoningWriteField]; has {
		t.Errorf("encode wrote the default field alongside the custom one: %#v", extra)
	}
}

// encode joins the readable thought texts in order, and returns nil when the
// thoughts carry no readable text, so the adapter sends no extra field.
func TestTextDialect_Encode(t *testing.T) {
	codec := NewTextDialect()

	extra := codec.EncodeReasoning([]*genai.Part{thoughtPart("one"), thoughtPart("two"), {Text: "", Thought: true}})
	if extra[reasoningWriteField] != "one\ntwo" {
		t.Errorf("encode = %#v, want the texts joined", extra)
	}
	if got := codec.EncodeReasoning([]*genai.Part{{Text: "", Thought: true}}); got != nil {
		t.Errorf("encode of text-less thoughts = %#v, want nil", got)
	}
	if got := codec.EncodeReasoning(nil); got != nil {
		t.Errorf("encode of no thoughts = %#v, want nil", got)
	}
}

// The text dialect accumulator concatenates the deltas' reasoning, which is what a
// stream of reasoning_content chunks must produce.
func TestTextDialect_Accumulator(t *testing.T) {
	acc := NewTextDialect().NewAccumulator()
	acc.AddDelta([]*genai.Part{thoughtPart("think")})
	acc.AddDelta([]*genai.Part{thoughtPart(" more")})
	acc.AddDelta(nil)

	parts := acc.Parts()
	if len(parts) != 1 || parts[0].Text != "think more" || !parts[0].Thought {
		t.Errorf("Parts = %#v, want the concatenated reasoning", parts)
	}

	empty := NewTextDialect().NewAccumulator()
	if got := empty.Parts(); got != nil {
		t.Errorf("empty accumulator Parts = %#v, want nil", got)
	}
}

// OpenRouterCodec decodes the reasoning_details array into one thought Part
// per block, keeping each block verbatim in PartMetadata. When the array and
// the plain-text copy both arrive the array wins: it carries strictly more.
func TestOpenRouterDialect_DecodeBlocks(t *testing.T) {
	raw := `{
		"role":"assistant",
		"content":"the answer",
		"reasoning":"plain copy",
		"reasoning_details":[
			{"type":"reasoning.text","text":"the thought","signature":"sig","id":"r1","format":"anthropic-claude-v1","index":0},
			{"type":"reasoning.encrypted","data":"blob","id":"r2","format":"anthropic-claude-v1","index":1}
		]
	}`

	parts := OpenRouter.DecodeMessage(raw)
	if len(parts) != 2 {
		t.Fatalf("expected 2 parts, got %d: %#v", len(parts), parts)
	}

	first, ok := reasoningDetailOf(parts[0])
	if !ok || first["signature"] != "sig" || parts[0].Text != "the thought" {
		t.Errorf("first part = %#v, block = %#v, want the text block verbatim", parts[0], first)
	}
	second, ok := reasoningDetailOf(parts[1])
	if !ok || second["data"] != "blob" {
		t.Errorf("second block = %#v, want the encrypted block verbatim", second)
	}
	if parts[1].Text != "" {
		t.Errorf("encrypted block Text = %q, want empty", parts[1].Text)
	}
}

// An envelope carrying only the plain-text field still yields a thought Part:
// OpenRouter serves models whose upstream only exposes text.
func TestOpenRouterDialect_DecodeTextFallback(t *testing.T) {
	parts := OpenRouter.DecodeMessage(`{"reasoning":"just text"}`)
	if len(parts) != 1 || parts[0].Text != "just text" || !parts[0].Thought {
		t.Fatalf("parts = %#v, want one thought part", parts)
	}
	if _, ok := reasoningDetailOf(parts[0]); ok {
		t.Errorf("text-only part carries a block: %#v", parts[0])
	}
	if got := OpenRouter.DecodeMessage(`{"content":"no reasoning here"}`); len(got) != 0 {
		t.Errorf("parts = %#v, want none", got)
	}
}

// A malformed or empty reasoning_details array degrades to the plain-text
// field rather than failing the whole turn.
func TestOpenRouterDialect_DecodeMalformedBlocks(t *testing.T) {
	parts := OpenRouter.DecodeMessage(`{"reasoning_details":"not an array","reasoning":"fallback"}`)
	if len(parts) != 1 || parts[0].Text != "fallback" {
		t.Errorf("parts = %#v, want the text fallback", parts)
	}
	parts = OpenRouter.DecodeMessage(`{"reasoning_details":[{}]}`)
	if len(parts) != 0 {
		t.Errorf("parts = %#v, want none for empty blocks", parts)
	}
}

// encode sends blocks back exactly as they arrived, under reasoning_details,
// and plain thoughts under the reasoning field; a Part feeds one of the two,
// never both.
func TestOpenRouterDialect_Encode(t *testing.T) {
	block := map[string]any{"type": "reasoning.text", "text": "kept", "signature": "sig", "index": float64(0)}
	blocked := &genai.Part{Text: "kept", Thought: true, PartMetadata: map[string]any{ReasoningDetailMetadataKey: block}}

	extra := OpenRouter.EncodeReasoning([]*genai.Part{blocked, thoughtPart("plain")})
	details, ok := extra[reasoningDetailsField].([]map[string]any)
	if !ok || len(details) != 1 || details[0]["signature"] != "sig" {
		t.Errorf("reasoning_details = %#v, want the block verbatim", extra[reasoningDetailsField])
	}
	if extra[reasoningWriteField] != "plain" {
		t.Errorf("reasoning field = %v, want the plain thought", extra[reasoningWriteField])
	}

	if got := OpenRouter.EncodeReasoning(nil); got != nil {
		t.Errorf("encode of no thoughts = %#v, want nil", got)
	}
	if got := OpenRouter.EncodeReasoning([]*genai.Part{{Text: "", Thought: true}}); got != nil {
		t.Errorf("encode of text-less unblocked thought = %#v, want nil", got)
	}
}

// The accumulator merges reasoning_details chunks by their reported index:
// text and data concatenate, signature takes the newest non-empty value so a
// signature arriving late is kept, and a null never erases an earlier value.
// Blocks without an index append in arrival order.
func TestOpenRouterDialect_AccumulatorMerge(t *testing.T) {
	acc := OpenRouter.NewAccumulator()

	acc.AddDelta([]*genai.Part{{Thought: true, PartMetadata: map[string]any{ReasoningDetailMetadataKey: map[string]any{
		"type": "reasoning.text", "text": "first ", "signature": nil, "index": float64(0),
	}}}})
	acc.AddDelta([]*genai.Part{{Thought: true, PartMetadata: map[string]any{ReasoningDetailMetadataKey: map[string]any{
		"type": "reasoning.text", "text": "second", "signature": "sig-late", "index": float64(0),
	}}}})
	acc.AddDelta([]*genai.Part{{Thought: true, PartMetadata: map[string]any{ReasoningDetailMetadataKey: map[string]any{
		"type": "reasoning.encrypted", "data": "blob", "index": float64(1),
	}}}})
	acc.AddDelta([]*genai.Part{thoughtPart("plain text too")})

	parts := acc.Parts()
	if len(parts) != 2 {
		t.Fatalf("expected 2 block parts, got %d: %#v", len(parts), parts)
	}
	first, _ := reasoningDetailOf(parts[0])
	if first["text"] != "first second" {
		t.Errorf("merged text = %v, want the concatenation", first["text"])
	}
	if first["signature"] != "sig-late" {
		t.Errorf("signature = %v, want the late value kept", first["signature"])
	}
	second, _ := reasoningDetailOf(parts[1])
	if second["data"] != "blob" {
		t.Errorf("second block data = %v, want blob", second["data"])
	}
}

// When a stream carries both blocks and plain text, the blocks win in the
// final rendering: OpenRouter populates both with the same reasoning, and the
// blocks carry strictly more.
func TestOpenRouterDialect_AccumulatorBlocksWin(t *testing.T) {
	acc := OpenRouter.NewAccumulator()
	acc.AddDelta([]*genai.Part{thoughtPart("plain")})
	acc.AddDelta([]*genai.Part{{Thought: true, PartMetadata: map[string]any{ReasoningDetailMetadataKey: map[string]any{
		"type": "reasoning.text", "text": "block text", "index": float64(0),
	}}}})

	parts := acc.Parts()
	if len(parts) != 1 {
		t.Fatalf("expected 1 part, got %d: %#v", len(parts), parts)
	}
	if parts[0].Text != "block text" {
		t.Errorf("part text = %q, want the block's, not the plain text", parts[0].Text)
	}
}
