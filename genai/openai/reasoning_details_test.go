// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package openai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/openai/openai-go/v3"
	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"
)

// newDetailsModel builds a model configured the way an OpenRouter caller has
// to: the plain-text reasoning arrives under "reasoning" there, and sending
// the structured array back is opt-in.
func newDetailsModel() *Model {
	return New(Config{
		ModelName:                "gpt-test",
		ReasoningField:           "reasoning",
		SupportsReasoningDetails: true,
	})
}

// A response carrying reasoning_details becomes one thought Part per block,
// in wire order, each keeping its block verbatim in PartMetadata. Order is
// part of OpenRouter's contract: the replayed sequence has to match what the
// model produced.
func TestConvertResponse_ReasoningDetails(t *testing.T) {
	raw := []byte(`{
        "id": "gen-1",
        "object": "chat.completion",
        "created": 0,
        "model": "anthropic/claude-sonnet-latest",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Based on my analysis, here is the answer.",
                "reasoning": "plaintext copy that must not be used when blocks exist",
                "reasoning_details": [
                    {"type": "reasoning.summary", "summary": "Broke the problem into components", "id": "s-1", "format": "anthropic-claude-v1", "index": 0},
                    {"type": "reasoning.encrypted", "data": "ZW5jcnlwdGVk", "id": "e-1", "format": "anthropic-claude-v1", "index": 1},
                    {"type": "reasoning.text", "text": "Working through it systematically", "signature": null, "id": "t-1", "format": "anthropic-claude-v1", "index": 2}
                ]
            },
            "finish_reason": "stop"
        }]
    }`)

	var resp openai.ChatCompletion
	if err := json.Unmarshal(raw, &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	got, err := newDetailsModel().convertResponse(&resp)
	if err != nil {
		t.Fatalf("convertResponse: %v", err)
	}

	// Three reasoning blocks plus the answer.
	if len(got.Content.Parts) != 4 {
		t.Fatalf("expected 4 parts, got %d", len(got.Content.Parts))
	}
	for i, part := range got.Content.Parts[:3] {
		if !part.Thought {
			t.Errorf("part %d Thought = false, want true", i)
		}
		if _, ok := reasoningDetailOf(part); !ok {
			t.Errorf("part %d carries no reasoning block: %#v", i, part.PartMetadata)
		}
	}

	// Readable text is mirrored onto the Part so Thought-filtering consumers
	// still see the reasoning; an encrypted block has none.
	if got := got.Content.Parts[0].Text; got != "Broke the problem into components" {
		t.Errorf("summary part text = %q", got)
	}
	if got := got.Content.Parts[1].Text; got != "" {
		t.Errorf("encrypted part text = %q, want empty", got)
	}
	if got := got.Content.Parts[2].Text; got != "Working through it systematically" {
		t.Errorf("text part text = %q", got)
	}

	// The plain-text field is ignored when blocks are present: both describe
	// the same reasoning and the blocks carry strictly more.
	for _, part := range got.Content.Parts {
		if strings.Contains(part.Text, "plaintext copy") {
			t.Errorf("plain-text reasoning leaked into the parts: %q", part.Text)
		}
	}

	// The block survives byte-for-byte, null signature included.
	block, _ := reasoningDetailOf(got.Content.Parts[2])
	want := map[string]any{
		"type": "reasoning.text", "text": "Working through it systematically",
		"signature": nil, "id": "t-1", "format": "anthropic-claude-v1", "index": float64(2),
	}
	if !reflect.DeepEqual(block, want) {
		t.Errorf("block = %#v\nwant %#v", block, want)
	}

	if got.Content.Parts[3].Thought || got.Content.Parts[3].Text != "Based on my analysis, here is the answer." {
		t.Errorf("last part should be the answer, got %#v", got.Content.Parts[3])
	}
}

// A block type this adapter has never heard of, and extra keys inside a known
// block, must survive ingest untouched. The schema is open and OpenRouter
// forbids modifying the sequence, so anything unrecognised is still replayed.
func TestConvertResponse_ReasoningDetailsUnknownShapes(t *testing.T) {
	raw := []byte(`{
        "id": "gen-2",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "answer",
                "reasoning_details": [
                    {"type": "reasoning.future", "payload": {"nested": true}, "index": 0},
                    {"type": "reasoning.text", "text": "known", "vendor_extra": "keep me", "index": 1}
                ]
            },
            "finish_reason": "stop"
        }]
    }`)

	var resp openai.ChatCompletion
	if err := json.Unmarshal(raw, &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	got, err := newDetailsModel().convertResponse(&resp)
	if err != nil {
		t.Fatalf("convertResponse: %v", err)
	}

	unknown, ok := reasoningDetailOf(got.Content.Parts[0])
	if !ok {
		t.Fatalf("unknown block was dropped")
	}
	if !reflect.DeepEqual(unknown["payload"], map[string]any{"nested": true}) {
		t.Errorf("nested payload not preserved: %#v", unknown["payload"])
	}
	if got.Content.Parts[0].Text != "" {
		t.Errorf("unknown block should offer no readable text, got %q", got.Content.Parts[0].Text)
	}

	known, _ := reasoningDetailOf(got.Content.Parts[1])
	if known["vendor_extra"] != "keep me" {
		t.Errorf("unknown key inside a known block was dropped: %#v", known)
	}
}

// Ingest is never gated by config: blocks are captured even when the adapter
// is not allowed to send them back, so enabling the option later still
// replays what earlier turns recorded.
func TestConvertResponse_ReasoningDetailsCapturedWhenEgressDisabled(t *testing.T) {
	raw := []byte(`{
        "id": "gen-3",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "answer",
                "reasoning_details": [{"type": "reasoning.text", "text": "thinking", "index": 0}]
            },
            "finish_reason": "stop"
        }]
    }`)

	var resp openai.ChatCompletion
	if err := json.Unmarshal(raw, &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	// Default config: reasoning details egress is off.
	m := New(Config{ModelName: "gpt-test"})
	got, err := m.convertResponse(&resp)
	if err != nil {
		t.Fatalf("convertResponse: %v", err)
	}
	if _, ok := reasoningDetailOf(got.Content.Parts[0]); !ok {
		t.Errorf("block must be captured regardless of the egress option")
	}
}

// On egress the blocks go back as a reasoning_details array, verbatim and in
// Part order, and the plain-text field is not populated from those same Parts.
func TestConvertContentToMessages_ReasoningDetails(t *testing.T) {
	m := newDetailsModel()

	msgs, err := m.convertContentToMessages(&genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			detailPart("reasoning.summary", map[string]any{"summary": "first", "index": float64(0)}),
			detailPart("reasoning.encrypted", map[string]any{"data": "ZW5j", "index": float64(1)}),
			{Text: "reply"},
		},
	})
	if err != nil {
		t.Fatalf("convertContentToMessages: %v", err)
	}

	assistant := msgs[0].OfAssistant
	fields := extraFields(t, assistant)
	if fields["content"] != "reply" {
		t.Errorf("content = %v, want the reply only", fields["content"])
	}
	if _, ok := fields["reasoning"]; ok {
		t.Errorf("plain-text field must not repeat block reasoning: %v", fields)
	}

	details, ok := fields[reasoningDetailsField].([]any)
	if !ok || len(details) != 2 {
		t.Fatalf("reasoning_details = %v, want 2 blocks", fields[reasoningDetailsField])
	}
	first, _ := details[0].(map[string]any)
	if first["type"] != "reasoning.summary" || first["summary"] != "first" {
		t.Errorf("first block = %#v", first)
	}
	second, _ := details[1].(map[string]any)
	if second["type"] != "reasoning.encrypted" || second["data"] != "ZW5j" {
		t.Errorf("second block = %#v", second)
	}
}

// Block order must be preserved exactly: OpenRouter rejects a rearranged
// sequence.
func TestConvertContentToMessages_ReasoningDetailsOrder(t *testing.T) {
	m := newDetailsModel()

	var parts []*genai.Part
	for _, id := range []string{"a", "b", "c", "d", "e"} {
		parts = append(parts, detailPart("reasoning.text", map[string]any{"text": id, "id": id}))
	}
	parts = append(parts, &genai.Part{Text: "reply"})

	msgs, err := m.convertContentToMessages(&genai.Content{Role: "model", Parts: parts})
	if err != nil {
		t.Fatalf("convertContentToMessages: %v", err)
	}

	details, _ := extraFields(t, msgs[0].OfAssistant)[reasoningDetailsField].([]any)
	if len(details) != 5 {
		t.Fatalf("expected 5 blocks, got %d", len(details))
	}
	for i, want := range []string{"a", "b", "c", "d", "e"} {
		block, _ := details[i].(map[string]any)
		if block["id"] != want {
			t.Errorf("block %d id = %v, want %q", i, block["id"], want)
		}
	}
}

// With the option off the array is not sent. The readable text of each block
// falls back into the plain-text field so the trace is carried as far as the
// backend allows; an encrypted block has no text and is lost, which is the
// unavoidable cost of a backend that cannot take the array.
func TestConvertContentToMessages_ReasoningDetailsEgressDisabled(t *testing.T) {
	m := New(Config{ModelName: "gpt-test", ReasoningField: "reasoning"})

	msgs, err := m.convertContentToMessages(&genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			detailPart("reasoning.text", map[string]any{"text": "readable"}),
			detailPart("reasoning.encrypted", map[string]any{"data": "ZW5j"}),
			{Text: "reply"},
		},
	})
	if err != nil {
		t.Fatalf("convertContentToMessages: %v", err)
	}

	fields := extraFields(t, msgs[0].OfAssistant)
	if _, ok := fields[reasoningDetailsField]; ok {
		t.Errorf("reasoning_details must not be sent when unsupported: %v", fields)
	}
	if fields["reasoning"] != "readable" {
		t.Errorf("reasoning = %v, want the readable block text", fields["reasoning"])
	}
}

// Think-tag mode sends no extra fields at all, so readable block text is
// inlined into content and encrypted blocks are dropped.
func TestConvertContentToMessages_ReasoningDetailsThinkTags(t *testing.T) {
	m := New(Config{
		ModelName:                "gpt-test",
		ReasoningEgress:          ReasoningEgressThinkTags,
		SupportsReasoningDetails: true,
	})

	msgs, err := m.convertContentToMessages(&genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			detailPart("reasoning.text", map[string]any{"text": "readable"}),
			detailPart("reasoning.encrypted", map[string]any{"data": "ZW5j"}),
			{Text: "reply"},
		},
	})
	if err != nil {
		t.Fatalf("convertContentToMessages: %v", err)
	}

	fields := extraFields(t, msgs[0].OfAssistant)
	if _, ok := fields[reasoningDetailsField]; ok {
		t.Errorf("think-tag mode must send no reasoning_details: %v", fields)
	}
	want := "<think>\nreadable\n</think>\nreply"
	if fields["content"] != want {
		t.Errorf("content = %v, want %q", fields["content"], want)
	}
}

// A history that mixes Parts carrying blocks with Parts carrying only text
// populates both fields, each from its own Parts, so nothing is duplicated
// and nothing is lost.
func TestConvertContentToMessages_ReasoningDetailsMixedWithPlainText(t *testing.T) {
	m := newDetailsModel()

	msgs, err := m.convertContentToMessages(&genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{Text: "plain reasoning from another provider", Thought: true},
			detailPart("reasoning.text", map[string]any{"text": "block reasoning"}),
			{Text: "reply"},
		},
	})
	if err != nil {
		t.Fatalf("convertContentToMessages: %v", err)
	}

	fields := extraFields(t, msgs[0].OfAssistant)
	if fields["reasoning"] != "plain reasoning from another provider" {
		t.Errorf("reasoning = %v, want only the block-less thought text", fields["reasoning"])
	}
	details, _ := fields[reasoningDetailsField].([]any)
	if len(details) != 1 {
		t.Fatalf("expected 1 block, got %v", fields[reasoningDetailsField])
	}
}

// An encrypted block has no text, and a text-less thought Part contributes
// nothing on its own. Carrying a block is what makes it count.
func TestConvertContentToMessages_TextLessBlockStillTravels(t *testing.T) {
	m := newDetailsModel()

	msgs, err := m.convertContentToMessages(&genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			detailPart("reasoning.encrypted", map[string]any{"data": "ZW5j"}),
			{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "check"}},
		},
	})
	if err != nil {
		t.Fatalf("convertContentToMessages: %v", err)
	}

	fields := extraFields(t, msgs[0].OfAssistant)
	details, _ := fields[reasoningDetailsField].([]any)
	if len(details) != 1 {
		t.Fatalf("encrypted block did not travel with the tool call: %v", fields)
	}
	if _, ok := fields["tool_calls"]; !ok {
		t.Errorf("tool_calls missing: %v", fields)
	}
}

// Reasoning blocks belong to an assistant turn only, the same rule the
// plain-text reasoning follows.
func TestConvertContentToMessages_ReasoningDetailsDroppedOutsideAssistant(t *testing.T) {
	m := newDetailsModel()

	msgs, err := m.convertContentToMessages(&genai.Content{
		Role: "user",
		Parts: []*genai.Part{
			detailPart("reasoning.text", map[string]any{"text": "foreign"}),
			{Text: "For context: something happened"},
		},
	})
	if err != nil {
		t.Fatalf("convertContentToMessages: %v", err)
	}
	if msgs[0].OfUser == nil {
		t.Fatalf("expected a user message")
	}
	if got := msgs[0].OfUser.Content.OfString.Value; got != "For context: something happened" {
		t.Errorf("content = %q, want the text part only", got)
	}
}

// The full loop: a provider response goes through ingest and straight back out
// as a request, and the blocks must come out byte-identical. This is the
// property OpenRouter actually requires.
func TestReasoningDetails_RoundTripIsByteIdentical(t *testing.T) {
	const blocksJSON = `[
        {"type":"reasoning.summary","summary":"summarised","id":"s","format":"openai-responses-v1","index":0},
        {"type":"reasoning.encrypted","data":"ZW5jcnlwdGVkLWJsb2I=","id":"e","format":"openai-responses-v1","index":1},
        {"type":"reasoning.text","text":"verbatim","signature":"sig-value","id":"t","format":"anthropic-claude-v1","index":2}
    ]`

	var captured []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"id":"x","object":"chat.completion","created":0,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok","reasoning_details":`+blocksJSON+`},"finish_reason":"stop"}]}`)
	}))
	defer srv.Close()

	m := New(Config{
		BaseURL:                  srv.URL,
		APIKey:                   "test-key",
		ModelName:                "gpt-test",
		ReasoningField:           "reasoning",
		SupportsReasoningDetails: true,
	})

	// First turn: read the blocks off the response.
	var ingested *genai.Content
	for resp, err := range m.GenerateContent(context.Background(), &model.LLMRequest{
		Config:   &genai.GenerateContentConfig{},
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "hi"}}}},
	}, false) {
		if err != nil {
			t.Fatalf("first turn: %v", err)
		}
		ingested = resp.Content
	}
	if ingested == nil {
		t.Fatalf("no content from the first turn")
	}

	// Second turn: replay them as history.
	for _, err := range m.GenerateContent(context.Background(), &model.LLMRequest{
		Config: &genai.GenerateContentConfig{},
		Contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: "hi"}}},
			ingested,
			{Role: "user", Parts: []*genai.Part{{Text: "and now?"}}},
		},
	}, false) {
		if err != nil {
			t.Fatalf("second turn: %v", err)
		}
	}

	var body map[string]any
	if err := json.Unmarshal(captured, &body); err != nil {
		t.Fatalf("unmarshal captured body: %v", err)
	}
	assistant := messageOfRole(t, body, "assistant")

	var want []any
	if err := json.Unmarshal([]byte(blocksJSON), &want); err != nil {
		t.Fatalf("unmarshal expected blocks: %v", err)
	}
	got, _ := assistant[reasoningDetailsField].([]any)
	if !reflect.DeepEqual(got, want) {
		t.Errorf("replayed blocks differ from what the provider sent:\ngot  %#v\nwant %#v", got, want)
	}
}

// detailPart builds a thought Part carrying a reasoning block, the shape
// ingest produces.
func detailPart(blockType string, fields map[string]any) *genai.Part {
	block := map[string]any{"type": blockType}
	for k, v := range fields {
		block[k] = v
	}
	return &genai.Part{
		Text:         readableReasoningDetail(block),
		Thought:      true,
		PartMetadata: map[string]any{ReasoningDetailMetadataKey: block},
	}
}
