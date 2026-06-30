// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package anthropic

import (
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"
)

// resolveThinkingMode is the single source of truth for which on-the-wire
// reasoning shape the adapter uses. An explicit ThinkingMode must win;
// otherwise it is deduced from whichever knob the caller set.
func TestResolveThinkingMode(t *testing.T) {
	cases := []struct {
		name  string
		model *Model
		want  string
	}{
		{"no reasoning", &Model{}, ""},
		{"budget only deduces enabled", &Model{thinkingBudgetTokens: 2048}, ThinkingModeEnabled},
		{"effort only deduces adaptive", &Model{thinkingEffort: "high"}, ThinkingModeAdaptive},
		{"explicit enabled wins over effort", &Model{thinkingMode: ThinkingModeEnabled, thinkingEffort: "high", thinkingBudgetTokens: 2048}, ThinkingModeEnabled},
		{"explicit adaptive wins over budget", &Model{thinkingMode: ThinkingModeAdaptive, thinkingBudgetTokens: 2048, thinkingEffort: "low"}, ThinkingModeAdaptive},
		{"unknown mode falls back to deduction", &Model{thinkingMode: "garbage", thinkingEffort: "high"}, ThinkingModeAdaptive},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.model.resolveThinkingMode(); got != tc.want {
				t.Errorf("resolveThinkingMode() = %q, want %q", got, tc.want)
			}
		})
	}
}

// The enabled (classic) mode must put a budget-bearing thinking block on
// the typed params. The adaptive mode must NOT (it is injected as raw
// JSON via reasoningRequestOptions instead), so params.Thinking stays
// zero for adaptive.
func TestBuildMessageParams_ThinkingBlock(t *testing.T) {
	t.Run("enabled sets budget thinking block", func(t *testing.T) {
		m := &Model{modelName: "claude-sonnet-4-5", thinkingBudgetTokens: 4096, maxOutputTokens: 8192}
		params, err := m.buildMessageParams(&model.LLMRequest{Config: &genai.GenerateContentConfig{}})
		if err != nil {
			t.Fatalf("buildMessageParams: %v", err)
		}
		if params.Thinking.OfEnabled == nil {
			t.Fatalf("expected OfEnabled to be set for enabled mode")
		}
		if params.Thinking.OfEnabled.BudgetTokens != 4096 {
			t.Errorf("BudgetTokens = %d, want 4096", params.Thinking.OfEnabled.BudgetTokens)
		}
	})

	t.Run("adaptive leaves typed thinking block unset", func(t *testing.T) {
		m := &Model{modelName: "claude-opus-4-8", thinkingEffort: "high", maxOutputTokens: 8192}
		params, err := m.buildMessageParams(&model.LLMRequest{Config: &genai.GenerateContentConfig{}})
		if err != nil {
			t.Fatalf("buildMessageParams: %v", err)
		}
		if params.Thinking.OfEnabled != nil {
			t.Errorf("adaptive mode must not set the typed OfEnabled block")
		}
	})

	t.Run("no reasoning leaves thinking unset", func(t *testing.T) {
		m := &Model{modelName: "claude-sonnet-4-5", maxOutputTokens: 8192}
		params, err := m.buildMessageParams(&model.LLMRequest{Config: &genai.GenerateContentConfig{}})
		if err != nil {
			t.Fatalf("buildMessageParams: %v", err)
		}
		if params.Thinking.OfEnabled != nil {
			t.Errorf("no reasoning must not set a thinking block")
		}
	})
}

// reasoningRequestOptions is the adaptive-only path: it must produce the
// raw-JSON options (thinking.type=adaptive + output_config.effort) for
// adaptive and nil for everything else.
func TestReasoningRequestOptions(t *testing.T) {
	if opts := (&Model{thinkingEffort: "high"}).reasoningRequestOptions(); len(opts) != 2 {
		t.Errorf("adaptive: expected 2 request options, got %d", len(opts))
	}
	if opts := (&Model{thinkingBudgetTokens: 4096}).reasoningRequestOptions(); opts != nil {
		t.Errorf("enabled: expected nil request options, got %v", opts)
	}
	if opts := (&Model{}).reasoningRequestOptions(); opts != nil {
		t.Errorf("no reasoning: expected nil request options, got %v", opts)
	}
}

// Misconfiguration must fail locally with a clear error instead of being
// sent to the server and bouncing as a 400.
func TestBuildMessageParams_ReasoningValidation(t *testing.T) {
	t.Run("adaptive without effort errors", func(t *testing.T) {
		m := &Model{modelName: "x", thinkingMode: ThinkingModeAdaptive}
		if _, err := m.buildMessageParams(&model.LLMRequest{Config: &genai.GenerateContentConfig{}}); err == nil {
			t.Fatalf("expected error for adaptive mode without effort")
		}
	})
	t.Run("enabled without budget errors", func(t *testing.T) {
		m := &Model{modelName: "x", thinkingMode: ThinkingModeEnabled}
		if _, err := m.buildMessageParams(&model.LLMRequest{Config: &genai.GenerateContentConfig{}}); err == nil {
			t.Fatalf("expected error for enabled mode without budget")
		}
	})
}

// mustBlock builds a ContentBlockUnion from real JSON. ContentBlockUnion
// dispatches via AsAny() by unmarshaling its private raw JSON, so a struct
// literal would leave JSON.raw empty and AsAny would yield nil. Tests must
// therefore feed real JSON, exactly as the SDK does off the wire.
func mustBlock(t *testing.T, raw string) anthropic.ContentBlockUnion {
	t.Helper()
	var b anthropic.ContentBlockUnion
	if err := b.UnmarshalJSON([]byte(raw)); err != nil {
		t.Fatalf("unmarshal block %q: %v", raw, err)
	}
	return b
}

// convertResponse must preserve a thinking block (text + signature) and a
// redacted_thinking block (opaque blob) so a later turn can echo them
// back; convertContentToMessage must reconstruct both into the right SDK
// block types. This is the round-trip Anthropic requires when extended
// thinking is combined with tool_use.
func TestThinkingBlockRoundTrip(t *testing.T) {
	m := &Model{modelName: "claude-opus-4-8"}

	resp := &anthropic.Message{
		Content: []anthropic.ContentBlockUnion{
			mustBlock(t, "{\"type\":\"thinking\",\"thinking\":\"let me think\",\"signature\":\"sig-abc\"}"),
			mustBlock(t, "{\"type\":\"redacted_thinking\",\"data\":\"encrypted-blob\"}"),
			mustBlock(t, "{\"type\":\"text\",\"text\":\"the answer\"}"),
		},
	}

	llmResp, err := m.convertResponse(resp)
	if err != nil {
		t.Fatalf("convertResponse: %v", err)
	}

	var thoughtText, thoughtSig, redactedBlob, answer string
	for _, p := range llmResp.Content.Parts {
		switch {
		case p.Thought && p.Text != "":
			thoughtText = p.Text
			thoughtSig = string(p.ThoughtSignature)
		case p.Thought && p.Text == "":
			redactedBlob = string(p.ThoughtSignature)
		case p.Text != "":
			answer = p.Text
		}
	}
	if thoughtText != "let me think" || thoughtSig != "sig-abc" {
		t.Errorf("thinking block not preserved: text=%q sig=%q", thoughtText, thoughtSig)
	}
	if redactedBlob != "encrypted-blob" {
		t.Errorf("redacted_thinking blob not preserved: %q", redactedBlob)
	}
	if answer != "the answer" {
		t.Errorf("text answer not preserved: %q", answer)
	}

	// Now send those parts back and confirm they rebuild as the proper
	// SDK block types in the right order (thinking before text).
	msg, err := m.convertContentToMessage(llmResp.Content)
	if err != nil {
		t.Fatalf("convertContentToMessage: %v", err)
	}
	if len(msg.Content) != 3 {
		t.Fatalf("expected 3 blocks back, got %d", len(msg.Content))
	}
	if msg.Content[0].OfThinking == nil {
		t.Errorf("block 0 should be a thinking block")
	} else {
		if msg.Content[0].OfThinking.Thinking != "let me think" {
			t.Errorf("thinking text = %q", msg.Content[0].OfThinking.Thinking)
		}
		if msg.Content[0].OfThinking.Signature != "sig-abc" {
			t.Errorf("thinking signature = %q", msg.Content[0].OfThinking.Signature)
		}
	}
	if msg.Content[1].OfRedactedThinking == nil {
		t.Errorf("block 1 should be a redacted_thinking block")
	} else if msg.Content[1].OfRedactedThinking.Data != "encrypted-blob" {
		t.Errorf("redacted data = %q", msg.Content[1].OfRedactedThinking.Data)
	}
	if msg.Content[2].OfText == nil {
		t.Errorf("block 2 should be a text block")
	}
}

// Thought parts under a user role must be dropped, not converted.
// The ADK contents processor rewrites another agent's events as
// user-role "For context:" content and passes signature-only thought
// parts through verbatim; Anthropic rejects any request that carries
// thinking/redacted_thinking blocks outside assistant messages
// ("thinking blocks may only be in `assistant` messages", HTTP 400).
func TestConvertContentToMessage_DropsThoughtPartsInUserRole(t *testing.T) {
	m := &Model{modelName: "claude-opus-4-8"}

	msg, err := m.convertContentToMessage(&genai.Content{
		Role: "user",
		Parts: []*genai.Part{
			{Text: "For context:"},
			{Text: "reasoning from another agent", Thought: true, ThoughtSignature: []byte("sig-foreign")},
			{Thought: true, ThoughtSignature: []byte("redacted-foreign")},
			{Text: "[coordinator-opus] said: hello"},
		},
	})
	if err != nil {
		t.Fatalf("convertContentToMessage: %v", err)
	}
	if msg == nil {
		t.Fatal("message with text parts was dropped entirely")
	}
	if len(msg.Content) != 2 {
		t.Fatalf("expected 2 text blocks, got %d", len(msg.Content))
	}
	for i, b := range msg.Content {
		if b.OfThinking != nil || b.OfRedactedThinking != nil {
			t.Errorf("block %d: thinking block leaked into a user message", i)
		}
		if b.OfText == nil {
			t.Errorf("block %d: expected a text block", i)
		}
	}

	// A user content carrying ONLY foreign thought parts must vanish
	// (nil message), not become an empty message Anthropic also rejects.
	msg, err = m.convertContentToMessage(&genai.Content{
		Role:  "user",
		Parts: []*genai.Part{{Thought: true, ThoughtSignature: []byte("redacted-only")}},
	})
	if err != nil {
		t.Fatalf("convertContentToMessage (thought-only): %v", err)
	}
	if msg != nil {
		t.Fatalf("thought-only user content should produce no message, got %+v", msg)
	}
}
