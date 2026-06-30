// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package anthropic

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"google.golang.org/genai"
)

// convertRoleToAnthropic must default unknown roles to "user" so we never emit
// an Anthropic message with an undefined role enum, which would be rejected by
// the SDK on marshal.
func TestConvertRoleToAnthropic(t *testing.T) {
	cases := []struct {
		role string
		want anthropic.MessageParamRole
	}{
		{"user", anthropic.MessageParamRoleUser},
		{"model", anthropic.MessageParamRoleAssistant},
		{"", anthropic.MessageParamRoleUser},
		{"system", anthropic.MessageParamRoleUser},
	}

	for _, c := range cases {
		t.Run(c.role, func(t *testing.T) {
			if got := convertRoleToAnthropic(c.role); got != c.want {
				t.Errorf("convertRoleToAnthropic(%q) = %q, want %q", c.role, got, c.want)
			}
		})
	}
}

// convertStopReason maps Anthropic's discrete stop reasons to genai's enum.
// Both ToolUse and StopSequence collapse to FinishReasonStop on purpose:
// downstream callers don't differentiate, and treating them as anything other
// than "stopped cleanly" would cause spurious retry loops.
func TestConvertStopReason(t *testing.T) {
	cases := []struct {
		name   string
		reason anthropic.StopReason
		want   genai.FinishReason
	}{
		{"end_turn", anthropic.StopReasonEndTurn, genai.FinishReasonStop},
		{"max_tokens", anthropic.StopReasonMaxTokens, genai.FinishReasonMaxTokens},
		{"stop_sequence", anthropic.StopReasonStopSequence, genai.FinishReasonStop},
		{"tool_use", anthropic.StopReasonToolUse, genai.FinishReasonStop},
		{"unknown_value", anthropic.StopReason("garbage"), genai.FinishReasonUnspecified},
		{"empty", anthropic.StopReason(""), genai.FinishReasonUnspecified},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := convertStopReason(c.reason); got != c.want {
				t.Errorf("convertStopReason(%q) = %v, want %v", c.reason, got, c.want)
			}
		})
	}
}

// convertToolInput is the inverse direction: raw payloads coming back from the
// model get normalised into a Go map for genai.FunctionCall.Args. We assert on
// equivalence (reflect.DeepEqual) instead of byte-for-byte JSON because the
// numeric type after Unmarshal is float64, which is fine for downstream callers.
func TestConvertToolInput(t *testing.T) {
	cases := []struct {
		name  string
		input any
		want  map[string]any
	}{
		{"nil", nil, map[string]any{}},
		{"already a map", map[string]any{"a": 1}, map[string]any{"a": 1}},
		{"raw message", json.RawMessage(`{"b":2}`), map[string]any{"b": float64(2)}},
		{"unmarshalable input falls back to empty", make(chan int), map[string]any{}},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := convertToolInput(c.input)
			if !reflect.DeepEqual(got, c.want) {
				t.Errorf("convertToolInput() = %#v, want %#v", got, c.want)
			}
		})
	}
}

// extractTextFromContent must concatenate only the text-bearing parts and
// preserve their order. Non-text parts (inline data, function calls/responses)
// must be skipped entirely so we don't leak placeholders into prompts.
func TestExtractTextFromContent(t *testing.T) {
	cases := []struct {
		name    string
		content *genai.Content
		want    string
	}{
		{"nil content", nil, ""},
		{"no parts", &genai.Content{Parts: []*genai.Part{}}, ""},
		{"single text", &genai.Content{Parts: []*genai.Part{{Text: "hello"}}}, "hello"},
		{
			name: "multiple text parts joined with newline",
			content: &genai.Content{Parts: []*genai.Part{
				{Text: "hello"},
				{Text: "world"},
			}},
			want: "hello\nworld",
		},
		{
			name: "non-text parts are skipped",
			content: &genai.Content{Parts: []*genai.Part{
				{Text: "before"},
				{InlineData: &genai.Blob{MIMEType: "image/png"}},
				{Text: "after"},
			}},
			want: "before\nafter",
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := extractTextFromContent(c.content); got != c.want {
				t.Errorf("extractTextFromContent() = %q, want %q", got, c.want)
			}
		})
	}
}

// sanitizeToolID guarantees Anthropic's regex constraint: ^[a-zA-Z0-9_-]+$.
// Valid IDs must pass through verbatim (so callers can reverse-correlate
// requests/responses); invalid IDs must be deterministically rewritten to a
// "toolu_" prefix plus 32 hex chars (16 bytes of SHA-256). We assert on the
// invariants rather than the exact length so the test survives any future
// SDK-level changes that don't break the contract.
func TestSanitizeToolID(t *testing.T) {
	const validIDPrefix = "toolu_"

	t.Run("valid id passes through", func(t *testing.T) {
		want := "tool_call-123_abc"
		if got := sanitizeToolID(want); got != want {
			t.Errorf("sanitizeToolID(%q) = %q, want passthrough", want, got)
		}
	})

	t.Run("invalid id gets rewritten deterministically", func(t *testing.T) {
		invalid := "invalid/tool:id with spaces"

		first := sanitizeToolID(invalid)
		second := sanitizeToolID(invalid)

		if first != second {
			t.Errorf("sanitizeToolID is non-deterministic: %q vs %q", first, second)
		}
		if !strings.HasPrefix(first, validIDPrefix) {
			t.Errorf("sanitizeToolID(%q) = %q, want %q prefix", invalid, first, validIDPrefix)
		}
		if !anthropicToolIDPattern.MatchString(first) {
			t.Errorf("sanitizeToolID(%q) = %q, does not match Anthropic ID pattern", invalid, first)
		}
	})

	t.Run("different inputs produce different outputs", func(t *testing.T) {
		a := sanitizeToolID("foo!")
		b := sanitizeToolID("bar?")
		if a == b {
			t.Errorf("expected different sanitized IDs, both produced %q", a)
		}
	})
}
