// Copyright 2025 achetronic
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

package openai

import (
	"reflect"
	"testing"

	"google.golang.org/genai"
)

// schemaTypeToString is the mapping that ResponseSchema and other strongly
// typed paths rely on. JSON Schema is case-sensitive — uppercase types are
// invalid — so we have to assert each one explicitly. Unknown types must
// degrade to "string" rather than empty/error to avoid sending a schema with
// an empty "type" field, which OpenAI rejects.
func TestSchemaTypeToString(t *testing.T) {
	cases := []struct {
		typ  genai.Type
		want string
	}{
		{genai.TypeString, "string"},
		{genai.TypeNumber, "number"},
		{genai.TypeInteger, "integer"},
		{genai.TypeBoolean, "boolean"},
		{genai.TypeArray, "array"},
		{genai.TypeObject, "object"},
		{genai.TypeUnspecified, "string"},
		{genai.Type("nonsense"), "string"},
	}

	for _, c := range cases {
		t.Run(string(c.typ), func(t *testing.T) {
			if got := schemaTypeToString(c.typ); got != c.want {
				t.Errorf("schemaTypeToString(%q) = %q, want %q", c.typ, got, c.want)
			}
		})
	}
}

// convertRole is intentionally minimal: it only renames "model" to
// "assistant" and passes everything else through untouched. The OpenAI SDK
// validates roles upstream, so this adapter shouldn't second-guess unknown
// values — it should hand them off as-is.
func TestConvertRole(t *testing.T) {
	cases := []struct {
		role string
		want string
	}{
		{"user", "user"},
		{"model", "assistant"},
		{"system", "system"},
		{"developer", "developer"},
		{"function", "function"},
		{"", ""},
	}

	for _, c := range cases {
		t.Run(c.role, func(t *testing.T) {
			if got := convertRole(c.role); got != c.want {
				t.Errorf("convertRole(%q) = %q, want %q", c.role, got, c.want)
			}
		})
	}
}

// convertFinishReason normalises OpenAI's stop reasons to the genai enum.
// "stop", "tool_calls" and "function_call" all collapse into FinishReasonStop
// because downstream agent code treats them identically (a clean turn end).
// Unknown reasons must fall through to Unspecified so callers can detect and
// log them rather than silently mapping to a misleading state.
func TestConvertFinishReason(t *testing.T) {
	cases := []struct {
		reason string
		want   genai.FinishReason
	}{
		{"stop", genai.FinishReasonStop},
		{"length", genai.FinishReasonMaxTokens},
		{"tool_calls", genai.FinishReasonStop},
		{"function_call", genai.FinishReasonStop},
		{"content_filter", genai.FinishReasonSafety},
		{"surprise_value", genai.FinishReasonUnspecified},
		{"", genai.FinishReasonUnspecified},
	}

	for _, c := range cases {
		t.Run(c.reason, func(t *testing.T) {
			if got := convertFinishReason(c.reason); got != c.want {
				t.Errorf("convertFinishReason(%q) = %v, want %v", c.reason, got, c.want)
			}
		})
	}
}

// joinTexts is just a thin wrapper but we still pin its contract: empty input
// must produce an empty string (not "\n"), and joining preserves order with a
// single newline between entries.
func TestJoinTexts(t *testing.T) {
	cases := []struct {
		name string
		in   []string
		want string
	}{
		{"empty", nil, ""},
		{"one", []string{"a"}, "a"},
		{"many", []string{"a", "b", "c"}, "a\nb\nc"},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := joinTexts(c.in); got != c.want {
				t.Errorf("joinTexts(%v) = %q, want %q", c.in, got, c.want)
			}
		})
	}
}

// parseJSONArgs must never return nil — agent code assumes the args map is
// always safe to index. Invalid JSON, empty input, and the literal "null"
// must all map to an empty map rather than propagate the parse error.
func TestParseJSONArgs(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want map[string]any
	}{
		{"empty", "", map[string]any{}},
		{"valid object", `{"a":1}`, map[string]any{"a": float64(1)}},
		{"invalid json", `{`, map[string]any{}},
		{"unrelated valid json", `[1,2,3]`, map[string]any{}},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := parseJSONArgs(c.in)
			if !reflect.DeepEqual(got, c.want) {
				t.Errorf("parseJSONArgs(%q) = %#v, want %#v", c.in, got, c.want)
			}
		})
	}
}

// extractText concatenates only the text parts of a Content with newlines.
// Non-text parts must be skipped — they have no string representation we'd
// want appearing inside an OpenAI message body.
func TestExtractText(t *testing.T) {
	cases := []struct {
		name    string
		content *genai.Content
		want    string
	}{
		{"nil content", nil, ""},
		{"no parts", &genai.Content{Parts: []*genai.Part{}}, ""},
		{
			name: "text parts joined with newline",
			content: &genai.Content{Parts: []*genai.Part{
				{Text: "hello"},
				{Text: "world"},
			}},
			want: "hello\nworld",
		},
		{
			name: "non-text parts skipped",
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
			if got := extractText(c.content); got != c.want {
				t.Errorf("extractText() = %q, want %q", got, c.want)
			}
		})
	}
}
