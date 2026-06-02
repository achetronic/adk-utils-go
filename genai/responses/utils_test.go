// Copyright 2025 achetronic
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

package responses

import (
	"testing"

	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
	"google.golang.org/genai"
)

func TestConvertRole(t *testing.T) {
	cases := []struct {
		in   string
		want responses.EasyInputMessageRole
	}{
		{"user", responses.EasyInputMessageRoleUser},
		{"model", responses.EasyInputMessageRoleAssistant},
		{"system", responses.EasyInputMessageRoleSystem},
		{"unknown", responses.EasyInputMessageRoleUser},
	}
	for _, c := range cases {
		t.Run(c.in, func(t *testing.T) {
			if got := convertRole(c.in); got != c.want {
				t.Errorf("convertRole(%q) = %q, want %q", c.in, got, c.want)
			}
		})
	}
}

func TestConvertStatus(t *testing.T) {
	cases := []struct {
		name    string
		status  responses.ResponseStatus
		details responses.ResponseIncompleteDetails
		want    genai.FinishReason
	}{
		{"completed", responses.ResponseStatusCompleted, responses.ResponseIncompleteDetails{}, genai.FinishReasonStop},
		{"incomplete max tokens", responses.ResponseStatusIncomplete, responses.ResponseIncompleteDetails{Reason: "max_output_tokens"}, genai.FinishReasonMaxTokens},
		{"incomplete content filter", responses.ResponseStatusIncomplete, responses.ResponseIncompleteDetails{Reason: "content_filter"}, genai.FinishReasonSafety},
		{"incomplete unknown reason", responses.ResponseStatusIncomplete, responses.ResponseIncompleteDetails{Reason: "other"}, genai.FinishReasonUnspecified},
		{"failed", responses.ResponseStatusFailed, responses.ResponseIncompleteDetails{}, genai.FinishReasonUnspecified},
		{"cancelled", responses.ResponseStatusCancelled, responses.ResponseIncompleteDetails{}, genai.FinishReasonUnspecified},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := convertStatus(c.status, c.details); got != c.want {
				t.Errorf("convertStatus(%q) = %v, want %v", c.status, got, c.want)
			}
		})
	}
}

// convertThinkingLevel maps genai's three-level enum to the Responses API
// reasoning effort. Anything outside Low/High defaults to Medium.
func TestConvertThinkingLevel(t *testing.T) {
	cases := []struct {
		level genai.ThinkingLevel
		want  shared.ReasoningEffort
	}{
		{genai.ThinkingLevelLow, shared.ReasoningEffortLow},
		{genai.ThinkingLevelHigh, shared.ReasoningEffortHigh},
		{genai.ThinkingLevel(""), shared.ReasoningEffortMedium},
		{genai.ThinkingLevel("invalid"), shared.ReasoningEffortMedium},
	}
	for _, c := range cases {
		t.Run(string(c.level), func(t *testing.T) {
			if got := convertThinkingLevel(c.level); got != c.want {
				t.Errorf("convertThinkingLevel(%q) = %q, want %q", c.level, got, c.want)
			}
		})
	}
}

func TestSchemaTypeToString(t *testing.T) {
	cases := []struct {
		in   genai.Type
		want string
	}{
		{genai.TypeString, "string"},
		{genai.TypeNumber, "number"},
		{genai.TypeInteger, "integer"},
		{genai.TypeBoolean, "boolean"},
		{genai.TypeArray, "array"},
		{genai.TypeObject, "object"},
		{genai.TypeUnspecified, "string"},
		{genai.Type("unknown"), "string"},
	}
	for _, c := range cases {
		t.Run(c.want, func(t *testing.T) {
			if got := schemaTypeToString(c.in); got != c.want {
				t.Errorf("schemaTypeToString(%v) = %q, want %q", c.in, got, c.want)
			}
		})
	}
}

func TestExtractText(t *testing.T) {
	cases := []struct {
		name    string
		content *genai.Content
		want    string
	}{
		{"nil content", nil, ""},
		{"empty parts", &genai.Content{Parts: []*genai.Part{}}, ""},
		{"single text", &genai.Content{Parts: []*genai.Part{{Text: "hello"}}}, "hello"},
		{"multiple texts", &genai.Content{Parts: []*genai.Part{{Text: "a"}, {Text: "b"}}}, "a\nb"},
		{"skips non-text", &genai.Content{Parts: []*genai.Part{
			{Text: "hi"},
			{FunctionCall: &genai.FunctionCall{Name: "fn"}},
			{Text: "bye"},
		}}, "hi\nbye"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := extractText(c.content); got != c.want {
				t.Errorf("extractText() = %q, want %q", got, c.want)
			}
		})
	}
}

func TestJoinTexts(t *testing.T) {
	cases := []struct {
		in   []string
		want string
	}{
		{nil, ""},
		{[]string{"a"}, "a"},
		{[]string{"a", "b", "c"}, "a\nb\nc"},
	}
	for _, c := range cases {
		if got := joinTexts(c.in); got != c.want {
			t.Errorf("joinTexts(%v) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestParseJSONArgs(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want int // expected number of keys
	}{
		{"empty string", "", 0},
		{"valid object", `{"a":1,"b":2}`, 2},
		{"malformed JSON", `{broken`, 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := parseJSONArgs(c.in)
			if len(got) != c.want {
				t.Errorf("parseJSONArgs(%q) has %d keys, want %d", c.in, len(got), c.want)
			}
		})
	}
}
