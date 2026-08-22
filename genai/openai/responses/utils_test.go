// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package responses

import (
	"encoding/json"
	"errors"
	"reflect"
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
		{"assistant", responses.EasyInputMessageRoleAssistant},
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
		{genai.ThinkingLevelMinimal, shared.ReasoningEffortMinimal},
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

// Malformed arguments must error instead of degrading to an empty map: a
// tool with side effects must not run with silently emptied arguments.
func TestParseJSONArgs(t *testing.T) {
	cases := []struct {
		name    string
		in      string
		want    int // expected number of keys
		wantErr bool
	}{
		{"empty string", "", 0, false},
		{"valid object", `{"a":1,"b":2}`, 2, false},
		{"malformed JSON", `{broken`, 0, true},
		{"truncated JSON", `{`, 0, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := parseJSONArgs(c.in)
			if c.wantErr {
				if err == nil {
					t.Fatalf("parseJSONArgs(%q) = %v, want error", c.in, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("parseJSONArgs(%q): %v", c.in, err)
			}
			if len(got) != c.want {
				t.Errorf("parseJSONArgs(%q) has %d keys, want %d", c.in, len(got), c.want)
			}
		})
	}
}

// A literal "null" is well-formed JSON some gateways emit for parameterless
// calls: it means no arguments, not a malformed payload.
func TestParseJSONArgs_NullMeansNoArguments(t *testing.T) {
	got, err := parseJSONArgs("null")
	if err != nil {
		t.Fatalf("parseJSONArgs(null): %v", err)
	}
	if got == nil || len(got) != 0 {
		t.Errorf("parseJSONArgs(null) = %#v, want a non-nil empty map", got)
	}
}

// TestFunctionCallArgs_Decoded exercises arguments through the SDK's real
// UnmarshalJSON path: a struct literal bypasses union decoding, so only a
// wire payload shows which union arm a given form populates.
func TestFunctionCallArgs_Decoded(t *testing.T) {
	cases := []struct {
		name    string
		raw     string
		want    map[string]any
		wantErr bool
	}{
		{
			name: "string arguments",
			raw:  `{"type":"function_call","name":"f","call_id":"c","arguments":"{\"city\":\"Paris\"}"}`,
			want: map[string]any{"city": "Paris"},
		},
		{
			// Some OpenAI-compatible gateways send a bare JSON object here;
			// those arguments must not be dropped.
			name: "object arguments",
			raw:  `{"type":"function_call","name":"f","call_id":"c","arguments":{"city":"Paris"}}`,
			want: map[string]any{"city": "Paris"},
		},
		{
			name: "empty object arguments",
			raw:  `{"type":"function_call","name":"f","call_id":"c","arguments":{}}`,
			want: map[string]any{},
		},
		{
			name: "empty string arguments",
			raw:  `{"type":"function_call","name":"f","call_id":"c","arguments":""}`,
			want: map[string]any{},
		},
		{
			name: "null arguments",
			raw:  `{"type":"function_call","name":"f","call_id":"c","arguments":null}`,
			want: map[string]any{},
		},
		{
			name: "absent arguments",
			raw:  `{"type":"function_call","name":"f","call_id":"c"}`,
			want: map[string]any{},
		},
		{
			name: "null string arguments",
			raw:  `{"type":"function_call","name":"f","call_id":"c","arguments":"null"}`,
			want: map[string]any{},
		},
		{
			// The SDK's union decode keeps the first duplicate key; the raw
			// bytes must go through the last-one-wins rule of encoding/json,
			// matching every other arguments consumer in the package.
			name: "duplicate keys keep the last",
			raw:  `{"type":"function_call","name":"f","call_id":"c","arguments":{"path":"/tmp/a","path":"/tmp/b"}}`,
			want: map[string]any{"path": "/tmp/b"},
		},
		{
			name:    "array arguments",
			raw:     `{"type":"function_call","name":"f","call_id":"c","arguments":[1,2]}`,
			wantErr: true,
		},
		{
			name:    "number arguments",
			raw:     `{"type":"function_call","name":"f","call_id":"c","arguments":42}`,
			wantErr: true,
		},
		{
			name:    "non-object string arguments",
			raw:     `{"type":"function_call","name":"f","call_id":"c","arguments":"[1,2]"}`,
			wantErr: true,
		},
		{
			// Overflows float64: must be rejected from the original bytes,
			// not resurface as a re-encoding failure of the SDK's +Inf value.
			name:    "out of range number",
			raw:     `{"type":"function_call","name":"f","call_id":"c","arguments":{"x":1e400}}`,
			wantErr: true,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var item responses.ResponseOutputItemUnion
			if err := json.Unmarshal([]byte(c.raw), &item); err != nil {
				t.Fatalf("unmarshal item: %v", err)
			}
			got, err := functionCallArgs(item)
			if c.wantErr {
				if !errors.Is(err, ErrFunctionCallArgs) {
					t.Fatalf("err = %v, want ErrFunctionCallArgs", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("functionCallArgs: %v", err)
			}
			if got == nil {
				t.Fatalf("Args = nil, want a non-nil map")
			}
			if !reflect.DeepEqual(got, c.want) {
				t.Errorf("Args = %#v, want %#v", got, c.want)
			}
		})
	}
}

// TestFunctionCallArgs_HandBuilt covers struct literals with no JSON
// metadata behind them, the form tests and local callers build.
func TestFunctionCallArgs_HandBuilt(t *testing.T) {
	cases := []struct {
		name string
		args responses.ResponseOutputItemUnionArguments
		want map[string]any
	}{
		{
			name: "string arm",
			args: responses.ResponseOutputItemUnionArguments{OfString: `{"city":"Paris"}`},
			want: map[string]any{"city": "Paris"},
		},
		{
			name: "bare arm",
			args: responses.ResponseOutputItemUnionArguments{OfResponseToolSearchCallArguments: map[string]any{"city": "Paris"}},
			want: map[string]any{"city": "Paris"},
		},
		{
			name: "zero value",
			args: responses.ResponseOutputItemUnionArguments{},
			want: map[string]any{},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			item := responses.ResponseOutputItemUnion{Type: "function_call", Name: "f", CallID: "c", Arguments: c.args}
			got, err := functionCallArgs(item)
			if err != nil {
				t.Fatalf("functionCallArgs: %v", err)
			}
			if !reflect.DeepEqual(got, c.want) {
				t.Errorf("Args = %#v, want %#v", got, c.want)
			}
		})
	}
}
