// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package openai

import (
	"reflect"
	"strings"
	"testing"

	"google.golang.org/genai"
)

// convertSchema is the strongly-typed path used by ResponseSchema. We assert
// on structural equality (a map[string]any tree) rather than serialised JSON
// because Go's json.Marshal happens to sort map keys alphabetically — relying
// on that ordering implicitly would couple the test to an implementation
// detail of the standard library that future Go versions don't have to
// preserve.
func TestConvertSchema(t *testing.T) {
	cases := []struct {
		name   string
		schema *genai.Schema
		want   map[string]any
	}{
		{
			name:   "nil schema yields a default empty object",
			schema: nil,
			want: map[string]any{
				"type":       "object",
				"properties": map[string]any{},
			},
		},
		{
			name: "primitive with description",
			schema: &genai.Schema{
				Type:        genai.TypeString,
				Description: "a string",
			},
			want: map[string]any{
				"type":        "string",
				"description": "a string",
			},
		},
		{
			name: "object with required and nested integer property",
			schema: &genai.Schema{
				Type:     genai.TypeObject,
				Required: []string{"a"},
				Properties: map[string]*genai.Schema{
					"a": {Type: genai.TypeInteger},
				},
			},
			want: map[string]any{
				"type":     "object",
				"required": []string{"a"},
				"properties": map[string]any{
					"a": map[string]any{"type": "integer"},
				},
			},
		},
		{
			name: "array with item schema",
			schema: &genai.Schema{
				Type:  genai.TypeArray,
				Items: &genai.Schema{Type: genai.TypeString},
			},
			want: map[string]any{
				"type":  "array",
				"items": map[string]any{"type": "string"},
			},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := convertSchema(c.schema)
			if err != nil {
				t.Fatalf("convertSchema() error = %v", err)
			}
			if !reflect.DeepEqual(got, c.want) {
				t.Errorf("convertSchema() = %#v\nwant %#v", got, c.want)
			}
		})
	}
}

// convertToFunctionParams handles user-provided tool schemas of any shape:
// genai.Schema, raw map[string]any, or anything that round-trips through
// JSON. Whichever path is taken, the resulting schema must:
//   - have all "type" fields lower-cased (Anthropic-style validation now
//     applies to recent OpenAI endpoints too)
//   - never produce an "object" without a "properties" field, because OpenAI
//     rejects tool definitions that omit it
func TestConvertToFunctionParams(t *testing.T) {
	cases := []struct {
		name string
		in   any
		want map[string]any
	}{
		{
			name: "uppercase types from genai schema get normalised and properties injected",
			in: map[string]any{
				"type": "OBJECT",
				"properties": map[string]any{
					"name": map[string]any{"type": "STRING"},
					"items": map[string]any{
						"type":  "ARRAY",
						"items": map[string]any{"type": "OBJECT"}, // missing properties
					},
				},
			},
			want: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"name": map[string]any{"type": "string"},
					"items": map[string]any{
						"type": "array",
						"items": map[string]any{
							"type":       "object",
							"properties": map[string]any{},
						},
					},
				},
			},
		},
		{
			name: "nil input returns nil",
			in:   nil,
			want: nil,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := convertToFunctionParams(c.in)
			if c.want == nil {
				if got != nil {
					t.Errorf("convertToFunctionParams() = %#v, want nil", got)
				}
				return
			}
			gotMap := map[string]any(got)
			if !reflect.DeepEqual(gotMap, c.want) {
				t.Errorf("convertToFunctionParams() = %#v\nwant %#v", gotMap, c.want)
			}
		})
	}
}

// lowercaseTypes walks the schema graph and lowercases every "type" string
// it finds, regardless of nesting. We test:
//   - the trivial root case
//   - traversal through "properties" maps
//   - traversal through "items" lists (legal in JSON Schema for tuple-style
//     arrays)
//   - that non-string "type" values and unrelated string fields are left
//     alone, so we don't mangle enum values that happen to contain capitals
func TestLowercaseTypes(t *testing.T) {
	cases := []struct {
		name string
		in   map[string]any
		want map[string]any
	}{
		{
			name: "root only",
			in:   map[string]any{"type": "OBJECT"},
			want: map[string]any{"type": "object"},
		},
		{
			name: "deeply nested",
			in: map[string]any{
				"type": "OBJECT",
				"properties": map[string]any{
					"name": map[string]any{"type": "STRING"},
					"tags": map[string]any{
						"type":  "ARRAY",
						"items": map[string]any{"type": "STRING"},
					},
				},
			},
			want: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"name": map[string]any{"type": "string"},
					"tags": map[string]any{
						"type":  "array",
						"items": map[string]any{"type": "string"},
					},
				},
			},
		},
		{
			name: "tuple-style array items",
			in: map[string]any{
				"type": "ARRAY",
				"items": []any{
					map[string]any{"type": "STRING"},
					map[string]any{"type": "INTEGER"},
				},
			},
			want: map[string]any{
				"type": "array",
				"items": []any{
					map[string]any{"type": "string"},
					map[string]any{"type": "integer"},
				},
			},
		},
		{
			name: "non-type fields untouched",
			in: map[string]any{
				"type":        "STRING",
				"description": "DO NOT TOUCH",
				"enum":        []any{"FOO", "BAR"},
			},
			want: map[string]any{
				"type":        "string",
				"description": "DO NOT TOUCH",
				"enum":        []any{"FOO", "BAR"},
			},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			lowercaseTypes(c.in)
			if !reflect.DeepEqual(c.in, c.want) {
				t.Errorf("after lowercaseTypes:\n got = %#v\nwant = %#v", c.in, c.want)
			}
		})
	}
}

// normalizeToolCallID hashes IDs longer than OpenAI's 40-char limit and
// remembers the mapping so denormalize can recover the original. Short IDs
// pass through untouched. The exact length of the shortened ID is part of
// the contract — OpenAI rejects anything beyond 40 chars — so we assert on
// the boundary explicitly.
func TestNormalizeToolCallID(t *testing.T) {
	const limit = 40

	t.Run("short ID passes through", func(t *testing.T) {
		m := newModelForTest()
		const id = "call_short"
		if got := m.normalizeToolCallID(id); got != id {
			t.Errorf("normalizeToolCallID(%q) = %q, want passthrough", id, got)
		}
	})

	t.Run("long ID is shortened deterministically and is reversible", func(t *testing.T) {
		m := newModelForTest()
		long := "call_" + strings.Repeat("x", 100)

		first := m.normalizeToolCallID(long)
		second := m.normalizeToolCallID(long)

		if first != second {
			t.Errorf("normalizeToolCallID is non-deterministic: %q vs %q", first, second)
		}
		if len(first) > limit {
			t.Errorf("normalizeToolCallID = %q (len %d), want <= %d", first, len(first), limit)
		}
		if !strings.HasPrefix(first, "tc_") {
			t.Errorf("normalizeToolCallID = %q, want \"tc_\" prefix", first)
		}
		if got := m.denormalizeToolCallID(first); got != long {
			t.Errorf("denormalizeToolCallID(%q) = %q, want original %q", first, got, long)
		}
	})

	t.Run("denormalize unknown ID returns input untouched", func(t *testing.T) {
		m := newModelForTest()
		if got := m.denormalizeToolCallID("never_seen"); got != "never_seen" {
			t.Errorf("denormalizeToolCallID() = %q, want passthrough", got)
		}
	})
}

func newModelForTest() *Model {
	return &Model{toolCallIDMap: make(map[string]string)}
}
