// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package anthropic

import (
	"reflect"
	"testing"

	"google.golang.org/genai"
)

// convertTools must produce schemas that satisfy Anthropic's JSON Schema
// validator, which strictly follows draft 2020-12. The genai SDK emits type
// names in upper-case (e.g. "ARRAY", "STRING"); leaving those upper-case in
// the wire payload causes Anthropic to reject the entire request with
// "tools.N.custom.input_schema: JSON schema is invalid". This test pins down
// the contract: every "type" field, at every depth, lands in lower-case.
//
// Issue: https://github.com/achetronic/adk-utils-go/issues/15
func TestConvertTools_LowercasesNestedSchemaTypes(t *testing.T) {
	m := &Model{modelName: "test-model"}

	tools := []*genai.Tool{
		{
			FunctionDeclarations: []*genai.FunctionDeclaration{
				{
					Name: "test_tool",
					Parameters: &genai.Schema{
						Type: genai.TypeObject,
						Properties: map[string]*genai.Schema{
							"artifact_names": {
								Type:  genai.TypeArray,
								Items: &genai.Schema{Type: genai.TypeString},
							},
							"options": {
								Type: genai.TypeObject,
								Properties: map[string]*genai.Schema{
									"deep": {
										Type:  genai.TypeArray,
										Items: &genai.Schema{Type: genai.TypeInteger},
									},
								},
							},
						},
					},
				},
			},
		},
	}

	converted, err := m.convertTools(tools)
	if err != nil {
		t.Fatalf("convertTools error: %v", err)
	}
	if len(converted) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(converted))
	}
	tool := converted[0].OfTool
	if tool == nil {
		t.Fatalf("expected OfTool to be set")
	}

	props, ok := tool.InputSchema.Properties.(map[string]any)
	if !ok {
		t.Fatalf("Properties is %T, want map[string]any", tool.InputSchema.Properties)
	}

	wantType := func(t *testing.T, m map[string]any, want string, where string) {
		t.Helper()
		got, ok := m["type"].(string)
		if !ok {
			t.Errorf("%s: missing or non-string \"type\" field, got %T", where, m["type"])
			return
		}
		if got != want {
			t.Errorf("%s: type = %q, want %q", where, got, want)
		}
	}

	artifactNames := props["artifact_names"].(map[string]any)
	wantType(t, artifactNames, "array", "artifact_names")
	wantType(t, artifactNames["items"].(map[string]any), "string", "artifact_names.items")

	options := props["options"].(map[string]any)
	wantType(t, options, "object", "options")
	deep := options["properties"].(map[string]any)["deep"].(map[string]any)
	wantType(t, deep, "array", "options.properties.deep")
	wantType(t, deep["items"].(map[string]any), "integer", "options.properties.deep.items")
}

// convertTools also propagates "required" from the genai schema. After the
// JSON round-trip required arrives as []interface{}; the adapter must still
// flatten it back to []string for the Anthropic SDK's strict typing.
func TestConvertTools_PreservesRequired(t *testing.T) {
	m := &Model{modelName: "test-model"}

	tools := []*genai.Tool{{
		FunctionDeclarations: []*genai.FunctionDeclaration{{
			Name: "with_required",
			Parameters: &genai.Schema{
				Type:     genai.TypeObject,
				Required: []string{"artifact_names", "kind"},
				Properties: map[string]*genai.Schema{
					"artifact_names": {Type: genai.TypeArray, Items: &genai.Schema{Type: genai.TypeString}},
					"kind":           {Type: genai.TypeString},
				},
			},
		}},
	}}

	got, err := m.convertTools(tools)
	if err != nil {
		t.Fatalf("convertTools error: %v", err)
	}

	required := got[0].OfTool.InputSchema.Required
	if len(required) != 2 {
		t.Fatalf("Required = %v, want 2 entries", required)
	}
	want := map[string]bool{"artifact_names": true, "kind": true}
	for _, r := range required {
		if !want[r] {
			t.Errorf("unexpected required field %q", r)
		}
	}
}

// lowercaseTypes must walk the entire schema graph regardless of how deep
// types are nested. We exercise object properties, array items, and the
// awkward-but-legal case of nested arrays-of-objects-of-arrays. Anything that
// isn't a "type" key must be left untouched (we don't want to rewrite enum
// values that happen to be strings, for example).
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
			name: "nested object properties",
			in: map[string]any{
				"type": "OBJECT",
				"properties": map[string]any{
					"name": map[string]any{"type": "STRING"},
				},
			},
			want: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"name": map[string]any{"type": "string"},
				},
			},
		},
		{
			name: "array of objects",
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
			name: "leaves non-type strings alone",
			in: map[string]any{
				"type":        "STRING",
				"description": "DO NOT TOUCH ME",
				"enum":        []any{"FOO", "BAR"},
			},
			want: map[string]any{
				"type":        "string",
				"description": "DO NOT TOUCH ME",
				"enum":        []any{"FOO", "BAR"},
			},
		},
		{
			name: "non-string type values are left alone",
			in:   map[string]any{"type": 42},
			want: map[string]any{"type": 42},
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
