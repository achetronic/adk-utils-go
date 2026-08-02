// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

// Copyright 2025 achetronic
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

package responses

import (
	"encoding/json"
	"fmt"
	"reflect"
	"testing"

	"google.golang.org/genai"
)

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

func TestConvertToFunctionParams(t *testing.T) {
	cases := []struct {
		name string
		in   any
		want map[string]any
	}{
		{
			name: "required auto-filled and optional fields made nullable",
			in: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"prompt":      map[string]any{"type": "string"},
					"heightRatio": map[string]any{"type": "integer"},
				},
				"required": []any{"prompt"},
			},
			want: map[string]any{
				"type":                 "object",
				"additionalProperties": false,
				"required":             []any{"heightRatio", "prompt"},
				"properties": map[string]any{
					"prompt":      map[string]any{"type": "string"},
					"heightRatio": map[string]any{"type": []any{"integer", "null"}},
				},
			},
		},
		{
			name: "nil input returns valid empty object schema",
			in:   nil,
			want: map[string]any{
				"type":                 "object",
				"properties":           map[string]any{},
				"required":             []any{},
				"additionalProperties": false,
			},
		},
		{
			name: "optional nested object gets strict treatment too",
			in: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"name": map[string]any{"type": "string"},
					"opts": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"color": map[string]any{"type": "string"},
						},
						"required": []any{"color"},
					},
				},
				"required": []any{"name"},
			},
			want: map[string]any{
				"type":                 "object",
				"additionalProperties": false,
				"required":             []any{"name", "opts"},
				"properties": map[string]any{
					"name": map[string]any{"type": "string"},
					"opts": map[string]any{
						"type":                 []any{"object", "null"},
						"additionalProperties": false,
						"required":             []any{"color"},
						"properties": map[string]any{
							"color": map[string]any{"type": "string"},
						},
					},
				},
			},
		},
		{
			name: "schema with properties and null required is treated as object",
			in: map[string]any{
				"properties": map[string]any{
					"env": map[string]any{"type": "string"},
				},
				"required": nil,
			},
			want: map[string]any{
				"type":                 "object",
				"additionalProperties": false,
				"required":             []any{"env"},
				"properties": map[string]any{
					"env": map[string]any{"type": []any{"string", "null"}},
				},
			},
		},
		{
			name: "bare const and enum get their literal type filled in",
			in: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"kind":  map[string]any{"const": "image"},
					"level": map[string]any{"enum": []any{float64(1), float64(2)}},
					"mixed": map[string]any{"enum": []any{"a", float64(1)}},
				},
				"required": []any{"kind", "level", "mixed"},
			},
			want: map[string]any{
				"type":                 "object",
				"additionalProperties": false,
				"required":             []any{"kind", "level", "mixed"},
				"properties": map[string]any{
					"kind":  map[string]any{"type": "string", "const": "image"},
					"level": map[string]any{"type": "number", "enum": []any{float64(1), float64(2)}},
					"mixed": map[string]any{"enum": []any{"a", float64(1)}},
				},
			},
		},
		{
			name: "$defs branches normalised and oneOf renamed to anyOf",
			in: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"item": map[string]any{"$ref": "#/$defs/item"},
				},
				"required": []any{"item"},
				"$defs": map[string]any{
					"item": map[string]any{
						"oneOf": []any{
							map[string]any{
								"type": "object",
								"properties": map[string]any{
									"text":    map[string]any{"type": "string"},
									"caption": map[string]any{"type": "string"},
									"source":  map[string]any{"$ref": "#/$defs/source"},
								},
								"required": []any{"text"},
							},
						},
					},
				},
			},
			want: map[string]any{
				"type":                 "object",
				"additionalProperties": false,
				"required":             []any{"item"},
				"properties": map[string]any{
					"item": map[string]any{"$ref": "#/$defs/item"},
				},
				"$defs": map[string]any{
					"item": map[string]any{
						"anyOf": []any{
							map[string]any{
								"type":                 "object",
								"additionalProperties": false,
								"required":             []any{"caption", "source", "text"},
								"properties": map[string]any{
									"text":    map[string]any{"type": "string"},
									"caption": map[string]any{"type": []any{"string", "null"}},
									"source": map[string]any{
										"anyOf": []any{
											map[string]any{"$ref": "#/$defs/source"},
											map[string]any{"type": "null"},
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			// Results are compared verbatim: the "required" order must be
			// deterministic (sorted), because an unstable order changes the
			// serialized tool definition on every request and breaks OpenAI
			// prompt-cache prefix matching.
			got := convertToStrictFunctionParams(c.in)
			if !reflect.DeepEqual(got, c.want) {
				t.Errorf("convertToStrictFunctionParams() = %#v\nwant %#v", got, c.want)
			}
		})
	}
}

// Schemas outside the strict subset must be sent as authored with
// strict=false instead of being normalized into a different meaning.
func TestConvertFunctionParamsStrictFallback(t *testing.T) {
	cases := []struct {
		name       string
		in         map[string]any
		wantStrict bool
	}{
		{
			name: "free-form object falls back to non-strict",
			in: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"set": map[string]any{"type": "object", "additionalProperties": true},
				},
			},
			wantStrict: false,
		},
		{
			name: "array without items falls back to non-strict",
			in: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"items": map[string]any{"type": "array"},
				},
			},
			wantStrict: false,
		},
		{
			name: "unsupported keyword falls back to non-strict",
			in: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"name": map[string]any{"type": "string", "minLength": float64(1)},
				},
			},
			wantStrict: false,
		},
		{
			name: "expressible schema stays strict",
			in: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"kind": map[string]any{"const": "image"},
					"tags": map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
				},
				"required": []any{"kind"},
			},
			wantStrict: true,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, strict := convertFunctionParams(c.in)
			if strict != c.wantStrict {
				t.Fatalf("strict = %v, want %v", strict, c.wantStrict)
			}
			if got == nil {
				t.Fatal("params = nil")
			}
			if !strict {
				// Non-strict schemas must pass through as authored.
				if !reflect.DeepEqual(got, c.in) {
					t.Errorf("non-strict schema was altered:\ngot  %#v\nwant %#v", got, c.in)
				}
			} else {
				if got["additionalProperties"] != false {
					t.Errorf("strict schema missing additionalProperties=false: %#v", got)
				}
			}
		})
	}
}

// convertToStrictFunctionParams must deep-copy the input so callers who
// reuse schemas across multiple tool registrations don't see mutations.
func TestConvertToStrictFunctionParams_DeepCopy(t *testing.T) {
	original := map[string]any{
		"type":       "object",
		"properties": map[string]any{"a": map[string]any{"type": "string"}},
	}

	_ = convertToStrictFunctionParams(original)

	// The original must still have a plain string type, not ["string","null"]
	prop := original["properties"].(map[string]any)["a"].(map[string]any)
	if _, ok := prop["type"].(string); !ok {
		t.Errorf("original schema was mutated: a.type = %#v, want string", prop["type"])
	}
	if _, has := original["additionalProperties"]; has {
		t.Errorf("original schema was mutated: has additionalProperties")
	}
}

// The normalised schema must serialize to identical bytes on every call:
// tool definitions are part of the prompt-cache prefix, so any instability
// (e.g. map-iteration order leaking into the "required" array) would cause
// a cache miss on every request.
func TestConvertToStrictFunctionParams_Deterministic(t *testing.T) {
	in := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"alpha": map[string]any{"type": "string"},
			"beta":  map[string]any{"type": "integer"},
			"gamma": map[string]any{"type": "boolean"},
			"delta": map[string]any{"type": "number"},
			"omega": map[string]any{"type": "string"},
		},
	}

	first, err := json.Marshal(convertToStrictFunctionParams(in))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	for range 50 {
		next, err := json.Marshal(convertToStrictFunctionParams(in))
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		if string(next) != string(first) {
			t.Fatalf("serialized schema is unstable:\nfirst = %s\n next = %s", first, next)
		}
	}
}

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

// Optional properties written as a typeless anyOf union must gain a null
// branch when strict mode moves them into required; without it a previously
// optional parameter silently becomes mandatory. A union that already has a
// null branch stays unchanged.
func TestNormalizeStrictSchema_OptionalAnyOfGetsNullBranch(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"target": map[string]any{"anyOf": []any{
				map[string]any{"type": "string"},
				map[string]any{"type": "integer"},
			}},
		},
	}

	normalizeStrictSchema(schema)
	normalizeStrictSchema(schema) // idempotent: a second pass adds nothing

	target := schema["properties"].(map[string]any)["target"].(map[string]any)
	branches, _ := target["anyOf"].([]any)
	if len(branches) != 3 {
		t.Fatalf("anyOf branches = %d, want 3 (original two plus null)", len(branches))
	}
	last, _ := branches[2].(map[string]any)
	if last["type"] != "null" {
		t.Errorf("last branch = %+v, want {type: null}", last)
	}
}

// OpenAPI-style "nullable" is not a JSON Schema keyword and the strict
// validator rejects it, so it must convert to a null type union (true) or
// simply disappear (false).
func TestNormalizeStrictSchema_NullableConvertsToNullUnion(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"note": map[string]any{"type": "string", "nullable": true},
			"tag":  map[string]any{"type": "string", "nullable": false},
		},
		"required": []any{"note", "tag"},
	}

	normalizeStrictSchema(schema)

	props := schema["properties"].(map[string]any)
	note := props["note"].(map[string]any)
	if _, ok := note["nullable"]; ok {
		t.Errorf("nullable key survived normalization: %+v", note)
	}
	if got, want := fmt.Sprintf("%v", note["type"]), "[string null]"; got != want {
		t.Errorf("note type = %v, want %v", note["type"], want)
	}
	tag := props["tag"].(map[string]any)
	if _, ok := tag["nullable"]; ok {
		t.Errorf("nullable key survived normalization: %+v", tag)
	}
	if tag["type"] != "string" {
		t.Errorf("tag type = %v, want string (nullable false adds nothing)", tag["type"])
	}
}

// The strict validator rejects a $ref with sibling keys. The reference
// hoists into a single-branch anyOf with the siblings kept on the node; a
// bare $ref stays untouched; combining $ref with a union goes non-strict
// because the conjunction cannot be expressed faithfully.
func TestNormalizeStrictSchema_RefWithSiblings(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"target": map[string]any{"$ref": "#/$defs/target", "description": "what to hit"},
			"backup": map[string]any{"$ref": "#/$defs/target"},
		},
		"required": []any{"target", "backup"},
		"$defs": map[string]any{
			"target": map[string]any{"type": "string"},
		},
	}

	normalizeStrictSchema(schema)

	props := schema["properties"].(map[string]any)
	target := props["target"].(map[string]any)
	if _, ok := target["$ref"]; ok {
		t.Errorf("$ref with siblings survived at top level: %+v", target)
	}
	if target["description"] != "what to hit" {
		t.Errorf("description lost: %+v", target)
	}
	branches, _ := target["anyOf"].([]any)
	if len(branches) != 1 || branches[0].(map[string]any)["$ref"] != "#/$defs/target" {
		t.Errorf("anyOf = %+v, want a single $ref branch", target["anyOf"])
	}
	backup := props["backup"].(map[string]any)
	if backup["$ref"] != "#/$defs/target" || len(backup) != 1 {
		t.Errorf("bare $ref should stay untouched, got %+v", backup)
	}
}

// An optional $ref property with siblings must end up nullable and keep its
// annotations: the hoisted anyOf gains a null branch.
func TestNormalizeStrictSchema_OptionalRefWithSiblings(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"target": map[string]any{"$ref": "#/$defs/target", "description": "what to hit"},
		},
		"$defs": map[string]any{
			"target": map[string]any{"type": "string"},
		},
	}

	normalizeStrictSchema(schema)

	target := schema["properties"].(map[string]any)["target"].(map[string]any)
	if target["description"] != "what to hit" {
		t.Errorf("description lost: %+v", target)
	}
	branches, _ := target["anyOf"].([]any)
	if len(branches) != 2 {
		t.Fatalf("anyOf = %+v, want $ref branch plus null branch", target["anyOf"])
	}
	if branches[0].(map[string]any)["$ref"] != "#/$defs/target" {
		t.Errorf("first branch = %+v, want the $ref", branches[0])
	}
	if branches[1].(map[string]any)["type"] != "null" {
		t.Errorf("second branch = %+v, want null", branches[1])
	}
}

// $ref combined with a union is a conjunction the strict subset cannot
// express, so the whole tool goes non-strict instead of being rewritten.
func TestConvertFunctionParams_RefWithUnionGoesNonStrict(t *testing.T) {
	params := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"target": map[string]any{
				"$ref":  "#/$defs/a",
				"anyOf": []any{map[string]any{"type": "string"}},
			},
		},
		"$defs": map[string]any{
			"a": map[string]any{"type": "string"},
		},
	}

	schemaMap, strict := convertFunctionParams(params)
	if schemaMap == nil {
		t.Fatalf("conversion failed")
	}
	if strict {
		t.Errorf("strict = true, want false for a $ref+union conjunction")
	}
}

// Go schema generators commonly emit a root-level $ref pointing into $defs.
// Strict mode wants a plain object root, so the reference inlines; shapes
// that cannot inline faithfully go non-strict instead of reaching the API
// as a root anyOf or a root $ref, which strict mode rejects.
func TestConvertFunctionParams_RootRef(t *testing.T) {
	t.Run("root ref with defs inlines and stays strict", func(t *testing.T) {
		params := map[string]any{
			"$ref": "#/$defs/args",
			"$defs": map[string]any{
				"args": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"query": map[string]any{"type": "string"},
						"child": map[string]any{"$ref": "#/$defs/args"},
					},
					"required": []any{"query"},
				},
			},
		}

		schemaMap, strict := convertFunctionParams(params)
		if !strict {
			t.Fatalf("strict = false, want true for an inlinable root ref")
		}
		if _, ok := schemaMap["$ref"]; ok {
			t.Errorf("root $ref survived: %+v", schemaMap)
		}
		if _, ok := schemaMap["anyOf"]; ok {
			t.Errorf("root anyOf is rejected by strict mode: %+v", schemaMap)
		}
		if schemaMap["type"] != "object" {
			t.Errorf("root type = %v, want object", schemaMap["type"])
		}
		if _, ok := schemaMap["$defs"].(map[string]any); !ok {
			t.Errorf("$defs lost from the root: %+v", schemaMap)
		}
		props, _ := schemaMap["properties"].(map[string]any)
		if _, ok := props["query"]; !ok {
			t.Errorf("inlined properties missing: %+v", schemaMap)
		}
	})

	t.Run("root ref with extra siblings goes non-strict", func(t *testing.T) {
		params := map[string]any{
			"$ref":        "#/$defs/args",
			"description": "tool arguments",
			"$defs": map[string]any{
				"args": map[string]any{"type": "object"},
			},
		}
		if _, strict := convertFunctionParams(params); strict {
			t.Errorf("strict = true, want false for a root ref with extra siblings")
		}
	})

	t.Run("unresolvable root ref goes non-strict", func(t *testing.T) {
		params := map[string]any{
			"$ref":  "#/definitions/args",
			"$defs": map[string]any{"args": map[string]any{"type": "object"}},
		}
		if _, strict := convertFunctionParams(params); strict {
			t.Errorf("strict = true, want false for an unresolvable root ref")
		}
	})

	t.Run("non-object root goes non-strict", func(t *testing.T) {
		params := map[string]any{
			"type":  "array",
			"items": map[string]any{"type": "string"},
		}
		if _, strict := convertFunctionParams(params); strict {
			t.Errorf("strict = true, want false for a non-object root")
		}
	})
}

// The API rejects anyOf at the root of a strict schema, and a root oneOf
// would be renamed into exactly that during normalization. Both go
// non-strict, whether written directly or reached by inlining a root $ref.
func TestConvertFunctionParams_RootUnionGoesNonStrict(t *testing.T) {
	t.Run("direct root anyOf", func(t *testing.T) {
		params := map[string]any{
			"type": "object",
			"anyOf": []any{
				map[string]any{"required": []any{"a"}},
			},
			"properties": map[string]any{"a": map[string]any{"type": "string"}},
		}
		m, strict := convertFunctionParams(params)
		if strict {
			t.Errorf("strict = true, want false for a root anyOf")
		}
		if m["anyOf"] == nil {
			t.Errorf("non-strict schema should pass through unchanged: %+v", m)
		}
	})

	t.Run("root oneOf reached through an inlined ref", func(t *testing.T) {
		params := map[string]any{
			"$ref": "#/$defs/args",
			"$defs": map[string]any{
				"args": map[string]any{
					"type": "object",
					"oneOf": []any{
						map[string]any{"properties": map[string]any{"a": map[string]any{"type": "string"}}},
					},
					"properties": map[string]any{"a": map[string]any{"type": "string"}},
				},
			},
		}
		if _, strict := convertFunctionParams(params); strict {
			t.Errorf("strict = true, want false for an inlined root oneOf")
		}
	})
}
