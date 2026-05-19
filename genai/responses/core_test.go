// Copyright 2025 achetronic
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

package responses

import (
	"fmt"
	"reflect"
	"sort"
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
				"required":            []any{"prompt", "heightRatio"},
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
				"properties":          map[string]any{},
				"required":            []any{},
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
				"required":            []any{"name", "opts"},
				"properties": map[string]any{
					"name": map[string]any{"type": "string"},
					"opts": map[string]any{
						"type":                 []any{"object", "null"},
						"additionalProperties": false,
						"required":            []any{"color"},
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
				"required":            []any{"env"},
				"properties": map[string]any{
					"env": map[string]any{"type": []any{"string", "null"}},
				},
			},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := convertToStrictFunctionParams(c.in)
			sortRequiredFields(got)
			sortRequiredFields(c.want)
			if !reflect.DeepEqual(got, c.want) {
				t.Errorf("convertToStrictFunctionParams() = %#v\nwant %#v", got, c.want)
			}
		})
	}
}

// sortRequiredFields normalises the "required" array order so map iteration
// non-determinism does not cause test flakes.
func sortRequiredFields(schema map[string]any) {
	if schema == nil {
		return
	}
	if req, ok := schema["required"].([]any); ok {
		sort.Slice(req, func(i, j int) bool {
			return fmt.Sprint(req[i]) < fmt.Sprint(req[j])
		})
	}
	if props, ok := schema["properties"].(map[string]any); ok {
		for _, v := range props {
			if m, ok := v.(map[string]any); ok {
				sortRequiredFields(m)
			}
		}
	}
	if items, ok := schema["items"].(map[string]any); ok {
		sortRequiredFields(items)
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
