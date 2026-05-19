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

	"google.golang.org/genai"
)

// convertTools maps genai FunctionDeclarations into Responses API
// FunctionToolParam entries. Nil tools are skipped. ParametersJsonSchema
// takes precedence over the legacy Parameters field.
func TestConvertTools(t *testing.T) {
	t.Run("single function declaration", func(t *testing.T) {
		tools := []*genai.Tool{{
			FunctionDeclarations: []*genai.FunctionDeclaration{{
				Name:        "get_weather",
				Description: "Get weather for a city",
				ParametersJsonSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"city": map[string]any{"type": "string"},
					},
				},
			}},
		}}

		got, err := convertTools(tools)
		if err != nil {
			t.Fatalf("convertTools: %v", err)
		}
		if len(got) != 1 {
			t.Fatalf("expected 1 tool, got %d", len(got))
		}
		fn := got[0].OfFunction
		if fn == nil {
			t.Fatalf("expected OfFunction to be set")
		}
		if fn.Name != "get_weather" {
			t.Errorf("Name = %q, want get_weather", fn.Name)
		}
	})

	t.Run("nil tool is skipped", func(t *testing.T) {
		tools := []*genai.Tool{nil, {
			FunctionDeclarations: []*genai.FunctionDeclaration{{
				Name: "fn",
			}},
		}}

		got, err := convertTools(tools)
		if err != nil {
			t.Fatalf("convertTools: %v", err)
		}
		if len(got) != 1 {
			t.Fatalf("expected 1 tool (nil skipped), got %d", len(got))
		}
	})

	t.Run("multiple declarations across tool groups", func(t *testing.T) {
		tools := []*genai.Tool{
			{FunctionDeclarations: []*genai.FunctionDeclaration{
				{Name: "a"}, {Name: "b"},
			}},
			{FunctionDeclarations: []*genai.FunctionDeclaration{
				{Name: "c"},
			}},
		}

		got, err := convertTools(tools)
		if err != nil {
			t.Fatalf("convertTools: %v", err)
		}
		if len(got) != 3 {
			t.Fatalf("expected 3 tools, got %d", len(got))
		}
	})
}

// convertInlineDataToPart routes different MIME types to the correct
// Responses API content part. Images use ResponseInputImageParam;
// PDFs, text, and audio use ResponseInputFileParam.
func TestConvertInlineDataToPart(t *testing.T) {
	cases := []struct {
		name     string
		blob     *genai.Blob
		wantType string // "image", "file", "error"
	}{
		{"png image", &genai.Blob{MIMEType: "image/png", Data: []byte("x")}, "image"},
		{"jpeg image", &genai.Blob{MIMEType: "image/jpeg", Data: []byte("x")}, "image"},
		{"webp image", &genai.Blob{MIMEType: "image/webp", Data: []byte("x")}, "image"},
		{"gif image", &genai.Blob{MIMEType: "image/gif", Data: []byte("x")}, "image"},
		{"pdf file", &genai.Blob{MIMEType: "application/pdf", Data: []byte("x")}, "file"},
		{"text file", &genai.Blob{MIMEType: "text/plain", Data: []byte("x")}, "file"},
		{"audio file", &genai.Blob{MIMEType: "audio/wav", Data: []byte("x")}, "file"},
		{"unsupported", &genai.Blob{MIMEType: "video/mp4", Data: []byte("x")}, "error"},
		{"nil blob", nil, "error"},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := convertInlineDataToPart(c.blob)

			if c.wantType == "error" {
				if err == nil {
					t.Errorf("expected error, got %+v", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			switch c.wantType {
			case "image":
				if got.OfInputImage == nil {
					t.Errorf("expected OfInputImage, got %+v", got)
				}
			case "file":
				if got.OfInputFile == nil {
					t.Errorf("expected OfInputFile, got %+v", got)
				}
			}
		})
	}
}
