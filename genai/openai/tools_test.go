// Copyright 2025 achetronic
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

package openai

import (
	"strings"
	"testing"

	"google.golang.org/genai"
)

// convertTools is a thin wrapper around convertToFunctionParams. We exercise
// it end-to-end here to lock in two contracts of the public method:
//   - one OpenAI tool is emitted per genai FunctionDeclaration, regardless of
//     how the caller groups them inside Tool aggregates
//   - nil entries in the input slice are skipped silently (defensive: ADK
//     can produce them when filtering tools at runtime), so the function
//     must not panic on them
//
// The schema rewriting itself (lowercasing types, injecting properties) is
// covered separately in core_test.go to avoid double-instrumenting the same
// path.
func TestConvertTools(t *testing.T) {
	t.Run("propagates name, description, and rewritten schema", func(t *testing.T) {
		m := newModelForTest()
		tools := []*genai.Tool{{
			FunctionDeclarations: []*genai.FunctionDeclaration{{
				Name:        "search",
				Description: "Search the corpus",
				Parameters: &genai.Schema{
					Type: genai.TypeObject,
					Properties: map[string]*genai.Schema{
						"q": {Type: genai.TypeString},
					},
					Required: []string{"q"},
				},
			}},
		}}

		got, err := m.convertTools(tools)
		if err != nil {
			t.Fatalf("error: %v", err)
		}
		if len(got) != 1 {
			t.Fatalf("tools = %d, want 1", len(got))
		}

		fn := got[0].OfFunction
		if fn == nil {
			t.Fatalf("OfFunction = nil")
		}
		if fn.Function.Name != "search" {
			t.Errorf("Name = %q, want %q", fn.Function.Name, "search")
		}
		if !fn.Function.Description.Valid() || fn.Function.Description.Value != "Search the corpus" {
			t.Errorf("Description = %#v, want \"Search the corpus\"", fn.Function.Description)
		}

		params := map[string]any(fn.Function.Parameters)
		if got, _ := params["type"].(string); got != "object" {
			t.Errorf("schema type = %q, want \"object\" (lowercased)", got)
		}
		if _, ok := params["properties"].(map[string]any); !ok {
			t.Errorf("missing properties map: %#v", params)
		}
	})

	t.Run("nil tool entries are skipped", func(t *testing.T) {
		m := newModelForTest()
		got, err := m.convertTools([]*genai.Tool{nil, nil})
		if err != nil {
			t.Fatalf("error: %v", err)
		}
		if len(got) != 0 {
			t.Errorf("tools = %d, want 0", len(got))
		}
	})

	t.Run("multiple declarations across multiple tool groups", func(t *testing.T) {
		m := newModelForTest()
		tools := []*genai.Tool{
			{FunctionDeclarations: []*genai.FunctionDeclaration{
				{Name: "a"}, {Name: "b"},
			}},
			{FunctionDeclarations: []*genai.FunctionDeclaration{
				{Name: "c"},
			}},
		}
		got, err := m.convertTools(tools)
		if err != nil {
			t.Fatalf("error: %v", err)
		}
		if len(got) != 3 {
			t.Fatalf("tools = %d, want 3", len(got))
		}
		names := []string{
			got[0].OfFunction.Function.Name,
			got[1].OfFunction.Function.Name,
			got[2].OfFunction.Function.Name,
		}
		if !equalStringSlices(names, []string{"a", "b", "c"}) {
			t.Errorf("declaration order not preserved: %v", names)
		}
	})

	t.Run("falls back to legacy Parameters when ParametersJsonSchema is nil", func(t *testing.T) {
		m := newModelForTest()
		tools := []*genai.Tool{{
			FunctionDeclarations: []*genai.FunctionDeclaration{{
				Name: "legacy",
				Parameters: &genai.Schema{
					Type: genai.TypeObject,
					Properties: map[string]*genai.Schema{
						"x": {Type: genai.TypeNumber},
					},
				},
			}},
		}}
		got, err := m.convertTools(tools)
		if err != nil {
			t.Fatalf("error: %v", err)
		}
		params := map[string]any(got[0].OfFunction.Function.Parameters)
		if params["type"] != "object" {
			t.Errorf("legacy fallback didn't produce a usable schema: %#v", params)
		}
	})
}

// convertInlineDataToPart emits the right SDK union variant for each MIME
// type bucket OpenAI supports: images go through OfImageURL with a data URI,
// audio through OfInputAudio with a format hint, and PDFs/text through
// OfFile. Anything outside those buckets must surface an error rather than
// silently dropping content; the caller (convertContentToMessages) lets that
// error propagate so the request fails loudly instead of being sent without
// the user's attachment.
func TestConvertInlineDataToPart(t *testing.T) {
	cases := []struct {
		name       string
		mime       string
		data       []byte
		wantKind   string
		wantErr    bool
		extraCheck func(t *testing.T, mime string, data []byte, p any)
	}{
		{
			name:     "image/png becomes a data-URI image part",
			mime:     "image/png",
			data:     []byte("fakepng"),
			wantKind: "image",
		},
		{
			name:     "image/jpeg also routes to image",
			mime:     "image/jpeg",
			data:     []byte("fake"),
			wantKind: "image",
		},
		{
			name:     "audio/wav becomes an input audio part with format=wav",
			mime:     "audio/wav",
			data:     []byte("riff"),
			wantKind: "audio_wav",
		},
		{
			name:     "audio/mpeg becomes an input audio part with format=mp3",
			mime:     "audio/mpeg",
			data:     []byte("id3"),
			wantKind: "audio_mp3",
		},
		{
			name:     "application/pdf becomes a file part",
			mime:     "application/pdf",
			data:     []byte("%PDF"),
			wantKind: "file",
		},
		{
			name:     "text/plain also becomes a file part",
			mime:     "text/plain",
			data:     []byte("hello"),
			wantKind: "file",
		},
		{
			name:    "unsupported MIME type returns an error",
			mime:    "video/mp4",
			data:    []byte("x"),
			wantErr: true,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := convertInlineDataToPart(&genai.Blob{MIMEType: c.mime, Data: c.data})
			if c.wantErr {
				if err == nil {
					t.Errorf("expected error for MIME %q, got %#v", c.mime, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got == nil {
				t.Fatalf("nil part returned")
			}

			switch c.wantKind {
			case "image":
				if got.OfImageURL == nil {
					t.Fatalf("expected OfImageURL, got %#v", got)
				}
				url := got.OfImageURL.ImageURL.URL
				if !strings.HasPrefix(url, "data:"+c.mime+";base64,") {
					t.Errorf("URL prefix mismatch: %q", url)
				}
			case "audio_wav":
				if got.OfInputAudio == nil {
					t.Fatalf("expected OfInputAudio, got %#v", got)
				}
				if got.OfInputAudio.InputAudio.Format != "wav" {
					t.Errorf("format = %q, want \"wav\"", got.OfInputAudio.InputAudio.Format)
				}
			case "audio_mp3":
				if got.OfInputAudio == nil {
					t.Fatalf("expected OfInputAudio, got %#v", got)
				}
				if got.OfInputAudio.InputAudio.Format != "mp3" {
					t.Errorf("format = %q, want \"mp3\"", got.OfInputAudio.InputAudio.Format)
				}
			case "file":
				if got.OfFile == nil {
					t.Fatalf("expected OfFile, got %#v", got)
				}
				fd := got.OfFile.File.FileData
				if !fd.Valid() || !strings.HasPrefix(fd.Value, "data:"+c.mime+";base64,") {
					t.Errorf("FileData = %#v, expected data URI prefix data:%s;base64,", fd, c.mime)
				}
			}
		})
	}

	t.Run("nil blob returns an error", func(t *testing.T) {
		_, err := convertInlineDataToPart(nil)
		if err == nil {
			t.Errorf("expected error for nil blob")
		}
	})
}
