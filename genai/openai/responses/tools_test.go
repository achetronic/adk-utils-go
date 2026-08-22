// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

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
// Responses API content part. Images use ResponseInputImageParam; documents
// use ResponseInputFileParam. Audio must error: the API's file inputs do not
// accept it.
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
		{"json file", &genai.Blob{MIMEType: "application/json", Data: []byte("x")}, "file"},
		{"audio rejected", &genai.Blob{MIMEType: "audio/wav", Data: []byte("x")}, "error"},
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

// convertFileDataToPart passes a remote image URL straight through to
// ResponseInputImageParam and a remote PDF to ResponseInputFileParam's
// file_url. The URI must be http(s); plain http is allowed for
// API-compatible gateways. Everything else (and a nil input) must return an
// error instead of being silently dropped.
func TestConvertFileDataToPart(t *testing.T) {
	const fileURI = "https://cdn.example.com/cat.png"

	cases := []struct {
		name     string
		data     *genai.FileData
		wantType string // "image", "file", "error"
	}{
		{"png image", &genai.FileData{MIMEType: "image/png", FileURI: fileURI}, "image"},
		{"jpeg image", &genai.FileData{MIMEType: "image/jpeg", FileURI: fileURI}, "image"},
		{"webp image", &genai.FileData{MIMEType: "image/webp", FileURI: fileURI}, "image"},
		{"plain http allowed for gateways", &genai.FileData{MIMEType: "image/png", FileURI: "http://localhost:8080/cat.png"}, "image"},
		{"MIME parameters stripped", &genai.FileData{MIMEType: "image/png; charset=utf-8", FileURI: fileURI}, "image"},
		{"pdf becomes input_file", &genai.FileData{MIMEType: "application/pdf", FileURI: "https://cdn.example.com/report.pdf"}, "file"},
		{"docx becomes input_file", &genai.FileData{MIMEType: "application/vnd.openxmlformats-officedocument.wordprocessingml.document", FileURI: "https://cdn.example.com/spec.docx"}, "file"},
		{"csv becomes input_file", &genai.FileData{MIMEType: "text/csv", FileURI: "https://cdn.example.com/data.csv"}, "file"},
		{"application/csv becomes input_file", &genai.FileData{MIMEType: "application/csv", FileURI: "https://cdn.example.com/data.csv"}, "file"},
		{"javascript becomes input_file", &genai.FileData{MIMEType: "application/javascript", FileURI: "https://cdn.example.com/app.js"}, "file"},
		{"rfc822 mail becomes input_file", &genai.FileData{MIMEType: "message/rfc822", FileURI: "https://cdn.example.com/mail.eml"}, "file"},
		{"google doc becomes input_file", &genai.FileData{MIMEType: "application/vnd.google-apps.document", FileURI: "https://cdn.example.com/doc"}, "file"},
		{"typescript becomes input_file", &genai.FileData{MIMEType: "application/typescript", FileURI: "https://cdn.example.com/app.ts"}, "file"},
		{"yaml becomes input_file", &genai.FileData{MIMEType: "application/yaml", FileURI: "https://cdn.example.com/conf.yaml"}, "file"},
		{"x-toml becomes input_file", &genai.FileData{MIMEType: "application/x-toml", FileURI: "https://cdn.example.com/conf.toml"}, "file"},
		{"sql becomes input_file", &genai.FileData{MIMEType: "application/x-sql", FileURI: "https://cdn.example.com/schema.sql"}, "file"},
		{"audio URL rejected", &genai.FileData{MIMEType: "audio/mpeg", FileURI: "https://cdn.example.com/talk.mp3"}, "error"},
		{"unsupported", &genai.FileData{MIMEType: "video/mp4", FileURI: fileURI}, "error"},
		{"empty MIME type rejected", &genai.FileData{MIMEType: "", FileURI: fileURI}, "error"},
		{"gs scheme rejected", &genai.FileData{MIMEType: "image/png", FileURI: "gs://bucket/cat.png"}, "error"},
		{"data URI rejected (use InlineData)", &genai.FileData{MIMEType: "image/png", FileURI: "data:image/png;base64,AAAA"}, "error"},
		{"empty URI rejected", &genai.FileData{MIMEType: "image/png", FileURI: ""}, "error"},
		{"nil file data", nil, "error"},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := convertFileDataToPart(c.data)

			if c.wantType == "error" {
				if err == nil {
					t.Errorf("expected error, got %+v", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if c.wantType == "file" {
				if got.OfInputFile == nil {
					t.Fatalf("expected OfInputFile, got %+v", got)
				}
				if url := got.OfInputFile.FileURL.Value; url != c.data.FileURI {
					t.Errorf("FileURL = %q, want the raw file URI %q", url, c.data.FileURI)
				}
				return
			}
			if got.OfInputImage == nil {
				t.Fatalf("expected OfInputImage, got %+v", got)
			}
			if url := got.OfInputImage.ImageURL.Value; url != c.data.FileURI {
				t.Errorf("ImageURL = %q, want the raw file URI %q", url, c.data.FileURI)
			}
		})
	}
}

// Tool parameters that cannot be serialized to JSON must fail loudly:
// silently sending the request without the tool definition surfaces as
// inexplicable model behaviour.
func TestConvertTools_UnserializableParams(t *testing.T) {
	tools := []*genai.Tool{{
		FunctionDeclarations: []*genai.FunctionDeclaration{{
			Name:                 "broken",
			ParametersJsonSchema: make(chan int),
		}},
	}}

	if _, err := convertTools(tools); err == nil {
		t.Fatalf("convertTools() error = nil, want serialization error")
	}
}

// Tool facets without a Responses API mapping must error instead of
// silently producing an empty tool list.
func TestConvertTools_UnsupportedFacetsError(t *testing.T) {
	_, err := convertTools([]*genai.Tool{{GoogleSearch: &genai.GoogleSearch{}}})
	if err == nil {
		t.Fatalf("expected an error for a GoogleSearch tool, got none")
	}
}

// Base64 documents must carry a filename with an extension: the API
// identifies the document type by it. DisplayName wins; a missing extension
// is filled in from the MIME type, and types with no unambiguous extension
// require the caller to name the file.
func TestConvertInlineDataToPart_DocumentFilename(t *testing.T) {
	cases := []struct {
		name string
		blob *genai.Blob
		want string // expected filename, "" means an error is expected
	}{
		{"MIME-derived default", &genai.Blob{MIMEType: "application/pdf", Data: []byte("x")}, "input.pdf"},
		{"DisplayName with extension kept", &genai.Blob{MIMEType: "text/csv", Data: []byte("x"), DisplayName: "report.csv"}, "report.csv"},
		{"DisplayName without extension gains one", &genai.Blob{MIMEType: "application/pdf", Data: []byte("x"), DisplayName: "季度报告"}, "季度报告.pdf"},
		{"python maps to py", &genai.Blob{MIMEType: "text/x-python", Data: []byte("x")}, "input.py"},
		{"ambiguous iwork requires DisplayName", &genai.Blob{MIMEType: "application/vnd.apple.iwork", Data: []byte("x")}, ""},
		{"ambiguous iwork with DisplayName passes", &genai.Blob{MIMEType: "application/vnd.apple.iwork", Data: []byte("x"), DisplayName: "deck.key"}, "deck.key"},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := convertInlineDataToPart(c.blob)
			if c.want == "" {
				if err == nil {
					t.Fatalf("expected an error, got %+v", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if fn := got.OfInputFile.Filename.Value; fn != c.want {
				t.Errorf("Filename = %q, want %q", fn, c.want)
			}
		})
	}
}

// extensionForMIME maps explicit cases, falls back to the stripped subtype,
// and returns empty for residues that are not usable extensions.
func TestExtensionForMIME(t *testing.T) {
	cases := map[string]string{
		"text/plain": "txt",
		"application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
		"application/x-sql":                    "sql",
		"application/json":                     "json",
		"text/csv":                             "csv",
		"application/x-yaml":                   "yaml",
		"message/rfc822":                       "eml",
		"text/x-python":                        "py",
		"text/x-typescript":                    "ts",
		"application/x-rust":                   "rs",
		"application/x-bash":                   "sh",
		"text/x-c++":                           "cpp",
		"text/x-golang":                        "go",
		"application/vnd.apple.iwork":          "",
		"application/vnd.google-apps.document": "",
	}
	for mime, want := range cases {
		if got := extensionForMIME(mime); got != want {
			t.Errorf("extensionForMIME(%q) = %q, want %q", mime, got, want)
		}
	}
}
