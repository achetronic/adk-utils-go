// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package anthropic

import (
	"encoding/base64"
	"testing"

	"google.golang.org/genai"
)

// convertInlineDataToBlock routes a genai.Blob to the right Anthropic content
// block based on MIME type:
//   - all image types Anthropic supports become OfImage with a base64 source
//   - application/pdf becomes OfDocument with a base64 PDF source
//   - text/* becomes OfDocument with a plain-text source (Anthropic accepts
//     raw text rather than base64 here, so we pass data.Data through verbatim)
//   - anything else is rejected with an error
//
// We assert on which OfXxx variant ends up populated, plus the relevant
// payload field, instead of stringifying the entire union - that keeps the
// tests stable across SDK refactors that may shuffle field names while
// preserving the discriminator.
func TestConvertInlineDataToBlock(t *testing.T) {
	cases := []struct {
		name     string
		mime     string
		data     []byte
		wantKind string
		wantErr  bool
	}{
		{name: "image/png", mime: "image/png", data: []byte("png"), wantKind: "image"},
		{name: "image/jpeg", mime: "image/jpeg", data: []byte("jpg"), wantKind: "image"},
		{name: "image/jpg alias", mime: "image/jpg", data: []byte("jpg"), wantKind: "image"},
		{name: "image/gif", mime: "image/gif", data: []byte("gif"), wantKind: "image"},
		{name: "image/webp", mime: "image/webp", data: []byte("wbp"), wantKind: "image"},
		{name: "application/pdf", mime: "application/pdf", data: []byte("%PDF"), wantKind: "pdf"},
		{name: "text/plain", mime: "text/plain", data: []byte("hello"), wantKind: "text"},
		{name: "text/html", mime: "text/html", data: []byte("<html/>"), wantKind: "text"},
		{name: "video/mp4 unsupported", mime: "video/mp4", data: []byte("x"), wantErr: true},
		{name: "audio/wav unsupported", mime: "audio/wav", data: []byte("x"), wantErr: true},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := convertInlineDataToBlock(&genai.Blob{MIMEType: c.mime, Data: c.data})
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
				t.Fatalf("nil block returned")
			}

			switch c.wantKind {
			case "image":
				img := got.OfImage
				if img == nil {
					t.Fatalf("expected OfImage variant, got %#v", got)
				}
				src := img.Source.OfBase64
				if src == nil {
					t.Fatalf("expected base64 image source")
				}
				if string(src.MediaType) != c.mime {
					t.Errorf("MediaType = %q, want %q", src.MediaType, c.mime)
				}
				if src.Data != base64.StdEncoding.EncodeToString(c.data) {
					t.Errorf("base64 payload mismatch")
				}
			case "pdf":
				doc := got.OfDocument
				if doc == nil {
					t.Fatalf("expected OfDocument variant")
				}
				src := doc.Source.OfBase64
				if src == nil {
					t.Fatalf("expected base64 PDF source")
				}
				if src.Data != base64.StdEncoding.EncodeToString(c.data) {
					t.Errorf("base64 payload mismatch")
				}
			case "text":
				doc := got.OfDocument
				if doc == nil {
					t.Fatalf("expected OfDocument variant")
				}
				txt := doc.Source.OfText
				if txt == nil {
					t.Fatalf("expected plain-text source")
				}
				// Anthropic's plain-text source carries raw bytes verbatim,
				// not base64 - that is the whole point of using OfText
				// instead of OfBase64.
				if txt.Data != string(c.data) {
					t.Errorf("plain text data = %q, want %q", txt.Data, string(c.data))
				}
			}
		})
	}

	t.Run("nil blob returns an error", func(t *testing.T) {
		_, err := convertInlineDataToBlock(nil)
		if err == nil {
			t.Errorf("expected error for nil blob")
		}
	})
}

// convertFileDataToBlock builds an OfImage block whose source is a remote URL
// (OfURL) rather than base64 bytes, passing the FileURI through verbatim. Only
// image MIME types are supported, and the URI must be http(s) — plain http is
// allowed because Anthropic-compatible gateways behind Config.BaseURL commonly
// fetch from local http endpoints (anthropic.com itself only fetches https and
// enforces that server-side). Other schemes fail here with a clear error
// instead of a provider-side 400.
func TestConvertFileDataToBlock(t *testing.T) {
	const fileURI = "https://cdn.example.com/cat.png"

	cases := []struct {
		name    string
		mime    string
		uri     string
		wantErr bool
	}{
		{name: "image/png", mime: "image/png", uri: fileURI},
		{name: "image/jpeg", mime: "image/jpeg", uri: fileURI},
		{name: "image/jpg alias", mime: "image/jpg", uri: fileURI},
		{name: "image/gif", mime: "image/gif", uri: fileURI},
		{name: "image/webp", mime: "image/webp", uri: fileURI},
		{name: "application/pdf unsupported", mime: "application/pdf", uri: fileURI, wantErr: true},
		{name: "text/plain unsupported", mime: "text/plain", uri: fileURI, wantErr: true},
		{name: "video/mp4 unsupported", mime: "video/mp4", uri: fileURI, wantErr: true},
		{name: "empty MIME type rejected", mime: "", uri: fileURI, wantErr: true},
		{name: "plain http is allowed for gateways", mime: "image/png", uri: "http://localhost:8080/cat.png"},
		{name: "gs scheme rejected", mime: "image/png", uri: "gs://bucket/cat.png", wantErr: true},
		{name: "empty URI rejected", mime: "image/png", uri: "", wantErr: true},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := convertFileDataToBlock(&genai.FileData{MIMEType: c.mime, FileURI: c.uri})
			if c.wantErr {
				if err == nil {
					t.Errorf("expected error for MIME %q URI %q, got %#v", c.mime, c.uri, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got == nil || got.OfImage == nil {
				t.Fatalf("expected OfImage variant, got %#v", got)
			}
			src := got.OfImage.Source.OfURL
			if src == nil {
				t.Fatalf("expected a URL image source, got %#v", got.OfImage.Source)
			}
			if src.URL != c.uri {
				t.Errorf("URL = %q, want %q (must be passed through verbatim)", src.URL, c.uri)
			}
		})
	}

	t.Run("nil file data returns an error", func(t *testing.T) {
		_, err := convertFileDataToBlock(nil)
		if err == nil {
			t.Errorf("expected error for nil file data")
		}
	})
}
