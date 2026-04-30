// Copyright 2025 achetronic
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

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
// payload field, instead of stringifying the entire union — that keeps the
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
				// not base64 — that is the whole point of using OfText
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
