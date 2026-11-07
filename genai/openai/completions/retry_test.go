// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package completions

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/achetronic/adk-utils-go/genai/common"
	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"
)

// TestRetryBudget pins the retry budget against a consistently 503ing backend:
// the SDK default makes three attempts, and a zero budget (or a clamped
// negative) makes exactly one.
func TestRetryBudget(t *testing.T) {
	zero := 0
	negative := -1

	cases := []struct {
		name      string
		options   common.Options
		wantCalls int32
	}{
		{name: "unset uses openai-go default (2 retries = 3 calls)", options: common.Options{}, wantCalls: 3},
		{name: "zero disables retries (1 call)", options: common.Options{MaxRetries: &zero}, wantCalls: 1},
		{name: "negative clamps to zero (1 call)", options: common.Options{MaxRetries: &negative}, wantCalls: 1},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var calls int32
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				atomic.AddInt32(&calls, 1)
				w.WriteHeader(http.StatusServiceUnavailable)
				io.WriteString(w, `{"error":{"message":"overloaded"}}`)
			}))
			defer srv.Close()

			req := &model.LLMRequest{
				Config: &genai.GenerateContentConfig{},
				Contents: []*genai.Content{
					{Role: "user", Parts: []*genai.Part{{Text: "hello"}}},
				},
			}

			cfg := Config{
				BaseURL: srv.URL, APIKey: "test-key", ModelName: "gpt-test",
				Options: tc.options,
			}
			m := New(cfg)

			for _, err := range m.GenerateContent(context.Background(), req, false) {
				_ = err // every attempt fails with 503; we assert on the call count
			}
			if got := atomic.LoadInt32(&calls); got != tc.wantCalls {
				t.Errorf("backend hit %d times, want %d (options=%+v)", got, tc.wantCalls, tc.options)
			}
		})
	}
}
