// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package anthropic

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"
)

// streamFinalResponse spins up a fake SSE Messages endpoint that replays the
// given events, fires one STREAMING request and returns the final aggregated
// response (the last non-partial one).
func streamFinalResponse(t *testing.T, events []string) *model.LLMResponse {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		for _, e := range events {
			_, _ = w.Write([]byte(e + "\n\n"))
		}
	}))
	defer srv.Close()

	m := New(Config{BaseURL: srv.URL, APIKey: "test-key", ModelName: "claude-test"})

	req := &model.LLMRequest{
		Config:   &genai.GenerateContentConfig{},
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "hi"}}}},
	}

	var final *model.LLMResponse
	for resp, err := range m.GenerateContent(context.Background(), req, true) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
		if !resp.Partial {
			final = resp
		}
	}
	if final == nil {
		t.Fatalf("stream yielded no final response")
	}
	return final
}

func sse(event, data string) string {
	return "event: " + event + "\ndata: " + data
}

// Gateways such as new-api (relaying an OpenAI-protocol upstream) send a
// message_start whose usage is a pre-generation ESTIMATE with no cache
// fields, and report the authoritative totals — including the prompt-caching
// split — only in the final message_delta. The final response must carry the
// delta's numbers, not the estimate. Event sequence captured from a live
// new-api endpoint.
func TestStreamUsage_FinalDeltaAuthoritative(t *testing.T) {
	final := streamFinalResponse(t, []string{
		sse("message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":14105,"output_tokens":0}}}`),
		sse("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`),
		sse("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}`),
		sse("content_block_stop", `{"type":"content_block_stop","index":0}`),
		sse("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":1695,"output_tokens":695,"cache_creation_input_tokens":0,"cache_read_input_tokens":10624}}`),
		sse("message_stop", `{"type":"message_stop"}`),
	})

	u := final.UsageMetadata
	if u == nil {
		t.Fatalf("final response has no usage metadata")
	}
	if got, want := u.PromptTokenCount, int32(1695+10624); got != want {
		t.Errorf("PromptTokenCount = %d, want %d (delta totals, not the message_start estimate)", got, want)
	}
	if got, want := u.CachedContentTokenCount, int32(10624); got != want {
		t.Errorf("CachedContentTokenCount = %d, want %d", got, want)
	}
	if got, want := u.CandidatesTokenCount, int32(695); got != want {
		t.Errorf("CandidatesTokenCount = %d, want %d", got, want)
	}
	if !strings.Contains(final.Content.Parts[0].Text, "ok") {
		t.Errorf("final text = %q, want it to contain the streamed delta", final.Content.Parts[0].Text)
	}
}

// A message_delta carrying only output_tokens (the classic Anthropic wire
// shape) must NOT clobber the usage from message_start: absent fields are
// left untouched.
func TestStreamUsage_SparseDeltaKeepsStartValues(t *testing.T) {
	final := streamFinalResponse(t, []string{
		sse("message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":100,"output_tokens":0,"cache_creation_input_tokens":0,"cache_read_input_tokens":30}}}`),
		sse("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`),
		sse("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}`),
		sse("content_block_stop", `{"type":"content_block_stop","index":0}`),
		sse("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":7}}`),
		sse("message_stop", `{"type":"message_stop"}`),
	})

	u := final.UsageMetadata
	if u == nil {
		t.Fatalf("final response has no usage metadata")
	}
	if got, want := u.PromptTokenCount, int32(100+30); got != want {
		t.Errorf("PromptTokenCount = %d, want %d (message_start values preserved)", got, want)
	}
	if got, want := u.CachedContentTokenCount, int32(30); got != want {
		t.Errorf("CachedContentTokenCount = %d, want %d", got, want)
	}
	if got, want := u.CandidatesTokenCount, int32(7); got != want {
		t.Errorf("CandidatesTokenCount = %d, want %d", got, want)
	}
}
