// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package openai

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"
)

// streamResponses drives one streaming request against a fake SSE endpoint
// serving the given data lines, and returns every LLMResponse yielded.
//
// Going through the public streaming path matters here: openai-go's
// accumulator keeps no raw JSON on the message it aggregates, so a test that
// fabricates an accumulator from a whole response proves nothing about a real
// stream. Only real chunks exercise the adapter's own accumulation.
func streamResponses(t *testing.T, cfg Config, dataLines []string) []*model.LLMResponse {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.WriteHeader(http.StatusOK)
		for _, line := range dataLines {
			io.WriteString(w, "data: "+line+"\n\n")
		}
		io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	cfg.BaseURL = srv.URL
	cfg.APIKey = "test-key"
	cfg.ModelName = "gpt-test"
	m := New(cfg)

	var out []*model.LLMResponse
	for resp, err := range m.GenerateContent(context.Background(), &model.LLMRequest{
		Config:   &genai.GenerateContentConfig{},
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "hi"}}}},
	}, true) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
		out = append(out, resp)
	}
	if len(out) == 0 {
		t.Fatalf("no responses yielded")
	}
	return out
}

// finalOf returns the terminal response of a streamed turn, the one whose
// content becomes conversation history.
func finalOf(t *testing.T, responses []*model.LLMResponse) *model.LLMResponse {
	t.Helper()
	final := responses[len(responses)-1]
	if final.Partial || !final.TurnComplete {
		t.Fatalf("last response is not the terminal one: %#v", final)
	}
	return final
}

// Plain-text reasoning streamed across chunks has to reach the terminal
// response, not just the partials. The terminal content is what gets
// persisted, so reasoning that only ever appeared on partials is lost from
// history.
func TestGenerateStream_ReasoningReachesFinalResponse(t *testing.T) {
	responses := streamResponses(t, Config{}, []string{
		`{"id":"x","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"first half "},"finish_reason":null}]}`,
		`{"id":"x","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"reasoning_content":"second half"},"finish_reason":null}]}`,
		`{"id":"x","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"content":"the answer"},"finish_reason":"stop"}]}`,
	})

	final := finalOf(t, responses)
	if len(final.Content.Parts) != 2 {
		t.Fatalf("expected reasoning + answer in the terminal response, got %d parts: %#v", len(final.Content.Parts), final.Content.Parts)
	}
	if !final.Content.Parts[0].Thought {
		t.Errorf("first terminal part Thought = false, want true")
	}
	if got := final.Content.Parts[0].Text; got != "first half second half" {
		t.Errorf("accumulated reasoning = %q, want the chunks concatenated", got)
	}
	if final.Content.Parts[1].Text != "the answer" {
		t.Errorf("answer part = %q", final.Content.Parts[1].Text)
	}
}

// Reasoning blocks stream in pieces keyed by index. OpenRouter builds the
// complete sequence by concatenating chunks in order, so the terminal
// response must carry one merged block per index, in first-seen order.
func TestGenerateStream_ReasoningDetailsMergeAcrossChunks(t *testing.T) {
	responses := streamResponses(t, Config{ReasoningField: "reasoning", SupportsReasoningDetails: true}, []string{
		`{"id":"x","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","reasoning_details":[{"type":"reasoning.text","text":"Let me think ","signature":null,"id":"t-1","index":0}]},"finish_reason":null}]}`,
		`{"id":"x","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"reasoning_details":[{"type":"reasoning.text","text":"about this.","signature":"sig-arrives-late","id":"t-1","index":0}]},"finish_reason":null}]}`,
		`{"id":"x","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"reasoning_details":[{"type":"reasoning.encrypted","data":"ZW5j","id":"e-1","index":1}]},"finish_reason":null}]}`,
		`{"id":"x","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"content":"done"},"finish_reason":"stop"}]}`,
	})

	final := finalOf(t, responses)
	if len(final.Content.Parts) != 3 {
		t.Fatalf("expected 2 blocks + answer, got %d parts: %#v", len(final.Content.Parts), final.Content.Parts)
	}

	text, ok := reasoningDetailOf(final.Content.Parts[0])
	if !ok {
		t.Fatalf("first part carries no block")
	}
	if got := text["text"]; got != "Let me think about this." {
		t.Errorf("streamed text block = %v, want the pieces concatenated", got)
	}
	// A signature that shows up in a later chunk must be kept, and the null
	// it replaced must not win.
	if got := text["signature"]; got != "sig-arrives-late" {
		t.Errorf("signature = %v, want the late non-null value", got)
	}

	encrypted, ok := reasoningDetailOf(final.Content.Parts[1])
	if !ok {
		t.Fatalf("second part carries no block")
	}
	if encrypted["type"] != "reasoning.encrypted" || encrypted["data"] != "ZW5j" {
		t.Errorf("encrypted block = %#v", encrypted)
	}

	if final.Content.Parts[2].Text != "done" {
		t.Errorf("answer part = %q", final.Content.Parts[2].Text)
	}
}

// The partial responses keep streaming reasoning as it arrives, so a UI can
// render the trace live rather than waiting for the turn to finish.
func TestGenerateStream_ReasoningOnPartials(t *testing.T) {
	responses := streamResponses(t, Config{ReasoningField: "reasoning", SupportsReasoningDetails: true}, []string{
		`{"id":"x","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","reasoning_details":[{"type":"reasoning.text","text":"thinking","index":0}]},"finish_reason":null}]}`,
		`{"id":"x","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"content":"answer"},"finish_reason":"stop"}]}`,
	})

	var sawPartialReasoning bool
	for _, resp := range responses[:len(responses)-1] {
		if !resp.Partial {
			continue
		}
		for _, part := range resp.Content.Parts {
			if part.Thought && part.Text == "thinking" {
				sawPartialReasoning = true
			}
		}
	}
	if !sawPartialReasoning {
		t.Errorf("no partial response carried the streamed reasoning")
	}
}

// A streamed turn whose blocks were captured must replay them on the next
// request, which is the whole point of accumulating them.
func TestGenerateStream_ReasoningDetailsReplayable(t *testing.T) {
	responses := streamResponses(t, Config{ReasoningField: "reasoning", SupportsReasoningDetails: true}, []string{
		`{"id":"x","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","reasoning_details":[{"type":"reasoning.encrypted","data":"ZW5j","id":"e-1","index":0}]},"finish_reason":null}]}`,
		`{"id":"x","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"content":"answer"},"finish_reason":"stop"}]}`,
	})

	final := finalOf(t, responses)

	body := captureBodyWithConfig(t, Config{ReasoningField: "reasoning", SupportsReasoningDetails: true}, &model.LLMRequest{
		Config: &genai.GenerateContentConfig{},
		Contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: "hi"}}},
			final.Content,
		},
	})

	assistant := messageOfRole(t, body, "assistant")
	details, _ := assistant[reasoningDetailsField].([]any)
	if len(details) != 1 {
		t.Fatalf("streamed block was not replayed: %v", assistant)
	}
	block, _ := details[0].(map[string]any)
	if block["data"] != "ZW5j" || block["id"] != "e-1" {
		t.Errorf("replayed block = %#v", block)
	}
}

// Without stream_options.include_usage a provider never sends the terminal
// usage chunk. The reasoning rework must not have disturbed that opt-in.
func TestGenerateStream_StillRequestsUsage(t *testing.T) {
	body := captureStreamBody(t, &model.LLMRequest{
		Config:   &genai.GenerateContentConfig{},
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "hi"}}}},
	})
	opts, ok := body["stream_options"].(map[string]any)
	if !ok || opts["include_usage"] != true {
		t.Errorf("stream_options.include_usage missing: %v", body["stream_options"])
	}
}

// A chunk carrying neither content nor reasoning must not yield a partial
// response with an empty part list.
func TestGenerateStream_SkipsEmptyChunks(t *testing.T) {
	responses := streamResponses(t, Config{}, []string{
		`{"id":"x","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}`,
		`{"id":"x","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"content":"answer"},"finish_reason":"stop"}]}`,
	})

	for _, resp := range responses {
		if resp.Partial && len(resp.Content.Parts) == 0 {
			t.Errorf("a partial response was yielded with no parts")
		}
	}
	final := finalOf(t, responses)
	if len(final.Content.Parts) != 1 || strings.TrimSpace(final.Content.Parts[0].Text) != "answer" {
		t.Errorf("terminal parts = %#v", final.Content.Parts)
	}
}
