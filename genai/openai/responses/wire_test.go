// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package responses

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"
)

// minimalResponseJSON is a complete-enough /v1/responses payload for
// convertResponse to succeed.
const minimalResponseJSON = `{
	"id": "resp_1",
	"status": "completed",
	"output": [{
		"type": "message",
		"id": "msg_1",
		"role": "assistant",
		"status": "completed",
		"content": [{"type": "output_text", "text": "ok"}]
	}],
	"usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
		"input_tokens_details": {"cached_tokens": 0},
		"output_tokens_details": {"reasoning_tokens": 0}}
}`

// captureBody points a Model at a local fake endpoint via BaseURL, fires one
// non-streaming request, and returns the JSON body the openai-go SDK actually
// put on the wire. Asserting on this (not on the pre-SDK params) is what
// proves the bytes a real Responses API server receives are correct.
func captureBody(t *testing.T, req *model.LLMRequest) map[string]any {
	return captureBodyFor(t, Config{}, req)
}

// captureBodyFor mirrors captureBody but lets the test override Config
// fields (the integration tier needs a real model name for its paid step);
// BaseURL always points at the local fake server.
func captureBodyFor(t *testing.T, cfg Config, req *model.LLMRequest) map[string]any {
	t.Helper()

	var captured []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, minimalResponseJSON)
	}))
	defer srv.Close()

	cfg.BaseURL = srv.URL
	if cfg.APIKey == "" {
		cfg.APIKey = "test-key"
	}
	if cfg.ModelName == "" {
		cfg.ModelName = "gpt-test"
	}
	m := New(cfg)

	for _, err := range m.GenerateContent(context.Background(), req, false) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
	}
	if len(captured) == 0 {
		t.Fatalf("server captured no request body")
	}
	var body map[string]any
	if err := json.Unmarshal(captured, &body); err != nil {
		t.Fatalf("unmarshal captured body: %v", err)
	}
	return body
}

// A replayed tool round-trip must reach the wire with store=false, the
// encrypted-reasoning include, and "{}" (never "null") for nil tool
// payloads: strict OpenAI-compatible parsers reject "null" there.
func TestWireBody_StatelessToolReplay(t *testing.T) {
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: "ping the tool"}}},
			{Role: "model", Parts: []*genai.Part{
				{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "ping"}},
			}},
			{Role: "user", Parts: []*genai.Part{
				{FunctionResponse: &genai.FunctionResponse{ID: "call_1", Name: "ping"}},
			}},
		},
	}

	body := captureBody(t, req)

	if body["model"] != "gpt-test" {
		t.Errorf("model = %v, want gpt-test", body["model"])
	}
	if body["store"] != false {
		t.Errorf("store = %v, want false", body["store"])
	}
	includes, _ := body["include"].([]any)
	found := false
	for _, inc := range includes {
		if inc == "reasoning.encrypted_content" {
			found = true
		}
	}
	if !found {
		t.Errorf("include = %v, want reasoning.encrypted_content", includes)
	}

	input, _ := body["input"].([]any)
	if len(input) != 3 {
		t.Fatalf("input items = %d, want 3", len(input))
	}
	call, _ := input[1].(map[string]any)
	if call["type"] != "function_call" || call["arguments"] != "{}" {
		t.Errorf("function_call item = %v, want arguments {}", call)
	}
	output, _ := input[2].(map[string]any)
	if output["type"] != "function_call_output" || output["output"] != "{}" {
		t.Errorf("function_call_output item = %v, want output {}", output)
	}
}

// Streaming requests must carry stream=true on the wire, and the terminal
// response.completed event must come back as the final aggregated turn.
func TestWireBody_StreamFlag(t *testing.T) {
	// SSE data fields are single-line, so the response JSON is compacted
	// before being embedded in the terminal event.
	var compact bytes.Buffer
	if err := json.Compact(&compact, []byte(minimalResponseJSON)); err != nil {
		t.Fatalf("compact: %v", err)
	}

	var captured []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "event: response.output_text.delta\n")
		io.WriteString(w, `data: {"type":"response.output_text.delta","delta":"ok","sequence_number":1}`+"\n\n")
		io.WriteString(w, "event: response.completed\n")
		io.WriteString(w, `data: {"type":"response.completed","sequence_number":2,"response":`+compact.String()+"}\n\n")
	}))
	defer srv.Close()

	m := New(Config{BaseURL: srv.URL, APIKey: "test-key", ModelName: "gpt-test"})

	var final *model.LLMResponse
	req := &model.LLMRequest{Contents: []*genai.Content{
		{Role: "user", Parts: []*genai.Part{{Text: "hi"}}},
	}}
	for resp, err := range m.GenerateContent(context.Background(), req, true) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
		if !resp.Partial {
			final = resp
		}
	}

	var body map[string]any
	if err := json.Unmarshal(captured, &body); err != nil {
		t.Fatalf("unmarshal captured body: %v", err)
	}
	if body["stream"] != true {
		t.Errorf("stream = %v, want true", body["stream"])
	}
	if final == nil {
		t.Fatalf("no final (non-partial) response was yielded")
	}
	if got := extractText(final.Content); got != "ok" {
		t.Errorf("final text = %q, want ok", got)
	}
}

// countingTransport records how many requests pass through so tests can
// prove an injected client actually carries the traffic.
type countingTransport struct {
	base http.RoundTripper
	hits int
}

func (c *countingTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	c.hits++
	return c.base.RoundTrip(r)
}

// A custom HTTP client injected through HTTPOptions must carry the request:
// consumers rely on it for OAuth transports, proxies, and test servers.
func TestCustomHTTPClientIsUsed(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, minimalResponseJSON)
	}))
	defer srv.Close()

	transport := &countingTransport{base: http.DefaultTransport}
	m := New(Config{
		BaseURL:     srv.URL,
		APIKey:      "test-key",
		ModelName:   "gpt-test",
		HTTPOptions: HTTPOptions{Client: &http.Client{Transport: transport}},
	})

	req := &model.LLMRequest{Contents: []*genai.Content{
		{Role: "user", Parts: []*genai.Part{{Text: "hi"}}},
	}}
	for _, err := range m.GenerateContent(context.Background(), req, false) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
	}
	if transport.hits != 1 {
		t.Errorf("injected transport hits = %d, want 1", transport.hits)
	}
}

// A terminal event whose payload carries malformed function arguments must
// surface as an error, not degrade into the delta fallback: gateways that
// omit output_item.done would otherwise silently drop the failure.
func TestWireBody_StreamMalformedTerminalArguments(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "event: response.completed\n")
		io.WriteString(w, `data: {"type":"response.completed","sequence_number":1,"response":{"id":"resp_1","status":"completed","output":[{"type":"function_call","id":"fc_1","call_id":"call_1","name":"delete_things","arguments":"{"}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2,"input_tokens_details":{"cached_tokens":0},"output_tokens_details":{"reasoning_tokens":0}}}}`+"\n\n")
	}))
	defer srv.Close()

	m := New(Config{BaseURL: srv.URL, APIKey: "test-key", ModelName: "gpt-test"})

	req := &model.LLMRequest{Contents: []*genai.Content{
		{Role: "user", Parts: []*genai.Part{{Text: "hi"}}},
	}}
	var gotErr error
	for _, err := range m.GenerateContent(context.Background(), req, true) {
		if err != nil {
			gotErr = err
		}
	}
	if gotErr == nil {
		t.Fatalf("expected an error for malformed terminal arguments, got none")
	}
}

// A gateway that streams function_call items with object-form arguments must
// deliver those arguments in the final response; reading only the string
// union arm would yield the call with an empty argument map.
func TestWireBody_StreamObjectArguments(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "event: response.output_item.done\n")
		io.WriteString(w, `data: {"type":"response.output_item.done","sequence_number":1,"output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"search","arguments":{"q":"weather"}}}`+"\n\n")
		io.WriteString(w, "event: response.completed\n")
		io.WriteString(w, `data: {"type":"response.completed","sequence_number":2,"response":{"id":"resp_1","status":"completed","output":[{"type":"function_call","id":"fc_1","call_id":"call_1","name":"search","arguments":{"q":"weather"}}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2,"input_tokens_details":{"cached_tokens":0},"output_tokens_details":{"reasoning_tokens":0}}}}`+"\n\n")
	}))
	defer srv.Close()

	m := New(Config{BaseURL: srv.URL, APIKey: "test-key", ModelName: "gpt-test"})

	req := &model.LLMRequest{Contents: []*genai.Content{
		{Role: "user", Parts: []*genai.Part{{Text: "hi"}}},
	}}
	var final *model.LLMResponse
	for resp, err := range m.GenerateContent(context.Background(), req, true) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
		if resp != nil && !resp.Partial {
			final = resp
		}
	}
	if final == nil {
		t.Fatalf("no final response yielded")
	}
	var fc *genai.FunctionCall
	for _, p := range final.Content.Parts {
		if p.FunctionCall != nil {
			fc = p.FunctionCall
		}
	}
	if fc == nil {
		t.Fatalf("no FunctionCall in final response: %#v", final.Content)
	}
	if want := map[string]any{"q": "weather"}; !reflect.DeepEqual(fc.Args, want) {
		t.Errorf("Args = %#v, want %#v", fc.Args, want)
	}
}

// A terminal event whose output holds only unknown item types, with nothing
// accumulated from the stream, must surface an error: yielding it as a
// completed turn would persist an empty event with no trace of what was
// dropped.
func TestWireBody_StreamUnknownOnlyTerminal(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "event: response.completed\n")
		io.WriteString(w, `data: {"type":"response.completed","sequence_number":1,"response":{"id":"resp_1","status":"completed","output":[{"type":"web_search_call","id":"ws_1","status":"completed"}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2,"input_tokens_details":{"cached_tokens":0},"output_tokens_details":{"reasoning_tokens":0}}}}`+"\n\n")
	}))
	defer srv.Close()

	m := New(Config{BaseURL: srv.URL, APIKey: "test-key", ModelName: "gpt-test"})

	req := &model.LLMRequest{Contents: []*genai.Content{
		{Role: "user", Parts: []*genai.Part{{Text: "hi"}}},
	}}
	var gotErr error
	for resp, err := range m.GenerateContent(context.Background(), req, true) {
		if err != nil {
			gotErr = err
			continue
		}
		if resp != nil && !resp.Partial && len(resp.Content.Parts) == 0 {
			t.Fatalf("an empty completed turn was yielded: %+v", resp)
		}
	}
	if !errors.Is(gotErr, ErrNoConsumableOutput) {
		t.Fatalf("err = %v, want ErrNoConsumableOutput", gotErr)
	}
}

// When a gateway sends complete output_item.done events but a terminal event
// without aggregated output, the fallback must rebuild the turn from those
// items: message IDs, phase, and encrypted reasoning survive, instead of
// degrading to bare delta text.
func TestWireBody_StreamFallbackPrefersDoneItems(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "event: response.output_text.delta\n")
		io.WriteString(w, `data: {"type":"response.output_text.delta","delta":"hello","sequence_number":1}`+"\n\n")
		io.WriteString(w, "event: response.output_item.done\n")
		io.WriteString(w, `data: {"type":"response.output_item.done","sequence_number":2,"item":{"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"enc-blob"}}`+"\n\n")
		io.WriteString(w, "event: response.output_item.done\n")
		io.WriteString(w, `data: {"type":"response.output_item.done","sequence_number":3,"item":{"type":"message","id":"msg_1","role":"assistant","status":"completed","phase":"final_answer","content":[{"type":"output_text","text":"hello"}]}}`+"\n\n")
		io.WriteString(w, "event: response.completed\n")
		io.WriteString(w, `data: {"type":"response.completed","sequence_number":4,"response":{"id":"resp_1","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2,"input_tokens_details":{"cached_tokens":0},"output_tokens_details":{"reasoning_tokens":0}}}}`+"\n\n")
	}))
	defer srv.Close()

	m := New(Config{BaseURL: srv.URL, APIKey: "test-key", ModelName: "gpt-test"})

	var final *model.LLMResponse
	req := &model.LLMRequest{Contents: []*genai.Content{
		{Role: "user", Parts: []*genai.Part{{Text: "hi"}}},
	}}
	for resp, err := range m.GenerateContent(context.Background(), req, true) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
		if !resp.Partial {
			final = resp
		}
	}
	if final == nil {
		t.Fatalf("no final response was yielded")
	}
	if len(final.Content.Parts) != 2 {
		t.Fatalf("expected 2 parts (reasoning + message), got %d: %+v", len(final.Content.Parts), final.Content.Parts)
	}
	thought := final.Content.Parts[0]
	if !thought.Thought || thought.PartMetadata["encrypted_content"] != "enc-blob" {
		t.Errorf("first part = %+v, want a thought carrying encrypted content", thought)
	}
	msg := final.Content.Parts[1]
	if msg.Text != "hello" || msg.PartMetadata["message_id"] != "msg_1" || msg.PartMetadata["phase"] != "final_answer" {
		t.Errorf("second part = %+v, want text with message_id and phase", msg)
	}
}

// closeTrackingBody flags when the response body is closed so tests can
// prove the stream does not leak the connection.
type closeTrackingBody struct {
	io.Reader
	closed *bool
}

func (b closeTrackingBody) Close() error {
	*b.closed = true
	return nil
}

type sseTransport struct {
	payload string
	closed  *bool
}

func (s *sseTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	return &http.Response{
		StatusCode: 200,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
		Body:       closeTrackingBody{Reader: bytes.NewReader([]byte(s.payload)), closed: s.closed},
		Request:    r,
	}, nil
}

// The stream returns early on terminal events, so the response body must be
// closed explicitly; leaving it open leaks the connection.
func TestStreamBodyClosed(t *testing.T) {
	var compact bytes.Buffer
	if err := json.Compact(&compact, []byte(minimalResponseJSON)); err != nil {
		t.Fatalf("compact: %v", err)
	}
	closed := false
	payload := "event: response.completed\n" +
		`data: {"type":"response.completed","sequence_number":1,"response":` + compact.String() + "}\n\n"
	m := New(Config{
		APIKey:      "test-key",
		ModelName:   "gpt-test",
		BaseURL:     "http://fake.local",
		HTTPOptions: HTTPOptions{Client: &http.Client{Transport: &sseTransport{payload: payload, closed: &closed}}},
	})

	req := &model.LLMRequest{Contents: []*genai.Content{
		{Role: "user", Parts: []*genai.Part{{Text: "hi"}}},
	}}
	for _, err := range m.GenerateContent(context.Background(), req, true) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
	}
	if !closed {
		t.Fatalf("stream body was not closed")
	}
}

// Done items that cover only part of the turn (e.g. reasoning) must not
// discard text that streamed as bare deltas: the consumer already saw it,
// so the final event and the next turn's history must keep it.
func TestWireBody_StreamFallbackMergesUncoveredDeltas(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "event: response.output_text.delta\n")
		io.WriteString(w, `data: {"type":"response.output_text.delta","delta":"hello","sequence_number":1}`+"\n\n")
		io.WriteString(w, "event: response.output_item.done\n")
		io.WriteString(w, `data: {"type":"response.output_item.done","sequence_number":2,"item":{"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"enc-blob"}}`+"\n\n")
		io.WriteString(w, "event: response.completed\n")
		io.WriteString(w, `data: {"type":"response.completed","sequence_number":3,"response":{"id":"resp_1","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2,"input_tokens_details":{"cached_tokens":0},"output_tokens_details":{"reasoning_tokens":0}}}}`+"\n\n")
	}))
	defer srv.Close()

	m := New(Config{BaseURL: srv.URL, APIKey: "test-key", ModelName: "gpt-test"})

	var final *model.LLMResponse
	req := &model.LLMRequest{Contents: []*genai.Content{
		{Role: "user", Parts: []*genai.Part{{Text: "hi"}}},
	}}
	for resp, err := range m.GenerateContent(context.Background(), req, true) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
		if !resp.Partial {
			final = resp
		}
	}
	if final == nil {
		t.Fatalf("no final response was yielded")
	}
	if len(final.Content.Parts) != 2 {
		t.Fatalf("expected 2 parts (reasoning + merged text), got %d: %+v", len(final.Content.Parts), final.Content.Parts)
	}
	if !final.Content.Parts[0].Thought || final.Content.Parts[0].PartMetadata["encrypted_content"] != "enc-blob" {
		t.Errorf("first part = %+v, want the reasoning item", final.Content.Parts[0])
	}
	if final.Content.Parts[1].Text != "hello" {
		t.Errorf("second part = %+v, want the merged delta text", final.Content.Parts[1])
	}
}

// Refusal deltas must be accumulated: without a terminal output or done
// item, the fallback would otherwise persist an empty response.
func TestWireBody_StreamRefusalDelta(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "event: response.refusal.delta\n")
		io.WriteString(w, `data: {"type":"response.refusal.delta","delta":"I cannot help.","item_id":"msg_1","content_index":0,"output_index":0,"sequence_number":1}`+"\n\n")
		io.WriteString(w, "event: response.completed\n")
		io.WriteString(w, `data: {"type":"response.completed","sequence_number":2,"response":{"id":"resp_1","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2,"input_tokens_details":{"cached_tokens":0},"output_tokens_details":{"reasoning_tokens":0}}}}`+"\n\n")
	}))
	defer srv.Close()

	m := New(Config{BaseURL: srv.URL, APIKey: "test-key", ModelName: "gpt-test"})

	var final *model.LLMResponse
	req := &model.LLMRequest{Contents: []*genai.Content{
		{Role: "user", Parts: []*genai.Part{{Text: "hi"}}},
	}}
	for resp, err := range m.GenerateContent(context.Background(), req, true) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
		if !resp.Partial {
			final = resp
		}
	}
	if final == nil {
		t.Fatalf("no final response was yielded")
	}
	if len(final.Content.Parts) != 1 {
		t.Fatalf("expected 1 part, got %d: %+v", len(final.Content.Parts), final.Content.Parts)
	}
	part := final.Content.Parts[0]
	if part.Text != "I cannot help." || part.PartMetadata["refusal"] != true {
		t.Errorf("part = %+v, want refusal text with refusal metadata", part)
	}
}

// A turn with two messages where only the first produced a done event must
// keep both: the covered message replays from its item (ID and phase
// intact), and the second message's delta text is merged in rather than
// masked by the first item's text.
func TestWireBody_StreamFallbackKeepsUncoveredMessage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "event: response.output_text.delta\n")
		io.WriteString(w, `data: {"type":"response.output_text.delta","delta":"let me look","item_id":"msg_1","output_index":0,"content_index":0,"sequence_number":1}`+"\n\n")
		io.WriteString(w, "event: response.output_item.done\n")
		io.WriteString(w, `data: {"type":"response.output_item.done","sequence_number":2,"output_index":0,"item":{"type":"message","id":"msg_1","role":"assistant","status":"completed","phase":"commentary","content":[{"type":"output_text","text":"let me look"}]}}`+"\n\n")
		io.WriteString(w, "event: response.output_text.delta\n")
		io.WriteString(w, `data: {"type":"response.output_text.delta","delta":"the answer is 4","item_id":"msg_2","output_index":1,"content_index":0,"sequence_number":3}`+"\n\n")
		io.WriteString(w, "event: response.completed\n")
		io.WriteString(w, `data: {"type":"response.completed","sequence_number":4,"response":{"id":"resp_1","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2,"input_tokens_details":{"cached_tokens":0},"output_tokens_details":{"reasoning_tokens":0}}}}`+"\n\n")
	}))
	defer srv.Close()

	m := New(Config{BaseURL: srv.URL, APIKey: "test-key", ModelName: "gpt-test"})

	var final *model.LLMResponse
	req := &model.LLMRequest{Contents: []*genai.Content{
		{Role: "user", Parts: []*genai.Part{{Text: "hi"}}},
	}}
	for resp, err := range m.GenerateContent(context.Background(), req, true) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
		if !resp.Partial {
			final = resp
		}
	}
	if final == nil {
		t.Fatalf("no final response was yielded")
	}
	if len(final.Content.Parts) != 2 {
		t.Fatalf("expected 2 parts (done message + uncovered text), got %d: %+v", len(final.Content.Parts), final.Content.Parts)
	}
	first := final.Content.Parts[0]
	if first.Text != "let me look" || first.PartMetadata["message_id"] != "msg_1" || first.PartMetadata["phase"] != "commentary" {
		t.Errorf("first part = %+v, want the done message with identity", first)
	}
	if second := final.Content.Parts[1]; second.Text != "the answer is 4" {
		t.Errorf("second part = %+v, want the uncovered final answer", second)
	}
}

// The reverse gap: the first message streamed only as deltas while a later
// message completed. The synthetic item must sort by output_index so the
// final order matches the stream, not msg_2 before msg_1.
func TestWireBody_StreamFallbackOrdersUncoveredMessage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "event: response.output_text.delta\n")
		io.WriteString(w, `data: {"type":"response.output_text.delta","delta":"let me look","item_id":"msg_1","output_index":0,"content_index":0,"sequence_number":1}`+"\n\n")
		io.WriteString(w, "event: response.output_item.done\n")
		io.WriteString(w, `data: {"type":"response.output_item.done","sequence_number":2,"output_index":1,"item":{"type":"message","id":"msg_2","role":"assistant","status":"completed","phase":"final_answer","content":[{"type":"output_text","text":"the answer is 4"}]}}`+"\n\n")
		io.WriteString(w, "event: response.completed\n")
		io.WriteString(w, `data: {"type":"response.completed","sequence_number":3,"response":{"id":"resp_1","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2,"input_tokens_details":{"cached_tokens":0},"output_tokens_details":{"reasoning_tokens":0}}}}`+"\n\n")
	}))
	defer srv.Close()

	m := New(Config{BaseURL: srv.URL, APIKey: "test-key", ModelName: "gpt-test"})

	var final *model.LLMResponse
	req := &model.LLMRequest{Contents: []*genai.Content{
		{Role: "user", Parts: []*genai.Part{{Text: "hi"}}},
	}}
	for resp, err := range m.GenerateContent(context.Background(), req, true) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
		if !resp.Partial {
			final = resp
		}
	}
	if final == nil {
		t.Fatalf("no final response was yielded")
	}
	if len(final.Content.Parts) != 2 {
		t.Fatalf("expected 2 parts, got %d: %+v", len(final.Content.Parts), final.Content.Parts)
	}
	first := final.Content.Parts[0]
	if first.Text != "let me look" || first.PartMetadata["message_id"] != "msg_1" {
		t.Errorf("first part = %+v, want the uncovered message with its ID", first)
	}
	second := final.Content.Parts[1]
	if second.Text != "the answer is 4" || second.PartMetadata["phase"] != "final_answer" {
		t.Errorf("second part = %+v, want the done message", second)
	}
}

// Base64 documents must reach the wire with both file_data and a filename:
// the API identifies the document type by the extension.
func TestWireBody_InlineDocumentCarriesFilename(t *testing.T) {
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{
				{Text: "summarize this"},
				{InlineData: &genai.Blob{MIMEType: "application/pdf", Data: []byte("fake-pdf")}},
			}},
		},
	}

	body := captureBody(t, req)

	input, _ := body["input"].([]any)
	if len(input) != 1 {
		t.Fatalf("input items = %d, want 1", len(input))
	}
	msg, _ := input[0].(map[string]any)
	contents, _ := msg["content"].([]any)
	if len(contents) != 2 {
		t.Fatalf("content parts = %d, want 2", len(contents))
	}
	file, _ := contents[1].(map[string]any)
	if file["type"] != "input_file" {
		t.Fatalf("second part = %v, want input_file", file)
	}
	if file["filename"] != "input.pdf" {
		t.Errorf("filename = %v, want input.pdf", file["filename"])
	}
	if data, _ := file["file_data"].(string); data == "" {
		t.Errorf("file_data missing from the wire body")
	}
}

// Reasoning that streamed only as summary deltas must be prepended when the
// done items carry no reasoning item of their own.
func TestWireBody_StreamFallbackPrependsDeltaReasoning(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "event: response.reasoning_summary_text.delta\n")
		io.WriteString(w, `data: {"type":"response.reasoning_summary_text.delta","delta":"thinking...","sequence_number":1}`+"\n\n")
		io.WriteString(w, "event: response.output_item.done\n")
		io.WriteString(w, `data: {"type":"response.output_item.done","sequence_number":2,"output_index":1,"item":{"type":"message","id":"msg_1","role":"assistant","status":"completed","phase":"final_answer","content":[{"type":"output_text","text":"done"}]}}`+"\n\n")
		io.WriteString(w, "event: response.completed\n")
		io.WriteString(w, `data: {"type":"response.completed","sequence_number":3,"response":{"id":"resp_1","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2,"input_tokens_details":{"cached_tokens":0},"output_tokens_details":{"reasoning_tokens":0}}}}`+"\n\n")
	}))
	defer srv.Close()

	m := New(Config{BaseURL: srv.URL, APIKey: "test-key", ModelName: "gpt-test"})

	var final *model.LLMResponse
	req := &model.LLMRequest{Contents: []*genai.Content{
		{Role: "user", Parts: []*genai.Part{{Text: "hi"}}},
	}}
	for resp, err := range m.GenerateContent(context.Background(), req, true) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
		if !resp.Partial {
			final = resp
		}
	}
	if final == nil {
		t.Fatalf("no final response was yielded")
	}
	if len(final.Content.Parts) != 2 {
		t.Fatalf("expected 2 parts (thought + message), got %d: %+v", len(final.Content.Parts), final.Content.Parts)
	}
	if p := final.Content.Parts[0]; !p.Thought || p.Text != "thinking..." {
		t.Errorf("first part = %+v, want the prepended delta reasoning", p)
	}
	if p := final.Content.Parts[1]; p.Text != "done" {
		t.Errorf("second part = %+v, want the done message", p)
	}
}

// response.failed and bare error events must surface as errors, not as an
// empty or truncated turn.
func TestWireBody_StreamFailureEvents(t *testing.T) {
	cases := []struct {
		name    string
		payload string
	}{
		{
			"response.failed",
			"event: response.failed\n" +
				`data: {"type":"response.failed","sequence_number":1,"response":{"id":"resp_1","status":"failed","error":{"code":"server_error","message":"backend exploded"},"output":[],"usage":{"input_tokens":0,"output_tokens":0,"total_tokens":0,"input_tokens_details":{"cached_tokens":0},"output_tokens_details":{"reasoning_tokens":0}}}}` + "\n\n",
		},
		{
			"error event",
			"event: error\n" +
				`data: {"type":"error","sequence_number":1,"code":"rate_limited","message":"slow down"}` + "\n\n",
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				io.ReadAll(r.Body)
				w.Header().Set("Content-Type", "text/event-stream")
				io.WriteString(w, c.payload)
			}))
			defer srv.Close()

			m := New(Config{BaseURL: srv.URL, APIKey: "test-key", ModelName: "gpt-test"})

			req := &model.LLMRequest{Contents: []*genai.Content{
				{Role: "user", Parts: []*genai.Part{{Text: "hi"}}},
			}}
			var gotErr error
			for _, err := range m.GenerateContent(context.Background(), req, true) {
				if err != nil {
					gotErr = err
				}
			}
			if gotErr == nil {
				t.Fatalf("expected an error, got none")
			}
		})
	}
}

// A history that ends mid-tool-call (the user cancelled the run) must not
// replay the unpaired function_call: the API rejects a call whose output
// never arrived. The reasoning item that led to the dropped call has no
// follower left and must go too.
func TestWireBody_DanglingCallDropped(t *testing.T) {
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: "do the thing"}}},
			{Role: "model", Parts: []*genai.Part{
				{
					Text:    "planning",
					Thought: true,
					PartMetadata: map[string]any{
						"reasoning_id":      "rs_1",
						"encrypted_content": "enc-blob",
						"reasoning_origin":  "wrong-origin",
					},
				},
				{FunctionCall: &genai.FunctionCall{ID: "call_cancelled", Name: "slow_tool"}},
			}},
			{Role: "user", Parts: []*genai.Part{{Text: "never mind, tell me a joke"}}},
		},
	}

	body := captureBody(t, req)

	input, _ := body["input"].([]any)
	if len(input) != 2 {
		t.Fatalf("input items = %d, want 2 (both user messages only): %v", len(input), input)
	}
	for i, raw := range input {
		item, _ := raw.(map[string]any)
		if item["type"] == "function_call" || item["type"] == "reasoning" {
			t.Errorf("input[%d] = %v, want the dangling call and its reasoning dropped", i, item)
		}
	}
}

// The reverse orphan: an output whose call was lost to compaction must be
// dropped too.
func TestWireBody_OrphanOutputDropped(t *testing.T) {
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{
				{FunctionResponse: &genai.FunctionResponse{ID: "call_lost", Name: "slow_tool"}},
				{Text: "carry on"},
			}},
		},
	}

	body := captureBody(t, req)

	input, _ := body["input"].([]any)
	if len(input) != 1 {
		t.Fatalf("input items = %d, want 1 (the text only): %v", len(input), input)
	}
	item, _ := input[0].(map[string]any)
	if item["type"] == "function_call_output" {
		t.Errorf("input[0] = %v, want the orphan output dropped", item)
	}
}

// Some gateways close the stream after the last delta without sending any
// terminal event. The final turn must still be synthesized from the
// accumulated deltas, or ADK raises "last event is not final" and the
// assistant turn is lost from history.
func TestWireBody_StreamNoTerminalEvent(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "event: response.output_text.delta\n")
		io.WriteString(w, `data: {"type":"response.output_text.delta","delta":"hel","sequence_number":1}`+"\n\n")
		io.WriteString(w, "event: response.output_text.delta\n")
		io.WriteString(w, `data: {"type":"response.output_text.delta","delta":"lo","sequence_number":2}`+"\n\n")
	}))
	defer srv.Close()

	m := New(Config{BaseURL: srv.URL, APIKey: "test-key", ModelName: "gpt-test"})

	var final *model.LLMResponse
	req := &model.LLMRequest{Contents: []*genai.Content{
		{Role: "user", Parts: []*genai.Part{{Text: "hi"}}},
	}}
	for resp, err := range m.GenerateContent(context.Background(), req, true) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
		if !resp.Partial {
			final = resp
		}
	}
	if final == nil {
		t.Fatalf("no final (non-partial) response was synthesized")
	}
	if got := extractText(final.Content); got != "hello" {
		t.Errorf("final text = %q, want the accumulated deltas %q", got, "hello")
	}
}

// Sampling settings must land on the wire: a config dropped between the ADK
// request and the SDK params would be invisible to callers until the model
// behaves differently. Values are float32-exact so the assertion is not at
// the mercy of float32-to-float64 widening.
func TestWireBody_GenerationConfig(t *testing.T) {
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: "hi"}}},
		},
		Config: &genai.GenerateContentConfig{
			Temperature:     genai.Ptr[float32](0.5),
			TopP:            genai.Ptr[float32](0.25),
			MaxOutputTokens: 128,
		},
	}

	body := captureBody(t, req)

	if body["temperature"] != 0.5 {
		t.Errorf("temperature = %v, want 0.5", body["temperature"])
	}
	if body["top_p"] != 0.25 {
		t.Errorf("top_p = %v, want 0.25", body["top_p"])
	}
	if body["max_output_tokens"] != float64(128) {
		t.Errorf("max_output_tokens = %v, want 128", body["max_output_tokens"])
	}
}

// A multi-name allowlist must reach the wire as a filtered tool list plus
// tool_choice required: "required" alone leaves every declared tool
// callable, which defeats the allowlist.
func TestWireBody_MultiNameAllowlistFiltersTools(t *testing.T) {
	tool := func(name string) *genai.Tool {
		return &genai.Tool{FunctionDeclarations: []*genai.FunctionDeclaration{{
			Name:                 name,
			ParametersJsonSchema: map[string]any{"type": "object"},
		}}}
	}
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: "hi"}}},
		},
		Config: &genai.GenerateContentConfig{
			Tools: []*genai.Tool{tool("search"), tool("delete"), tool("rename")},
			ToolConfig: &genai.ToolConfig{FunctionCallingConfig: &genai.FunctionCallingConfig{
				Mode:                 genai.FunctionCallingConfigModeAny,
				AllowedFunctionNames: []string{"search", "rename"},
			}},
		},
	}

	body := captureBody(t, req)

	if body["tool_choice"] != "required" {
		t.Errorf("tool_choice = %v, want required", body["tool_choice"])
	}
	tools, _ := body["tools"].([]any)
	if len(tools) != 2 {
		t.Fatalf("tools on the wire = %d, want the 2 allowed: %v", len(tools), tools)
	}
	names := map[string]bool{}
	for _, raw := range tools {
		toolMap, _ := raw.(map[string]any)
		names[toolMap["name"].(string)] = true
	}
	if !names["search"] || !names["rename"] || names["delete"] {
		t.Errorf("tool names on the wire = %v, want search and rename only", names)
	}
}

// Assistant history carrying a phase but no message ID must replay as a
// plain input message with the phase kept: OutputMessage requires an id,
// so building one with an empty id is a request the API rejects.
func TestWireBody_PhaseOnlyReplaysAsInputMessage(t *testing.T) {
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: "hi"}}},
			{Role: "model", Parts: []*genai.Part{{
				Text:         "thinking out loud",
				PartMetadata: map[string]any{"phase": "commentary"},
			}}},
		},
	}

	body := captureBody(t, req)

	input, _ := body["input"].([]any)
	if len(input) != 2 {
		t.Fatalf("input items = %d, want 2: %v", len(input), input)
	}
	msg, _ := input[1].(map[string]any)
	if msg["type"] != "message" || msg["role"] != "assistant" {
		t.Fatalf("input[1] = %v, want an assistant message", msg)
	}
	if _, hasID := msg["id"]; hasID {
		t.Errorf("input[1] = %v, want no id field (OutputMessage requires a real one)", msg)
	}
	if _, isString := msg["content"].(string); !isString {
		t.Errorf("content = %v, want the plain input message string form", msg["content"])
	}
	if msg["phase"] != "commentary" {
		t.Errorf("phase = %v, want commentary kept on the input message", msg["phase"])
	}
}

// A stream that answers 200 and closes without delivering any event must
// surface an error instead of ending as a silent zero-event iteration:
// overloaded gateways answer exactly this shape.
func TestWireBody_StreamEmptyIsAnError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
	}))
	defer srv.Close()

	m := New(Config{BaseURL: srv.URL, APIKey: "test-key", ModelName: "gpt-test"})

	req := &model.LLMRequest{Contents: []*genai.Content{
		{Role: "user", Parts: []*genai.Part{{Text: "hi"}}},
	}}
	var streamErr error
	var responseCount int
	for resp, err := range m.GenerateContent(context.Background(), req, true) {
		if err != nil {
			streamErr = err
			break
		}
		if resp != nil {
			responseCount++
		}
	}
	if responseCount != 0 {
		t.Errorf("responses yielded = %d, want none from an empty stream", responseCount)
	}
	if !errors.Is(streamErr, ErrNoConsumableOutput) {
		t.Errorf("err = %v, want ErrNoConsumableOutput", streamErr)
	}
}
