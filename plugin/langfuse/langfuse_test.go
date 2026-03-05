// Copyright 2025 achetronic
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package langfuse

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"strings"
	"sync"
	"testing"

	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	oteltrace "go.opentelemetry.io/otel/trace"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

type mockState struct{ data map[string]any }

func newMockState() *mockState { return &mockState{data: make(map[string]any)} }

func (s *mockState) Get(key string) (any, error) {
	v, ok := s.data[key]
	if !ok {
		return nil, fmt.Errorf("key not found: %s", key)
	}
	return v, nil
}
func (s *mockState) Set(key string, val any) error { s.data[key] = val; return nil }
func (s *mockState) All() iter.Seq2[string, any] {
	return func(yield func(string, any) bool) {
		for k, v := range s.data {
			if !yield(k, v) {
				return
			}
		}
	}
}

type mockArtifacts struct{}

func (a *mockArtifacts) Save(_ context.Context, _ string, _ *genai.Part) (*artifact.SaveResponse, error) {
	return nil, nil
}
func (a *mockArtifacts) List(_ context.Context) (*artifact.ListResponse, error) { return nil, nil }
func (a *mockArtifacts) Load(_ context.Context, _ string) (*artifact.LoadResponse, error) {
	return nil, nil
}
func (a *mockArtifacts) LoadVersion(_ context.Context, _ string, _ int) (*artifact.LoadResponse, error) {
	return nil, nil
}

type mockCallbackContext struct {
	context.Context
	agentName    string
	invocationID string
	branch       string
	userID       string
	sessionID    string
	userContent  *genai.Content
	state        session.State
}

func newMockCtx() *mockCallbackContext {
	return &mockCallbackContext{
		Context:      context.Background(),
		agentName:    "test-agent",
		invocationID: "inv-1",
		userID:       "user-1",
		sessionID:    "session-1",
		state:        newMockState(),
	}
}

func (m *mockCallbackContext) UserContent() *genai.Content          { return m.userContent }
func (m *mockCallbackContext) InvocationID() string                 { return m.invocationID }
func (m *mockCallbackContext) AgentName() string                    { return m.agentName }
func (m *mockCallbackContext) ReadonlyState() session.ReadonlyState { return m.state }
func (m *mockCallbackContext) UserID() string                       { return m.userID }
func (m *mockCallbackContext) AppName() string                      { return "test-app" }
func (m *mockCallbackContext) SessionID() string                    { return m.sessionID }
func (m *mockCallbackContext) Branch() string                       { return m.branch }
func (m *mockCallbackContext) Artifacts() agent.Artifacts           { return &mockArtifacts{} }
func (m *mockCallbackContext) State() session.State                 { return m.state }

func mockCtxWithSpan(tp oteltrace.TracerProvider, name string) (*mockCallbackContext, oteltrace.Span) {
	tracer := tp.Tracer("test")
	ctx, span := tracer.Start(context.Background(), name)
	m := newMockCtx()
	m.Context = ctx
	return m, span
}

// recordingExporter collects exported spans for assertions.
type recordingExporter struct {
	mu    sync.Mutex
	spans []sdktrace.ReadOnlySpan
}

func (e *recordingExporter) ExportSpans(_ context.Context, spans []sdktrace.ReadOnlySpan) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.spans = append(e.spans, spans...)
	return nil
}
func (e *recordingExporter) Shutdown(_ context.Context) error { return nil }

func (e *recordingExporter) Spans() []sdktrace.ReadOnlySpan {
	e.mu.Lock()
	defer e.mu.Unlock()
	cp := make([]sdktrace.ReadOnlySpan, len(e.spans))
	copy(cp, e.spans)
	return cp
}

func newTestTP(exp sdktrace.SpanExporter) *sdktrace.TracerProvider {
	return sdktrace.NewTracerProvider(sdktrace.WithSyncer(exp))
}

func spanAttr(s sdktrace.ReadOnlySpan, key string) (attribute.Value, bool) {
	for _, a := range s.Attributes() {
		if string(a.Key) == key {
			return a.Value, true
		}
	}
	return attribute.Value{}, false
}

func assertAttr(t *testing.T, s sdktrace.ReadOnlySpan, key, want string) {
	t.Helper()
	v, ok := spanAttr(s, key)
	if !ok {
		t.Errorf("missing attribute %q on span %q", key, s.Name())
		return
	}
	if got := v.AsString(); got != want {
		t.Errorf("attr %q = %q, want %q", key, got, want)
	}
}

// ---------------------------------------------------------------------------
// Config tests
// ---------------------------------------------------------------------------

func TestConfig_IsEnabled(t *testing.T) {
	tests := []struct {
		name string
		cfg  *Config
		want bool
	}{
		{"nil config", nil, false},
		{"empty keys", &Config{}, false},
		{"only public", &Config{PublicKey: "pk"}, false},
		{"only secret", &Config{SecretKey: "sk"}, false},
		{"both keys", &Config{PublicKey: "pk", SecretKey: "sk"}, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.cfg.IsEnabled(); got != tt.want {
				t.Errorf("IsEnabled() = %v, want %v", got, tt.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Context helpers tests
// ---------------------------------------------------------------------------

func TestContextHelpers(t *testing.T) {
	ctx := context.Background()

	if v := UserIDFromContext(ctx); v != "" {
		t.Errorf("empty ctx should return empty UserID, got %q", v)
	}
	ctx2 := WithUserID(ctx, "alice")
	if v := UserIDFromContext(ctx2); v != "alice" {
		t.Errorf("expected alice, got %q", v)
	}
	if v := UserIDFromContext(ctx); v != "" {
		t.Error("original ctx must be unmodified")
	}

	if v := TagsFromContext(ctx); v != nil {
		t.Errorf("empty ctx should return nil tags, got %v", v)
	}
	ctx3 := WithTags(ctx, []string{"a", "b"})
	tags := TagsFromContext(ctx3)
	if len(tags) != 2 || tags[0] != "a" || tags[1] != "b" {
		t.Errorf("unexpected tags: %v", tags)
	}

	if v := TraceMetadataFromContext(ctx); v != nil {
		t.Errorf("empty ctx should return nil metadata, got %v", v)
	}
	ctx4 := WithTraceMetadata(ctx, map[string]string{"k": "v"})
	md := TraceMetadataFromContext(ctx4)
	if md["k"] != "v" {
		t.Errorf("unexpected metadata: %v", md)
	}

	if v := EnvironmentFromContext(ctx); v != "" {
		t.Errorf("empty ctx should return empty environment, got %q", v)
	}
	ctx5 := WithEnvironment(ctx, "prod")
	if v := EnvironmentFromContext(ctx5); v != "prod" {
		t.Errorf("expected prod, got %q", v)
	}

	if v := ReleaseFromContext(ctx); v != "" {
		t.Errorf("empty ctx should return empty release, got %q", v)
	}
	ctx6 := WithRelease(ctx, "v1.0")
	if v := ReleaseFromContext(ctx6); v != "v1.0" {
		t.Errorf("expected v1.0, got %q", v)
	}

	if v := TraceNameFromContext(ctx); v != "" {
		t.Errorf("empty ctx should return empty trace name, got %q", v)
	}
	ctx7 := WithTraceName(ctx, "my-trace")
	if v := TraceNameFromContext(ctx7); v != "my-trace" {
		t.Errorf("expected my-trace, got %q", v)
	}
}

func TestContextHelpers_Concurrent(t *testing.T) {
	base := context.Background()
	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			ctx := WithUserID(base, fmt.Sprintf("user-%d", id))
			ctx = WithTags(ctx, []string{fmt.Sprintf("tag-%d", id)})
			ctx = WithEnvironment(ctx, fmt.Sprintf("env-%d", id))

			if got := UserIDFromContext(ctx); got != fmt.Sprintf("user-%d", id) {
				t.Errorf("goroutine %d: expected user-%d, got %s", id, id, got)
			}
			tags := TagsFromContext(ctx)
			if len(tags) != 1 || tags[0] != fmt.Sprintf("tag-%d", id) {
				t.Errorf("goroutine %d: unexpected tags %v", id, tags)
			}
			if got := EnvironmentFromContext(ctx); got != fmt.Sprintf("env-%d", id) {
				t.Errorf("goroutine %d: expected env-%d, got %s", id, id, got)
			}
		}(i)
	}
	wg.Wait()
}

// ---------------------------------------------------------------------------
// Helper function tests
// ---------------------------------------------------------------------------

func TestContentToText(t *testing.T) {
	tests := []struct {
		name string
		c    *genai.Content
		want string
	}{
		{"nil", nil, ""},
		{"empty parts", &genai.Content{}, ""},
		{"single text", &genai.Content{Parts: []*genai.Part{{Text: "hello"}}}, "hello"},
		{"multi text", &genai.Content{Parts: []*genai.Part{{Text: "a"}, {Text: "b"}}}, "a\nb"},
		{"function call", &genai.Content{Parts: []*genai.Part{{
			FunctionCall: &genai.FunctionCall{Name: "search", Args: map[string]any{"q": "foo"}},
		}}}, `[tool_call: search({"q":"foo"})]`},
		{"function response", &genai.Content{Parts: []*genai.Part{{
			FunctionResponse: &genai.FunctionResponse{Name: "search", Response: map[string]any{"ok": true}},
		}}}, `[tool_response: search → {"ok":true}]`},
		{"mixed", &genai.Content{Parts: []*genai.Part{
			{Text: "hi"},
			{FunctionCall: &genai.FunctionCall{Name: "f", Args: map[string]any{}}},
		}}, "hi\n[tool_call: f({})]"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := contentToText(tt.c); got != tt.want {
				t.Errorf("contentToText() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestHasFunctionCalls(t *testing.T) {
	if hasFunctionCalls(nil) {
		t.Error("nil should not have function calls")
	}
	if hasFunctionCalls(&genai.Content{Parts: []*genai.Part{{Text: "hi"}}}) {
		t.Error("text-only should not have function calls")
	}
	if !hasFunctionCalls(&genai.Content{Parts: []*genai.Part{{
		FunctionCall: &genai.FunctionCall{Name: "f"},
	}}}) {
		t.Error("should detect function call")
	}
}

func TestMarshalLLMRequest(t *testing.T) {
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: "hello"}}},
			{Role: "model", Parts: []*genai.Part{{
				FunctionCall: &genai.FunctionCall{Name: "search", Args: map[string]any{"q": "test"}},
			}}},
			{Role: "user", Parts: []*genai.Part{{
				FunctionResponse: &genai.FunctionResponse{Name: "search", Response: map[string]any{"r": "ok"}},
			}}},
			nil,
		},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: &genai.Content{Parts: []*genai.Part{{Text: "You are helpful."}}},
		},
	}

	result := marshalLLMRequest(req)
	msgs, ok := result["messages"].([]map[string]any)
	if !ok {
		t.Fatal("messages should be a slice of maps")
	}
	if len(msgs) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(msgs))
	}
	if msgs[0]["role"] != "user" || msgs[0]["content"] != "hello" {
		t.Errorf("msg[0] unexpected: %v", msgs[0])
	}
	if msgs[1]["role"] != "model" {
		t.Errorf("msg[1] role: %v", msgs[1]["role"])
	}
	tc, ok := msgs[1]["tool_call"].(map[string]any)
	if !ok || tc["name"] != "search" {
		t.Errorf("msg[1] tool_call unexpected: %v", msgs[1])
	}
	if msgs[2]["role"] != "tool" {
		t.Errorf("msg[2] role: %v", msgs[2]["role"])
	}
	if result["system"] != "You are helpful." {
		t.Errorf("system: %v", result["system"])
	}
}

func TestMarshalLLMRequest_NoSystem(t *testing.T) {
	req := &model.LLMRequest{
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "hi"}}}},
	}
	result := marshalLLMRequest(req)
	if _, ok := result["system"]; ok {
		t.Error("should not have system key when no system instruction")
	}
}

// ---------------------------------------------------------------------------
// branchKey tests
// ---------------------------------------------------------------------------

func TestBranchKey(t *testing.T) {
	ctx := newMockCtx()
	ctx.invocationID = "inv-42"
	ctx.branch = ""
	if got := branchKey(ctx); got != "inv-42" {
		t.Errorf("no branch: expected inv-42, got %q", got)
	}
	ctx.branch = "parallel-0"
	if got := branchKey(ctx); got != "inv-42:parallel-0" {
		t.Errorf("with branch: expected inv-42:parallel-0, got %q", got)
	}
}

// ---------------------------------------------------------------------------
// spanEnricher: agent span stack management
// ---------------------------------------------------------------------------

func TestSpanEnricher_BeforeAfterAgent_StackManagement(t *testing.T) {
	rec := &recordingExporter{}
	tp := newTestTP(rec)
	e := newSpanEnricher()

	ctx1, span1 := mockCtxWithSpan(tp, "agent1")
	ctx1.invocationID = "inv-1"
	e.beforeAgent(ctx1)

	e.mu.Lock()
	if len(e.agentSpans["inv-1"]) != 1 {
		t.Fatalf("expected 1 span on stack, got %d", len(e.agentSpans["inv-1"]))
	}
	e.mu.Unlock()

	ctx2, span2 := mockCtxWithSpan(tp, "agent2")
	ctx2.invocationID = "inv-1"
	e.beforeAgent(ctx2)

	e.mu.Lock()
	if len(e.agentSpans["inv-1"]) != 2 {
		t.Fatalf("expected 2 spans, got %d", len(e.agentSpans["inv-1"]))
	}
	e.mu.Unlock()

	e.afterAgent(ctx2)
	e.mu.Lock()
	if len(e.agentSpans["inv-1"]) != 1 {
		t.Fatalf("after pop: expected 1, got %d", len(e.agentSpans["inv-1"]))
	}
	e.mu.Unlock()

	e.afterAgent(ctx1)
	e.mu.Lock()
	if _, ok := e.agentSpans["inv-1"]; ok {
		t.Error("stack should be deleted after last pop")
	}
	e.mu.Unlock()

	span1.End()
	span2.End()
}

func TestSpanEnricher_ParallelBranches_Isolation(t *testing.T) {
	rec := &recordingExporter{}
	tp := newTestTP(rec)
	e := newSpanEnricher()

	ctxA, spanA := mockCtxWithSpan(tp, "branchA")
	ctxA.invocationID = "inv-1"
	ctxA.branch = "branch-a"

	ctxB, spanB := mockCtxWithSpan(tp, "branchB")
	ctxB.invocationID = "inv-1"
	ctxB.branch = "branch-b"

	e.beforeAgent(ctxA)
	e.beforeAgent(ctxB)

	e.mu.Lock()
	if len(e.agentSpans) != 2 {
		t.Fatalf("expected 2 branch keys, got %d", len(e.agentSpans))
	}
	e.mu.Unlock()

	e.afterAgent(ctxA)
	e.afterAgent(ctxB)

	e.mu.Lock()
	if len(e.agentSpans) != 0 {
		t.Error("all branches should be cleaned up")
	}
	e.mu.Unlock()

	spanA.End()
	spanB.End()
}

// ---------------------------------------------------------------------------
// spanEnricher: Langfuse attributes
// ---------------------------------------------------------------------------

func TestSpanEnricher_BeforeAgent_SetsAllAttributes(t *testing.T) {
	rec := &recordingExporter{}
	tp := newTestTP(rec)
	e := newSpanEnricher()

	ctx, span := mockCtxWithSpan(tp, "invoke_agent")
	ctx.userID = "adk-user"
	ctx.sessionID = "sess-42"
	ctx.Context = WithUserID(ctx.Context, "langfuse-user")
	ctx.Context = WithTags(ctx.Context, []string{"tag1", "tag2"})
	ctx.Context = WithTraceMetadata(ctx.Context, map[string]string{"team": "sre"})
	ctx.Context = WithEnvironment(ctx.Context, "staging")
	ctx.Context = WithRelease(ctx.Context, "v2.0")
	ctx.Context = WithTraceName(ctx.Context, "my-trace")
	ctx.userContent = &genai.Content{Parts: []*genai.Part{{Text: "hello agent"}}}

	e.beforeAgent(ctx)
	span.End()

	spans := rec.Spans()
	if len(spans) != 1 {
		t.Fatalf("expected 1 span, got %d", len(spans))
	}
	s := spans[0]

	assertAttr(t, s, "langfuse.user.id", "langfuse-user")
	assertAttr(t, s, "langfuse.session.id", "sess-42")
	assertAttr(t, s, "langfuse.environment", "staging")
	assertAttr(t, s, "langfuse.release", "v2.0")
	assertAttr(t, s, "langfuse.trace.name", "my-trace")
	assertAttr(t, s, "langfuse.trace.metadata.team", "sre")

	v, ok := spanAttr(s, "langfuse.trace.tags")
	if !ok {
		t.Fatal("missing langfuse.trace.tags")
	}
	if sl := v.AsStringSlice(); len(sl) != 2 || sl[0] != "tag1" {
		t.Errorf("unexpected tags: %v", sl)
	}
	if _, ok := spanAttr(s, "langfuse.trace.input"); !ok {
		t.Error("missing langfuse.trace.input")
	}

	e.afterAgent(ctx)
}

func TestSpanEnricher_BeforeAgent_FallbackToADKUserID(t *testing.T) {
	rec := &recordingExporter{}
	tp := newTestTP(rec)
	e := newSpanEnricher()

	ctx, span := mockCtxWithSpan(tp, "invoke_agent")
	ctx.userID = "adk-user-id"

	e.beforeAgent(ctx)
	span.End()

	assertAttr(t, rec.Spans()[0], "langfuse.user.id", "adk-user-id")
	e.afterAgent(ctx)
}

// ---------------------------------------------------------------------------
// spanEnricher: model callback + pending queue
// ---------------------------------------------------------------------------

func TestSpanEnricher_BeforeAfterModel_PendingQueue(t *testing.T) {
	rec := &recordingExporter{}
	tp := newTestTP(rec)
	e := newSpanEnricher()

	ctx, span := mockCtxWithSpan(tp, "invoke_agent")
	spanID := span.SpanContext().SpanID().String()

	req := &model.LLMRequest{
		Model:    "claude-sonnet-4-6",
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "hello"}}}},
	}
	e.beforeModel(ctx, req)

	e.mu.Lock()
	if len(e.pending[spanID]) != 1 {
		t.Fatalf("expected 1 pending, got %d", len(e.pending[spanID]))
	}
	if e.pending[spanID][0].model != "claude-sonnet-4-6" {
		t.Errorf("model: %q", e.pending[spanID][0].model)
	}
	e.mu.Unlock()

	resp := &model.LLMResponse{
		Content: &genai.Content{Role: "model", Parts: []*genai.Part{{Text: "hi there"}}},
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount: 100, CandidatesTokenCount: 20, TotalTokenCount: 120,
		},
	}
	e.afterModel(ctx, resp, nil)

	e.mu.Lock()
	call := e.pending[spanID][0]
	e.mu.Unlock()

	if call.response != "hi there" {
		t.Errorf("response: %q", call.response)
	}
	if call.inputTokens != 100 || call.outputTokens != 20 || call.totalTokens != 120 {
		t.Errorf("tokens: in=%d out=%d total=%d", call.inputTokens, call.outputTokens, call.totalTokens)
	}
	span.End()
}

func TestSpanEnricher_AfterModel_CapturesError(t *testing.T) {
	rec := &recordingExporter{}
	tp := newTestTP(rec)
	e := newSpanEnricher()

	ctx, span := mockCtxWithSpan(tp, "invoke_agent")
	spanID := span.SpanContext().SpanID().String()

	e.beforeModel(ctx, &model.LLMRequest{
		Model:    "gpt-4o",
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "fail"}}}},
	})
	e.afterModel(ctx, nil, errors.New("rate limit exceeded"))

	e.mu.Lock()
	if e.pending[spanID][0].response != "rate limit exceeded" {
		t.Errorf("error response: %q", e.pending[spanID][0].response)
	}
	e.mu.Unlock()
	span.End()
}

func TestSpanEnricher_AfterModel_PartialSkipsOutputPropagation(t *testing.T) {
	rec := &recordingExporter{}
	tp := newTestTP(rec)
	e := newSpanEnricher()

	ctx, span := mockCtxWithSpan(tp, "invoke_agent")
	e.beforeAgent(ctx)
	e.beforeModel(ctx, &model.LLMRequest{
		Model:    "m",
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "q"}}}},
	})
	e.afterModel(ctx, &model.LLMResponse{
		Partial: true,
		Content: &genai.Content{Parts: []*genai.Part{{Text: "partial"}}},
	}, nil)

	span.End()
	for _, s := range rec.Spans() {
		if _, ok := spanAttr(s, "langfuse.trace.output"); ok {
			t.Error("partial response should NOT set trace output")
		}
	}
	e.afterAgent(ctx)
}

func TestSpanEnricher_AfterModel_FunctionCallSkipsOutputPropagation(t *testing.T) {
	rec := &recordingExporter{}
	tp := newTestTP(rec)
	e := newSpanEnricher()

	ctx, span := mockCtxWithSpan(tp, "invoke_agent")
	e.beforeAgent(ctx)
	e.beforeModel(ctx, &model.LLMRequest{
		Model:    "m",
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "search"}}}},
	})
	e.afterModel(ctx, &model.LLMResponse{
		Content: &genai.Content{Parts: []*genai.Part{
			{Text: "let me search"},
			{FunctionCall: &genai.FunctionCall{Name: "search", Args: map[string]any{}}},
		}},
	}, nil)

	span.End()
	for _, s := range rec.Spans() {
		if _, ok := spanAttr(s, "langfuse.trace.output"); ok {
			t.Error("function call should NOT set trace output")
		}
	}
	e.afterAgent(ctx)
}

func TestSpanEnricher_AfterModel_FinalTextSetsOutput(t *testing.T) {
	rec := &recordingExporter{}
	tp := newTestTP(rec)
	e := newSpanEnricher()

	ctx, span := mockCtxWithSpan(tp, "invoke_agent")
	e.beforeAgent(ctx)
	e.beforeModel(ctx, &model.LLMRequest{
		Model:    "m",
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "hello"}}}},
	})
	e.afterModel(ctx, &model.LLMResponse{
		Content: &genai.Content{Parts: []*genai.Part{{Text: "final answer"}}},
	}, nil)

	span.End()
	found := false
	for _, s := range rec.Spans() {
		if v, ok := spanAttr(s, "langfuse.trace.output"); ok {
			found = true
			if !strings.Contains(v.AsString(), "final answer") {
				t.Errorf("unexpected output: %s", v.AsString())
			}
		}
	}
	if !found {
		t.Error("expected langfuse.trace.output on agent span")
	}
	e.afterAgent(ctx)
}

// ---------------------------------------------------------------------------
// popCall tests
// ---------------------------------------------------------------------------

func TestPopCall_FIFO(t *testing.T) {
	e := newSpanEnricher()
	e.pending["span-1"] = []llmCall{
		{request: "req1", model: "m1"},
		{request: "req2", model: "m2"},
		{request: "req3", model: "m3"},
	}

	for i, want := range []string{"req1", "req2", "req3"} {
		call, ok := e.popCall("span-1")
		if !ok || call.request != want {
			t.Errorf("pop %d: ok=%v request=%q", i, ok, call.request)
		}
	}
	if _, ok := e.popCall("span-1"); ok {
		t.Error("should return false after queue empty")
	}
	e.mu.Lock()
	if _, exists := e.pending["span-1"]; exists {
		t.Error("key should be deleted after last pop")
	}
	e.mu.Unlock()
}

func TestPopCall_Nonexistent(t *testing.T) {
	e := newSpanEnricher()
	if _, ok := e.popCall("nope"); ok {
		t.Error("should return false")
	}
}

// ---------------------------------------------------------------------------
// enrichingExporter: end-to-end with real spans
// ---------------------------------------------------------------------------

func TestEnrichingExporter_EnrichesGenerateContent(t *testing.T) {
	rec := &recordingExporter{}
	e := newSpanEnricher()
	ex := &enrichingExporter{inner: rec, enricher: e}
	tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(ex))
	tracer := tp.Tracer("test")

	invokeCtx, invokeSpan := tracer.Start(context.Background(), "invoke_agent")
	parentID := invokeSpan.SpanContext().SpanID().String()

	e.pending[parentID] = []llmCall{{
		request:      `{"messages":[{"role":"user","content":"hi"}]}`,
		response:     "hello!",
		model:        "claude-sonnet-4-6",
		inputTokens:  50,
		outputTokens: 10,
	}}

	_, genSpan := tracer.Start(invokeCtx, "generate_content")
	genSpan.End()
	invokeSpan.End()
	tp.Shutdown(context.Background())

	var genS sdktrace.ReadOnlySpan
	for _, s := range rec.Spans() {
		if strings.HasPrefix(s.Name(), "generate_content") {
			genS = s
		}
	}
	if genS == nil {
		t.Fatal("no generate_content span found")
	}

	assertAttr(t, genS, "gcp.vertex.agent.llm_request", `{"messages":[{"role":"user","content":"hi"}]}`)
	assertAttr(t, genS, "gcp.vertex.agent.llm_response", "hello!")
	assertAttr(t, genS, "gen_ai.request.model", "claude-sonnet-4-6")

	v, ok := spanAttr(genS, "gen_ai.usage.input_tokens")
	if !ok || v.AsInt64() != 50 {
		t.Errorf("input_tokens: ok=%v val=%v", ok, v)
	}
	v, ok = spanAttr(genS, "gen_ai.usage.output_tokens")
	if !ok || v.AsInt64() != 10 {
		t.Errorf("output_tokens: ok=%v val=%v", ok, v)
	}
}

func TestEnrichingExporter_PassesThroughNonGenerateContent(t *testing.T) {
	rec := &recordingExporter{}
	e := newSpanEnricher()
	ex := &enrichingExporter{inner: rec, enricher: e}
	tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(ex))
	tracer := tp.Tracer("test")

	_, span := tracer.Start(context.Background(), "invoke_agent")
	span.SetAttributes(attribute.String("existing", "value"))
	span.End()
	tp.Shutdown(context.Background())

	spans := rec.Spans()
	if len(spans) != 1 {
		t.Fatalf("expected 1 span, got %d", len(spans))
	}
	if _, ok := spanAttr(spans[0], "gcp.vertex.agent.llm_request"); ok {
		t.Error("non-generate_content should not be enriched")
	}
	assertAttr(t, spans[0], "existing", "value")
}

func TestEnrichingExporter_NoPendingCall(t *testing.T) {
	rec := &recordingExporter{}
	e := newSpanEnricher()
	ex := &enrichingExporter{inner: rec, enricher: e}
	tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(ex))
	tracer := tp.Tracer("test")

	parentCtx, parentSpan := tracer.Start(context.Background(), "invoke_agent")
	_, genSpan := tracer.Start(parentCtx, "generate_content")
	genSpan.End()
	parentSpan.End()
	tp.Shutdown(context.Background())

	for _, s := range rec.Spans() {
		if strings.HasPrefix(s.Name(), "generate_content") {
			if _, ok := spanAttr(s, "gcp.vertex.agent.llm_request"); ok {
				t.Error("should not enrich when no pending call")
			}
		}
	}
}

func TestEnrichingExporter_ZeroTokensNotSet(t *testing.T) {
	rec := &recordingExporter{}
	e := newSpanEnricher()
	ex := &enrichingExporter{inner: rec, enricher: e}
	tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(ex))
	tracer := tp.Tracer("test")

	parentCtx, parentSpan := tracer.Start(context.Background(), "invoke_agent")
	parentID := parentSpan.SpanContext().SpanID().String()
	e.pending[parentID] = []llmCall{{request: "req", model: "m", inputTokens: 0, outputTokens: 0}}

	_, genSpan := tracer.Start(parentCtx, "generate_content")
	genSpan.End()
	parentSpan.End()
	tp.Shutdown(context.Background())

	for _, s := range rec.Spans() {
		if strings.HasPrefix(s.Name(), "generate_content") {
			if _, ok := spanAttr(s, "gen_ai.usage.input_tokens"); ok {
				t.Error("zero tokens should not be set")
			}
		}
	}
}

func TestEnrichingExporter_MultipleBatch(t *testing.T) {
	rec := &recordingExporter{}
	e := newSpanEnricher()
	ex := &enrichingExporter{inner: rec, enricher: e}
	tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(ex))
	tracer := tp.Tracer("test")

	parent1Ctx, parent1 := tracer.Start(context.Background(), "invoke_agent_1")
	parent2Ctx, parent2 := tracer.Start(context.Background(), "invoke_agent_2")

	e.pending[parent1.SpanContext().SpanID().String()] = []llmCall{{model: "m1", request: "r1", inputTokens: 10, outputTokens: 5}}
	e.pending[parent2.SpanContext().SpanID().String()] = []llmCall{{model: "m2", request: "r2", inputTokens: 20, outputTokens: 15}}

	_, g1 := tracer.Start(parent1Ctx, "generate_content")
	_, g2 := tracer.Start(parent2Ctx, "generate_content")
	g1.End()
	g2.End()
	parent1.End()
	parent2.End()
	tp.Shutdown(context.Background())

	models := map[string]bool{}
	for _, s := range rec.Spans() {
		if v, ok := spanAttr(s, "gen_ai.request.model"); ok {
			models[v.AsString()] = true
		}
	}
	if !models["m1"] || !models["m2"] {
		t.Errorf("expected both m1 and m2, got %v", models)
	}
}

// ---------------------------------------------------------------------------
// enrichedSpan tests
// ---------------------------------------------------------------------------

func TestEnrichedSpan_AttributesMerge(t *testing.T) {
	rec := &recordingExporter{}
	e := newSpanEnricher()
	ex := &enrichingExporter{inner: rec, enricher: e}
	tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(ex))
	tracer := tp.Tracer("test")

	parentCtx, parentSpan := tracer.Start(context.Background(), "invoke_agent")
	parentID := parentSpan.SpanContext().SpanID().String()
	e.pending[parentID] = []llmCall{{request: "r", model: "m", inputTokens: 1, outputTokens: 1}}

	_, genSpan := tracer.Start(parentCtx, "generate_content")
	genSpan.SetAttributes(attribute.String("original", "yes"))
	genSpan.End()
	parentSpan.End()
	tp.Shutdown(context.Background())

	for _, s := range rec.Spans() {
		if strings.HasPrefix(s.Name(), "generate_content") {
			assertAttr(t, s, "original", "yes")
			assertAttr(t, s, "gen_ai.request.model", "m")
		}
	}
}

// ---------------------------------------------------------------------------
// Full flow: callbacks + exporter end-to-end
// ---------------------------------------------------------------------------

func TestFullFlow_SingleTurn(t *testing.T) {
	rec := &recordingExporter{}
	e := newSpanEnricher()
	ex := &enrichingExporter{inner: rec, enricher: e}
	tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(ex))
	tracer := tp.Tracer("test")

	invokeCtx, invokeSpan := tracer.Start(context.Background(), "invoke_agent")

	mctx := newMockCtx()
	mctx.Context = invokeCtx
	mctx.Context = WithUserID(mctx.Context, "alice")
	mctx.sessionID = "sess-1"
	mctx.userContent = &genai.Content{Parts: []*genai.Part{{Text: "what is k8s?"}}}

	e.beforeAgent(mctx)

	req := &model.LLMRequest{
		Model:    "claude-sonnet-4-6",
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "what is k8s?"}}}},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: &genai.Content{Parts: []*genai.Part{{Text: "You are a k8s expert."}}},
		},
	}
	e.beforeModel(mctx, req)

	resp := &model.LLMResponse{
		Content: &genai.Content{Role: "model", Parts: []*genai.Part{{Text: "Kubernetes is a container orchestration platform."}}},
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount: 150, CandidatesTokenCount: 30, TotalTokenCount: 180,
		},
	}
	e.afterModel(mctx, resp, nil)

	_, genSpan := tracer.Start(invokeCtx, "generate_content [claude-sonnet-4-6]")
	genSpan.End()

	e.afterAgent(mctx)
	invokeSpan.End()
	tp.Shutdown(context.Background())

	var invokeS, genS sdktrace.ReadOnlySpan
	for _, s := range rec.Spans() {
		switch {
		case strings.HasPrefix(s.Name(), "generate_content"):
			genS = s
		case s.Name() == "invoke_agent":
			invokeS = s
		}
	}
	if invokeS == nil || genS == nil {
		t.Fatal("missing expected spans")
	}

	assertAttr(t, invokeS, "langfuse.user.id", "alice")
	assertAttr(t, invokeS, "langfuse.session.id", "sess-1")

	if v, ok := spanAttr(invokeS, "langfuse.trace.output"); !ok {
		t.Error("missing trace output on invoke span")
	} else if !strings.Contains(v.AsString(), "container orchestration") {
		t.Errorf("unexpected output: %s", v.AsString())
	}

	assertAttr(t, genS, "gen_ai.request.model", "claude-sonnet-4-6")
	assertAttr(t, genS, "gcp.vertex.agent.llm_response", "Kubernetes is a container orchestration platform.")

	v, _ := spanAttr(genS, "gen_ai.usage.input_tokens")
	if v.AsInt64() != 150 {
		t.Errorf("input_tokens: %d", v.AsInt64())
	}
	v, _ = spanAttr(genS, "gen_ai.usage.output_tokens")
	if v.AsInt64() != 30 {
		t.Errorf("output_tokens: %d", v.AsInt64())
	}

	var reqData map[string]any
	rv, _ := spanAttr(genS, "gcp.vertex.agent.llm_request")
	json.Unmarshal([]byte(rv.AsString()), &reqData)
	if reqData["system"] != "You are a k8s expert." {
		t.Errorf("system in request: %v", reqData["system"])
	}
}

func TestFullFlow_ToolCallThenFinalAnswer(t *testing.T) {
	rec := &recordingExporter{}
	e := newSpanEnricher()
	ex := &enrichingExporter{inner: rec, enricher: e}
	tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(ex))
	tracer := tp.Tracer("test")

	invokeCtx, invokeSpan := tracer.Start(context.Background(), "invoke_agent")
	mctx := newMockCtx()
	mctx.Context = invokeCtx

	e.beforeAgent(mctx)

	// Turn 1: tool call
	e.beforeModel(mctx, &model.LLMRequest{
		Model:    "gpt-4o",
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "list pods"}}}},
	})
	e.afterModel(mctx, &model.LLMResponse{
		Content: &genai.Content{Parts: []*genai.Part{
			{FunctionCall: &genai.FunctionCall{Name: "list_resources", Args: map[string]any{"kind": "Pod"}}},
		}},
	}, nil)
	_, gen1 := tracer.Start(invokeCtx, "generate_content [gpt-4o]")
	gen1.End()

	// Turn 2: final answer
	e.beforeModel(mctx, &model.LLMRequest{
		Model: "gpt-4o",
		Contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: "list pods"}}},
			{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{Name: "list_resources", Args: map[string]any{"kind": "Pod"}}}}},
			{Role: "user", Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{Name: "list_resources", Response: map[string]any{"pods": 3}}}}},
		},
	})
	e.afterModel(mctx, &model.LLMResponse{
		Content: &genai.Content{Parts: []*genai.Part{{Text: "There are 3 pods running."}}},
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount: 500, CandidatesTokenCount: 25, TotalTokenCount: 525,
		},
	}, nil)
	_, gen2 := tracer.Start(invokeCtx, "generate_content [gpt-4o]")
	gen2.End()

	e.afterAgent(mctx)
	invokeSpan.End()
	tp.Shutdown(context.Background())

	var invokeS sdktrace.ReadOnlySpan
	var genSpans []sdktrace.ReadOnlySpan
	for _, s := range rec.Spans() {
		if s.Name() == "invoke_agent" {
			invokeS = s
		}
		if strings.HasPrefix(s.Name(), "generate_content") {
			genSpans = append(genSpans, s)
		}
	}

	if invokeS == nil {
		t.Fatal("missing invoke_agent")
	}
	if len(genSpans) != 2 {
		t.Fatalf("expected 2 generate_content spans, got %d", len(genSpans))
	}

	if v, ok := spanAttr(invokeS, "langfuse.trace.output"); !ok {
		t.Error("missing trace output")
	} else if !strings.Contains(v.AsString(), "3 pods") {
		t.Errorf("output should mention 3 pods: %s", v.AsString())
	}

	// First gen span: tool call turn
	assertAttr(t, genSpans[0], "gen_ai.request.model", "gpt-4o")

	// Second gen span: final answer with tokens
	assertAttr(t, genSpans[1], "gen_ai.request.model", "gpt-4o")
	v, _ := spanAttr(genSpans[1], "gen_ai.usage.input_tokens")
	if v.AsInt64() != 500 {
		t.Errorf("turn2 input_tokens: %d", v.AsInt64())
	}
}

// ---------------------------------------------------------------------------
// Concurrency
// ---------------------------------------------------------------------------

func TestSpanEnricher_ConcurrentAccess(t *testing.T) {
	e := newSpanEnricher()
	rec := &recordingExporter{}
	tp := newTestTP(rec)

	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			ctx, span := mockCtxWithSpan(tp, fmt.Sprintf("agent-%d", id))
			ctx.invocationID = fmt.Sprintf("inv-%d", id)

			e.beforeAgent(ctx)
			e.beforeModel(ctx, &model.LLMRequest{
				Model:    "m",
				Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: fmt.Sprintf("q%d", id)}}}},
			})
			e.afterModel(ctx, &model.LLMResponse{
				Content: &genai.Content{Parts: []*genai.Part{{Text: fmt.Sprintf("a%d", id)}}},
				UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
					PromptTokenCount: int32(id * 10), CandidatesTokenCount: int32(id), TotalTokenCount: int32(id*10 + id),
				},
			}, nil)

			spanID := span.SpanContext().SpanID().String()
			call, ok := e.popCall(spanID)
			if !ok {
				t.Errorf("goroutine %d: no pending call", id)
				return
			}
			if call.response != fmt.Sprintf("a%d", id) {
				t.Errorf("goroutine %d: response %q", id, call.response)
			}

			e.afterAgent(ctx)
			span.End()
		}(i)
	}
	wg.Wait()

	e.mu.Lock()
	if len(e.agentSpans) != 0 {
		t.Errorf("expected empty agentSpans, got %d", len(e.agentSpans))
	}
	if len(e.pending) != 0 {
		t.Errorf("expected empty pending, got %d", len(e.pending))
	}
	e.mu.Unlock()
}
