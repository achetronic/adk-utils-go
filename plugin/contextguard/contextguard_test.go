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

package contextguard

import (
	"context"
	"fmt"
	"iter"
	"strings"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
)

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

type mockState struct {
	data map[string]any
}

func newMockState() *mockState {
	return &mockState{data: make(map[string]any)}
}

func (s *mockState) Get(key string) (any, error) {
	v, ok := s.data[key]
	if !ok {
		return nil, fmt.Errorf("key not found: %s", key)
	}
	return v, nil
}

func (s *mockState) Set(key string, val any) error {
	s.data[key] = val
	return nil
}

func (s *mockState) All() iter.Seq2[string, any] {
	return func(yield func(string, any) bool) {
		for k, v := range s.data {
			if !yield(k, v) {
				return
			}
		}
	}
}

type mockCallbackContext struct {
	context.Context
	agentName string
	sessionID string
	state     session.State
}

func newMockCallbackContext(agentName string) *mockCallbackContext {
	return &mockCallbackContext{
		Context:   context.Background(),
		agentName: agentName,
		sessionID: "test-session",
		state:     newMockState(),
	}
}

func (m *mockCallbackContext) UserContent() *genai.Content            { return nil }
func (m *mockCallbackContext) InvocationID() string                   { return "inv-1" }
func (m *mockCallbackContext) AgentName() string                      { return m.agentName }
func (m *mockCallbackContext) ReadonlyState() session.ReadonlyState   { return m.state }
func (m *mockCallbackContext) UserID() string                         { return "user-1" }
func (m *mockCallbackContext) AppName() string                        { return "test-app" }
func (m *mockCallbackContext) SessionID() string                      { return m.sessionID }
func (m *mockCallbackContext) Branch() string                         { return "" }
func (m *mockCallbackContext) Artifacts() agent.Artifacts             { return &mockArtifacts{} }
func (m *mockCallbackContext) State() session.State                   { return m.state }

type mockArtifacts struct{}

func (a *mockArtifacts) Save(_ context.Context, _ string, _ *genai.Part) (*artifact.SaveResponse, error) {
	return nil, nil
}
func (a *mockArtifacts) List(_ context.Context) (*artifact.ListResponse, error) {
	return nil, nil
}
func (a *mockArtifacts) Load(_ context.Context, _ string) (*artifact.LoadResponse, error) {
	return nil, nil
}
func (a *mockArtifacts) LoadVersion(_ context.Context, _ string, _ int) (*artifact.LoadResponse, error) {
	return nil, nil
}

type mockLLM struct {
	name     string
	response string
}

func (m *mockLLM) Name() string { return m.name }

func (m *mockLLM) GenerateContent(_ context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	resp := m.response
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{
			Content: &genai.Content{
				Role:  "model",
				Parts: []*genai.Part{{Text: resp}},
			},
		}, nil)
	}
}

type mockRegistry struct {
	contextWindows map[string]int
	maxTokens      map[string]int
}

func newMockRegistry() *mockRegistry {
	return &mockRegistry{
		contextWindows: map[string]int{
			"claude-sonnet-4-5-20250929": 200_000,
			"gpt-4o":                     128_000,
			"small-model":                8_000,
		},
		maxTokens: map[string]int{
			"claude-sonnet-4-5-20250929": 8192,
			"gpt-4o":                     4096,
			"small-model":                1024,
		},
	}
}

func (r *mockRegistry) ContextWindow(modelID string) int {
	if v, ok := r.contextWindows[modelID]; ok {
		return v
	}
	return 128_000
}

func (r *mockRegistry) DefaultMaxTokens(modelID string) int {
	if v, ok := r.maxTokens[modelID]; ok {
		return v
	}
	return 4096
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func textContent(role, text string) *genai.Content {
	return &genai.Content{
		Role:  role,
		Parts: []*genai.Part{{Text: text}},
	}
}

func toolCallContent(name string) *genai.Content {
	return &genai.Content{
		Role: "model",
		Parts: []*genai.Part{{
			FunctionCall: &genai.FunctionCall{Name: name, Args: map[string]any{"q": "test"}},
		}},
	}
}

func toolResultContent(name string) *genai.Content {
	return &genai.Content{
		Role: "user",
		Parts: []*genai.Part{{
			FunctionResponse: &genai.FunctionResponse{Name: name, Response: map[string]any{"result": "ok"}},
		}},
	}
}

func makeConversation(turns int) []*genai.Content {
	contents := make([]*genai.Content, 0, turns*2)
	for i := 0; i < turns; i++ {
		contents = append(contents,
			textContent("user", fmt.Sprintf("User message %d with some padding text to increase token count", i)),
			textContent("model", fmt.Sprintf("Model response %d with more text to simulate real content", i)),
		)
	}
	return contents
}

func makeLargeConversation(approxTokens int) []*genai.Content {
	var contents []*genai.Content
	msgSize := 400
	msgsNeeded := (approxTokens * 4) / msgSize
	for i := 0; i < msgsNeeded/2; i++ {
		contents = append(contents,
			textContent("user", strings.Repeat("x", msgSize)),
			textContent("model", strings.Repeat("y", msgSize)),
		)
	}
	return contents
}

// ---------------------------------------------------------------------------
// Tests: estimateTokens
// ---------------------------------------------------------------------------

func TestEstimateTokens_EmptyRequest(t *testing.T) {
	req := &model.LLMRequest{}
	if got := estimateTokens(req); got != 0 {
		t.Errorf("estimateTokens(empty) = %d, want 0", got)
	}
}

func TestEstimateTokens_TextOnly(t *testing.T) {
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			textContent("user", strings.Repeat("a", 400)),
		},
	}
	got := estimateTokens(req)
	if got != 100 {
		t.Errorf("estimateTokens(400 chars) = %d, want 100", got)
	}
}

func TestEstimateTokens_WithSystemInstruction(t *testing.T) {
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			textContent("user", strings.Repeat("a", 400)),
		},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: &genai.Content{
				Parts: []*genai.Part{{Text: strings.Repeat("s", 200)}},
			},
		},
	}
	got := estimateTokens(req)
	want := 100 + 50
	if got != want {
		t.Errorf("estimateTokens(text+system) = %d, want %d", got, want)
	}
}

func TestEstimateTokens_WithFunctionCallAndResponse(t *testing.T) {
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			toolCallContent("search"),
			toolResultContent("search"),
		},
	}
	got := estimateTokens(req)
	if got <= 0 {
		t.Errorf("estimateTokens(tool call+result) = %d, want > 0", got)
	}
}

func TestEstimateTokens_NilContents(t *testing.T) {
	req := &model.LLMRequest{
		Contents: []*genai.Content{nil, textContent("user", "hello"), nil},
	}
	got := estimateTokens(req)
	if got != len("hello")/4 {
		t.Errorf("estimateTokens(with nils) = %d, want %d", got, len("hello")/4)
	}
}

// ---------------------------------------------------------------------------
// Tests: estimateContentTokens
// ---------------------------------------------------------------------------

func TestEstimateContentTokens(t *testing.T) {
	contents := []*genai.Content{
		textContent("user", strings.Repeat("a", 800)),
		textContent("model", strings.Repeat("b", 400)),
		nil,
	}
	got := estimateContentTokens(contents)
	want := 300
	if got != want {
		t.Errorf("estimateContentTokens = %d, want %d", got, want)
	}
}

// ---------------------------------------------------------------------------
// Tests: computeBuffer
// ---------------------------------------------------------------------------

func TestComputeBuffer(t *testing.T) {
	tests := []struct {
		name          string
		contextWindow int
		want          int
	}{
		{"large window (250k)", 250_000, 20_000},
		{"exactly at threshold (200k)", 200_000, 40_000},
		{"small window (50k)", 50_000, 10_000},
		{"tiny window (10k)", 10_000, 2_000},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := computeBuffer(tt.contextWindow)
			if got != tt.want {
				t.Errorf("computeBuffer(%d) = %d, want %d", tt.contextWindow, got, tt.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Tests: contentHasFunctionCall / contentHasFunctionResponse
// ---------------------------------------------------------------------------

func TestContentHasFunctionCall(t *testing.T) {
	if !contentHasFunctionCall(toolCallContent("test")) {
		t.Error("expected true for tool call content")
	}
	if contentHasFunctionCall(textContent("model", "hello")) {
		t.Error("expected false for text content")
	}
}

func TestContentHasFunctionResponse(t *testing.T) {
	if !contentHasFunctionResponse(toolResultContent("test")) {
		t.Error("expected true for tool result content")
	}
	if contentHasFunctionResponse(textContent("user", "hello")) {
		t.Error("expected false for text content")
	}
}

// ---------------------------------------------------------------------------
// Tests: findSplitIndex / safeSplitIndex
// ---------------------------------------------------------------------------

func TestFindSplitIndex_BasicSplit(t *testing.T) {
	contents := makeConversation(20)
	budget := 1000
	idx := findSplitIndex(contents, budget)
	if idx <= 0 || idx >= len(contents) {
		t.Errorf("findSplitIndex returned %d, expected between 1 and %d", idx, len(contents)-1)
	}
}

func TestFindSplitIndex_SmallConversation(t *testing.T) {
	contents := []*genai.Content{
		textContent("user", "hello"),
		textContent("model", "hi"),
	}
	idx := findSplitIndex(contents, 10000)
	if idx < 0 || idx > len(contents) {
		t.Errorf("findSplitIndex on 2-message conversation returned %d", idx)
	}
}

func TestSafeSplitIndex_SkipsToolChain(t *testing.T) {
	contents := []*genai.Content{
		textContent("user", "hello"),
		textContent("model", "let me search"),
		toolCallContent("search"),
		toolResultContent("search"),
		textContent("model", "here are the results"),
		textContent("user", "thanks"),
	}

	idx := safeSplitIndex(contents, 3)
	if idx > 2 {
		t.Errorf("safeSplitIndex should back up past tool chain, got %d", idx)
	}
}

func TestSafeSplitIndex_BoundsCheck(t *testing.T) {
	contents := makeConversation(5)

	if got := safeSplitIndex(contents, 0); got != 0 {
		t.Errorf("safeSplitIndex(0) = %d, want 0", got)
	}
	if got := safeSplitIndex(contents, len(contents)); got != len(contents) {
		t.Errorf("safeSplitIndex(len) = %d, want %d", got, len(contents))
	}
}

// ---------------------------------------------------------------------------
// Tests: injectSummary
// ---------------------------------------------------------------------------

func TestInjectSummary_AddsSummary(t *testing.T) {
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			textContent("user", "hello"),
		},
	}
	injectSummary(req, "previous conversation about Go testing")

	if len(req.Contents) != 2 {
		t.Fatalf("expected 2 contents, got %d", len(req.Contents))
	}
	if !strings.HasPrefix(req.Contents[0].Parts[0].Text, "[Previous conversation summary]") {
		t.Error("summary not injected as first content")
	}
	if req.Contents[0].Role != "user" {
		t.Errorf("summary role = %q, want 'user'", req.Contents[0].Role)
	}
}

func TestInjectSummary_NoDuplicate(t *testing.T) {
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			textContent("user", "[Previous conversation summary]\nold summary\n[End of summary — conversation continues below]"),
			textContent("user", "hello"),
		},
	}
	injectSummary(req, "new summary")

	if len(req.Contents) != 2 {
		t.Fatalf("expected no duplicate, got %d contents", len(req.Contents))
	}
}

// ---------------------------------------------------------------------------
// Tests: replaceSummary
// ---------------------------------------------------------------------------

func TestReplaceSummary(t *testing.T) {
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			textContent("user", "old1"),
			textContent("model", "old2"),
			textContent("user", "recent1"),
			textContent("model", "recent2"),
		},
	}
	recent := req.Contents[2:]
	replaceSummary(req, "summary of old messages", recent)

	if len(req.Contents) != 3 {
		t.Fatalf("expected 3 contents (summary + 2 recent), got %d", len(req.Contents))
	}
	if !strings.Contains(req.Contents[0].Parts[0].Text, "summary of old messages") {
		t.Error("summary not in first content")
	}
	if req.Contents[1].Parts[0].Text != "recent1" {
		t.Errorf("recent content[0] = %q, want 'recent1'", req.Contents[1].Parts[0].Text)
	}
}

// ---------------------------------------------------------------------------
// Tests: buildSummarizePrompt
// ---------------------------------------------------------------------------

func TestBuildSummarizePrompt_WithoutPreviousSummary(t *testing.T) {
	contents := []*genai.Content{
		textContent("user", "What is Go?"),
		textContent("model", "Go is a programming language."),
	}
	prompt := buildSummarizePrompt(contents, "", nil)

	if !strings.Contains(prompt, "Provide a detailed summary") {
		t.Error("missing summary instruction")
	}
	if !strings.Contains(prompt, "What is Go?") {
		t.Error("missing user message in transcript")
	}
	if !strings.Contains(prompt, "Go is a programming language.") {
		t.Error("missing model message in transcript")
	}
	if strings.Contains(prompt, "Previous summary") {
		t.Error("should not contain previous summary block")
	}
}

func TestBuildSummarizePrompt_WithPreviousSummary(t *testing.T) {
	contents := []*genai.Content{
		textContent("user", "Tell me more"),
	}
	prompt := buildSummarizePrompt(contents, "Earlier we discussed Go.", nil)

	if !strings.Contains(prompt, "Earlier we discussed Go.") {
		t.Error("missing previous summary")
	}
	if !strings.Contains(prompt, "Incorporate the previous summary") {
		t.Error("missing incorporation instruction")
	}
}

func TestBuildSummarizePrompt_WithToolCalls(t *testing.T) {
	contents := []*genai.Content{
		toolCallContent("search"),
		toolResultContent("search"),
	}
	prompt := buildSummarizePrompt(contents, "", nil)

	if !strings.Contains(prompt, "[called tool: search]") {
		t.Error("missing tool call in transcript")
	}
	if !strings.Contains(prompt, "[tool search returned a result]") {
		t.Error("missing tool result in transcript")
	}
}

func TestBuildSummarizePrompt_NilContents(t *testing.T) {
	contents := []*genai.Content{nil, textContent("user", "hello"), nil}
	prompt := buildSummarizePrompt(contents, "", nil)
	if !strings.Contains(prompt, "hello") {
		t.Error("should include non-nil content")
	}
}

// ---------------------------------------------------------------------------
// Tests: buildFallbackSummary
// ---------------------------------------------------------------------------

func TestBuildFallbackSummary_Basic(t *testing.T) {
	contents := []*genai.Content{
		textContent("user", "hello"),
		textContent("model", "hi there"),
	}
	summary := buildFallbackSummary(contents, "")

	if !strings.Contains(summary, "user: hello") {
		t.Error("missing user message in fallback")
	}
	if !strings.Contains(summary, "model: hi there") {
		t.Error("missing model message in fallback")
	}
}

func TestBuildFallbackSummary_TruncatesLongMessages(t *testing.T) {
	longMsg := strings.Repeat("x", 300)
	contents := []*genai.Content{textContent("user", longMsg)}
	summary := buildFallbackSummary(contents, "")

	if !strings.HasSuffix(strings.TrimSpace(summary), "...") {
		t.Error("should truncate long messages with ...")
	}
	if strings.Contains(summary, longMsg) {
		t.Error("should not contain full long message")
	}
}

func TestBuildFallbackSummary_WithPrevious(t *testing.T) {
	contents := []*genai.Content{textContent("user", "new")}
	summary := buildFallbackSummary(contents, "previous context")

	if !strings.HasPrefix(summary, "previous context") {
		t.Error("should start with previous summary")
	}
	if !strings.Contains(summary, "---") {
		t.Error("should contain separator")
	}
	if !strings.Contains(summary, "user: new") {
		t.Error("should contain new content")
	}
}

func TestBuildFallbackSummary_EmptyRole(t *testing.T) {
	contents := []*genai.Content{
		{Role: "", Parts: []*genai.Part{{Text: "orphan message"}}},
	}
	summary := buildFallbackSummary(contents, "")
	if !strings.Contains(summary, "unknown: orphan message") {
		t.Error("empty role should become 'unknown'")
	}
}

// ---------------------------------------------------------------------------
// Tests: Session state helpers
// ---------------------------------------------------------------------------

func TestLoadSummary_Empty(t *testing.T) {
	ctx := newMockCallbackContext("agent1")
	s := loadSummary(ctx)
	if s != "" {
		t.Errorf("loadSummary on empty state = %q, want empty", s)
	}
}

func TestPersistAndLoadSummary(t *testing.T) {
	ctx := newMockCallbackContext("agent1")
	persistSummary(ctx, "test summary", 5000)

	s := loadSummary(ctx)
	if s != "test summary" {
		t.Errorf("loadSummary = %q, want 'test summary'", s)
	}
}

func TestLoadSummary_AgentNameSuffix(t *testing.T) {
	ctx1 := newMockCallbackContext("agent1")
	ctx2 := &mockCallbackContext{
		Context:   context.Background(),
		agentName: "agent2",
		sessionID: "test-session",
		state:     ctx1.state,
	}

	persistSummary(ctx1, "summary for agent1", 1000)
	persistSummary(ctx2, "summary for agent2", 2000)

	s1 := loadSummary(ctx1)
	s2 := loadSummary(ctx2)

	if s1 != "summary for agent1" {
		t.Errorf("agent1 summary = %q", s1)
	}
	if s2 != "summary for agent2" {
		t.Errorf("agent2 summary = %q", s2)
	}
}

func TestLoadContentsAtCompaction_Empty(t *testing.T) {
	ctx := newMockCallbackContext("agent1")
	if got := loadContentsAtCompaction(ctx); got != 0 {
		t.Errorf("loadContentsAtCompaction on empty = %d, want 0", got)
	}
}

func TestPersistAndLoadContentsAtCompaction(t *testing.T) {
	ctx := newMockCallbackContext("agent1")
	persistContentsAtCompaction(ctx, 42)

	got := loadContentsAtCompaction(ctx)
	if got != 42 {
		t.Errorf("loadContentsAtCompaction = %d, want 42", got)
	}
}

func TestLoadContentsAtCompaction_Float64Conversion(t *testing.T) {
	ctx := newMockCallbackContext("agent1")
	ctx.state.Set(stateKeyPrefixContentsAtCompaction+"agent1", float64(99))

	got := loadContentsAtCompaction(ctx)
	if got != 99 {
		t.Errorf("loadContentsAtCompaction(float64) = %d, want 99", got)
	}
}

// ---------------------------------------------------------------------------
// Tests: ContextGuard builder API
// ---------------------------------------------------------------------------

func TestNew(t *testing.T) {
	registry := newMockRegistry()
	guard := New(registry)

	if guard == nil {
		t.Fatal("New returned nil")
	}
	if guard.registry != registry {
		t.Error("registry not set")
	}
	if guard.strategies == nil {
		t.Error("strategies map not initialized")
	}
}

func TestAdd_DefaultThreshold(t *testing.T) {
	guard := New(newMockRegistry())
	llm := &mockLLM{name: "gpt-4o"}
	guard.Add("assistant", llm)

	s, ok := guard.strategies["assistant"]
	if !ok {
		t.Fatal("strategy not registered for 'assistant'")
	}
	if s.Name() != StrategyThreshold {
		t.Errorf("default strategy = %q, want %q", s.Name(), StrategyThreshold)
	}
}

func TestAdd_WithSlidingWindow(t *testing.T) {
	guard := New(newMockRegistry())
	llm := &mockLLM{name: "gpt-4o"}
	guard.Add("researcher", llm, WithSlidingWindow(30))

	s, ok := guard.strategies["researcher"]
	if !ok {
		t.Fatal("strategy not registered for 'researcher'")
	}
	if s.Name() != StrategySlidingWindow {
		t.Errorf("strategy = %q, want %q", s.Name(), StrategySlidingWindow)
	}
}

func TestAdd_WithSlidingWindowDefault(t *testing.T) {
	guard := New(newMockRegistry())
	llm := &mockLLM{name: "gpt-4o"}
	guard.Add("agent", llm, WithSlidingWindow(0))

	s := guard.strategies["agent"].(*slidingWindowStrategy)
	if s.maxTurns != defaultMaxTurns {
		t.Errorf("maxTurns = %d, want default %d", s.maxTurns, defaultMaxTurns)
	}
}

func TestAdd_WithMaxTokens(t *testing.T) {
	guard := New(newMockRegistry())
	llm := &mockLLM{name: "gpt-4o"}
	guard.Add("assistant", llm, WithMaxTokens(500_000))

	s := guard.strategies["assistant"].(*thresholdStrategy)
	if s.maxTokens != 500_000 {
		t.Errorf("maxTokens = %d, want 500000", s.maxTokens)
	}
}

func TestAdd_MultipleAgents(t *testing.T) {
	guard := New(newMockRegistry())
	llm1 := &mockLLM{name: "gpt-4o"}
	llm2 := &mockLLM{name: "claude-sonnet-4-5-20250929"}

	guard.Add("assistant", llm1)
	guard.Add("researcher", llm2, WithSlidingWindow(50))

	if len(guard.strategies) != 2 {
		t.Fatalf("expected 2 strategies, got %d", len(guard.strategies))
	}
	if guard.strategies["assistant"].Name() != StrategyThreshold {
		t.Error("assistant should use threshold")
	}
	if guard.strategies["researcher"].Name() != StrategySlidingWindow {
		t.Error("researcher should use sliding_window")
	}
}

func TestPluginConfig_ReturnsValid(t *testing.T) {
	guard := New(newMockRegistry())
	guard.Add("assistant", &mockLLM{name: "gpt-4o"})

	cfg := guard.PluginConfig()
	if len(cfg.Plugins) != 1 {
		t.Fatalf("expected 1 plugin, got %d", len(cfg.Plugins))
	}
}

// ---------------------------------------------------------------------------
// Tests: beforeModel callback
// ---------------------------------------------------------------------------

func TestBeforeModel_NilRequest(t *testing.T) {
	g := &contextGuard{strategies: map[string]Strategy{}}
	ctx := newMockCallbackContext("agent1")

	resp, err := g.beforeModel(ctx, nil)
	if resp != nil || err != nil {
		t.Errorf("nil request should return (nil, nil), got (%v, %v)", resp, err)
	}
}

func TestBeforeModel_EmptyContents(t *testing.T) {
	g := &contextGuard{strategies: map[string]Strategy{}}
	ctx := newMockCallbackContext("agent1")
	req := &model.LLMRequest{Contents: []*genai.Content{}}

	resp, err := g.beforeModel(ctx, req)
	if resp != nil || err != nil {
		t.Errorf("empty contents should return (nil, nil), got (%v, %v)", resp, err)
	}
}

func TestBeforeModel_UnknownAgent(t *testing.T) {
	g := &contextGuard{strategies: map[string]Strategy{}}
	ctx := newMockCallbackContext("unknown-agent")
	req := &model.LLMRequest{
		Contents: []*genai.Content{textContent("user", "hello")},
	}

	resp, err := g.beforeModel(ctx, req)
	if resp != nil || err != nil {
		t.Errorf("unknown agent should return (nil, nil), got (%v, %v)", resp, err)
	}
}

// ---------------------------------------------------------------------------
// Tests: Threshold strategy (Compact)
// ---------------------------------------------------------------------------

func TestThresholdStrategy_BelowThreshold(t *testing.T) {
	registry := newMockRegistry()
	llm := &mockLLM{name: "gpt-4o", response: "summary"}
	s := newThresholdStrategy(registry, llm, 0)
	ctx := newMockCallbackContext("agent1")

	req := &model.LLMRequest{
		Model:    "gpt-4o",
		Contents: []*genai.Content{textContent("user", "short message")},
	}

	originalLen := len(req.Contents)
	err := s.Compact(ctx, req)
	if err != nil {
		t.Fatalf("Compact error: %v", err)
	}
	if len(req.Contents) != originalLen {
		t.Error("should not modify contents below threshold")
	}
}

func TestThresholdStrategy_ExceedsThreshold(t *testing.T) {
	registry := &mockRegistry{
		contextWindows: map[string]int{"small-model": 1_000},
		maxTokens:      map[string]int{"small-model": 512},
	}
	llm := &mockLLM{name: "small-model", response: "Summarized conversation"}
	s := newThresholdStrategy(registry, llm, 0)
	ctx := newMockCallbackContext("agent1")

	req := &model.LLMRequest{
		Model:    "small-model",
		Contents: makeLargeConversation(2_000),
	}

	originalLen := len(req.Contents)
	err := s.Compact(ctx, req)
	if err != nil {
		t.Fatalf("Compact error: %v", err)
	}
	if len(req.Contents) >= originalLen {
		t.Error("should have compacted the conversation")
	}
	if len(req.Contents) != 2 {
		t.Errorf("expected 2 contents (summary + continuation), got %d", len(req.Contents))
	}
	if !strings.Contains(req.Contents[0].Parts[0].Text, "Summarized conversation") {
		t.Error("first content should be the summary")
	}
}

func TestThresholdStrategy_WithMaxTokensOverride(t *testing.T) {
	registry := newMockRegistry()
	llm := &mockLLM{name: "gpt-4o", response: "summary"}
	s := newThresholdStrategy(registry, llm, 500)
	ctx := newMockCallbackContext("agent1")

	req := &model.LLMRequest{
		Model:    "gpt-4o",
		Contents: makeLargeConversation(1_000),
	}

	err := s.Compact(ctx, req)
	if err != nil {
		t.Fatalf("Compact error: %v", err)
	}
	if !strings.Contains(req.Contents[0].Parts[0].Text, "summary") {
		t.Error("should compact with manual maxTokens=500")
	}
}

func TestThresholdStrategy_InjectsExistingSummary(t *testing.T) {
	registry := newMockRegistry()
	llm := &mockLLM{name: "gpt-4o", response: "new summary"}
	s := newThresholdStrategy(registry, llm, 0)
	ctx := newMockCallbackContext("agent1")

	persistSummary(ctx, "old summary from last compaction", 5000)

	req := &model.LLMRequest{
		Model:    "gpt-4o",
		Contents: []*genai.Content{textContent("user", "short")},
	}

	err := s.Compact(ctx, req)
	if err != nil {
		t.Fatalf("Compact error: %v", err)
	}

	if len(req.Contents) < 2 {
		t.Fatal("should have injected summary")
	}
	if !strings.Contains(req.Contents[0].Parts[0].Text, "old summary from last compaction") {
		t.Error("should inject existing summary")
	}
}

// ---------------------------------------------------------------------------
// Tests: Sliding window strategy (Compact)
// ---------------------------------------------------------------------------

func TestSlidingWindowStrategy_BelowLimit(t *testing.T) {
	registry := newMockRegistry()
	llm := &mockLLM{name: "gpt-4o", response: "summary"}
	s := newSlidingWindowStrategy(registry, llm, 50)
	ctx := newMockCallbackContext("agent1")

	req := &model.LLMRequest{
		Model:    "gpt-4o",
		Contents: makeConversation(5),
	}

	originalLen := len(req.Contents)
	err := s.Compact(ctx, req)
	if err != nil {
		t.Fatalf("Compact error: %v", err)
	}
	if len(req.Contents) != originalLen {
		t.Error("should not compact below turn limit")
	}
}

func TestSlidingWindowStrategy_ExceedsLimit(t *testing.T) {
	registry := newMockRegistry()
	llm := &mockLLM{name: "gpt-4o", response: "Sliding window summary"}
	s := newSlidingWindowStrategy(registry, llm, 5)
	ctx := newMockCallbackContext("agent1")

	req := &model.LLMRequest{
		Model:    "gpt-4o",
		Contents: makeConversation(20),
	}

	originalLen := len(req.Contents)
	err := s.Compact(ctx, req)
	if err != nil {
		t.Fatalf("Compact error: %v", err)
	}
	if len(req.Contents) >= originalLen {
		t.Error("should have compacted the conversation")
	}
	if !strings.Contains(req.Contents[0].Parts[0].Text, "Sliding window summary") {
		t.Error("first content should be the summary")
	}

	watermark := loadContentsAtCompaction(ctx)
	if watermark != originalLen {
		t.Errorf("watermark = %d, want %d", watermark, originalLen)
	}
}

func TestSlidingWindowStrategy_InjectsExistingSummary(t *testing.T) {
	registry := newMockRegistry()
	llm := &mockLLM{name: "gpt-4o", response: "new summary"}
	s := newSlidingWindowStrategy(registry, llm, 100)
	ctx := newMockCallbackContext("agent1")

	persistSummary(ctx, "previous sliding window summary", 3000)

	req := &model.LLMRequest{
		Model:    "gpt-4o",
		Contents: makeConversation(5),
	}

	err := s.Compact(ctx, req)
	if err != nil {
		t.Fatalf("Compact error: %v", err)
	}
	if !strings.Contains(req.Contents[0].Parts[0].Text, "previous sliding window summary") {
		t.Error("should inject existing summary when below limit")
	}
}

func TestSlidingWindowStrategy_RespectsWatermark(t *testing.T) {
	registry := newMockRegistry()
	llm := &mockLLM{name: "gpt-4o", response: "summary"}
	s := newSlidingWindowStrategy(registry, llm, 10)
	ctx := newMockCallbackContext("agent1")

	persistContentsAtCompaction(ctx, 35)

	req := &model.LLMRequest{
		Model:    "gpt-4o",
		Contents: makeConversation(20),
	}

	originalLen := len(req.Contents)
	err := s.Compact(ctx, req)
	if err != nil {
		t.Fatalf("Compact error: %v", err)
	}
	if len(req.Contents) != originalLen {
		t.Error("should NOT compact: turnsSinceCompaction = 40-35 = 5, below maxTurns=10")
	}
}

// ---------------------------------------------------------------------------
// Tests: CrushRegistry (unit, no network)
// ---------------------------------------------------------------------------

func TestCrushRegistry_DefaultValues(t *testing.T) {
	r := NewCrushRegistry()

	if got := r.ContextWindow("nonexistent-model"); got != crushDefaultCtxWindow {
		t.Errorf("ContextWindow(unknown) = %d, want %d", got, crushDefaultCtxWindow)
	}
	if got := r.DefaultMaxTokens("nonexistent-model"); got != crushDefaultMaxTokens {
		t.Errorf("DefaultMaxTokens(unknown) = %d, want %d", got, crushDefaultMaxTokens)
	}
}

func TestCrushRegistry_StartStop(t *testing.T) {
	r := NewCrushRegistry()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	r.Start(ctx)
	r.Stop()
}

func TestCrushRegistry_StopWithoutStart(t *testing.T) {
	r := NewCrushRegistry()
	r.Stop()
}

// ---------------------------------------------------------------------------
// Tests: AgentOption functions
// ---------------------------------------------------------------------------

func TestWithSlidingWindow_SetsFields(t *testing.T) {
	cfg := &agentConfig{}
	WithSlidingWindow(42)(cfg)
	if cfg.strategy != StrategySlidingWindow {
		t.Errorf("strategy = %q, want %q", cfg.strategy, StrategySlidingWindow)
	}
	if cfg.maxTurns != 42 {
		t.Errorf("maxTurns = %d, want 42", cfg.maxTurns)
	}
}

func TestWithMaxTokens_SetsField(t *testing.T) {
	cfg := &agentConfig{}
	WithMaxTokens(1_000_000)(cfg)
	if cfg.maxTokens != 1_000_000 {
		t.Errorf("maxTokens = %d, want 1000000", cfg.maxTokens)
	}
	if cfg.strategy != "" {
		t.Errorf("strategy should not be set, got %q", cfg.strategy)
	}
}

// ---------------------------------------------------------------------------
// Helper: makeToolOnlyConversation
// ---------------------------------------------------------------------------

func makeToolOnlyConversation(pairs int) []*genai.Content {
	contents := make([]*genai.Content, 0, pairs*2)
	for i := 0; i < pairs; i++ {
		contents = append(contents,
			toolCallContent(fmt.Sprintf("tool_%d", i)),
			toolResultContent(fmt.Sprintf("tool_%d", i)),
		)
	}
	return contents
}

func makeLargeToolResultContent(name string, size int) *genai.Content {
	return &genai.Content{
		Role: "user",
		Parts: []*genai.Part{{
			FunctionResponse: &genai.FunctionResponse{
				Name:     name,
				Response: map[string]any{"result": strings.Repeat("x", size)},
			},
		}},
	}
}

// ---------------------------------------------------------------------------
// Tests: Fix 1 — safeSplitIndex must never regress to 0
// ---------------------------------------------------------------------------

func TestSafeSplitIndex_AllToolCalls_NeverReturnsZero(t *testing.T) {
	contents := makeToolOnlyConversation(10)

	for candidateIdx := 1; candidateIdx < len(contents); candidateIdx++ {
		got := safeSplitIndex(contents, candidateIdx)
		if got <= 0 {
			t.Errorf("safeSplitIndex(contents, %d) = %d; must never be 0 for all-tool conversations", candidateIdx, got)
		}
	}
}

func TestSafeSplitIndex_AllToolCalls_FloorAtOne(t *testing.T) {
	contents := makeToolOnlyConversation(5)

	got := safeSplitIndex(contents, 1)
	if got < 1 {
		t.Errorf("safeSplitIndex(contents, 1) = %d; expected >= 1", got)
	}
}

func TestSafeSplitIndex_ForwardWalk_LandsBetweenPairs(t *testing.T) {
	contents := makeToolOnlyConversation(5)

	got := safeSplitIndex(contents, 3)

	if got <= 0 || got >= len(contents) {
		t.Fatalf("safeSplitIndex(contents, 3) = %d; out of bounds", got)
	}
	if got%2 != 0 {
		t.Errorf("safeSplitIndex(contents, 3) = %d; expected even index (pair boundary), tool pairs are [0,1], [2,3], [4,5]...", got)
	}
}

func TestSafeSplitIndex_MixedConversation_BackwardStillWorks(t *testing.T) {
	contents := []*genai.Content{
		textContent("user", "hello"),
		textContent("model", "let me check"),
		toolCallContent("search"),
		toolResultContent("search"),
		textContent("model", "results"),
		textContent("user", "thanks"),
	}

	got := safeSplitIndex(contents, 3)
	if got != 1 {
		t.Errorf("safeSplitIndex should walk back past tool chain to text at idx 1, got %d", got)
	}
}

func TestSafeSplitIndex_ForwardWalk_SkipsCompletePairs(t *testing.T) {
	contents := []*genai.Content{
		toolCallContent("a"),
		toolResultContent("a"),
		toolCallContent("b"),
		toolResultContent("b"),
		toolCallContent("c"),
		toolResultContent("c"),
		textContent("user", "done"),
	}

	got := safeSplitIndex(contents, 1)
	if got < 2 {
		t.Errorf("safeSplitIndex(1) should walk forward to pair boundary, got %d", got)
	}
	if got > 6 {
		t.Errorf("safeSplitIndex(1) should not go past the text message at 6, got %d", got)
	}
}

// ---------------------------------------------------------------------------
// Tests: Iterative compaction (Proposal 2)
// ---------------------------------------------------------------------------

func TestThresholdStrategy_IterativeCompaction(t *testing.T) {
	registry := &mockRegistry{
		contextWindows: map[string]int{"small-model": 2_000},
		maxTokens:      map[string]int{"small-model": 512},
	}
	llm := &mockLLM{name: "small-model", response: "compact summary"}
	s := newThresholdStrategy(registry, llm, 0)
	ctx := newMockCallbackContext("agent1")

	req := &model.LLMRequest{
		Model:    "small-model",
		Contents: makeLargeConversation(5_000),
	}

	tokensBefore := estimateTokens(req)
	buffer := computeBuffer(2_000)
	threshold := 2_000 - buffer

	if tokensBefore <= threshold {
		t.Skip("test data too small to trigger compaction")
	}

	err := s.Compact(ctx, req)
	if err != nil {
		t.Fatalf("Compact error: %v", err)
	}

	if !strings.Contains(req.Contents[0].Parts[0].Text, "[Previous conversation summary]") {
		t.Error("first content should be the summary")
	}

	summary := loadSummary(ctx)
	if summary == "" {
		t.Error("summary should have been persisted")
	}
}

// ---------------------------------------------------------------------------
// Tests: Fix 2 — findSplitIndex must count FunctionCall/FunctionResponse tokens
// ---------------------------------------------------------------------------

func TestFindSplitIndex_ToolCallsCountTokens(t *testing.T) {
	contents := []*genai.Content{
		textContent("user", "hello"),
		textContent("model", "let me check"),
		toolCallContent("big_tool"),
		makeLargeToolResultContent("big_tool", 4000),
		toolCallContent("another_tool"),
		makeLargeToolResultContent("another_tool", 4000),
		textContent("user", "ok"),
		textContent("model", "done"),
	}

	budget := 500
	idx := findSplitIndex(contents, budget)

	recentContents := contents[idx:]
	recentTextTokens := 0
	for _, c := range recentContents {
		if c == nil {
			continue
		}
		for _, p := range c.Parts {
			if p == nil {
				continue
			}
			if p.Text != "" {
				recentTextTokens += len(p.Text) / 4
			}
			if p.FunctionCall != nil {
				recentTextTokens += len(p.FunctionCall.Name) / 4
				for k, v := range p.FunctionCall.Args {
					recentTextTokens += len(k) / 4
					recentTextTokens += len(fmt.Sprintf("%v", v)) / 4
				}
			}
			if p.FunctionResponse != nil {
				recentTextTokens += len(p.FunctionResponse.Name) / 4
				recentTextTokens += len(fmt.Sprintf("%v", p.FunctionResponse.Response)) / 4
			}
		}
	}

	if recentTextTokens > budget*2 {
		t.Errorf("findSplitIndex kept too many recent tokens: %d (budget was %d); tool call tokens not counted?",
			recentTextTokens, budget)
	}
}

// ---------------------------------------------------------------------------
// Tests: End-to-end — threshold strategy compacts all-tool conversations
// ---------------------------------------------------------------------------

func TestThresholdStrategy_AllToolCalls_StillCompacts(t *testing.T) {
	registry := &mockRegistry{
		contextWindows: map[string]int{"small-model": 1_000},
		maxTokens:      map[string]int{"small-model": 512},
	}
	llm := &mockLLM{name: "small-model", response: "Summarized tool conversation"}
	s := newThresholdStrategy(registry, llm, 0)
	ctx := newMockCallbackContext("agent1")

	var contents []*genai.Content
	for i := 0; i < 50; i++ {
		contents = append(contents,
			toolCallContent(fmt.Sprintf("kubectl_%d", i)),
			&genai.Content{
				Role: "user",
				Parts: []*genai.Part{{
					FunctionResponse: &genai.FunctionResponse{
						Name:     fmt.Sprintf("kubectl_%d", i),
						Response: map[string]any{"result": strings.Repeat("pod-data-", 50)},
					},
				}},
			},
		)
	}

	req := &model.LLMRequest{
		Model:    "small-model",
		Contents: contents,
	}

	err := s.Compact(ctx, req)
	if err != nil {
		t.Fatalf("Compact error: %v", err)
	}

	if !strings.Contains(req.Contents[0].Parts[0].Text, "Summarized tool conversation") {
		t.Error("first content should be the summary after compaction")
	}

	summary := loadSummary(ctx)
	if summary == "" {
		t.Error("summary should have been persisted to state")
	}
}

func TestSlidingWindowStrategy_AllToolCalls_StillCompacts(t *testing.T) {
	registry := newMockRegistry()
	llm := &mockLLM{name: "gpt-4o", response: "Summarized sliding window"}
	s := newSlidingWindowStrategy(registry, llm, 5)
	ctx := newMockCallbackContext("agent1")

	req := &model.LLMRequest{
		Model:    "gpt-4o",
		Contents: makeToolOnlyConversation(20),
	}

	err := s.Compact(ctx, req)
	if err != nil {
		t.Fatalf("Compact error: %v", err)
	}

	if !strings.Contains(req.Contents[0].Parts[0].Text, "Summarized sliding window") {
		t.Error("first content should be the summary")
	}

	summary := loadSummary(ctx)
	if summary == "" {
		t.Error("summary should have been persisted to state")
	}

	watermark := loadContentsAtCompaction(ctx)
	if watermark <= 0 {
		t.Error("watermark should have been written after compaction")
	}
}

// ---------------------------------------------------------------------------
// Tests: Proposal 5.1 — Real token counts via AfterModelCallback
// ---------------------------------------------------------------------------

func TestPersistAndLoadRealTokens(t *testing.T) {
	ctx := newMockCallbackContext("agent1")

	if got := loadRealTokens(ctx); got != 0 {
		t.Errorf("loadRealTokens on empty state = %d, want 0", got)
	}

	persistRealTokens(ctx, 150_000)

	if got := loadRealTokens(ctx); got != 150_000 {
		t.Errorf("loadRealTokens = %d, want 150000", got)
	}
}

func TestLoadRealTokens_Float64Conversion(t *testing.T) {
	ctx := newMockCallbackContext("agent1")
	ctx.state.Set(stateKeyPrefixRealTokens+"agent1", float64(42_000))

	if got := loadRealTokens(ctx); got != 42_000 {
		t.Errorf("loadRealTokens(float64) = %d, want 42000", got)
	}
}

func TestTokenCount_PrefersRealTokens(t *testing.T) {
	ctx := newMockCallbackContext("agent1")
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			textContent("user", strings.Repeat("a", 400)),
		},
	}

	heuristic := estimateTokens(req)
	if heuristic != 100 {
		t.Errorf("raw heuristic = %d, want 100", heuristic)
	}

	got := tokenCount(ctx, req)
	wantDefault := int(float64(100) * defaultHeuristicCorrectionFactor)
	if got != wantDefault {
		t.Errorf("tokenCount without real = %d, want %d (heuristic * correction)", got, wantDefault)
	}

	persistRealTokens(ctx, 200_000)

	real := tokenCount(ctx, req)
	if real != 200_000 {
		t.Errorf("tokenCount with real (no lastHeuristic) = %d, want 200000", real)
	}
}

func TestTokenCount_CalibratedHeuristic(t *testing.T) {
	ctx := newMockCallbackContext("agent1")

	persistRealTokens(ctx, 1000)
	persistLastHeuristic(ctx, 500)

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			textContent("user", strings.Repeat("a", 3000)),
		},
	}
	currentHeuristic := estimateTokens(req)
	if currentHeuristic != 750 {
		t.Fatalf("raw heuristic = %d, want 750", currentHeuristic)
	}

	got := tokenCount(ctx, req)
	wantCalibrated := int(float64(750) * (float64(1000) / float64(500)))
	if got != wantCalibrated {
		t.Errorf("tokenCount calibrated = %d, want %d (750 * 2.0)", got, wantCalibrated)
	}
}

func TestTokenCount_CorrectionFloorAtOne(t *testing.T) {
	ctx := newMockCallbackContext("agent1")

	persistRealTokens(ctx, 300)
	persistLastHeuristic(ctx, 600)

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			textContent("user", strings.Repeat("a", 2000)),
		},
	}
	currentHeuristic := estimateTokens(req)

	got := tokenCount(ctx, req)
	if got != currentHeuristic {
		t.Errorf("tokenCount with correction<1 = %d, want %d (correction clamped to 1.0)", got, currentHeuristic)
	}
}

func TestTokenCount_RealTokensWinWhenLarger(t *testing.T) {
	ctx := newMockCallbackContext("agent1")

	persistRealTokens(ctx, 100_000)
	persistLastHeuristic(ctx, 50_000)

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			textContent("user", strings.Repeat("a", 400)),
		},
	}

	got := tokenCount(ctx, req)
	if got != 100_000 {
		t.Errorf("tokenCount = %d, want 100000 (realTokens should win over small calibrated)", got)
	}
}

func TestPersistAndLoadLastHeuristic(t *testing.T) {
	ctx := newMockCallbackContext("agent1")

	if got := loadLastHeuristic(ctx); got != 0 {
		t.Errorf("loadLastHeuristic empty = %d, want 0", got)
	}

	persistLastHeuristic(ctx, 42_000)

	if got := loadLastHeuristic(ctx); got != 42_000 {
		t.Errorf("loadLastHeuristic = %d, want 42000", got)
	}
}

func TestAfterModel_PersistsTokenCount(t *testing.T) {
	guard := New(newMockRegistry())
	llm := &mockLLM{name: "gpt-4o"}
	guard.Add("agent1", llm)

	g := &contextGuard{strategies: guard.strategies}
	ctx := newMockCallbackContext("agent1")

	resp := &model.LLMResponse{
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     120_000,
			CandidatesTokenCount: 5_000,
		},
	}

	result, err := g.afterModel(ctx, resp, nil)
	if result != nil || err != nil {
		t.Errorf("afterModel should return (nil, nil), got (%v, %v)", result, err)
	}

	got := loadRealTokens(ctx)
	if got != 120_000 {
		t.Errorf("real tokens = %d, want 120000 (only PromptTokenCount)", got)
	}
}

func TestAfterModel_NilUsageMetadata(t *testing.T) {
	guard := New(newMockRegistry())
	llm := &mockLLM{name: "gpt-4o"}
	guard.Add("agent1", llm)

	g := &contextGuard{strategies: guard.strategies}
	ctx := newMockCallbackContext("agent1")

	result, err := g.afterModel(ctx, &model.LLMResponse{}, nil)
	if result != nil || err != nil {
		t.Errorf("afterModel with nil usage should return (nil, nil)")
	}

	if got := loadRealTokens(ctx); got != 0 {
		t.Errorf("should not persist when UsageMetadata is nil, got %d", got)
	}
}

func TestAfterModel_SkipsPartials(t *testing.T) {
	guard := New(newMockRegistry())
	llm := &mockLLM{name: "gpt-4o"}
	guard.Add("agent1", llm)

	g := &contextGuard{strategies: guard.strategies}
	ctx := newMockCallbackContext("agent1")

	partial := &model.LLMResponse{
		Partial: true,
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount: 100_000,
		},
	}

	result, err := g.afterModel(ctx, partial, nil)
	if result != nil || err != nil {
		t.Errorf("afterModel with partial should return (nil, nil)")
	}

	if got := loadRealTokens(ctx); got != 0 {
		t.Errorf("should not persist for partial responses, got %d", got)
	}
}

func TestAfterModel_UnknownAgent(t *testing.T) {
	g := &contextGuard{strategies: map[string]Strategy{}}
	ctx := newMockCallbackContext("unknown")

	resp := &model.LLMResponse{
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     100_000,
			CandidatesTokenCount: 1_000,
		},
	}

	result, err := g.afterModel(ctx, resp, nil)
	if result != nil || err != nil {
		t.Errorf("afterModel for unknown agent should return (nil, nil)")
	}

	if got := loadRealTokens(ctx); got != 0 {
		t.Errorf("should not persist for unknown agent, got %d", got)
	}
}

func TestThresholdStrategy_UsesRealTokens(t *testing.T) {
	registry := newMockRegistry()
	llm := &mockLLM{name: "gpt-4o", response: "compacted summary"}
	s := newThresholdStrategy(registry, llm, 0)
	ctx := newMockCallbackContext("agent1")

	req := &model.LLMRequest{
		Model:    "gpt-4o",
		Contents: []*genai.Content{textContent("user", "short message")},
	}

	persistRealTokens(ctx, 110_000)

	err := s.Compact(ctx, req)
	if err != nil {
		t.Fatalf("Compact error: %v", err)
	}

	if !strings.Contains(req.Contents[0].Parts[0].Text, "[Previous conversation summary]") {
		t.Error("should have compacted: real tokens (110k) exceed threshold (128k - 25.6k = 102.4k)")
	}
}

// ---------------------------------------------------------------------------
// Tests: Full summary — threshold always summarizes everything (no recent tail)
// ---------------------------------------------------------------------------

func TestThresholdStrategy_NoRecentTail(t *testing.T) {
	registry := &mockRegistry{
		contextWindows: map[string]int{"small-model": 1_000},
		maxTokens:      map[string]int{"small-model": 512},
	}
	llm := &mockLLM{name: "small-model", response: "Full summary of everything"}
	s := newThresholdStrategy(registry, llm, 0)
	ctx := newMockCallbackContext("agent1")

	req := &model.LLMRequest{
		Model:    "small-model",
		Contents: makeLargeConversation(2_000),
	}

	err := s.Compact(ctx, req)
	if err != nil {
		t.Fatalf("Compact error: %v", err)
	}

	if !strings.Contains(req.Contents[0].Parts[0].Text, "Full summary of everything") {
		t.Error("first content should be the summary")
	}

	// Always full summary: [summary] + [continuation] = 2 messages
	if len(req.Contents) != 2 {
		t.Errorf("expected 2 contents (summary + continuation), got %d", len(req.Contents))
	}

	last := req.Contents[len(req.Contents)-1]
	if !strings.Contains(last.Parts[0].Text, "conversation was compacted") {
		t.Error("last content should be the continuation instruction")
	}
}

// ---------------------------------------------------------------------------
// Tests: Proposal 5.3 — Continuation context injection
// ---------------------------------------------------------------------------

func TestInjectContinuation_WithUserContent(t *testing.T) {
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			textContent("user", "summary here"),
		},
	}
	userContent := textContent("user", "Fix the auth middleware bug")

	injectContinuation(req, userContent)

	if len(req.Contents) != 2 {
		t.Fatalf("expected 2 contents, got %d", len(req.Contents))
	}

	last := req.Contents[1]
	if last.Role != "user" {
		t.Errorf("continuation role = %q, want 'user'", last.Role)
	}
	if !strings.Contains(last.Parts[0].Text, "Fix the auth middleware bug") {
		t.Error("continuation should contain original user request")
	}
	if !strings.Contains(last.Parts[0].Text, "conversation was compacted") {
		t.Error("continuation should explain compaction")
	}
}

func TestInjectContinuation_NilUserContent(t *testing.T) {
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			textContent("user", "summary here"),
		},
	}

	injectContinuation(req, nil)

	if len(req.Contents) != 2 {
		t.Fatalf("expected 2 contents, got %d", len(req.Contents))
	}

	last := req.Contents[1]
	if !strings.Contains(last.Parts[0].Text, "Continue working") {
		t.Error("continuation should still be injected without user content")
	}
}

func TestThresholdStrategy_InjectsContinuation(t *testing.T) {
	registry := &mockRegistry{
		contextWindows: map[string]int{"small-model": 1_000},
		maxTokens:      map[string]int{"small-model": 512},
	}
	llm := &mockLLM{name: "small-model", response: "summary"}
	s := newThresholdStrategy(registry, llm, 0)

	ctx := &mockCallbackContext{
		Context:   context.Background(),
		agentName: "agent1",
		sessionID: "test-session",
		state:     newMockState(),
	}

	req := &model.LLMRequest{
		Model:    "small-model",
		Contents: makeLargeConversation(2_000),
	}

	err := s.Compact(ctx, req)
	if err != nil {
		t.Fatalf("Compact error: %v", err)
	}

	last := req.Contents[len(req.Contents)-1]
	if !strings.Contains(last.Parts[0].Text, "conversation was compacted") {
		t.Error("last content should be the continuation instruction after compaction")
	}
}

// ---------------------------------------------------------------------------
// Tests: Proposal 5.4 — Todo preservation in summary prompt
// ---------------------------------------------------------------------------

func TestLoadTodos_Empty(t *testing.T) {
	ctx := newMockCallbackContext("agent1")
	todos := loadTodos(ctx)
	if todos != nil {
		t.Errorf("loadTodos on empty state should return nil, got %v", todos)
	}
}

func TestLoadTodos_TypedSlice(t *testing.T) {
	ctx := newMockCallbackContext("agent1")
	items := []TodoItem{
		{Content: "Fix bug", Status: "completed"},
		{Content: "Write tests", Status: "in_progress", ActiveForm: "Writing tests"},
	}
	ctx.state.Set("todos", items)

	got := loadTodos(ctx)
	if len(got) != 2 {
		t.Fatalf("loadTodos = %d items, want 2", len(got))
	}
	if got[0].Content != "Fix bug" || got[0].Status != "completed" {
		t.Errorf("todo[0] = %+v", got[0])
	}
	if got[1].Content != "Write tests" || got[1].ActiveForm != "Writing tests" {
		t.Errorf("todo[1] = %+v", got[1])
	}
}

func TestLoadTodos_MapSlice(t *testing.T) {
	ctx := newMockCallbackContext("agent1")
	raw := []any{
		map[string]any{"content": "Deploy", "status": "pending", "active_form": "Deploying"},
		map[string]any{"content": "Monitor", "status": "completed"},
	}
	ctx.state.Set("todos", raw)

	got := loadTodos(ctx)
	if len(got) != 2 {
		t.Fatalf("loadTodos from []any = %d items, want 2", len(got))
	}
	if got[0].Content != "Deploy" || got[0].Status != "pending" || got[0].ActiveForm != "Deploying" {
		t.Errorf("todo[0] = %+v", got[0])
	}
	if got[1].Content != "Monitor" || got[1].Status != "completed" {
		t.Errorf("todo[1] = %+v", got[1])
	}
}

func TestBuildSummarizePrompt_WithTodos(t *testing.T) {
	contents := []*genai.Content{
		textContent("user", "Working on deployment"),
	}
	todos := []TodoItem{
		{Content: "Fix auth bug", Status: "completed"},
		{Content: "Deploy to staging", Status: "in_progress"},
		{Content: "Write docs", Status: "pending"},
	}

	prompt := buildSummarizePrompt(contents, "", todos)

	if !strings.Contains(prompt, "[Current todo list]") {
		t.Error("should contain todo list header")
	}
	if !strings.Contains(prompt, "- [completed] Fix auth bug") {
		t.Error("should contain completed todo")
	}
	if !strings.Contains(prompt, "- [in_progress] Deploy to staging") {
		t.Error("should contain in_progress todo")
	}
	if !strings.Contains(prompt, "- [pending] Write docs") {
		t.Error("should contain pending todo")
	}
	if !strings.Contains(prompt, "todos") {
		t.Error("should instruct to restore todos")
	}
}

func TestBuildSummarizePrompt_WithoutTodos(t *testing.T) {
	contents := []*genai.Content{
		textContent("user", "hello"),
	}

	prompt := buildSummarizePrompt(contents, "", nil)

	if strings.Contains(prompt, "[Current todo list]") {
		t.Error("should not contain todo list when nil")
	}
}

// ---------------------------------------------------------------------------
// Tests: PluginConfig wires both callbacks
// ---------------------------------------------------------------------------

func TestPluginConfig_HasAfterModelCallback(t *testing.T) {
	guard := New(newMockRegistry())
	guard.Add("agent1", &mockLLM{name: "gpt-4o"})

	cfg := guard.PluginConfig()
	if len(cfg.Plugins) != 1 {
		t.Fatalf("expected 1 plugin, got %d", len(cfg.Plugins))
	}
}

// ---------------------------------------------------------------------------
// Tests: End-to-end — real tokens trigger compaction that heuristic misses
// ---------------------------------------------------------------------------

func TestThresholdStrategy_RealTokens_TriggersWhereHeuristicFails(t *testing.T) {
	registry := newMockRegistry()
	llm := &mockLLM{name: "gpt-4o", response: "compacted"}
	s := newThresholdStrategy(registry, llm, 150_000)
	ctx := newMockCallbackContext("agent1")

	// Create a request where len/4 estimates ~50k tokens but "real" is 130k.
	// With maxTokens=150k, threshold = 150k - 30k = 120k.
	// Heuristic: 50k < 120k → no compaction.
	// Real: 130k >= 120k → compaction.
	req := &model.LLMRequest{
		Model:    "gpt-4o",
		Contents: makeLargeConversation(50_000),
	}

	heuristic := estimateTokens(req)
	if heuristic >= 120_000 {
		t.Skip("heuristic too high for this test scenario")
	}

	persistRealTokens(ctx, 130_000)

	err := s.Compact(ctx, req)
	if err != nil {
		t.Fatalf("Compact error: %v", err)
	}

	if !strings.Contains(req.Contents[0].Parts[0].Text, "compacted") {
		t.Error("real token count should have triggered compaction where heuristic wouldn't")
	}
}
