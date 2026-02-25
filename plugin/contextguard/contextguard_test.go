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
	prompt := buildSummarizePrompt(contents, "")

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
	prompt := buildSummarizePrompt(contents, "Earlier we discussed Go.")

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
	prompt := buildSummarizePrompt(contents, "")

	if !strings.Contains(prompt, "[called tool: search]") {
		t.Error("missing tool call in transcript")
	}
	if !strings.Contains(prompt, "[tool search returned a result]") {
		t.Error("missing tool result in transcript")
	}
}

func TestBuildSummarizePrompt_NilContents(t *testing.T) {
	contents := []*genai.Content{nil, textContent("user", "hello"), nil}
	prompt := buildSummarizePrompt(contents, "")
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
