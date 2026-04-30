// Copyright 2025 achetronic
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

package openai

import (
	"errors"
	"reflect"
	"testing"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/shared"
	"google.golang.org/genai"
)

// convertResponse rebuilds a genai.LLMResponse from the SDK's
// ChatCompletion. The non-trivial cases are:
//   - empty Choices must surface ErrNoChoicesInResponse so the caller can
//     treat it as a hard failure (an "empty" response is undefined behaviour
//     for OpenAI; better to fail loudly than to bubble up an empty turn)
//   - both text and tool_calls in the same choice must coexist in the
//     resulting Content (the order matters: text first, then tool calls,
//     mirroring how the genai-side iteration consumes parts)
//   - Args inside the function call must be JSON-decoded (parseJSONArgs),
//     not passed through verbatim as a string
func TestConvertResponse(t *testing.T) {
	t.Run("empty choices returns ErrNoChoicesInResponse", func(t *testing.T) {
		m := newModelForTest()
		resp := &openai.ChatCompletion{}
		_, err := m.convertResponse(resp)
		if !errors.Is(err, ErrNoChoicesInResponse) {
			t.Errorf("err = %v, want %v", err, ErrNoChoicesInResponse)
		}
	})

	t.Run("text only response", func(t *testing.T) {
		m := newModelForTest()
		resp := &openai.ChatCompletion{
			Choices: []openai.ChatCompletionChoice{{
				Message:      openai.ChatCompletionMessage{Content: "hello"},
				FinishReason: "stop",
			}},
			Usage: openai.CompletionUsage{PromptTokens: 1, CompletionTokens: 2, TotalTokens: 3},
		}
		got, err := m.convertResponse(resp)
		if err != nil {
			t.Fatalf("error: %v", err)
		}
		if !got.TurnComplete {
			t.Errorf("TurnComplete = false, want true")
		}
		if got.FinishReason != genai.FinishReasonStop {
			t.Errorf("FinishReason = %v, want Stop", got.FinishReason)
		}
		if got.Content == nil || len(got.Content.Parts) != 1 || got.Content.Parts[0].Text != "hello" {
			t.Errorf("Content parts = %#v, want single text \"hello\"", got.Content)
		}
		if got.UsageMetadata == nil || got.UsageMetadata.TotalTokenCount != 3 {
			t.Errorf("UsageMetadata = %#v, want TotalTokenCount=3", got.UsageMetadata)
		}
	})

	t.Run("tool calls plus text", func(t *testing.T) {
		m := newModelForTest()
		resp := &openai.ChatCompletion{
			Choices: []openai.ChatCompletionChoice{{
				Message: openai.ChatCompletionMessage{
					Content: "looking up",
					ToolCalls: []openai.ChatCompletionMessageToolCallUnion{{
						ID:   "call_42",
						Type: "function",
						Function: openai.ChatCompletionMessageFunctionToolCallFunction{
							Name:      "search",
							Arguments: `{"q":"weather"}`,
						},
					}},
				},
				FinishReason: "tool_calls",
			}},
		}
		got, err := m.convertResponse(resp)
		if err != nil {
			t.Fatalf("error: %v", err)
		}
		if got.FinishReason != genai.FinishReasonStop {
			t.Errorf("tool_calls finish should map to Stop, got %v", got.FinishReason)
		}
		if len(got.Content.Parts) != 2 {
			t.Fatalf("expected 2 parts (text + call), got %d", len(got.Content.Parts))
		}
		if got.Content.Parts[0].Text != "looking up" {
			t.Errorf("first part text = %q", got.Content.Parts[0].Text)
		}
		fc := got.Content.Parts[1].FunctionCall
		if fc == nil {
			t.Fatalf("expected FunctionCall on second part")
		}
		if fc.ID != "call_42" || fc.Name != "search" {
			t.Errorf("FunctionCall = %#v, want id=call_42 name=search", fc)
		}
		if got, want := fc.Args, map[string]any{"q": "weather"}; !reflect.DeepEqual(got, want) {
			t.Errorf("Args = %#v, want %#v", got, want)
		}
	})

	t.Run("usage with zero total tokens is dropped", func(t *testing.T) {
		m := newModelForTest()
		resp := &openai.ChatCompletion{
			Choices: []openai.ChatCompletionChoice{{
				Message:      openai.ChatCompletionMessage{Content: "x"},
				FinishReason: "stop",
			}},
			// Usage zero-valued: providers like Ollama don't always report tokens.
		}
		got, err := m.convertResponse(resp)
		if err != nil {
			t.Fatalf("error: %v", err)
		}
		if got.UsageMetadata != nil {
			t.Errorf("UsageMetadata = %#v, want nil when no tokens reported", got.UsageMetadata)
		}
	})
}

// buildStreamFinalResponse mirrors convertResponse but reads from the
// streaming Accumulator. It must be tolerant of an empty accumulator (no
// choices yet) by returning an empty Content rather than panicking, and it
// must always set TurnComplete=true and Partial=false because, by the time
// it's called, the stream has already drained.
func TestBuildStreamFinalResponse(t *testing.T) {
	t.Run("empty accumulator returns an empty but valid response", func(t *testing.T) {
		m := newModelForTest()
		got := m.buildStreamFinalResponse(&openai.ChatCompletionAccumulator{})
		if got == nil {
			t.Fatalf("expected non-nil response")
		}
		if got.Partial {
			t.Errorf("Partial = true, want false at end of stream")
		}
		if !got.TurnComplete {
			t.Errorf("TurnComplete = false, want true at end of stream")
		}
		if got.Content == nil || len(got.Content.Parts) != 0 {
			t.Errorf("Content = %#v, want empty parts", got.Content)
		}
	})

	t.Run("populated accumulator collapses text and tool calls", func(t *testing.T) {
		m := newModelForTest()
		acc := &openai.ChatCompletionAccumulator{
			ChatCompletion: openai.ChatCompletion{
				Choices: []openai.ChatCompletionChoice{{
					Message: openai.ChatCompletionMessage{
						Content: "streamed",
						ToolCalls: []openai.ChatCompletionMessageToolCallUnion{{
							ID: "tc",
							Function: openai.ChatCompletionMessageFunctionToolCallFunction{
								Name:      "fn",
								Arguments: `{"a":1}`,
							},
						}},
					},
					FinishReason: "stop",
				}},
				Usage: openai.CompletionUsage{TotalTokens: 9},
			},
		}
		got := m.buildStreamFinalResponse(acc)
		if got.FinishReason != genai.FinishReasonStop {
			t.Errorf("FinishReason = %v, want Stop", got.FinishReason)
		}
		if len(got.Content.Parts) != 2 {
			t.Fatalf("expected 2 parts, got %d", len(got.Content.Parts))
		}
		if got.UsageMetadata == nil || got.UsageMetadata.TotalTokenCount != 9 {
			t.Errorf("UsageMetadata = %#v, want TotalTokenCount=9", got.UsageMetadata)
		}
	})
}

// convertUsageMetadata returns nil when the provider didn't report any
// tokens (TotalTokens == 0). Ollama and a few self-hosted backends behave
// this way, and surfacing a zeroed usage struct downstream poisons cost
// reporting; we'd rather return nil so callers can detect the absence.
func TestConvertUsageMetadata(t *testing.T) {
	cases := []struct {
		name   string
		usage  openai.CompletionUsage
		want   *genai.GenerateContentResponseUsageMetadata
		wantOK bool
	}{
		{
			name:   "no tokens reported returns nil",
			usage:  openai.CompletionUsage{},
			wantOK: false,
		},
		{
			name:  "populated usage maps 1:1",
			usage: openai.CompletionUsage{PromptTokens: 11, CompletionTokens: 22, TotalTokens: 33},
			want: &genai.GenerateContentResponseUsageMetadata{
				PromptTokenCount:     11,
				CandidatesTokenCount: 22,
				TotalTokenCount:      33,
			},
			wantOK: true,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := convertUsageMetadata(c.usage)
			if !c.wantOK {
				if got != nil {
					t.Errorf("got = %#v, want nil", got)
				}
				return
			}
			if got == nil {
				t.Fatalf("got nil, want %#v", c.want)
			}
			if !reflect.DeepEqual(got, c.want) {
				t.Errorf("got = %#v, want %#v", got, c.want)
			}
		})
	}
}

// convertThinkingLevel maps genai's three-level enum to OpenAI's reasoning
// effort. Anything outside Low/High must default to Medium so callers can
// safely use the enum's zero value (Unspecified) without inadvertently
// disabling reasoning.
func TestConvertThinkingLevel(t *testing.T) {
	cases := []struct {
		level genai.ThinkingLevel
		want  shared.ReasoningEffort
	}{
		{genai.ThinkingLevelLow, shared.ReasoningEffortLow},
		{genai.ThinkingLevelHigh, shared.ReasoningEffortHigh},
		{genai.ThinkingLevel(""), shared.ReasoningEffortMedium},
		{genai.ThinkingLevel("invalid"), shared.ReasoningEffortMedium},
	}
	for _, c := range cases {
		t.Run(string(c.level), func(t *testing.T) {
			if got := convertThinkingLevel(c.level); got != c.want {
				t.Errorf("convertThinkingLevel(%q) = %q, want %q", c.level, got, c.want)
			}
		})
	}
}
