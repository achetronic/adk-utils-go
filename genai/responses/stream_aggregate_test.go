// Copyright 2025 achetronic
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

package responses

import (
	"testing"

	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// TestStreamAccumulatorFinalResponse verifies that the accumulated deltas can be
// rebuilt into a single non-partial, TurnComplete final response, with the
// reasoning part ordered before the answer part. This is the core of the fix:
// when the terminal event omits the aggregated output, the accumulated deltas
// are used to produce the event ADK persists.
func TestStreamAccumulatorFinalResponse(t *testing.T) {
	var acc streamAccumulator
	acc.reasoning.WriteString("think first, ")
	acc.reasoning.WriteString("then answer.")
	acc.text.WriteString("hello ")
	acc.text.WriteString("world")

	resp := acc.finalResponse(genai.FinishReasonStop, nil)

	if resp.Partial {
		t.Errorf("Partial = true, want false (only non-partial events are persisted by ADK)")
	}
	if !resp.TurnComplete {
		t.Errorf("TurnComplete = false, want true")
	}
	if resp.FinishReason != genai.FinishReasonStop {
		t.Errorf("FinishReason = %v, want Stop", resp.FinishReason)
	}
	if got := len(resp.Content.Parts); got != 2 {
		t.Fatalf("len(Parts) = %d, want 2 (reasoning + answer)", got)
	}
	if p := resp.Content.Parts[0]; !p.Thought || p.Text != "think first, then answer." {
		t.Errorf("Parts[0] = %+v, want reasoning part {Thought:true, Text:\"think first, then answer.\"}", p)
	}
	if p := resp.Content.Parts[1]; p.Thought || p.Text != "hello world" {
		t.Errorf("Parts[1] = %+v, want answer part {Thought:false, Text:\"hello world\"}", p)
	}
}

// TestStreamAccumulatorOnlyText verifies that with an answer but no reasoning,
// the final response contains a single text part.
func TestStreamAccumulatorOnlyText(t *testing.T) {
	var acc streamAccumulator
	acc.text.WriteString("ok")

	resp := acc.finalResponse(genai.FinishReasonStop, nil)
	if got := len(resp.Content.Parts); got != 1 {
		t.Fatalf("len(Parts) = %d, want 1", got)
	}
	if p := resp.Content.Parts[0]; p.Thought || p.Text != "ok" {
		t.Errorf("Parts[0] = %+v, want {Thought:false, Text:\"ok\"}", p)
	}
}

// TestStreamAccumulatorHasContent checks hasContent for empty and non-empty
// accumulators.
func TestStreamAccumulatorHasContent(t *testing.T) {
	var empty streamAccumulator
	if empty.hasContent() {
		t.Errorf("empty accumulator hasContent() = true, want false")
	}

	var withReasoning streamAccumulator
	withReasoning.reasoning.WriteString("x")
	if !withReasoning.hasContent() {
		t.Errorf("reasoning-only accumulator hasContent() = false, want true")
	}

	var withText streamAccumulator
	withText.text.WriteString("x")
	if !withText.hasContent() {
		t.Errorf("text-only accumulator hasContent() = false, want true")
	}
}

// TestHasNoContent verifies the fallback predicate: a nil response, a nil
// Content, or empty Parts all count as "no content", in which case
// generateStream falls back to the accumulated deltas.
func TestHasNoContent(t *testing.T) {
	cases := []struct {
		name string
		resp *model.LLMResponse
		want bool
	}{
		{"nil response", nil, true},
		{"nil Content", &model.LLMResponse{}, true},
		{"empty Parts", &model.LLMResponse{Content: &genai.Content{Parts: []*genai.Part{}}}, true},
		{"has text part", &model.LLMResponse{Content: &genai.Content{Parts: []*genai.Part{{Text: "hi"}}}}, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := hasNoContent(c.resp); got != c.want {
				t.Errorf("hasNoContent() = %v, want %v", got, c.want)
			}
		})
	}
}
