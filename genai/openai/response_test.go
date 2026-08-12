// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package openai

import (
	"encoding/json"
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
		got := m.buildStreamFinalResponse(&openai.ChatCompletionAccumulator{}, nil)
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
		got := m.buildStreamFinalResponse(acc, nil)
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

// probeString is the raw-JSON probe TextCodec uses to find the plain-text
// reasoning field: it returns the first configured field present as a
// non-empty string, "" otherwise, and "" for an unparseable envelope. The
// degradation is deliberate: at the response-conversion layer dropping a
// thought Part is the safe outcome of a field the codec cannot read.
func TestProbeString(t *testing.T) {
	fields := []string{"reasoning_content", "reasoning"}
	cases := []struct {
		name string
		raw  string
		want string
	}{
		{
			name: "empty raw returns empty",
			raw:  "",
			want: "",
		},
		{
			name: "raw without a reasoning field returns empty",
			raw:  `{"role":"assistant","content":"hello"}`,
			want: "",
		},
		{
			name: "raw with reasoning_content returns the value",
			raw:  `{"role":"assistant","content":"hello","reasoning_content":"thinking step by step"}`,
			want: "thinking step by step",
		},
		{
			name: "raw with reasoning returns the value",
			raw:  `{"role":"assistant","content":"hello","reasoning":"openrouter trace"}`,
			want: "openrouter trace",
		},
		{
			name: "first configured field wins over a later one",
			raw:  `{"reasoning_content":"from content","reasoning":"from reasoning"}`,
			want: "from content",
		},
		{
			name: "raw with empty reasoning_content returns empty",
			raw:  `{"role":"assistant","content":"hello","reasoning_content":""}`,
			want: "",
		},
		{
			name: "malformed JSON returns empty rather than panicking",
			raw:  `{"role":"assistant"`,
			want: "",
		},
		{
			name: "reasoning field of wrong type (non-string) returns empty",
			raw:  `{"reasoning_content":123}`,
			want: "",
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := probeString(probeRawJSON(c.raw), fields); got != c.want {
				t.Errorf("probeString(probeRawJSON(%q)) = %q, want %q", c.raw, got, c.want)
			}
		})
	}
}

// convertResponse must surface reasoning_content as a separate Part with
// Thought=true so downstream consumers (Google ADK's llmagent pipeline,
// which filters `if part.Text != "" && !part.Thought`) can distinguish
// chain-of-thought tokens from the final answer.
//
// Tests below build the response via json.Unmarshal because openai-go
// stores the raw JSON envelope in an unexported `JSON.raw` field, populated
// only by the generated UnmarshalJSON. Constructing the struct literally
// would leave RawJSON() empty and bypass the field we are testing - a
// false-positive trap that would let a regression slip through.
func TestConvertResponse_WithReasoningContent(t *testing.T) {
	t.Run("reasoning_content yields a leading Thought part", func(t *testing.T) {
		raw := []byte(`{
            "id": "chatcmpl-x",
            "object": "chat.completion",
            "created": 0,
            "model": "kimi-k2.6",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The answer is 42.",
                    "reasoning_content": "User asked for the meaning of life. The canonical joke answer is 42."
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
        }`)

		var resp openai.ChatCompletion
		if err := json.Unmarshal(raw, &resp); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}

		m := newTextModelForTest()
		got, err := m.convertResponse(&resp)
		if err != nil {
			t.Fatalf("convertResponse: %v", err)
		}

		if len(got.Content.Parts) != 2 {
			t.Fatalf("expected 2 parts (thought + answer), got %d", len(got.Content.Parts))
		}
		if !got.Content.Parts[0].Thought {
			t.Errorf("first part Thought = false, want true")
		}
		if got.Content.Parts[0].Text != "User asked for the meaning of life. The canonical joke answer is 42." {
			t.Errorf("first part text = %q, want reasoning content", got.Content.Parts[0].Text)
		}
		if got.Content.Parts[1].Thought {
			t.Errorf("second part Thought = true, want false (it is the final answer)")
		}
		if got.Content.Parts[1].Text != "The answer is 42." {
			t.Errorf("second part text = %q, want final answer", got.Content.Parts[1].Text)
		}
	})

	t.Run("no reasoning_content leaves the response unchanged", func(t *testing.T) {
		// Sanity check: providers without thinking mode (vanilla GPT-4o,
		// Ollama, etc.) must keep producing single-part responses. A
		// regression here would mean every non-reasoning response suddenly
		// grows a phantom empty Thought part.
		raw := []byte(`{
            "id": "chatcmpl-y",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "plain reply"},
                "finish_reason": "stop"
            }]
        }`)

		var resp openai.ChatCompletion
		if err := json.Unmarshal(raw, &resp); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}

		m := newTextModelForTest()
		got, err := m.convertResponse(&resp)
		if err != nil {
			t.Fatalf("convertResponse: %v", err)
		}

		if len(got.Content.Parts) != 1 {
			t.Fatalf("expected 1 part, got %d", len(got.Content.Parts))
		}
		if got.Content.Parts[0].Thought {
			t.Errorf("Thought = true, want false for a vanilla response")
		}
	})

	t.Run("reasoning_content coexists with a tool call", func(t *testing.T) {
		// The combined case matters because reasoning models often emit
		// chain-of-thought *and* a tool call in the same turn. The Part
		// order must be: thought -> text -> function call, reflecting the
		// temporal order the model produced them.
		raw := []byte(`{
            "id": "chatcmpl-z",
            "object": "chat.completion",
            "created": 0,
            "model": "deepseek-r1",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Looking up weather.",
                    "reasoning_content": "Need fresh data - call the weather tool.",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"city\":\"Madrid\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }`)

		var resp openai.ChatCompletion
		if err := json.Unmarshal(raw, &resp); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}

		m := newTextModelForTest()
		got, err := m.convertResponse(&resp)
		if err != nil {
			t.Fatalf("convertResponse: %v", err)
		}

		if len(got.Content.Parts) != 3 {
			t.Fatalf("expected 3 parts (thought + text + tool call), got %d", len(got.Content.Parts))
		}
		if !got.Content.Parts[0].Thought || got.Content.Parts[0].Text == "" {
			t.Errorf("first part should be a thought; got %#v", got.Content.Parts[0])
		}
		if got.Content.Parts[1].Thought || got.Content.Parts[1].Text == "" {
			t.Errorf("second part should be plain text; got %#v", got.Content.Parts[1])
		}
		if got.Content.Parts[2].FunctionCall == nil {
			t.Errorf("third part should be a FunctionCall; got %#v", got.Content.Parts[2])
		}
	})
}

// buildStreamFinalResponse surfaces the reasoning the adapter accumulated
// itself as leading Thought parts. It does not read reasoning off the SDK
// accumulator: that aggregate merges chunks field by field and keeps no raw
// JSON, so on a live stream there is no reasoning left to read there. This
// test builds the accumulator by hand, which is exactly the path the real
// stream takes.
func TestBuildStreamFinalResponse_WithReasoning(t *testing.T) {
	t.Run("accumulated reasoning surfaces as a leading thought part", func(t *testing.T) {
		acc := &openai.ChatCompletionAccumulator{
			ChatCompletion: openai.ChatCompletion{
				Choices: []openai.ChatCompletionChoice{{
					Message:      openai.ChatCompletionMessage{Content: "Sure, here is the result."},
					FinishReason: "stop",
				}},
			},
		}
		reasoning := NewTextDialect().NewAccumulator()
		reasoning.AddDelta([]*genai.Part{thoughtPart("Streamed chain-of-thought.")})

		m := newModelForTest()
		got := m.buildStreamFinalResponse(acc, reasoning)

		if len(got.Content.Parts) != 2 {
			t.Fatalf("expected 2 parts, got %d", len(got.Content.Parts))
		}
		if !got.Content.Parts[0].Thought {
			t.Errorf("first part Thought = false, want true")
		}
		if got.Content.Parts[0].Text != "Streamed chain-of-thought." {
			t.Errorf("first part text = %q", got.Content.Parts[0].Text)
		}
		if got.Content.Parts[1].Thought {
			t.Errorf("second part Thought = true, want false")
		}
		if got.Content.Parts[1].Text != "Sure, here is the result." {
			t.Errorf("second part text = %q", got.Content.Parts[1].Text)
		}
	})
}

// convertUsageMetadata must preserve the provider's detailed token buckets.
// This is the only path through which billing/usage systems downstream (e.g.
// Langfuse) can apply reasoning-output and cache-read prices. Dropping either
// field silently makes cost dashboards inaccurate.
func TestConvertUsageMetadata_WithTokenDetails(t *testing.T) {
	t.Run("reasoning and cached tokens map to genai details", func(t *testing.T) {
		usage := openai.CompletionUsage{
			PromptTokens:     10,
			CompletionTokens: 50,
			TotalTokens:      60,
			CompletionTokensDetails: openai.CompletionUsageCompletionTokensDetails{
				ReasoningTokens: 42,
			},
			PromptTokensDetails: openai.CompletionUsagePromptTokensDetails{
				CachedTokens: 8,
			},
		}
		got := convertUsageMetadata(usage)
		if got == nil {
			t.Fatalf("got nil")
		}
		if got.ThoughtsTokenCount != 42 {
			t.Errorf("ThoughtsTokenCount = %d, want 42", got.ThoughtsTokenCount)
		}
		if got.CachedContentTokenCount != 8 {
			t.Errorf("CachedContentTokenCount = %d, want 8", got.CachedContentTokenCount)
		}
		// Ensure the rest of the mapping is unchanged: a regression that
		// rewrites the field assignments could silently zero one of these.
		if got.PromptTokenCount != 10 || got.CandidatesTokenCount != 50 || got.TotalTokenCount != 60 {
			t.Errorf("usage mapping drifted: %#v", got)
		}
	})

	t.Run("missing details emit zero detail counts", func(t *testing.T) {
		// Providers that don't emit CompletionTokensDetails (Ollama,
		// vanilla GPT-4o) must keep detail counts at zero. genai's
		// `omitempty` then drops the fields on serialisation so the
		// observability backends don't see misleading zero metrics.
		usage := openai.CompletionUsage{
			PromptTokens:     1,
			CompletionTokens: 2,
			TotalTokens:      3,
		}
		got := convertUsageMetadata(usage)
		if got == nil {
			t.Fatalf("got nil")
		}
		if got.ThoughtsTokenCount != 0 {
			t.Errorf("ThoughtsTokenCount = %d, want 0", got.ThoughtsTokenCount)
		}
		if got.CachedContentTokenCount != 0 {
			t.Errorf("CachedContentTokenCount = %d, want 0", got.CachedContentTokenCount)
		}
	})
}
