// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package openai

import (
	"testing"

	"google.golang.org/genai"
)

// The helper's own contract is tested in genai/common. These pin that the
// OpenAI converter actually routes both tool sides through it.

// nil Args (a tool with no parameters, e.g. exit_loop) must encode to "{}".
func TestConvertContentToMessages_NilFunctionCallArgsBecomeEmptyObject(t *testing.T) {
	m := newModelForTest()
	content := &genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "exit_loop"}}, // Args is nil
		},
	}

	msgs, err := m.convertContentToMessages(content)
	if err != nil {
		t.Fatalf("convertContentToMessages error: %v", err)
	}
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %d", len(msgs))
	}

	tc := msgs[0].OfAssistant.ToolCalls[0].OfFunction
	if got := tc.Function.Arguments; got != "{}" {
		t.Errorf("arguments = %q, want %q (a nil Args map must not serialise to \"null\")", got, "{}")
	}
}

// Same on the response side: nil Response must encode to "{}".
func TestConvertContentToMessages_NilFunctionResponseBecomesEmptyObject(t *testing.T) {
	m := newModelForTest()
	content := &genai.Content{
		Role: "user",
		Parts: []*genai.Part{
			{FunctionResponse: &genai.FunctionResponse{ID: "call_1"}}, // Response is nil
		},
	}

	msgs, err := m.convertContentToMessages(content)
	if err != nil {
		t.Fatalf("convertContentToMessages error: %v", err)
	}
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %d", len(msgs))
	}

	if got := msgs[0].OfTool.Content.OfString.Value; got != "{}" {
		t.Errorf("tool content = %q, want %q (a nil Response map must not serialise to \"null\")", got, "{}")
	}
}

// Canary: the converter must not mutate the caller's Content (a shared Content
// would corrupt and concurrent conversions would race).
func TestConvertContentToMessages_DoesNotMutateInputPayloads(t *testing.T) {
	m := newModelForTest()
	call := &genai.FunctionCall{ID: "call_1", Name: "exit_loop"} // Args nil
	resp := &genai.FunctionResponse{ID: "call_1"}                // Response nil
	content := &genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{FunctionCall: call},
			{FunctionResponse: resp},
		},
	}

	if _, err := m.convertContentToMessages(content); err != nil {
		t.Fatalf("convertContentToMessages error: %v", err)
	}

	if call.Args != nil {
		t.Errorf("FunctionCall.Args was mutated to %#v, want it left nil", call.Args)
	}
	if resp.Response != nil {
		t.Errorf("FunctionResponse.Response was mutated to %#v, want it left nil", resp.Response)
	}
}
