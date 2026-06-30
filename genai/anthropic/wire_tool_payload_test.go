// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package anthropic

import (
	"testing"

	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"
)

// On the wire, nil tool payloads must serialise to {} (a JSON object), never
// the literal null. A tool_use turn followed by its tool_result keeps
// repairMessageHistory from dropping the call, so both blocks reach the body.
func TestWireBody_NilToolPayloads(t *testing.T) {
	req := &model.LLMRequest{
		Config: &genai.GenerateContentConfig{},
		Contents: []*genai.Content{
			{Role: "model", Parts: []*genai.Part{
				{FunctionCall: &genai.FunctionCall{ID: "toolu_1", Name: "exit_loop"}},
			}},
			{Role: "user", Parts: []*genai.Part{
				{FunctionResponse: &genai.FunctionResponse{ID: "toolu_1"}},
			}},
		},
	}
	body := captureBodyFor(t, Config{}, req)

	var input, result any
	messages, _ := body["messages"].([]any)
	for _, raw := range messages {
		msg, _ := raw.(map[string]any)
		blocks, _ := msg["content"].([]any)
		for _, b := range blocks {
			block, _ := b.(map[string]any)
			switch block["type"] {
			case "tool_use":
				input = block["input"]
			case "tool_result":
				result = block["content"]
			}
		}
	}

	if input == nil {
		t.Fatal("no tool_use block on the wire")
	}
	if m, ok := input.(map[string]any); !ok || len(m) != 0 {
		t.Errorf("tool_use.input = %#v, want empty object {}", input)
	}

	if result == nil {
		t.Fatal("no tool_result block on the wire")
	}
	// The SDK sends tool_result content as an array of text blocks; the payload
	// is the text of the single block.
	blocks, ok := result.([]any)
	if !ok || len(blocks) != 1 {
		t.Fatalf("tool_result.content = %#v, want a one-block array", result)
	}
	block, _ := blocks[0].(map[string]any)
	if block["text"] != "{}" {
		t.Errorf("tool_result text = %#v, want empty object string", block["text"])
	}
}
