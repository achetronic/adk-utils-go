// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package anthropic

import (
	"encoding/json"
	"strings"
	"testing"

	"google.golang.org/genai"
)

// nil Response must encode to "{}" in the tool_result block, never "null".
func TestConvertContentToMessage_NilFunctionResponseBecomesEmptyObject(t *testing.T) {
	m := &Model{modelName: "test-model"}
	content := &genai.Content{
		Role: "user",
		Parts: []*genai.Part{
			{FunctionResponse: &genai.FunctionResponse{ID: "toolu_1"}}, // Response is nil
		},
	}

	msg, err := m.convertContentToMessage(content)
	if err != nil {
		t.Fatalf("convertContentToMessage error: %v", err)
	}
	if msg == nil || len(msg.Content) != 1 {
		t.Fatalf("expected a message with 1 block, got %#v", msg)
	}

	wire, err := json.Marshal(msg.Content[0])
	if err != nil {
		t.Fatalf("marshalling tool_result block: %v", err)
	}
	if !strings.Contains(string(wire), "{}") {
		t.Errorf("tool_result content does not carry %q: %s", "{}", wire)
	}
	if strings.Contains(string(wire), "null") {
		t.Errorf("tool_result content leaked a \"null\" payload: %s", wire)
	}
}

// Canary: a nil Response must stay nil; the converter must not mutate its input.
func TestConvertContentToMessage_DoesNotMutateNilResponse(t *testing.T) {
	m := &Model{modelName: "test-model"}
	resp := &genai.FunctionResponse{ID: "toolu_1"} // Response nil
	content := &genai.Content{
		Role:  "user",
		Parts: []*genai.Part{{FunctionResponse: resp}},
	}

	if _, err := m.convertContentToMessage(content); err != nil {
		t.Fatalf("convertContentToMessage error: %v", err)
	}

	if resp.Response != nil {
		t.Errorf("FunctionResponse.Response was mutated to %#v, want it left nil", resp.Response)
	}
}
