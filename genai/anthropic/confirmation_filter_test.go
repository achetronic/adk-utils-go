// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package anthropic

import (
	"reflect"
	"testing"

	"google.golang.org/adk/v2/tool/toolconfirmation"
	"google.golang.org/genai"
)

// ADK's HITL confirmation protocol stores internal function calls and
// responses (name "adk_request_confirmation") in the session. They are
// framework bookkeeping, not tools the model declared; replaying them breaks
// tool_use/tool_result pairing on the wire. The conversion must drop them
// while keeping the real tool exchange intact.
func TestConvertContentToMessage_DropsADKConfirmationParts(t *testing.T) {
	m := &Model{}

	assistant := &genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{FunctionCall: &genai.FunctionCall{
				ID:   "call_real",
				Name: "create_enrollment",
				Args: map[string]any{"email": "a@b.c"},
			}},
			{FunctionCall: &genai.FunctionCall{
				ID:   "adk_confirm_1",
				Name: toolconfirmation.FunctionCallName,
				Args: map[string]any{"confirmed": true},
			}},
		},
	}

	msg, err := m.convertContentToMessage(assistant)
	if err != nil {
		t.Fatalf("convertContentToMessage error: %v", err)
	}
	if msg == nil {
		t.Fatal("expected a message, got nil")
	}
	if got, want := blockTypes(msg.Content), []string{"tool_use"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("block types = %v, want %v", got, want)
	}
	if got := msg.Content[0].OfToolUse.Name; got != "create_enrollment" {
		t.Errorf("remaining tool_use name = %q, want %q", got, "create_enrollment")
	}

	user := &genai.Content{
		Role: "user",
		Parts: []*genai.Part{
			{FunctionResponse: &genai.FunctionResponse{
				ID:       "adk_confirm_1",
				Name:     toolconfirmation.FunctionCallName,
				Response: map[string]any{"confirmed": true},
			}},
			{FunctionResponse: &genai.FunctionResponse{
				ID:       "call_real",
				Name:     "create_enrollment",
				Response: map[string]any{"ok": true},
			}},
		},
	}

	msg, err = m.convertContentToMessage(user)
	if err != nil {
		t.Fatalf("convertContentToMessage error: %v", err)
	}
	if msg == nil {
		t.Fatal("expected a message, got nil")
	}
	if got, want := blockTypes(msg.Content), []string{"tool_result"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("block types = %v, want %v", got, want)
	}
	if got := msg.Content[0].OfToolResult.ToolUseID; got != sanitizeToolID("call_real") {
		t.Errorf("remaining tool_result ToolUseID = %q, want %q", got, sanitizeToolID("call_real"))
	}
}

// A confirmation-only turn must yield no message (nil, nil), matching the
// empty-content contract the caller relies on to drop the turn.
func TestConvertContentToMessage_ConfirmationOnlyTurnProducesNothing(t *testing.T) {
	m := &Model{}

	content := &genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{FunctionCall: &genai.FunctionCall{
				ID:   "adk_confirm_1",
				Name: toolconfirmation.FunctionCallName,
				Args: map[string]any{},
			}},
		},
	}

	msg, err := m.convertContentToMessage(content)
	if err != nil {
		t.Fatalf("convertContentToMessage error: %v", err)
	}
	if msg != nil {
		t.Errorf("expected nil message, got %#v", msg)
	}
}
