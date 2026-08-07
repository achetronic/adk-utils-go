// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package openai

import (
	"testing"

	"google.golang.org/adk/v2/tool/toolconfirmation"
	"google.golang.org/genai"
)

// ADK's HITL confirmation protocol stores internal function calls and
// responses (name "adk_request_confirmation") in the session. They are
// framework bookkeeping, not tools the model declared; forwarding them makes
// OpenAI reject the request because tool messages must pair with tool_calls.
// The conversion must drop them while keeping the real tool exchange intact.
func TestConvertContentToMessages_DropsADKConfirmationParts(t *testing.T) {
	m := newModelForTest()

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

	msgs, err := m.convertContentToMessages(assistant)
	if err != nil {
		t.Fatalf("convertContentToMessages error: %v", err)
	}
	if len(msgs) != 1 {
		t.Fatalf("expected 1 assistant message, got %d", len(msgs))
	}
	calls := msgs[0].OfAssistant.ToolCalls
	if len(calls) != 1 {
		t.Fatalf("expected 1 tool call after filtering, got %d", len(calls))
	}
	if got := calls[0].OfFunction.Function.Name; got != "create_enrollment" {
		t.Errorf("remaining tool call name = %q, want %q", got, "create_enrollment")
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

	msgs, err = m.convertContentToMessages(user)
	if err != nil {
		t.Fatalf("convertContentToMessages error: %v", err)
	}
	if len(msgs) != 1 {
		t.Fatalf("expected 1 tool message after filtering, got %d", len(msgs))
	}
	if got := msgs[0].OfTool.ToolCallID; got != "call_real" {
		t.Errorf("remaining tool message ToolCallID = %q, want %q", got, "call_real")
	}
}

// A confirmation-only turn must produce no messages at all, so the replayed
// history carries no empty assistant/user shells for the framework parts.
func TestConvertContentToMessages_ConfirmationOnlyTurnProducesNothing(t *testing.T) {
	m := newModelForTest()

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

	msgs, err := m.convertContentToMessages(content)
	if err != nil {
		t.Fatalf("convertContentToMessages error: %v", err)
	}
	if len(msgs) != 0 {
		t.Errorf("expected no messages, got %d", len(msgs))
	}
}
