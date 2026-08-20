// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package responses

import (
	"testing"

	"google.golang.org/adk/v2/tool/toolconfirmation"
	"google.golang.org/genai"
)

// ADK's HITL confirmation protocol stores internal function calls and
// responses (name "adk_request_confirmation") in the session. They are
// framework bookkeeping, not tools the model declared; replaying them makes
// the API reject the request over undeclared call/output items. The
// conversion must drop them while keeping the real tool exchange intact.
func TestConvertContentToInputItems_DropsADKConfirmationParts(t *testing.T) {
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

	items, err := convertContentToInputItems(assistant, "test-origin", nil)
	if err != nil {
		t.Fatalf("convertContentToInputItems error: %v", err)
	}
	if len(items) != 1 {
		t.Fatalf("expected 1 item after filtering, got %d: %+v", len(items), items)
	}
	if items[0].OfFunctionCall == nil {
		t.Fatalf("expected a function call item, got %+v", items[0])
	}
	if got := items[0].OfFunctionCall.Name; got != "create_enrollment" {
		t.Errorf("remaining function call name = %q, want %q", got, "create_enrollment")
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

	items, err = convertContentToInputItems(user, "test-origin", nil)
	if err != nil {
		t.Fatalf("convertContentToInputItems error: %v", err)
	}
	if len(items) != 1 {
		t.Fatalf("expected 1 item after filtering, got %d: %+v", len(items), items)
	}
	if items[0].OfFunctionCallOutput == nil {
		t.Fatalf("expected a function call output item, got %+v", items[0])
	}
	if got := items[0].OfFunctionCallOutput.CallID; got != "call_real" {
		t.Errorf("remaining output CallID = %q, want %q", got, "call_real")
	}
}

// A confirmation-only turn must produce no items at all, so the replayed
// history carries no leftovers of the framework exchange.
func TestConvertContentToInputItems_ConfirmationOnlyTurnProducesNothing(t *testing.T) {
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

	items, err := convertContentToInputItems(content, "test-origin", nil)
	if err != nil {
		t.Fatalf("convertContentToInputItems error: %v", err)
	}
	if len(items) != 0 {
		t.Errorf("expected no items, got %d: %+v", len(items), items)
	}
}

// A filtered confirmation call produces no item, so it cannot serve as the
// same-turn follower a replayed reasoning item needs. The reasoning must be
// skipped with it, or the API rejects the request over a dangling item.
func TestConvertContentToInputItems_ConfirmationCallExcludedFromFollowers(t *testing.T) {
	content := &genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{
				Text:    "deciding whether to ask",
				Thought: true,
				PartMetadata: map[string]any{
					"reasoning_id":      "rs_1",
					"encrypted_content": "enc-blob",
					"reasoning_origin":  "test-origin",
				},
			},
			{FunctionCall: &genai.FunctionCall{
				ID:   "adk_confirm_1",
				Name: toolconfirmation.FunctionCallName,
			}},
		},
	}

	items, err := convertContentToInputItems(content, "test-origin", nil)
	if err != nil {
		t.Fatalf("convertContentToInputItems error: %v", err)
	}
	if len(items) != 0 {
		t.Fatalf("expected 0 items (call dropped, reasoning left without a follower), got %d: %+v", len(items), items)
	}
}
