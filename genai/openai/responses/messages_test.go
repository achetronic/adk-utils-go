// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package responses

import (
	"testing"

	"github.com/openai/openai-go/v3/responses"
	"google.golang.org/genai"
)

// convertContentToInputItems bridges ADK Content into Responses API typed
// input items. A single Content may produce multiple items: text/media
// coalesce into a message, while FunctionCall and FunctionResponse become
// separate typed items.
func TestConvertContentToInputItems(t *testing.T) {
	cases := []struct {
		name      string
		content   *genai.Content
		wantCount int
		wantTypes []string // expected item type discriminators
		assert    func(t *testing.T, items []responses.ResponseInputItemUnionParam)
	}{
		{
			name: "user text becomes a single message item",
			content: &genai.Content{
				Role:  "user",
				Parts: []*genai.Part{{Text: "hello"}},
			},
			wantCount: 1,
			assert: func(t *testing.T, items []responses.ResponseInputItemUnionParam) {
				if items[0].OfMessage == nil {
					t.Errorf("expected EasyInputMessage, got %+v", items[0])
				}
			},
		},
		{
			name: "model text becomes an assistant message",
			content: &genai.Content{
				Role:  "model",
				Parts: []*genai.Part{{Text: "hi back"}},
			},
			wantCount: 1,
			assert: func(t *testing.T, items []responses.ResponseInputItemUnionParam) {
				if items[0].OfMessage == nil {
					t.Errorf("expected EasyInputMessage, got %+v", items[0])
				}
				if items[0].OfMessage.Role != responses.EasyInputMessageRoleAssistant {
					t.Errorf("role = %q, want assistant", items[0].OfMessage.Role)
				}
			},
		},
		{
			name: "function response becomes a function_call_output item",
			content: &genai.Content{
				Role: "user",
				Parts: []*genai.Part{
					{FunctionResponse: &genai.FunctionResponse{
						ID:       "call_42",
						Response: map[string]any{"ok": true},
					}},
				},
			},
			wantCount: 1,
			assert: func(t *testing.T, items []responses.ResponseInputItemUnionParam) {
				if items[0].OfFunctionCallOutput == nil {
					t.Errorf("expected FunctionCallOutput, got %+v", items[0])
				}
			},
		},
		{
			name: "model text plus function call produce two items",
			content: &genai.Content{
				Role: "model",
				Parts: []*genai.Part{
					{Text: "calling tool"},
					{FunctionCall: &genai.FunctionCall{
						ID:   "call_xyz",
						Name: "do_thing",
						Args: map[string]any{"foo": "bar"},
					}},
				},
			},
			wantCount: 2,
			assert: func(t *testing.T, items []responses.ResponseInputItemUnionParam) {
				if items[0].OfMessage == nil {
					t.Errorf("first item should be a message")
				}
				if items[1].OfFunctionCall == nil {
					t.Errorf("second item should be a function call")
				}
				if items[1].OfFunctionCall.Name != "do_thing" {
					t.Errorf("function name = %q, want do_thing", items[1].OfFunctionCall.Name)
				}
			},
		},
		{
			name: "user text plus inline image produces a single multi-part message",
			content: &genai.Content{
				Role: "user",
				Parts: []*genai.Part{
					{Text: "describe this"},
					{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte("fakepng")}},
				},
			},
			wantCount: 1,
			assert: func(t *testing.T, items []responses.ResponseInputItemUnionParam) {
				if items[0].OfMessage == nil {
					t.Errorf("expected message item")
				}
			},
		},
		{
			name: "function response then text produce two items",
			content: &genai.Content{
				Role: "model",
				Parts: []*genai.Part{
					{FunctionResponse: &genai.FunctionResponse{ID: "tool_1", Response: map[string]any{"result": "ok"}}},
					{Text: "all done"},
				},
			},
			wantCount: 2,
			assert: func(t *testing.T, items []responses.ResponseInputItemUnionParam) {
				if items[0].OfFunctionCallOutput == nil {
					t.Errorf("first item should be function_call_output")
				}
				if items[1].OfMessage == nil {
					t.Errorf("second item should be a message")
				}
			},
		},
		{
			name: "empty parts produce no items",
			content: &genai.Content{
				Role:  "user",
				Parts: []*genai.Part{{}},
			},
			wantCount: 0,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := convertContentToInputItems(c.content, "test-origin", nil)
			if err != nil {
				t.Fatalf("convertContentToInputItems error: %v", err)
			}
			if len(got) != c.wantCount {
				t.Fatalf("item count = %d, want %d", len(got), c.wantCount)
			}
			if c.assert != nil {
				c.assert(t, got)
			}
		})
	}
}

// Phase must survive the replay either way, keyed on the message ID:
// with one, the content replays as an OutputMessage carrying its full
// identity; without one it must fall back to a plain input message (an
// OutputMessage with an empty id is a request the API rejects), which
// carries phase itself. Dropping phase degrades GPT-5.3-Codex+ models.
func TestConvertContentToInputItems_PhasePreserved(t *testing.T) {
	t.Run("with message ID", func(t *testing.T) {
		content := &genai.Content{
			Role: "model",
			Parts: []*genai.Part{
				{
					Text:         "thinking...",
					PartMetadata: map[string]any{"phase": "commentary", "message_id": "msg_1"},
				},
			},
		}

		items, err := convertContentToInputItems(content, "test-origin", nil)
		if err != nil {
			t.Fatalf("error: %v", err)
		}
		if len(items) != 1 {
			t.Fatalf("expected 1 item, got %d", len(items))
		}
		msg := items[0].OfOutputMessage
		if msg == nil {
			t.Fatalf("expected OutputMessage for identity-carrying content, got %+v", items[0])
		}
		if msg.ID != "msg_1" {
			t.Errorf("ID = %q, want msg_1", msg.ID)
		}
		if msg.Phase != "commentary" {
			t.Errorf("Phase = %q, want commentary", msg.Phase)
		}
	})

	t.Run("phase only", func(t *testing.T) {
		content := &genai.Content{
			Role: "model",
			Parts: []*genai.Part{
				{
					Text:         "thinking...",
					PartMetadata: map[string]any{"phase": "commentary"},
				},
			},
		}

		items, err := convertContentToInputItems(content, "test-origin", nil)
		if err != nil {
			t.Fatalf("error: %v", err)
		}
		if len(items) != 1 {
			t.Fatalf("expected 1 item, got %d", len(items))
		}
		msg := items[0].OfMessage
		if msg == nil {
			t.Fatalf("expected a plain input message without an id, got %+v", items[0])
		}
		if msg.Phase != "commentary" {
			t.Errorf("Phase = %q, want commentary kept on the input message", msg.Phase)
		}
	})
}

// Thought parts without encrypted content reference server-side IDs that only
// resolve in the originating response, so they must be silently dropped to
// avoid "Item not found" errors in stateless flows.
func TestConvertContentToInputItems_ThoughtPartsWithoutEncryptedContentSkipped(t *testing.T) {
	content := &genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{
				Text:         "Let me think.",
				Thought:      true,
				PartMetadata: map[string]any{"reasoning_id": "rs-1"},
			},
			{Text: "The answer."},
		},
	}

	items, err := convertContentToInputItems(content, "test-origin", nil)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if len(items) != 1 {
		t.Fatalf("expected 1 item (thought skipped), got %d", len(items))
	}
	if items[0].OfMessage == nil {
		t.Fatalf("item should be message, got %+v", items[0])
	}
}

// Thought parts carrying encrypted content must be replayed as reasoning
// items, preceding the rest of the assistant turn: reasoning models require
// the reasoning item that led to a function_call to be present in the input.
func TestConvertContentToInputItems_ThoughtPartsWithEncryptedContentReplayed(t *testing.T) {
	content := &genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{
				Text:    "Let me think.",
				Thought: true,
				PartMetadata: map[string]any{
					"reasoning_id":      "rs-1",
					"encrypted_content": "enc-blob",
					"reasoning_origin":  "test-origin",
				},
			},
			{FunctionCall: &genai.FunctionCall{ID: "call-1", Name: "get_weather"}},
		},
	}

	items, err := convertContentToInputItems(content, "test-origin", nil)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if len(items) != 2 {
		t.Fatalf("expected 2 items (reasoning + function call), got %d", len(items))
	}
	reasoning := items[0].OfReasoning
	if reasoning == nil {
		t.Fatalf("first item should be a reasoning item, got %+v", items[0])
	}
	if reasoning.ID != "rs-1" {
		t.Errorf("reasoning ID = %q, want rs-1", reasoning.ID)
	}
	if !reasoning.EncryptedContent.Valid() || reasoning.EncryptedContent.Value != "enc-blob" {
		t.Errorf("EncryptedContent = %+v, want enc-blob", reasoning.EncryptedContent)
	}
	if len(reasoning.Summary) != 1 || reasoning.Summary[0].Text != "Let me think." {
		t.Errorf("Summary = %+v, want single summary with original text", reasoning.Summary)
	}
	if items[1].OfFunctionCall == nil {
		t.Fatalf("second item should be the function call, got %+v", items[1])
	}
}

// A reasoning item with encrypted content but no summary text (common for
// reasoning models) must still be replayed, with an empty summary list.
func TestConvertContentToInputItems_EncryptedThoughtWithoutSummary(t *testing.T) {
	content := &genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{
				Thought: true,
				PartMetadata: map[string]any{
					"reasoning_id":      "rs-2",
					"encrypted_content": "enc-blob-2",
					"reasoning_origin":  "test-origin",
				},
			},
			{Text: "The answer."},
		},
	}

	items, err := convertContentToInputItems(content, "test-origin", nil)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if len(items) != 2 {
		t.Fatalf("expected 2 items (reasoning + message), got %d", len(items))
	}
	reasoning := items[0].OfReasoning
	if reasoning == nil {
		t.Fatalf("first item should be a reasoning item, got %+v", items[0])
	}
	if len(reasoning.Summary) != 0 {
		t.Errorf("Summary = %+v, want empty", reasoning.Summary)
	}
}

// Encrypted reasoning content is bound to the provider/API key/model that
// produced it; replaying it through another channel fails with a 400
// invalid_encrypted_content. Thoughts whose recorded origin does not match
// the requesting channel must be skipped instead of replayed.
func TestConvertContentToInputItems_EncryptedThoughtFromOtherOriginSkipped(t *testing.T) {
	content := &genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{
				Text:    "Let me think.",
				Thought: true,
				PartMetadata: map[string]any{
					"reasoning_id":      "rs-1",
					"encrypted_content": "enc-blob",
					"reasoning_origin":  "azure-origin",
				},
			},
			{Text: "The answer."},
		},
	}

	items, err := convertContentToInputItems(content, "openai-origin", nil)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if len(items) != 1 {
		t.Fatalf("expected 1 item (foreign thought skipped), got %d", len(items))
	}
	if items[0].OfMessage == nil {
		t.Fatalf("item should be message, got %+v", items[0])
	}
}

// Thought parts recorded before origin tracking existed have encrypted
// content but no reasoning_origin; they must be skipped, not replayed on
// the assumption they match.
func TestConvertContentToInputItems_EncryptedThoughtWithoutOriginSkipped(t *testing.T) {
	content := &genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{
				Thought: true,
				PartMetadata: map[string]any{
					"reasoning_id":      "rs-1",
					"encrypted_content": "enc-blob",
				},
			},
			{Text: "The answer."},
		},
	}

	items, err := convertContentToInputItems(content, "test-origin", nil)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if len(items) != 1 {
		t.Fatalf("expected 1 item (origin-less thought skipped), got %d", len(items))
	}
}

// In multi-agent histories another agent's output (thoughts included) shows
// up under non-assistant roles, where a reasoning item is invalid input.
// Thought parts outside assistant turns must never be replayed.
func TestConvertContentToInputItems_ThoughtInUserTurnSkipped(t *testing.T) {
	content := &genai.Content{
		Role: "user",
		Parts: []*genai.Part{
			{
				Text:    "Copied agent reasoning.",
				Thought: true,
				PartMetadata: map[string]any{
					"reasoning_id":      "rs-1",
					"encrypted_content": "enc-blob",
					"reasoning_origin":  "test-origin",
				},
			},
			{Text: "User question."},
		},
	}

	items, err := convertContentToInputItems(content, "test-origin", nil)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if len(items) != 1 {
		t.Fatalf("expected 1 item (thought in user turn skipped), got %d", len(items))
	}
	if items[0].OfMessage == nil {
		t.Fatalf("item should be message, got %+v", items[0])
	}
}

// A replayed reasoning item must be followed by another item from the same
// turn or the API rejects it as dangling. A trailing thought (e.g. from an
// interrupted turn) must be skipped.
func TestConvertContentToInputItems_TrailingThoughtSkipped(t *testing.T) {
	content := &genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{
				Text:    "Thinking, then the turn was cut short.",
				Thought: true,
				PartMetadata: map[string]any{
					"reasoning_id":      "rs-1",
					"encrypted_content": "enc-blob",
					"reasoning_origin":  "test-origin",
				},
			},
		},
	}

	items, err := convertContentToInputItems(content, "test-origin", nil)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if len(items) != 0 {
		t.Fatalf("expected 0 items (dangling thought skipped), got %d", len(items))
	}
}

// Model output without phase should use the simpler EasyInputMessage path.
func TestConvertContentToInputItems_NoPhaseFallback(t *testing.T) {
	content := &genai.Content{
		Role:  "model",
		Parts: []*genai.Part{{Text: "plain response"}},
	}

	items, err := convertContentToInputItems(content, "test-origin", nil)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if len(items) != 1 {
		t.Fatalf("expected 1 item, got %d", len(items))
	}
	if items[0].OfMessage == nil {
		t.Fatalf("expected EasyInputMessage for non-phase content, got %+v", items[0])
	}
}

// Nil tool payloads must serialise as "{}", never "null": strict
// OpenAI-compatible parsers (Qwen on vLLM/llama.cpp) reject "null" where
// they expect an object.
func TestConvertContentToInputItems_NilToolPayloadsBecomeEmptyObject(t *testing.T) {
	content := &genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "ping"}},
			{FunctionResponse: &genai.FunctionResponse{ID: "call_1", Name: "ping"}},
		},
	}

	items, err := convertContentToInputItems(content, "test-origin", nil)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if len(items) != 2 {
		t.Fatalf("expected 2 items, got %d", len(items))
	}
	if got := items[0].OfFunctionCall.Arguments; got != "{}" {
		t.Errorf("Arguments = %q, want {}", got)
	}
	if got := items[1].OfFunctionCallOutput.Output.OfString.Value; got != "{}" {
		t.Errorf("Output = %q, want {}", got)
	}
}

// Parts from different output messages must be replayed as separate items
// with their own ID and phase. Coalescing them would relabel commentary as
// the final answer and drop the message identity models like gpt-5.3-codex
// expect to see again.
func TestConvertContentToInputItems_MessageBoundariesPreserved(t *testing.T) {
	content := &genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{
				Text:         "let me check",
				PartMetadata: map[string]any{"phase": "commentary", "message_id": "msg_1"},
			},
			{
				Text:         "the answer is 4",
				PartMetadata: map[string]any{"phase": "final_answer", "message_id": "msg_2"},
			},
		},
	}

	items, err := convertContentToInputItems(content, "test-origin", nil)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if len(items) != 2 {
		t.Fatalf("expected 2 items, got %d", len(items))
	}
	first, second := items[0].OfOutputMessage, items[1].OfOutputMessage
	if first == nil || second == nil {
		t.Fatalf("expected two OutputMessages, got %+v", items)
	}
	if first.ID != "msg_1" || second.ID != "msg_2" {
		t.Errorf("IDs = %q, %q, want msg_1, msg_2", first.ID, second.ID)
	}
	if first.Phase != "commentary" || second.Phase != "final_answer" {
		t.Errorf("phases = %q, %q, want commentary, final_answer", first.Phase, second.Phase)
	}
}

// Parts carrying the same message_id belong to one output message and must
// still coalesce into a single replayed item.
func TestConvertContentToInputItems_SameMessagePartsCoalesce(t *testing.T) {
	content := &genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{
				Text:         "first half",
				PartMetadata: map[string]any{"phase": "final_answer", "message_id": "msg_1"},
			},
			{
				Text:         "second half",
				PartMetadata: map[string]any{"phase": "final_answer", "message_id": "msg_1"},
			},
		},
	}

	items, err := convertContentToInputItems(content, "test-origin", nil)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if len(items) != 1 {
		t.Fatalf("expected 1 item, got %d", len(items))
	}
	msg := items[0].OfOutputMessage
	if msg == nil {
		t.Fatalf("expected an OutputMessage, got %+v", items[0])
	}
	if len(msg.Content) != 2 {
		t.Errorf("content parts = %d, want 2", len(msg.Content))
	}
}

// A message ID without phase must still replay as an OutputMessage so the
// item identity survives the stateless round trip.
func TestConvertContentToInputItems_MessageIDWithoutPhase(t *testing.T) {
	content := &genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{
				Text:         "plain but identified",
				PartMetadata: map[string]any{"message_id": "msg_9"},
			},
		},
	}

	items, err := convertContentToInputItems(content, "test-origin", nil)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if len(items) != 1 {
		t.Fatalf("expected 1 item, got %d", len(items))
	}
	msg := items[0].OfOutputMessage
	if msg == nil {
		t.Fatalf("expected an OutputMessage, got %+v", items[0])
	}
	if msg.ID != "msg_9" {
		t.Errorf("ID = %q, want msg_9", msg.ID)
	}
	if msg.Phase != "" {
		t.Errorf("Phase = %q, want empty", msg.Phase)
	}
}

// A refusal part must replay as a refusal content part with its message
// status, not as completed output_text.
func TestConvertContentToInputItems_RefusalAndStatusReplayed(t *testing.T) {
	content := &genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{
				Text: "I cannot help with that.",
				PartMetadata: map[string]any{
					"message_id": "msg_1",
					"status":     "incomplete",
					"refusal":    true,
				},
			},
		},
	}

	items, err := convertContentToInputItems(content, "test-origin", nil)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if len(items) != 1 {
		t.Fatalf("expected 1 item, got %d", len(items))
	}
	msg := items[0].OfOutputMessage
	if msg == nil {
		t.Fatalf("expected an OutputMessage, got %+v", items[0])
	}
	if msg.Status != "incomplete" {
		t.Errorf("Status = %q, want incomplete", msg.Status)
	}
	if len(msg.Content) != 1 || msg.Content[0].OfRefusal == nil {
		t.Fatalf("expected a refusal content part, got %+v", msg.Content)
	}
	if msg.Content[0].OfRefusal.Refusal != "I cannot help with that." {
		t.Errorf("Refusal text = %q", msg.Content[0].OfRefusal.Refusal)
	}
}

// Interleaved text and media must keep their original order: regrouping
// them would change which image a "describe the above" refers to.
func TestConvertContentToInputItems_InterleavedOrderPreserved(t *testing.T) {
	content := &genai.Content{
		Role: "user",
		Parts: []*genai.Part{
			{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte("a")}},
			{Text: "describe the image above"},
			{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte("b")}},
			{Text: "now compare both"},
		},
	}

	items, err := convertContentToInputItems(content, "test-origin", nil)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if len(items) != 1 {
		t.Fatalf("expected 1 item, got %d", len(items))
	}
	contents := items[0].OfMessage.Content.OfInputItemContentList
	if len(contents) != 4 {
		t.Fatalf("expected 4 content parts, got %d", len(contents))
	}
	wantKinds := []string{"image", "text", "image", "text"}
	for i, want := range wantKinds {
		isImage := contents[i].OfInputImage != nil
		isText := contents[i].OfInputText != nil
		if (want == "image") != isImage || (want == "text") != isText {
			t.Errorf("content[%d]: want %s, got image=%v text=%v", i, want, isImage, isText)
		}
	}
}

// An identity-bearing assistant message that also holds media keeps the
// media through the content-list path: output message content cannot carry
// media, and dropping parts silently would change the prompt.
func TestConvertContentToInputItems_IdentityWithMediaKeepsMedia(t *testing.T) {
	content := &genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{
				Text:         "here is the image",
				PartMetadata: map[string]any{"message_id": "msg_1", "phase": "final_answer"},
			},
			{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte("x")}},
		},
	}

	items, err := convertContentToInputItems(content, "test-origin", nil)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if len(items) != 1 {
		t.Fatalf("expected 1 item, got %d", len(items))
	}
	msg := items[0].OfMessage
	if msg == nil {
		t.Fatalf("expected the content-list message path, got %+v", items[0])
	}
	contents := msg.Content.OfInputItemContentList
	if len(contents) != 2 || contents[0].OfInputText == nil || contents[1].OfInputImage == nil {
		t.Fatalf("expected text plus image in order, got %+v", contents)
	}
	if msg.Phase != "final_answer" {
		t.Errorf("Phase = %q, want final_answer kept through the degradation", msg.Phase)
	}
}

// The reverse order: media first, then identity-bearing text. The media-only
// prefix has no identity yet and must adopt the incoming one instead of
// splitting off into a separate phase-less message.
func TestConvertContentToInputItems_MediaBeforeIdentityText(t *testing.T) {
	content := &genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte("x")}},
			{
				Text:         "the chart above says it all",
				PartMetadata: map[string]any{"message_id": "msg_1", "phase": "final_answer"},
			},
		},
	}

	items, err := convertContentToInputItems(content, "test-origin", nil)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if len(items) != 1 {
		t.Fatalf("expected 1 item, got %d: %+v", len(items), items)
	}
	msg := items[0].OfMessage
	if msg == nil {
		t.Fatalf("expected the content-list message path, got %+v", items[0])
	}
	if msg.Phase != "final_answer" {
		t.Errorf("Phase = %q, want final_answer adopted by the whole message", msg.Phase)
	}
	contents := msg.Content.OfInputItemContentList
	if len(contents) != 2 || contents[0].OfInputImage == nil || contents[1].OfInputText == nil {
		t.Fatalf("expected image then text, got %+v", contents)
	}
}

// Phase metadata under a user role (multi-agent histories relabel other
// agents' output) must not reach the wire: the API rejects phase on user
// messages.
func TestConvertContentToInputItems_UserRoleSuppressesPhase(t *testing.T) {
	content := &genai.Content{
		Role: "user",
		Parts: []*genai.Part{
			{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte("x")}},
			{
				Text:         "another agent said this",
				PartMetadata: map[string]any{"message_id": "msg_1", "phase": "final_answer"},
			},
		},
	}

	items, err := convertContentToInputItems(content, "test-origin", nil)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if len(items) != 1 {
		t.Fatalf("expected 1 item, got %d", len(items))
	}
	msg := items[0].OfMessage
	if msg == nil {
		t.Fatalf("expected the content-list message path, got %+v", items[0])
	}
	if msg.Phase != "" {
		t.Errorf("Phase = %q, want empty on a user message", msg.Phase)
	}
	if msg.Role != "user" {
		t.Errorf("Role = %q, want user", msg.Role)
	}
}

// A dropped dangling call no longer counts as a follower: the reasoning
// item that led to it must be skipped too, or the API rejects it as
// dangling in turn.
func TestConvertContentToInputItems_DanglingCallExcludedFromFollowers(t *testing.T) {
	content := &genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{
				Text:    "planning",
				Thought: true,
				PartMetadata: map[string]any{
					"reasoning_id":      "rs_1",
					"encrypted_content": "enc-blob",
					"reasoning_origin":  "test-origin",
				},
			},
			{FunctionCall: &genai.FunctionCall{ID: "call_cancelled", Name: "slow_tool"}},
		},
	}

	items, err := convertContentToInputItems(content, "test-origin", map[string]bool{"call_cancelled": true})
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if len(items) != 0 {
		t.Fatalf("expected 0 items (call dropped, reasoning left without a follower), got %d: %+v", len(items), items)
	}
}
