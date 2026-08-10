// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package responses

import (
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"
)

// convertResponse rebuilds a genai.LLMResponse from the Responses API
// Response. Tests build the response via json.Unmarshal because the SDK
// populates internal JSON metadata through its generated UnmarshalJSON,
// which typed union dispatch (AsAny) depends on.

func TestConvertResponse_EmptyOutput(t *testing.T) {
	resp := &responses.Response{}
	_, err := convertResponse(resp, "test-origin")
	if !errors.Is(err, ErrNoOutputInResponse) {
		t.Errorf("err = %v, want %v", err, ErrNoOutputInResponse)
	}
}

// A response whose output items are all of unknown types must fail loud
// instead of passing as a completed turn with empty content; the error
// names the skipped types so the operator can see what was dropped.
func TestConvertResponse_UnknownOnlyOutput(t *testing.T) {
	raw := []byte(`{
		"id": "resp-u1",
		"status": "completed",
		"output": [
			{"type": "web_search_call", "id": "ws-1", "status": "completed"},
			{"type": "image_generation_call", "id": "ig-1", "status": "completed"}
		]
	}`)

	var resp responses.Response
	if err := json.Unmarshal(raw, &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	_, err := convertResponse(&resp, "test-origin")
	if !errors.Is(err, ErrNoConsumableOutput) {
		t.Fatalf("err = %v, want ErrNoConsumableOutput", err)
	}
	if !strings.Contains(err.Error(), "image_generation_call, web_search_call") {
		t.Errorf("err = %q, want the sorted skipped item types listed", err)
	}
}

// Unknown item types alongside consumable ones are skipped silently: the
// forward-compatibility contract is to ignore extensions, not to fail.
func TestConvertResponse_UnknownAlongsideKnown(t *testing.T) {
	raw := []byte(`{
		"id": "resp-u2",
		"status": "completed",
		"output": [
			{"type": "web_search_call", "id": "ws-1", "status": "completed"},
			{"type": "message", "id": "msg-1", "role": "assistant", "status": "completed",
				"content": [{"type": "output_text", "text": "found it"}]}
		]
	}`)

	var resp responses.Response
	if err := json.Unmarshal(raw, &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	got, err := convertResponse(&resp, "test-origin")
	if err != nil {
		t.Fatalf("convertResponse: %v", err)
	}
	if len(got.Content.Parts) != 1 || got.Content.Parts[0].Text != "found it" {
		t.Errorf("Content parts = %#v, want the single message text", got.Content)
	}
}

func TestConvertResponse_TextOnly(t *testing.T) {
	raw := []byte(`{
		"id": "resp-1",
		"status": "completed",
		"output": [{
			"type": "message",
			"id": "msg-1",
			"role": "assistant",
			"status": "completed",
			"content": [{"type": "output_text", "text": "hello world"}]
		}],
		"usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
			"input_tokens_details": {"cached_tokens": 0},
			"output_tokens_details": {"reasoning_tokens": 0}}
	}`)

	var resp responses.Response
	if err := json.Unmarshal(raw, &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	got, err := convertResponse(&resp, "test-origin")
	if err != nil {
		t.Fatalf("convertResponse: %v", err)
	}
	if !got.TurnComplete {
		t.Errorf("TurnComplete = false, want true")
	}
	if got.FinishReason != genai.FinishReasonStop {
		t.Errorf("FinishReason = %v, want Stop", got.FinishReason)
	}
	if len(got.Content.Parts) != 1 || got.Content.Parts[0].Text != "hello world" {
		t.Errorf("Content parts = %#v, want single text", got.Content)
	}
	if got.UsageMetadata == nil || got.UsageMetadata.TotalTokenCount != 15 {
		t.Errorf("UsageMetadata = %#v, want TotalTokenCount=15", got.UsageMetadata)
	}
}

func TestConvertResponse_ToolCall(t *testing.T) {
	raw := []byte(`{
		"id": "resp-2",
		"status": "completed",
		"output": [{
			"type": "function_call",
			"id": "fc-1",
			"call_id": "call_42",
			"name": "search",
			"arguments": "{\"q\":\"weather\"}"
		}],
		"usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
			"input_tokens_details": {"cached_tokens": 0},
			"output_tokens_details": {"reasoning_tokens": 0}}
	}`)

	var resp responses.Response
	if err := json.Unmarshal(raw, &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	got, err := convertResponse(&resp, "test-origin")
	if err != nil {
		t.Fatalf("convertResponse: %v", err)
	}
	if len(got.Content.Parts) != 1 {
		t.Fatalf("expected 1 part, got %d", len(got.Content.Parts))
	}
	fc := got.Content.Parts[0].FunctionCall
	if fc == nil {
		t.Fatalf("expected FunctionCall")
	}
	if fc.ID != "call_42" || fc.Name != "search" {
		t.Errorf("FunctionCall = %#v, want id=call_42 name=search", fc)
	}
	if want := map[string]any{"q": "weather"}; !reflect.DeepEqual(fc.Args, want) {
		t.Errorf("Args = %#v, want %#v", fc.Args, want)
	}
}

func TestConvertResponse_TextPlusToolCall(t *testing.T) {
	raw := []byte(`{
		"id": "resp-3",
		"status": "completed",
		"output": [
			{
				"type": "message",
				"id": "msg-1",
				"role": "assistant",
				"status": "completed",
				"content": [{"type": "output_text", "text": "looking up"}]
			},
			{
				"type": "function_call",
				"id": "fc-1",
				"call_id": "call_1",
				"name": "get_weather",
				"arguments": "{\"city\":\"Madrid\"}"
			}
		],
		"usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
			"input_tokens_details": {"cached_tokens": 0},
			"output_tokens_details": {"reasoning_tokens": 0}}
	}`)

	var resp responses.Response
	if err := json.Unmarshal(raw, &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	got, err := convertResponse(&resp, "test-origin")
	if err != nil {
		t.Fatalf("convertResponse: %v", err)
	}
	if len(got.Content.Parts) != 2 {
		t.Fatalf("expected 2 parts, got %d", len(got.Content.Parts))
	}
	if got.Content.Parts[0].Text != "looking up" {
		t.Errorf("first part text = %q", got.Content.Parts[0].Text)
	}
	if got.Content.Parts[1].FunctionCall == nil {
		t.Errorf("second part should be a FunctionCall")
	}
}

// Reasoning items (ResponseReasoningItem) appear before messages in the
// output array. Their summary texts become Parts with Thought=true,
// preserving the temporal order.
func TestConvertResponse_WithReasoning(t *testing.T) {
	raw := []byte(`{
		"id": "resp-4",
		"status": "completed",
		"output": [
			{
				"type": "reasoning",
				"id": "rs-1",
				"summary": [{"type": "summary_text", "text": "Let me think about this."}]
			},
			{
				"type": "message",
				"id": "msg-1",
				"role": "assistant",
				"status": "completed",
				"content": [{"type": "output_text", "text": "The answer is 42."}]
			}
		],
		"usage": {"input_tokens": 5, "output_tokens": 10, "total_tokens": 15,
			"input_tokens_details": {"cached_tokens": 0},
			"output_tokens_details": {"reasoning_tokens": 7}}
	}`)

	var resp responses.Response
	if err := json.Unmarshal(raw, &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	got, err := convertResponse(&resp, "test-origin")
	if err != nil {
		t.Fatalf("convertResponse: %v", err)
	}
	if len(got.Content.Parts) != 2 {
		t.Fatalf("expected 2 parts, got %d", len(got.Content.Parts))
	}
	if !got.Content.Parts[0].Thought || got.Content.Parts[0].Text != "Let me think about this." {
		t.Errorf("first part should be a thought: %#v", got.Content.Parts[0])
	}
	if got.Content.Parts[1].Thought || got.Content.Parts[1].Text != "The answer is 42." {
		t.Errorf("second part should be plain text: %#v", got.Content.Parts[1])
	}
	if got.UsageMetadata == nil || got.UsageMetadata.ThoughtsTokenCount != 7 {
		t.Errorf("ThoughtsTokenCount = %v, want 7", got.UsageMetadata)
	}
	// Reasoning parts must carry reasoning_id for stateless round-tripping
	pm := got.Content.Parts[0].PartMetadata
	if pm == nil || pm["reasoning_id"] != "rs-1" {
		t.Errorf("PartMetadata = %v, want reasoning_id=rs-1", pm)
	}
}

// A reasoning item with encrypted content but an empty summary (common for
// reasoning models) must still produce a thought part: dropping it would
// lose the encrypted content needed to replay the item on the next turn.
func TestConvertResponse_EncryptedReasoningWithoutSummary(t *testing.T) {
	raw := []byte(`{
		"id": "resp-6",
		"status": "completed",
		"output": [
			{
				"type": "reasoning",
				"id": "rs-1",
				"summary": [],
				"encrypted_content": "enc-blob"
			},
			{
				"type": "message",
				"id": "msg-1",
				"role": "assistant",
				"status": "completed",
				"content": [{"type": "output_text", "text": "The answer is 42."}]
			}
		],
		"usage": {"input_tokens": 5, "output_tokens": 10, "total_tokens": 15,
			"input_tokens_details": {"cached_tokens": 0},
			"output_tokens_details": {"reasoning_tokens": 7}}
	}`)

	var resp responses.Response
	if err := json.Unmarshal(raw, &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	got, err := convertResponse(&resp, "test-origin")
	if err != nil {
		t.Fatalf("convertResponse: %v", err)
	}
	if len(got.Content.Parts) != 2 {
		t.Fatalf("expected 2 parts, got %d", len(got.Content.Parts))
	}
	thought := got.Content.Parts[0]
	if !thought.Thought || thought.Text != "" {
		t.Errorf("first part should be an empty-text thought: %#v", thought)
	}
	pm := thought.PartMetadata
	if pm == nil || pm["reasoning_id"] != "rs-1" || pm["encrypted_content"] != "enc-blob" {
		t.Errorf("PartMetadata = %v, want reasoning_id=rs-1 and encrypted_content=enc-blob", pm)
	}
	// The origin must be recorded so replay is restricted to the channel
	// that produced the encrypted content.
	if pm["reasoning_origin"] != "test-origin" {
		t.Errorf("reasoning_origin = %v, want test-origin", pm["reasoning_origin"])
	}
}

// The origin fingerprint gates encrypted-reasoning replay to the channel
// that produced it: it must be stable for identical configs and change when
// any of base URL, API key, or model changes.
func TestComputeOrigin(t *testing.T) {
	base := computeOrigin("https://api.openai.com/v1", "sk-a", "gpt-5.5")

	if computeOrigin("https://api.openai.com/v1", "sk-a", "gpt-5.5") != base {
		t.Errorf("origin is not stable for identical inputs")
	}
	variants := map[string]string{
		"base URL": computeOrigin("https://azure.example.com/v1", "sk-a", "gpt-5.5"),
		"API key":  computeOrigin("https://api.openai.com/v1", "sk-b", "gpt-5.5"),
		"model":    computeOrigin("https://api.openai.com/v1", "sk-a", "gpt-5.6"),
	}
	for dim, got := range variants {
		if got == base {
			t.Errorf("origin did not change when %s changed", dim)
		}
	}
	// Field boundaries must be unambiguous: shifting a character across the
	// URL/key boundary must not collide.
	if computeOrigin("ab", "c", "m") == computeOrigin("a", "bc", "m") {
		t.Errorf("origin collides across field boundaries")
	}
}

// Every request must run statelessly: store=false so nothing persists
// server-side, and encrypted reasoning requested so reasoning items can be
// replayed across turns.
func TestBuildResponseParams_StatelessDefaults(t *testing.T) {
	m := New(Config{APIKey: "test", ModelName: "gpt-5.5"})

	params, err := m.buildResponseParams(&model.LLMRequest{})
	if err != nil {
		t.Fatalf("buildResponseParams: %v", err)
	}
	if !params.Store.Valid() || params.Store.Value {
		t.Errorf("Store = %+v, want false", params.Store)
	}
	found := false
	for _, inc := range params.Include {
		if inc == responses.ResponseIncludableReasoningEncryptedContent {
			found = true
		}
	}
	if !found {
		t.Errorf("Include = %v, want it to contain reasoning.encrypted_content", params.Include)
	}
}

// Phase metadata on assistant messages must be preserved in PartMetadata
// so that subsequent requests can echo it back to the model.
func TestConvertResponse_PhaseMetadata(t *testing.T) {
	raw := []byte(`{
		"id": "resp-5",
		"status": "completed",
		"output": [{
			"type": "message",
			"id": "msg-1",
			"role": "assistant",
			"status": "completed",
			"phase": "commentary",
			"content": [{"type": "output_text", "text": "intermediate"}]
		}],
		"usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
			"input_tokens_details": {"cached_tokens": 0},
			"output_tokens_details": {"reasoning_tokens": 0}}
	}`)

	var resp responses.Response
	if err := json.Unmarshal(raw, &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	got, err := convertResponse(&resp, "test-origin")
	if err != nil {
		t.Fatalf("convertResponse: %v", err)
	}
	if len(got.Content.Parts) != 1 {
		t.Fatalf("expected 1 part, got %d", len(got.Content.Parts))
	}
	pm := got.Content.Parts[0].PartMetadata
	if pm == nil {
		t.Fatalf("PartMetadata is nil, want phase and message_id")
	}
	if pm["phase"] != "commentary" {
		t.Errorf("PartMetadata[\"phase\"] = %v, want commentary", pm["phase"])
	}
	if pm["message_id"] != "msg-1" {
		t.Errorf("PartMetadata[\"message_id\"] = %v, want msg-1", pm["message_id"])
	}
}

// Messages without a phase still carry message_id in PartMetadata for
// round-tripping, but must not have a "phase" key.
func TestConvertResponse_NoPhase(t *testing.T) {
	raw := []byte(`{
		"id": "resp-6",
		"status": "completed",
		"output": [{
			"type": "message",
			"id": "msg-1",
			"role": "assistant",
			"status": "completed",
			"content": [{"type": "output_text", "text": "plain"}]
		}],
		"usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
			"input_tokens_details": {"cached_tokens": 0},
			"output_tokens_details": {"reasoning_tokens": 0}}
	}`)

	var resp responses.Response
	if err := json.Unmarshal(raw, &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	got, err := convertResponse(&resp, "test-origin")
	if err != nil {
		t.Fatalf("convertResponse: %v", err)
	}
	pm := got.Content.Parts[0].PartMetadata
	if pm != nil {
		if _, hasPhase := pm["phase"]; hasPhase {
			t.Errorf("PartMetadata has phase key, want absent for messages without phase")
		}
		if pm["message_id"] != "msg-1" {
			t.Errorf("message_id = %v, want msg-1", pm["message_id"])
		}
	}
}

func TestConvertUsageMetadata(t *testing.T) {
	t.Run("populated usage maps correctly", func(t *testing.T) {
		raw := []byte(`{
			"input_tokens": 11, "output_tokens": 22, "total_tokens": 33,
			"input_tokens_details": {"cached_tokens": 5},
			"output_tokens_details": {"reasoning_tokens": 8}
		}`)
		var usage responses.ResponseUsage
		if err := json.Unmarshal(raw, &usage); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		got := convertUsageMetadata(usage)
		if got == nil {
			t.Fatalf("got nil")
		}
		want := &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:        11,
			CandidatesTokenCount:    22,
			TotalTokenCount:         33,
			ThoughtsTokenCount:      8,
			CachedContentTokenCount: 5,
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("got = %#v, want %#v", got, want)
		}
	})

	t.Run("zero total tokens returns nil", func(t *testing.T) {
		usage := responses.ResponseUsage{}
		if got := convertUsageMetadata(usage); got != nil {
			t.Errorf("got = %#v, want nil", got)
		}
	})
}

// Conversion failures in the generation config must propagate out of
// buildResponseParams instead of being swallowed, so callers see the broken
// tool definition immediately.
func TestBuildResponseParams_InvalidToolPropagates(t *testing.T) {
	m := New(Config{APIKey: "test", ModelName: "gpt-5.5"})
	req := &model.LLMRequest{
		Config: &genai.GenerateContentConfig{
			Tools: []*genai.Tool{{
				FunctionDeclarations: []*genai.FunctionDeclaration{{
					Name:                 "broken",
					ParametersJsonSchema: make(chan int),
				}},
			}},
		},
	}

	if _, err := m.buildResponseParams(req); err == nil {
		t.Fatalf("buildResponseParams() error = nil, want tool conversion error")
	}
}

// Reasoning summaries are only requested when the caller asks for thoughts;
// the effort level maps regardless.
func TestBuildResponseParams_ReasoningSummaryFollowsIncludeThoughts(t *testing.T) {
	m := New(Config{APIKey: "test", ModelName: "gpt-5.5"})

	req := &model.LLMRequest{Config: &genai.GenerateContentConfig{
		ThinkingConfig: &genai.ThinkingConfig{
			IncludeThoughts: true,
			ThinkingLevel:   genai.ThinkingLevelLow,
		},
	}}
	params, err := m.buildResponseParams(req)
	if err != nil {
		t.Fatalf("buildResponseParams: %v", err)
	}
	if params.Reasoning.Summary != shared.ReasoningSummaryAuto {
		t.Errorf("Summary = %q, want auto", params.Reasoning.Summary)
	}
	if params.Reasoning.Effort != shared.ReasoningEffortLow {
		t.Errorf("Effort = %q, want low", params.Reasoning.Effort)
	}

	req.Config.ThinkingConfig.IncludeThoughts = false
	params, err = m.buildResponseParams(req)
	if err != nil {
		t.Fatalf("buildResponseParams: %v", err)
	}
	if params.Reasoning.Summary != "" {
		t.Errorf("Summary = %q, want unset when thoughts are not requested", params.Reasoning.Summary)
	}
	if params.Reasoning.Effort != shared.ReasoningEffortLow {
		t.Errorf("Effort = %q, want low", params.Reasoning.Effort)
	}
}

// Malformed function-call arguments must fail the conversion: a tool with
// side effects must not run with silently emptied arguments.
func TestConvertResponse_MalformedArgumentsError(t *testing.T) {
	raw := []byte(`{
		"id": "resp-7",
		"status": "completed",
		"output": [{
			"type": "function_call",
			"id": "fc-1",
			"call_id": "call_1",
			"name": "delete_things",
			"arguments": "{"
		}],
		"usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
			"input_tokens_details": {"cached_tokens": 0},
			"output_tokens_details": {"reasoning_tokens": 0}}
	}`)

	var resp responses.Response
	if err := json.Unmarshal(raw, &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if _, err := convertResponse(&resp, "test-origin"); err == nil {
		t.Fatalf("expected an error for malformed arguments, got none")
	}
}

// A raw JSON schema set via ResponseJsonSchema must reach the request as a
// json_schema response format: strict when it fits the subset, non-strict
// with the schema intact otherwise. Ignoring it silently would let the model
// answer unconstrained while the caller believes a schema is enforced.
func TestBuildResponseParams_ResponseJsonSchema(t *testing.T) {
	m := New(Config{APIKey: "test", ModelName: "gpt-5.5"})

	strictSchema := map[string]any{
		"type":       "object",
		"properties": map[string]any{"score": map[string]any{"type": "number"}},
	}
	params, err := m.buildResponseParams(&model.LLMRequest{Config: &genai.GenerateContentConfig{
		ResponseJsonSchema: strictSchema,
	}})
	if err != nil {
		t.Fatalf("buildResponseParams: %v", err)
	}
	format := params.Text.Format.OfJSONSchema
	if format == nil {
		t.Fatalf("expected a json_schema response format, got %+v", params.Text.Format)
	}
	if !format.Strict.Valid() || !format.Strict.Value {
		t.Errorf("Strict = %+v, want true for a subset-compatible schema", format.Strict)
	}

	lossySchema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"code": map[string]any{"type": "string", "minLength": 2},
		},
	}
	params, err = m.buildResponseParams(&model.LLMRequest{Config: &genai.GenerateContentConfig{
		ResponseJsonSchema: lossySchema,
	}})
	if err != nil {
		t.Fatalf("buildResponseParams: %v", err)
	}
	format = params.Text.Format.OfJSONSchema
	if format == nil {
		t.Fatalf("expected a json_schema response format, got %+v", params.Text.Format)
	}
	if !format.Strict.Valid() || format.Strict.Value {
		t.Errorf("Strict = %+v, want false when the subset cannot express the schema", format.Strict)
	}
	code := format.Schema["properties"].(map[string]any)["code"].(map[string]any)
	if code["minLength"] != float64(2) && code["minLength"] != 2 {
		t.Errorf("minLength = %v, want preserved verbatim", code["minLength"])
	}
}

// Refusal content must keep its identity in PartMetadata: replaying it as
// plain output_text would lose the refusal semantics, and message status
// must survive so incomplete messages do not come back as completed.
func TestConvertResponse_RefusalAndStatusMetadata(t *testing.T) {
	raw := []byte(`{
		"id": "resp-8",
		"status": "completed",
		"output": [{
			"type": "message",
			"id": "msg-1",
			"role": "assistant",
			"status": "incomplete",
			"content": [{"type": "refusal", "refusal": "I cannot help with that."}]
		}],
		"usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
			"input_tokens_details": {"cached_tokens": 0},
			"output_tokens_details": {"reasoning_tokens": 0}}
	}`)

	var resp responses.Response
	if err := json.Unmarshal(raw, &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	got, err := convertResponse(&resp, "test-origin")
	if err != nil {
		t.Fatalf("convertResponse: %v", err)
	}
	if len(got.Content.Parts) != 1 {
		t.Fatalf("expected 1 part, got %d", len(got.Content.Parts))
	}
	pm := got.Content.Parts[0].PartMetadata
	if pm == nil || pm["refusal"] != true || pm["status"] != "incomplete" || pm["message_id"] != "msg-1" {
		t.Errorf("PartMetadata = %v, want refusal=true status=incomplete message_id=msg-1", pm)
	}
}

// A failed response must surface the server's error instead of degrading
// into "no output items" or, worse, passing partial output as a completed
// turn.
func TestConvertResponse_FailedStatus(t *testing.T) {
	raw := []byte(`{
		"id": "resp-9",
		"status": "failed",
		"error": {"code": "server_error", "message": "the backend fell over"},
		"output": [{
			"type": "message",
			"id": "msg-1",
			"role": "assistant",
			"status": "incomplete",
			"content": [{"type": "output_text", "text": "partial"}]
		}],
		"usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
			"input_tokens_details": {"cached_tokens": 0},
			"output_tokens_details": {"reasoning_tokens": 0}}
	}`)

	var resp responses.Response
	if err := json.Unmarshal(raw, &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	_, err := convertResponse(&resp, "test-origin")
	if err == nil {
		t.Fatalf("expected an error for a failed response, got none")
	}
	if got := err.Error(); got != "openai responses api: the backend fell over" {
		t.Errorf("err = %q, want the server message", got)
	}
}

// A typed ResponseSchema with a non-object root must go non-strict: strict
// mode requires an object root, and the typed schema allows primitives and
// arrays.
func TestBuildResponseParams_TypedSchemaNonObjectRoot(t *testing.T) {
	m := New(Config{APIKey: "test", ModelName: "gpt-5.5"})

	params, err := m.buildResponseParams(&model.LLMRequest{Config: &genai.GenerateContentConfig{
		ResponseSchema: &genai.Schema{
			Type:  genai.TypeArray,
			Items: &genai.Schema{Type: genai.TypeString},
		},
	}})
	if err != nil {
		t.Fatalf("buildResponseParams: %v", err)
	}
	format := params.Text.Format.OfJSONSchema
	if format == nil {
		t.Fatalf("expected a json_schema response format, got %+v", params.Text.Format)
	}
	if !format.Strict.Valid() || format.Strict.Value {
		t.Errorf("Strict = %+v, want false for a non-object root", format.Strict)
	}
	if format.Schema["type"] != "array" {
		t.Errorf("schema root = %v, want the array kept verbatim", format.Schema["type"])
	}
}
