// Copyright 2025 achetronic
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

package bedrock

import (
	"encoding/json"
	"testing"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"google.golang.org/genai"
)

func TestConvertRole(t *testing.T) {
	cases := []struct {
		role string
		want types.ConversationRole
	}{
		{genai.RoleModel, types.ConversationRoleAssistant},
		{genai.RoleUser, types.ConversationRoleUser},
		{"", types.ConversationRoleUser},
		{"system", types.ConversationRoleUser}, // unknown roles fall back to user
	}
	for _, c := range cases {
		if got := convertRole(c.role); got != c.want {
			t.Errorf("convertRole(%q) = %q, want %q", c.role, got, c.want)
		}
	}
}

func TestSanitizeToolID(t *testing.T) {
	// Already-valid IDs pass through unchanged.
	valid := "call_abc123-XYZ"
	if got := sanitizeToolID(valid); got != valid {
		t.Errorf("sanitizeToolID(%q) = %q, want unchanged", valid, got)
	}

	// Invalid IDs (e.g. containing ':' or '.') get a stable, deterministic
	// replacement so a tool_use/tool_result pair still matches.
	invalid := "call:weird.id/with-slash"
	got1 := sanitizeToolID(invalid)
	got2 := sanitizeToolID(invalid)
	if got1 != got2 {
		t.Errorf("sanitizeToolID(%q) not deterministic: %q != %q", invalid, got1, got2)
	}
	if !toolIDPattern.MatchString(got1) {
		t.Errorf("sanitizeToolID(%q) = %q, does not match toolIDPattern", invalid, got1)
	}

	// Over-length IDs (>64 chars) must also be sanitized.
	long := ""
	for i := 0; i < 100; i++ {
		long += "a"
	}
	if got := sanitizeToolID(long); !toolIDPattern.MatchString(got) {
		t.Errorf("sanitizeToolID(long 100-char id) = %q, does not match toolIDPattern", got)
	}
}

func textBlock(s string) types.ContentBlock {
	return &types.ContentBlockMemberText{Value: s}
}

func toolUseBlock(id, name string) types.ContentBlock {
	return &types.ContentBlockMemberToolUse{
		Value: types.ToolUseBlock{
			ToolUseId: awsString(id),
			Name:      awsString(name),
		},
	}
}

func toolResultBlock(id string) types.ContentBlock {
	return &types.ContentBlockMemberToolResult{
		Value: types.ToolResultBlock{ToolUseId: awsString(id)},
	}
}

func TestRepairMessageHistory_DropsOrphanedToolUse(t *testing.T) {
	messages := []types.Message{
		{Role: types.ConversationRoleUser, Content: []types.ContentBlock{textBlock("hi")}},
		{Role: types.ConversationRoleAssistant, Content: []types.ContentBlock{
			textBlock("let me check"),
			toolUseBlock("tool_1", "get_weather"),
		}},
		// No matching tool result follows: history ends mid-tool-call.
	}

	repaired := repairMessageHistory(messages)

	if len(repaired) != 2 {
		t.Fatalf("len(repaired) = %d, want 2", len(repaired))
	}
	for _, block := range repaired[1].Content {
		if _, ok := block.(*types.ContentBlockMemberToolUse); ok {
			t.Errorf("orphaned toolUse block was not dropped")
		}
	}
	// The text block in the same turn must survive.
	if len(repaired[1].Content) != 1 {
		t.Errorf("expected the text block to survive, got %d blocks", len(repaired[1].Content))
	}
}

func TestRepairMessageHistory_KeepsMatchedToolUse(t *testing.T) {
	messages := []types.Message{
		{Role: types.ConversationRoleAssistant, Content: []types.ContentBlock{
			toolUseBlock("tool_1", "get_weather"),
		}},
		{Role: types.ConversationRoleUser, Content: []types.ContentBlock{
			toolResultBlock("tool_1"),
		}},
	}

	repaired := repairMessageHistory(messages)

	if len(repaired) != 2 {
		t.Fatalf("len(repaired) = %d, want 2 (matched toolUse must survive)", len(repaired))
	}
	if _, ok := repaired[0].Content[0].(*types.ContentBlockMemberToolUse); !ok {
		t.Errorf("matched toolUse block was unexpectedly dropped")
	}
}

func TestTrimFinalAssistantWhitespace(t *testing.T) {
	messages := []types.Message{
		{Role: types.ConversationRoleUser, Content: []types.ContentBlock{textBlock("hi")}},
		{Role: types.ConversationRoleAssistant, Content: []types.ContentBlock{textBlock("hello there   \n")}},
	}

	trimmed := trimFinalAssistantWhitespace(messages)

	last := trimmed[len(trimmed)-1]
	text, ok := last.Content[0].(*types.ContentBlockMemberText)
	if !ok {
		t.Fatalf("expected a text block")
	}
	if text.Value != "hello there" {
		t.Errorf("trimmed text = %q, want %q", text.Value, "hello there")
	}
}

func TestTrimFinalAssistantWhitespace_DropsEmptyTrailingBlock(t *testing.T) {
	messages := []types.Message{
		{Role: types.ConversationRoleAssistant, Content: []types.ContentBlock{
			textBlock("real content"),
			textBlock("   \n\t"),
		}},
	}

	trimmed := trimFinalAssistantWhitespace(messages)

	last := trimmed[len(trimmed)-1]
	if len(last.Content) != 1 {
		t.Fatalf("len(content) = %d, want 1 (whitespace-only block should be dropped)", len(last.Content))
	}
	text, ok := last.Content[0].(*types.ContentBlockMemberText)
	if !ok || text.Value != "real content" {
		t.Errorf("unexpected remaining content: %#v", last.Content[0])
	}
}

func TestTrimFinalAssistantWhitespace_IgnoresNonTrailingAssistant(t *testing.T) {
	messages := []types.Message{
		{Role: types.ConversationRoleAssistant, Content: []types.ContentBlock{textBlock("hi   ")}},
		{Role: types.ConversationRoleUser, Content: []types.ContentBlock{textBlock("ok")}},
	}

	trimmed := trimFinalAssistantWhitespace(messages)

	text := trimmed[0].Content[0].(*types.ContentBlockMemberText)
	if text.Value != "hi   " {
		t.Errorf("non-trailing assistant turn was modified: %q", text.Value)
	}
}

func TestConvertStopReason(t *testing.T) {
	cases := []struct {
		reason types.StopReason
		want   genai.FinishReason
	}{
		{types.StopReasonEndTurn, genai.FinishReasonStop},
		{types.StopReasonStopSequence, genai.FinishReasonStop},
		{types.StopReasonToolUse, genai.FinishReasonStop},
		{types.StopReasonMaxTokens, genai.FinishReasonMaxTokens},
		{types.StopReasonModelContextWindowExceeded, genai.FinishReasonMaxTokens},
		{types.StopReasonGuardrailIntervened, genai.FinishReasonSafety},
		{types.StopReasonContentFiltered, genai.FinishReasonSafety},
		{types.StopReasonMalformedModelOutput, genai.FinishReasonOther},
		{types.StopReason("something_new"), genai.FinishReasonUnspecified},
	}
	for _, c := range cases {
		if got := convertStopReason(c.reason); got != c.want {
			t.Errorf("convertStopReason(%q) = %q, want %q", c.reason, got, c.want)
		}
	}
}

func TestGuardrailMetadata(t *testing.T) {
	// No intervention, no trace requested: nothing to surface.
	if meta := guardrailMetadata(types.StopReasonEndTurn, nil); meta != nil {
		t.Errorf("expected nil metadata for a normal turn with no trace, got %#v", meta)
	}

	// Intervention without trace: still surface the intervened flag.
	meta := guardrailMetadata(types.StopReasonGuardrailIntervened, nil)
	if meta == nil {
		t.Fatal("expected non-nil metadata when the guardrail intervened")
	}
	g := meta["bedrockGuardrail"].(map[string]any)
	if intervened, _ := g["intervened"].(bool); !intervened {
		t.Errorf("intervened = %v, want true", g["intervened"])
	}
	if !Intervened(meta) {
		t.Errorf("Intervened(meta) = false, want true")
	}

	// Trace requested but no intervention: surface trace with
	// intervened=false.
	trace := &types.GuardrailTraceAssessment{ActionReason: awsString("none")}
	meta = guardrailMetadata(types.StopReasonEndTurn, trace)
	if meta == nil {
		t.Fatal("expected non-nil metadata when a trace is present")
	}
	g = meta["bedrockGuardrail"].(map[string]any)
	if intervened, _ := g["intervened"].(bool); intervened {
		t.Errorf("intervened = true, want false")
	}
	if _, ok := g["trace"]; !ok {
		t.Errorf("expected trace key to be present")
	}
}

func TestIntervened_NonBedrockResponse(t *testing.T) {
	if Intervened(map[string]any{}) {
		t.Errorf("Intervened on empty metadata should be false")
	}
	if Intervened(map[string]any{"someOtherKey": "value"}) {
		t.Errorf("Intervened on unrelated metadata should be false")
	}
}

func TestImageFormatFromMIMEType(t *testing.T) {
	cases := []struct {
		mime   string
		want   types.ImageFormat
		wantOK bool
	}{
		{"image/png", types.ImageFormatPng, true},
		{"image/jpeg", types.ImageFormatJpeg, true},
		{"image/jpg", types.ImageFormatJpeg, true},
		{"image/gif", types.ImageFormatGif, true},
		{"image/webp", types.ImageFormatWebp, true},
		{"IMAGE/PNG", types.ImageFormatPng, true}, // case-insensitive
		{"application/pdf", "", false},
		{"", "", false},
	}
	for _, c := range cases {
		got, ok := imageFormatFromMIMEType(c.mime)
		if ok != c.wantOK || (ok && got != c.want) {
			t.Errorf("imageFormatFromMIMEType(%q) = (%q, %v), want (%q, %v)", c.mime, got, ok, c.want, c.wantOK)
		}
	}
}

func TestConvertPart_TextAndFunctionCallAndResponse(t *testing.T) {
	// Text part.
	block, err := convertPart(&genai.Part{Text: "hello"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if tb, ok := block.(*types.ContentBlockMemberText); !ok || tb.Value != "hello" {
		t.Errorf("unexpected text block: %#v", block)
	}

	// Function call with nil args must serialize to {} (D1 parity, via
	// common.MarshalToolPayload), not be dropped or panic.
	block, err = convertPart(&genai.Part{FunctionCall: &genai.FunctionCall{
		ID:   "call_1",
		Name: "get_weather",
	}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	tu, ok := block.(*types.ContentBlockMemberToolUse)
	if !ok {
		t.Fatalf("expected a toolUse block, got %#v", block)
	}
	if derefOr(tu.Value.ToolUseId) != "call_1" || derefOr(tu.Value.Name) != "get_weather" {
		t.Errorf("unexpected toolUse block: %#v", tu.Value)
	}

	// Function response carrying an "error" key maps to ToolResultStatusError.
	block, err = convertPart(&genai.Part{FunctionResponse: &genai.FunctionResponse{
		ID:       "call_1",
		Name:     "get_weather",
		Response: map[string]any{"error": "timeout"},
	}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	tr, ok := block.(*types.ContentBlockMemberToolResult)
	if !ok {
		t.Fatalf("expected a toolResult block, got %#v", block)
	}
	if tr.Value.Status != types.ToolResultStatusError {
		t.Errorf("Status = %q, want error", tr.Value.Status)
	}

	// Thought parts are dropped.
	block, err = convertPart(&genai.Part{Text: "reasoning...", Thought: true})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if block != nil {
		t.Errorf("expected thought part to be dropped, got %#v", block)
	}
}

func TestConvertToolConfig_ToolChoiceMapping(t *testing.T) {
	tools := []*genai.Tool{{
		FunctionDeclarations: []*genai.FunctionDeclaration{
			{Name: "get_weather", Description: "Get the weather"},
		},
	}}

	cases := []struct {
		name string
		cfg  *genai.ToolConfig
		// assert is a small predicate over the resulting ToolChoice; nil
		// means "ToolChoice should be unset".
		assert func(t *testing.T, tc types.ToolChoice)
	}{
		{
			name:   "no ToolConfig leaves tool_choice unset",
			cfg:    nil,
			assert: func(t *testing.T, tc types.ToolChoice) { wantNilChoice(t, tc) },
		},
		{
			name: "ModeAuto -> auto",
			cfg:  &genai.ToolConfig{FunctionCallingConfig: &genai.FunctionCallingConfig{Mode: genai.FunctionCallingConfigModeAuto}},
			assert: func(t *testing.T, tc types.ToolChoice) {
				if _, ok := tc.(*types.ToolChoiceMemberAuto); !ok {
					t.Errorf("expected ToolChoiceMemberAuto, got %#v", tc)
				}
			},
		},
		{
			name: "ModeAny without allow-list -> any",
			cfg:  &genai.ToolConfig{FunctionCallingConfig: &genai.FunctionCallingConfig{Mode: genai.FunctionCallingConfigModeAny}},
			assert: func(t *testing.T, tc types.ToolChoice) {
				if _, ok := tc.(*types.ToolChoiceMemberAny); !ok {
					t.Errorf("expected ToolChoiceMemberAny, got %#v", tc)
				}
			},
		},
		{
			name: "ModeAny with exactly one allowed name -> named tool choice",
			cfg: &genai.ToolConfig{FunctionCallingConfig: &genai.FunctionCallingConfig{
				Mode:                 genai.FunctionCallingConfigModeAny,
				AllowedFunctionNames: []string{"get_weather"},
			}},
			assert: func(t *testing.T, tc types.ToolChoice) {
				named, ok := tc.(*types.ToolChoiceMemberTool)
				if !ok {
					t.Fatalf("expected ToolChoiceMemberTool, got %#v", tc)
				}
				if derefOr(named.Value.Name) != "get_weather" {
					t.Errorf("named tool = %q, want get_weather", derefOr(named.Value.Name))
				}
			},
		},
		{
			name: "ModeAny with multiple allowed names falls back to any",
			cfg: &genai.ToolConfig{FunctionCallingConfig: &genai.FunctionCallingConfig{
				Mode:                 genai.FunctionCallingConfigModeAny,
				AllowedFunctionNames: []string{"get_weather", "get_time"},
			}},
			assert: func(t *testing.T, tc types.ToolChoice) {
				if _, ok := tc.(*types.ToolChoiceMemberAny); !ok {
					t.Errorf("expected fallback to ToolChoiceMemberAny, got %#v", tc)
				}
			},
		},
		{
			name:   "ModeNone leaves tool_choice unset (Converse has no none variant)",
			cfg:    &genai.ToolConfig{FunctionCallingConfig: &genai.FunctionCallingConfig{Mode: genai.FunctionCallingConfigModeNone}},
			assert: func(t *testing.T, tc types.ToolChoice) { wantNilChoice(t, tc) },
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			toolConfig, err := convertToolConfig(tools, c.cfg)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if toolConfig == nil {
				t.Fatal("expected non-nil ToolConfiguration (tools were provided)")
			}
			c.assert(t, toolConfig.ToolChoice)
		})
	}
}

func wantNilChoice(t *testing.T, tc types.ToolChoice) {
	t.Helper()
	if tc != nil {
		t.Errorf("expected unset ToolChoice, got %#v", tc)
	}
}

func TestConvertFunctionDeclaration_SchemaTypeForcedToObjectAndLowercased(t *testing.T) {
	decl := &genai.FunctionDeclaration{
		Name:        "get_weather",
		Description: "Get the weather for a city",
		Parameters: &genai.Schema{
			Type: "OBJECT",
			Properties: map[string]*genai.Schema{
				"city": {Type: "STRING"},
			},
			Required: []string{"city"},
		},
	}

	spec, err := convertFunctionDeclaration(decl)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if derefOr(spec.Name) != "get_weather" {
		t.Errorf("Name = %q", derefOr(spec.Name))
	}

	jsonSchema, ok := spec.InputSchema.(*types.ToolInputSchemaMemberJson)
	if !ok {
		t.Fatalf("expected ToolInputSchemaMemberJson, got %#v", spec.InputSchema)
	}

	var decoded map[string]any
	raw, err := jsonSchema.Value.MarshalSmithyDocument()
	if err != nil {
		t.Fatalf("marshalling schema document: %v", err)
	}
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("unmarshalling schema document: %v", err)
	}
	if decoded["type"] != "object" {
		t.Errorf("top-level type = %v, want \"object\"", decoded["type"])
	}
	props, ok := decoded["properties"].(map[string]any)
	if !ok {
		t.Fatalf("expected properties map, got %#v", decoded["properties"])
	}
	cityProp, ok := props["city"].(map[string]any)
	if !ok {
		t.Fatalf("expected city property map, got %#v", props["city"])
	}
	if cityProp["type"] != "string" {
		t.Errorf("city.type = %v, want lowercase \"string\"", cityProp["type"])
	}
}

func TestConvertContentToMessage_EmptyContentReturnsNil(t *testing.T) {
	msg, err := convertContentToMessage(&genai.Content{Role: genai.RoleUser})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if msg != nil {
		t.Errorf("expected nil message for empty content, got %#v", msg)
	}
}

func TestGuardrailConfig_ToConverseConfig(t *testing.T) {
	if (&GuardrailConfig{}).toConverseConfig() != nil {
		t.Errorf("empty GuardrailConfig should yield a nil Converse config")
	}

	g := &GuardrailConfig{Identifier: "gr-123", Version: "1", Trace: "enabled"}
	cfg := g.toConverseConfig()
	if cfg == nil {
		t.Fatal("expected non-nil config")
	}
	if derefOr(cfg.GuardrailIdentifier) != "gr-123" || derefOr(cfg.GuardrailVersion) != "1" {
		t.Errorf("unexpected identifier/version: %#v", cfg)
	}
	if cfg.Trace != types.GuardrailTraceEnabled {
		t.Errorf("Trace = %q, want enabled", cfg.Trace)
	}

	streamCfg := g.toConverseStreamConfig()
	if streamCfg.StreamProcessingMode != types.GuardrailStreamProcessingModeSync {
		t.Errorf("StreamProcessingMode = %q, want sync", streamCfg.StreamProcessingMode)
	}
}
