// Copyright 2025 achetronic
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

package anthropic

import (
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// Anthropic's ToolChoiceUnionParam is a discriminated union with four
// mutually-exclusive variants (OfAuto / OfAny / OfTool / OfNone). We assert
// on the variant pointers directly because the nested Type field uses
// constant.* types whose in-memory zero value is the empty string — their
// real discriminator is injected during marshaling, not at construction.
// Checking the pointer-set pattern matches what GetType() does internally
// and is what the marshaler keys off of.
type toolChoiceVariant int

const (
	variantUnset toolChoiceVariant = iota
	variantAuto
	variantAny
	variantTool
	variantNone
)

// readVariant inspects the ToolChoiceUnionParam and returns which variant
// (if any) is populated. Mirrors the discriminator logic the SDK applies
// during JSON marshaling but avoids going through the marshaler so the test
// stays fast and free of side effects.
func readVariant(tc anthropic.ToolChoiceUnionParam) toolChoiceVariant {
	switch {
	case tc.OfAuto != nil:
		return variantAuto
	case tc.OfAny != nil:
		return variantAny
	case tc.OfTool != nil:
		return variantTool
	case tc.OfNone != nil:
		return variantNone
	default:
		return variantUnset
	}
}

// buildMessageParams must translate genai.ToolConfig.FunctionCallingConfig
// into Anthropic's tool_choice field. Without this, models that drift into
// plain-text replies (the same production symptom that motivated the OpenAI
// adapter fix) cannot be forced into tool-calling mode via the ADK config —
// the setting is silently dropped before it reaches the wire.
func TestBuildMessageParams_ToolChoice(t *testing.T) {
	cases := []struct {
		name         string
		cfg          *genai.GenerateContentConfig
		wantVariant  toolChoiceVariant
		wantToolName string // only checked when wantVariant == variantTool
	}{
		{
			name:        "ModeAuto translates to auto",
			cfg:         &genai.GenerateContentConfig{ToolConfig: &genai.ToolConfig{FunctionCallingConfig: &genai.FunctionCallingConfig{Mode: genai.FunctionCallingConfigModeAuto}}},
			wantVariant: variantAuto,
		},
		{
			name:        "ModeNone translates to none",
			cfg:         &genai.GenerateContentConfig{ToolConfig: &genai.ToolConfig{FunctionCallingConfig: &genai.FunctionCallingConfig{Mode: genai.FunctionCallingConfigModeNone}}},
			wantVariant: variantNone,
		},
		{
			name:        "ModeAny without allowed names translates to any",
			cfg:         &genai.GenerateContentConfig{ToolConfig: &genai.ToolConfig{FunctionCallingConfig: &genai.FunctionCallingConfig{Mode: genai.FunctionCallingConfigModeAny}}},
			wantVariant: variantAny,
		},
		{
			name: "ModeAny with a single allowed name pins to that tool",
			cfg: &genai.GenerateContentConfig{ToolConfig: &genai.ToolConfig{FunctionCallingConfig: &genai.FunctionCallingConfig{
				Mode:                 genai.FunctionCallingConfigModeAny,
				AllowedFunctionNames: []string{"my_tool"},
			}}},
			wantVariant:  variantTool,
			wantToolName: "my_tool",
		},
		{
			name: "ModeAny with multiple allowed names falls back to any",
			cfg: &genai.GenerateContentConfig{ToolConfig: &genai.ToolConfig{FunctionCallingConfig: &genai.FunctionCallingConfig{
				Mode:                 genai.FunctionCallingConfigModeAny,
				AllowedFunctionNames: []string{"a", "b"},
			}}},
			wantVariant: variantAny,
		},
		{
			name:        "No ToolConfig leaves tool_choice unset",
			cfg:         &genai.GenerateContentConfig{},
			wantVariant: variantUnset,
		},
		{
			name:        "Nil FunctionCallingConfig leaves tool_choice unset",
			cfg:         &genai.GenerateContentConfig{ToolConfig: &genai.ToolConfig{}},
			wantVariant: variantUnset,
		},
	}

	m := &Model{modelName: "test-model"}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Exercise the full path from LLMRequest to SDK params rather than
			// poking at a private helper — keeps the test stable if the mapping
			// is ever refactored into its own function.
			params, err := m.buildMessageParams(&model.LLMRequest{Config: tc.cfg})
			if err != nil {
				t.Fatalf("buildMessageParams returned error: %v", err)
			}

			if got := readVariant(params.ToolChoice); got != tc.wantVariant {
				t.Errorf("ToolChoice variant = %d, want %d", got, tc.wantVariant)
			}

			if tc.wantVariant == variantTool {
				if params.ToolChoice.OfTool == nil {
					t.Fatalf("expected OfTool to be set")
				}
				if got := params.ToolChoice.OfTool.Name; got != tc.wantToolName {
					t.Errorf("ToolChoice.OfTool.Name = %q, want %q", got, tc.wantToolName)
				}
			}
		})
	}
}
