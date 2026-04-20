// Copyright 2025 achetronic
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

package openai

import (
	"testing"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
	"google.golang.org/genai"
)

// applyGenerationConfig must translate genai.ToolConfig.FunctionCallingConfig
// into OpenAI's tool_choice field. Without this, models that drift into
// plain-text replies under tool_choice="auto" (the OpenAI default) cannot be
// forced into tool-calling mode via the ADK config — the setting is silently
// dropped before it reaches the wire.
func TestApplyGenerationConfig_ToolChoice(t *testing.T) {
	cases := []struct {
		name         string
		cfg          *genai.GenerateContentConfig
		wantAuto     string // expected OfAuto value; "" means should be unset
		wantFunction string // expected OfFunctionToolChoice.Function.Name; "" means should be unset
	}{
		{
			name:     "ModeAuto translates to auto",
			cfg:      &genai.GenerateContentConfig{ToolConfig: &genai.ToolConfig{FunctionCallingConfig: &genai.FunctionCallingConfig{Mode: genai.FunctionCallingConfigModeAuto}}},
			wantAuto: "auto",
		},
		{
			name:     "ModeNone translates to none",
			cfg:      &genai.GenerateContentConfig{ToolConfig: &genai.ToolConfig{FunctionCallingConfig: &genai.FunctionCallingConfig{Mode: genai.FunctionCallingConfigModeNone}}},
			wantAuto: "none",
		},
		{
			name:     "ModeAny without allowed names translates to required",
			cfg:      &genai.GenerateContentConfig{ToolConfig: &genai.ToolConfig{FunctionCallingConfig: &genai.FunctionCallingConfig{Mode: genai.FunctionCallingConfigModeAny}}},
			wantAuto: "required",
		},
		{
			name: "ModeAny with a single allowed name pins to that function",
			cfg: &genai.GenerateContentConfig{ToolConfig: &genai.ToolConfig{FunctionCallingConfig: &genai.FunctionCallingConfig{
				Mode:                 genai.FunctionCallingConfigModeAny,
				AllowedFunctionNames: []string{"my_tool"},
			}}},
			wantFunction: "my_tool",
		},
		{
			name: "ModeAny with multiple allowed names falls back to required",
			cfg: &genai.GenerateContentConfig{ToolConfig: &genai.ToolConfig{FunctionCallingConfig: &genai.FunctionCallingConfig{
				Mode:                 genai.FunctionCallingConfigModeAny,
				AllowedFunctionNames: []string{"a", "b"},
			}}},
			wantAuto: "required",
		},
		{
			name:     "No ToolConfig leaves tool_choice unset",
			cfg:      &genai.GenerateContentConfig{},
			wantAuto: "",
		},
		{
			name:     "Nil FunctionCallingConfig leaves tool_choice unset",
			cfg:      &genai.GenerateContentConfig{ToolConfig: &genai.ToolConfig{}},
			wantAuto: "",
		},
	}

	m := &Model{modelName: "test-model"}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var params openai.ChatCompletionNewParams
			m.applyGenerationConfig(&params, tc.cfg)

			gotAuto := ""
			if !param.IsOmitted(params.ToolChoice.OfAuto) {
				gotAuto = params.ToolChoice.OfAuto.Value
			}
			if gotAuto != tc.wantAuto {
				t.Errorf("OfAuto = %q, want %q", gotAuto, tc.wantAuto)
			}

			gotFunction := ""
			if params.ToolChoice.OfFunctionToolChoice != nil {
				gotFunction = params.ToolChoice.OfFunctionToolChoice.Function.Name
			}
			if gotFunction != tc.wantFunction {
				t.Errorf("OfFunctionToolChoice.Function.Name = %q, want %q", gotFunction, tc.wantFunction)
			}
		})
	}
}
