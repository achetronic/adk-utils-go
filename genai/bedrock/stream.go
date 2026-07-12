// Copyright 2025 achetronic
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package bedrock

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"
)

// generateStream sends a Converse request and yields partial responses as
// they arrive, then a final complete one - mirroring genai/anthropic's
// generateStream shape so callers see the same Partial/TurnComplete
// semantics regardless of provider.
func (m *Model) generateStream(ctx context.Context, req *model.LLMRequest) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		input, err := m.buildConverseInput(req)
		if err != nil {
			yield(nil, err)
			return
		}

		streamInput := &bedrockruntime.ConverseStreamInput{
			ModelId:                      input.ModelId,
			Messages:                     input.Messages,
			System:                       input.System,
			InferenceConfig:              input.InferenceConfig,
			ToolConfig:                   input.ToolConfig,
			GuardrailConfig:              m.guardrail.toConverseStreamConfig(),
			AdditionalModelRequestFields: input.AdditionalModelRequestFields,
		}

		out, err := m.client.ConverseStream(ctx, streamInput)
		if err != nil {
			yield(nil, err)
			return
		}
		stream := out.GetStream()
		defer stream.Close()

		// Accumulated state for the final, non-partial response.
		var (
			finalParts   []*genai.Part
			textBuf      string
			toolUseID    string
			toolUseName  string
			toolUseInput string
			inToolUse    bool
			stopReason   types.StopReason
			usage        *types.TokenUsage
			trace        *types.ConverseStreamTrace
		)

		flushText := func() {
			if textBuf != "" {
				finalParts = append(finalParts, &genai.Part{Text: textBuf})
				textBuf = ""
			}
		}

		flushToolUse := func() {
			if inToolUse {
				finalParts = append(finalParts, &genai.Part{
					FunctionCall: &genai.FunctionCall{
						ID:   toolUseID,
						Name: toolUseName,
						Args: parseToolUseArgs(toolUseInput),
					},
				})
				toolUseID, toolUseName, toolUseInput = "", "", ""
				inToolUse = false
			}
		}

		for event := range stream.Events() {
			switch v := event.(type) {
			case *types.ConverseStreamOutputMemberContentBlockStart:
				if tu, ok := v.Value.Start.(*types.ContentBlockStartMemberToolUse); ok {
					flushText()
					inToolUse = true
					toolUseID = derefOr(tu.Value.ToolUseId)
					toolUseName = derefOr(tu.Value.Name)
				}

			case *types.ConverseStreamOutputMemberContentBlockDelta:
				switch d := v.Value.Delta.(type) {
				case *types.ContentBlockDeltaMemberText:
					textBuf += d.Value
					part := &genai.Part{Text: d.Value}
					llmResp := &model.LLMResponse{
						Content:      &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{part}},
						Partial:      true,
						TurnComplete: false,
					}
					if !yield(llmResp, nil) {
						return
					}
				case *types.ContentBlockDeltaMemberToolUse:
					if d.Value.Input != nil {
						toolUseInput += *d.Value.Input
					}
				}

			case *types.ConverseStreamOutputMemberContentBlockStop:
				flushText()
				flushToolUse()

			case *types.ConverseStreamOutputMemberMessageStop:
				stopReason = v.Value.StopReason

			case *types.ConverseStreamOutputMemberMetadata:
				if v.Value.Usage != nil {
					usage = v.Value.Usage
				}
				if v.Value.Trace != nil {
					trace = v.Value.Trace
				}
			}
		}

		if err := stream.Err(); err != nil {
			yield(nil, fmt.Errorf("bedrock: converse stream: %w", err))
			return
		}

		// Safety net in case the stream ended without an explicit
		// content_block_stop for a trailing block.
		flushText()
		flushToolUse()

		finalResp := &model.LLMResponse{
			Content:      &genai.Content{Role: genai.RoleModel, Parts: finalParts},
			Partial:      false,
			TurnComplete: true,
			FinishReason: convertStopReason(stopReason),
		}
		if usage != nil {
			finalResp.UsageMetadata = &genai.GenerateContentResponseUsageMetadata{
				PromptTokenCount:     int32derefOr(usage.InputTokens),
				CandidatesTokenCount: int32derefOr(usage.OutputTokens),
			}
		}
		var guardrailTrace *types.GuardrailTraceAssessment
		if trace != nil {
			guardrailTrace = trace.Guardrail
		}
		if meta := guardrailMetadata(stopReason, guardrailTrace); meta != nil {
			finalResp.CustomMetadata = meta
		}

		yield(finalResp, nil)
	}
}

// parseToolUseArgs decodes the accumulated tool_use input JSON string from a
// stream (built by concatenating ToolUseBlockDelta.Input chunks) into a
// map[string]any. An empty or malformed accumulation (e.g. a tool with no
// arguments, which some models stream as an empty string rather than "{}")
// falls back to an empty map rather than failing the whole turn.
func parseToolUseArgs(raw string) map[string]any {
	if raw == "" {
		return map[string]any{}
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(raw), &args); err != nil || args == nil {
		return map[string]any{}
	}
	return args
}
