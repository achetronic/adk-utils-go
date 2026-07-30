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
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"github.com/achetronic/adk-utils-go/genai/common"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"
)

// toolIDPattern matches the character set Bedrock's Converse API accepts for
// toolUseId: letters, digits, underscore and hyphen, 1-64 chars. Mirrors the
// equivalent rule in genai/anthropic (see DECISIONS.md A1); the same fix-up
// is needed here because tool-call IDs often originate from a different
// provider's adapter when an ADK session is migrated or replayed.
var toolIDPattern = regexp.MustCompile(`^[a-zA-Z0-9_-]{1,64}$`)

// sanitizeToolID returns id unchanged if it already satisfies Bedrock's
// toolUseId character/length rules, otherwise a stable, deterministic
// replacement derived from a hash of id so repeated calls (and the matching
// tool_use/tool_result pair) stay consistent.
func sanitizeToolID(id string) string {
	if toolIDPattern.MatchString(id) {
		return id
	}
	sum := sha256.Sum256([]byte(id))
	return "toolu_" + hex.EncodeToString(sum[:])[:16]
}

// buildConverseInput converts an LLMRequest into Bedrock's Converse API
// format (system prompt, messages, inference config, tools, guardrail).
func (m *Model) buildConverseInput(req *model.LLMRequest) (*bedrockruntime.ConverseInput, error) {
	input := &bedrockruntime.ConverseInput{
		ModelId: &m.modelID,
	}

	// System instruction.
	if req.Config != nil && req.Config.SystemInstruction != nil {
		if text := extractTextFromContent(req.Config.SystemInstruction); text != "" {
			input.System = []types.SystemContentBlock{
				&types.SystemContentBlockMemberText{Value: text},
			}
		}
	}

	// Messages.
	messages := make([]types.Message, 0, len(req.Contents))
	for _, content := range req.Contents {
		msg, err := convertContentToMessage(content)
		if err != nil {
			return nil, err
		}
		if msg != nil {
			messages = append(messages, *msg)
		}
	}

	// Repair message history to comply with Bedrock's requirements (every
	// toolUse needs a matching toolResult), mirroring genai/anthropic's A2.
	// redactBlockedInputs runs first so a guardrail-blocked user turn doesn't
	// re-trigger the guardrail on every subsequent request.
	messages = redactBlockedInputs(messages)
	messages = repairMessageHistory(messages)
	messages = trimFinalAssistantWhitespace(messages)
	input.Messages = messages

	// Inference config + tools + guardrail.
	inferenceConfig := &types.InferenceConfiguration{}
	hasInferenceConfig := false

	if m.maxOutputTokens > 0 {
		maxTokens := int32(m.maxOutputTokens)
		inferenceConfig.MaxTokens = &maxTokens
		hasInferenceConfig = true
	}

	if req.Config != nil {
		if req.Config.MaxOutputTokens > 0 {
			maxTokens := req.Config.MaxOutputTokens
			inferenceConfig.MaxTokens = &maxTokens
			hasInferenceConfig = true
		}
		if req.Config.Temperature != nil {
			inferenceConfig.Temperature = req.Config.Temperature
			hasInferenceConfig = true
		}
		if req.Config.TopP != nil {
			inferenceConfig.TopP = req.Config.TopP
			hasInferenceConfig = true
		}
		if len(req.Config.StopSequences) > 0 {
			inferenceConfig.StopSequences = req.Config.StopSequences
			hasInferenceConfig = true
		}

		if len(req.Config.Tools) > 0 {
			toolConfig, err := convertToolConfig(req.Config.Tools, req.Config.ToolConfig)
			if err != nil {
				return nil, err
			}
			input.ToolConfig = toolConfig
		}
	}

	if hasInferenceConfig {
		input.InferenceConfig = inferenceConfig
	}

	if m.guardrail != nil {
		input.GuardrailConfig = m.guardrail.toConverseConfig()
	}

	if len(m.additionalModelRequestFields) > 0 {
		input.AdditionalModelRequestFields = document.NewLazyDocument(m.additionalModelRequestFields)
	}

	return input, nil
}

// convertRole maps a genai role to Bedrock's ConversationRole. Converse only
// has user/assistant (system is a dedicated top-level field, not a message
// role); unknown roles fall back to "user", mirroring genai/anthropic's A7.
func convertRole(role string) types.ConversationRole {
	if role == genai.RoleModel {
		return types.ConversationRoleAssistant
	}
	return types.ConversationRoleUser
}

// convertContentToMessage transforms a genai.Content (text, images, tool
// calls/results) into a Bedrock Converse Message. Returns (nil, nil) for
// content that yields no blocks (e.g. an empty turn), so the caller can skip
// it without sending an invalid empty message.
func convertContentToMessage(content *genai.Content) (*types.Message, error) {
	if content == nil {
		return nil, nil
	}

	blocks := make([]types.ContentBlock, 0, len(content.Parts))
	for _, part := range content.Parts {
		block, err := convertPart(part)
		if err != nil {
			return nil, err
		}
		if block != nil {
			blocks = append(blocks, block)
		}
	}

	if len(blocks) == 0 {
		return nil, nil
	}

	return &types.Message{
		Role:    convertRole(content.Role),
		Content: blocks,
	}, nil
}

// convertPart converts a single genai.Part into a Bedrock ContentBlock.
// Thought parts (extended-thinking traces from another provider) are
// dropped: Converse's common content-block set has no reasoning-block
// variant outside provider-specific reasoningContent, and echoing foreign
// thought signatures back is both meaningless and liable to be rejected.
func convertPart(part *genai.Part) (types.ContentBlock, error) {
	if part == nil || part.Thought {
		return nil, nil
	}

	switch {
	case part.Text != "":
		return &types.ContentBlockMemberText{Value: part.Text}, nil

	case part.InlineData != nil:
		format, ok := imageFormatFromMIMEType(part.InlineData.MIMEType)
		if !ok {
			return nil, fmt.Errorf("bedrock: unsupported inline data MIME type %q", part.InlineData.MIMEType)
		}
		return &types.ContentBlockMemberImage{
			Value: types.ImageBlock{
				Format: format,
				Source: &types.ImageSourceMemberBytes{Value: part.InlineData.Data},
			},
		}, nil

	case part.FunctionCall != nil:
		call := part.FunctionCall
		payload, err := common.MarshalToolPayload(call.Args)
		if err != nil {
			return nil, fmt.Errorf("bedrock: marshalling tool call args for %q: %w", call.Name, err)
		}
		args, err := rawJSONToDocument(payload)
		if err != nil {
			return nil, err
		}
		return &types.ContentBlockMemberToolUse{
			Value: types.ToolUseBlock{
				ToolUseId: awsString(sanitizeToolID(toolCallID(call))),
				Name:      awsString(call.Name),
				Input:     args,
			},
		}, nil

	case part.FunctionResponse != nil:
		resp := part.FunctionResponse
		payload, err := common.MarshalToolPayload(resp.Response)
		if err != nil {
			return nil, fmt.Errorf("bedrock: marshalling tool result for %q: %w", resp.Name, err)
		}
		result, err := rawJSONToDocument(payload)
		if err != nil {
			return nil, err
		}
		return &types.ContentBlockMemberToolResult{
			Value: types.ToolResultBlock{
				ToolUseId: awsString(sanitizeToolID(toolResponseID(resp))),
				Content: []types.ToolResultContentBlock{
					&types.ToolResultContentBlockMemberJson{Value: result},
				},
				Status: toolResultStatus(resp.Response),
			},
		}, nil

	default:
		// Unsupported part kind (executable code, video, server-side tool
		// call/response, etc.). Skip rather than fail the whole request.
		return nil, nil
	}
}

// toolCallID returns the ID to use for a tool_use block, falling back to the
// function name when the model didn't supply one (some adapters/providers
// omit FunctionCall.ID for single-shot, non-resumable calls).
func toolCallID(call *genai.FunctionCall) string {
	if call.ID != "" {
		return call.ID
	}
	return call.Name
}

// toolResponseID mirrors toolCallID for the matching FunctionResponse so the
// toolUse/toolResult pair keeps matching IDs after sanitization.
func toolResponseID(resp *genai.FunctionResponse) string {
	if resp.ID != "" {
		return resp.ID
	}
	return resp.Name
}

// toolResultStatus inspects a function response payload for an "error" key
// (the genai.FunctionResponse convention: "use 'error' key to specify error
// details") and maps it to Bedrock's ToolResultStatus. Several models
// (Nova, Claude 3/4) use this to decide how to react to a failed tool call.
func toolResultStatus(response map[string]any) types.ToolResultStatus {
	if response == nil {
		return ""
	}
	if _, hasError := response["error"]; hasError {
		return types.ToolResultStatusError
	}
	return ""
}

// rawJSONToDocument decodes canonical JSON bytes (as produced by
// common.MarshalToolPayload) into a generic Go value and wraps it in a
// Bedrock document.Interface. Going through a decode step (rather than
// handing the json.RawMessage straight to document.NewLazyDocument) avoids
// depending on whether Bedrock's smithy-generated JSON encoder special-cases
// json.RawMessage/Marshaler the same way encoding/json does.
func rawJSONToDocument(raw json.RawMessage) (document.Interface, error) {
	var v any
	if err := json.Unmarshal(raw, &v); err != nil {
		return nil, fmt.Errorf("bedrock: decoding tool payload: %w", err)
	}
	return document.NewLazyDocument(v), nil
}

// imageFormatFromMIMEType maps a genai Blob's MIME type to Bedrock's
// ImageFormat enum. Converse only accepts these four raster formats.
func imageFormatFromMIMEType(mimeType string) (types.ImageFormat, bool) {
	switch strings.ToLower(strings.TrimSpace(mimeType)) {
	case "image/png":
		return types.ImageFormatPng, true
	case "image/jpeg", "image/jpg":
		return types.ImageFormatJpeg, true
	case "image/gif":
		return types.ImageFormatGif, true
	case "image/webp":
		return types.ImageFormatWebp, true
	default:
		return "", false
	}
}

// extractTextFromContent concatenates all text parts of a Content into a
// single string, used for the system instruction.
func extractTextFromContent(content *genai.Content) string {
	if content == nil {
		return ""
	}
	var sb strings.Builder
	for _, part := range content.Parts {
		if part != nil && part.Text != "" {
			sb.WriteString(part.Text)
		}
	}
	return sb.String()
}

// awsString is a tiny local helper so call sites read as `awsString(x)`
// instead of repeating `&x` over a temporary; keeps convertPart readable
// given how many *string fields the Bedrock SDK types use.
func awsString(s string) *string {
	return &s
}
