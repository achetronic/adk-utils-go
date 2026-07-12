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
	"encoding/json"
	"strings"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"google.golang.org/genai"
)

// convertToolConfig converts genai tool/function declarations and the
// caller's tool_choice preference into Bedrock's ToolConfiguration.
func convertToolConfig(genaiTools []*genai.Tool, toolConfig *genai.ToolConfig) (*types.ToolConfiguration, error) {
	tools := make([]types.Tool, 0, len(genaiTools))
	for _, tool := range genaiTools {
		if tool == nil {
			continue
		}
		for _, decl := range tool.FunctionDeclarations {
			spec, err := convertFunctionDeclaration(decl)
			if err != nil {
				return nil, err
			}
			tools = append(tools, &types.ToolMemberToolSpec{Value: *spec})
		}
	}

	if len(tools) == 0 {
		return nil, nil
	}

	cfg := &types.ToolConfiguration{Tools: tools}

	// ToolConfig -> tool_choice. Mirrors the mapping table in
	// genai/anthropic and genai/openai (see AGENTS.md "LLM Adapters -
	// tool_choice Mapping"):
	//
	//   ModeUnspecified (zero value) -> unset
	//   ModeAuto                     -> {auto}
	//   ModeNone                     -> unset (Converse has no "none"; the
	//                                   caller already controls this by not
	//                                   sending tools at all when it wants
	//                                   them fully disabled for a turn)
	//   ModeAny, no allow-list       -> {any}
	//   ModeAny, exactly one name    -> {tool: name}
	//   ModeAny, multiple names      -> fallback to {any}
	if toolConfig != nil && toolConfig.FunctionCallingConfig != nil {
		fcc := toolConfig.FunctionCallingConfig
		switch fcc.Mode {
		case genai.FunctionCallingConfigModeAuto:
			cfg.ToolChoice = &types.ToolChoiceMemberAuto{}
		case genai.FunctionCallingConfigModeAny:
			if len(fcc.AllowedFunctionNames) == 1 {
				cfg.ToolChoice = &types.ToolChoiceMemberTool{
					Value: types.SpecificToolChoice{Name: awsString(fcc.AllowedFunctionNames[0])},
				}
			} else {
				cfg.ToolChoice = &types.ToolChoiceMemberAny{}
			}
		}
	}

	return cfg, nil
}

// convertFunctionDeclaration converts one genai.FunctionDeclaration into a
// Bedrock ToolSpecification. The top-level schema type is forced to
// "object", which Converse requires (mirrors genai/anthropic A6).
func convertFunctionDeclaration(decl *genai.FunctionDeclaration) (*types.ToolSpecification, error) {
	if decl == nil {
		return nil, nil
	}

	// ParametersJsonSchema (typically *jsonschema.Schema) takes precedence
	// over the typed Parameters (*genai.Schema) when both are set, mirroring
	// genai/anthropic's convertTools.
	params := decl.ParametersJsonSchema
	if params == nil {
		params = decl.Parameters
	}

	schema := map[string]any{"type": "object"}
	if params != nil {
		var decoded map[string]any
		if asMap, ok := params.(map[string]any); ok {
			decoded = asMap
		} else if raw, err := json.Marshal(params); err == nil {
			_ = json.Unmarshal(raw, &decoded)
		}
		if decoded != nil {
			lowercaseSchemaTypes(decoded)
			decoded["type"] = "object"
			schema = decoded
		}
	}

	return &types.ToolSpecification{
		Name:        awsString(decl.Name),
		Description: awsString(decl.Description),
		InputSchema: &types.ToolInputSchemaMemberJson{
			Value: document.NewLazyDocument(schema),
		},
	}, nil
}

// lowercaseSchemaTypes lowercases every "type" value in a JSON schema map.
// genai.Schema's Type field marshals using Gemini's uppercase enum
// convention (e.g. "OBJECT", "STRING"), but standard JSON Schema - which is
// what Bedrock's ToolInputSchema expects - uses lowercase. Mirrors
// genai/anthropic's lowercaseTypes.
func lowercaseSchemaTypes(m map[string]any) {
	for k, v := range m {
		switch {
		case k == "type":
			if s, ok := v.(string); ok {
				m[k] = strings.ToLower(s)
			}
		case isMap(v):
			lowercaseSchemaTypes(v.(map[string]any))
		case isList(v):
			for _, item := range v.([]any) {
				if itemMap, ok := item.(map[string]any); ok {
					lowercaseSchemaTypes(itemMap)
				}
			}
		}
	}
}

func isMap(v any) bool {
	_, ok := v.(map[string]any)
	return ok
}

func isList(v any) bool {
	_, ok := v.([]any)
	return ok
}

// dropEmptyMessages removes messages with no content blocks. These arise when
// a Bedrock Guardrail blocks an assistant turn: ADK replays the empty response
// as history on the next request, and Bedrock rejects requests that contain
// empty-content turns.
func dropEmptyMessages(messages []types.Message) []types.Message {
	out := make([]types.Message, 0, len(messages))
	for _, msg := range messages {
		if len(msg.Content) > 0 {
			out = append(out, msg)
		}
	}
	return out
}

// repairMessageHistory drops orphaned toolUse blocks (those without a
// matching toolResult in the immediately following user message) before
// sending, mirroring genai/anthropic's A2: Bedrock rejects a request where
// an assistant toolUse isn't followed by its toolResult, and ADK histories
// can end mid-tool-call (cancel, compaction, agent switch).
func repairMessageHistory(messages []types.Message) []types.Message {
	repaired := make([]types.Message, 0, len(messages))

	for i, msg := range messages {
		if msg.Role != types.ConversationRoleAssistant {
			repaired = append(repaired, msg)
			continue
		}

		pendingIDs := map[string]bool{}
		for _, block := range msg.Content {
			if tu, ok := block.(*types.ContentBlockMemberToolUse); ok && tu.Value.ToolUseId != nil {
				pendingIDs[*tu.Value.ToolUseId] = true
			}
		}

		if len(pendingIDs) == 0 {
			repaired = append(repaired, msg)
			continue
		}

		// Look at the immediately following message for matching
		// toolResult blocks.
		if i+1 < len(messages) && messages[i+1].Role == types.ConversationRoleUser {
			for _, block := range messages[i+1].Content {
				if tr, ok := block.(*types.ContentBlockMemberToolResult); ok && tr.Value.ToolUseId != nil {
					delete(pendingIDs, *tr.Value.ToolUseId)
				}
			}
		}

		if len(pendingIDs) == 0 {
			repaired = append(repaired, msg)
			continue
		}

		// Drop the unmatched toolUse blocks but keep any other content
		// (text, matched toolUse blocks) in the turn.
		kept := make([]types.ContentBlock, 0, len(msg.Content))
		for _, block := range msg.Content {
			if tu, ok := block.(*types.ContentBlockMemberToolUse); ok && tu.Value.ToolUseId != nil && pendingIDs[*tu.Value.ToolUseId] {
				continue
			}
			kept = append(kept, block)
		}
		if len(kept) > 0 {
			msg.Content = kept
			repaired = append(repaired, msg)
		}
	}

	return repaired
}

// trimFinalAssistantWhitespace right-trims the last text block of a
// trailing assistant message, mirroring genai/anthropic's A9: Anthropic
// (and, on Bedrock, other models in prefill-sensitive configurations)
// rejects a request whose final assistant content ends in whitespace. A
// block left empty afterwards is dropped (empty text blocks are also
// commonly rejected).
func trimFinalAssistantWhitespace(messages []types.Message) []types.Message {
	if len(messages) == 0 {
		return messages
	}

	last := len(messages) - 1
	if messages[last].Role != types.ConversationRoleAssistant {
		return messages
	}

	content := messages[last].Content
	for i := len(content) - 1; i >= 0; i-- {
		text, ok := content[i].(*types.ContentBlockMemberText)
		if !ok {
			break
		}
		trimmed := strings.TrimRight(text.Value, " \t\n\r")
		if trimmed == "" {
			content = append(content[:i], content[i+1:]...)
			continue
		}
		content[i] = &types.ContentBlockMemberText{Value: trimmed}
		break
	}
	messages[last].Content = content

	return messages
}
