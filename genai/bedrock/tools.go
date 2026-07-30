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

// guardrailMarker is an invisible (zero-width space) prefix stamped on the
// first text block of a guardrail-blocked response (see markGuardrailBlocked
// in response.go/stream.go). It lets redactBlockedInputs recognize a blocked
// turn on a later request without hardcoding any particular guardrail's
// refusal wording — Bedrock's actual guardrail message text (e.g. "Sorry,
// the model cannot answer this question.") is guardrail-config-specific and
// varies, so it can't be matched literally. The marker is invisible to users
// (renders as nothing) but survives being stored and replayed as history.
const guardrailMarker = "​"

// guardrailFallbackText is used only when Bedrock's blocked response has no
// content at all (empty message). Bedrock and ADK both reject/mishandle
// empty-content turns (see B8), so something non-empty is required.
const guardrailFallbackText = "[blocked by guardrail]"

// guardrailBlockedStage reports which stage of the Converse guardrail
// pipeline actually blocked the turn: "input" (the guardrail denied the
// prompt before the model ever ran) or "output" (the model generated a
// response and the guardrail denied that). trace.ModelOutput is the
// reliable signal: it's only populated when the model actually produced
// output for the guardrail to assess, so an empty ModelOutput means
// generation never happened. Returns "" when trace is nil (the caller
// didn't request a trace, so the stage can't be determined) or the turn
// wasn't guardrail-blocked at all.
func guardrailBlockedStage(trace *types.GuardrailTraceAssessment) string {
	if trace == nil {
		return ""
	}
	if len(trace.ModelOutput) == 0 {
		return "input"
	}
	return "output"
}

// markGuardrailBlocked collapses parts (a guardrail-blocked response) down
// to a single text part tagged with guardrailMarker, so a later request can
// recognize this turn without altering the guardrail's actual message text.
//
// Collapsing to one part is deliberate, not just tagging in place: Bedrock
// can return the same guardrail message across more than one content block
// when both the input and output stages produced a denial message (or
// duplicate blocks for the same stage), and passing all of them through
// would display the identical message twice, back to back, with no
// separator.
//
// stage (from guardrailBlockedStage) picks which occurrence to keep when
// there's more than one text part: "input" keeps the first (the guardrail's
// pre-generation denial message, which has no prior generated text to
// follow it), "output" keeps the last (any generated text that slipped out
// before the guardrail caught it is expected to precede the final blocked-
// output message). An empty stage (no trace available) falls back to the
// first non-empty part. Falls back to guardrailFallbackText when Bedrock
// returned no text content at all.
func markGuardrailBlocked(parts []*genai.Part, stage string) []*genai.Part {
	var nonEmpty []*genai.Part
	for _, p := range parts {
		if p != nil && p.Text != "" {
			nonEmpty = append(nonEmpty, p)
		}
	}
	if len(nonEmpty) == 0 {
		return []*genai.Part{{Text: guardrailMarker + guardrailFallbackText}}
	}

	chosen := nonEmpty[0]
	if stage == "output" {
		chosen = nonEmpty[len(nonEmpty)-1]
	}
	return []*genai.Part{{Text: guardrailMarker + chosen.Text}}
}

// redactBlockedInputs replaces the user turn immediately preceding a
// guardrail-blocked assistant turn (identified by guardrailMarker) with a
// placeholder. The user message that triggered the guardrail is otherwise
// still in history with its original content, and Bedrock re-evaluates it on
// every subsequent request — re-blocking the conversation indefinitely.
// Redacting it stops the re-triggering while keeping the alternating
// user/assistant turn structure intact (mirrors Strands'
// guardrail_redact_input=True).
func redactBlockedInputs(messages []types.Message) []types.Message {
	for i, msg := range messages {
		if msg.Role != types.ConversationRoleAssistant || len(msg.Content) == 0 {
			continue
		}
		text, ok := msg.Content[0].(*types.ContentBlockMemberText)
		if !ok || !strings.HasPrefix(text.Value, guardrailMarker) {
			continue
		}
		if i > 0 && messages[i-1].Role == types.ConversationRoleUser {
			messages[i-1].Content = []types.ContentBlock{
				&types.ContentBlockMemberText{Value: "[User input redacted by guardrail.]"},
			}
		}
	}
	return messages
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
