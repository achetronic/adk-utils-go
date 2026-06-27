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

package anthropic

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"net/http"
	"regexp"
	"strings"

	"github.com/achetronic/adk-utils-go/genai/common"
	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

var _ model.LLM = &Model{}

var (
	ErrNoContentInResponse = errors.New("no content in Anthropic response")
)

// anthropicToolIDPattern matches valid Anthropic tool_use IDs: ^[a-zA-Z0-9_-]+$
var anthropicToolIDPattern = regexp.MustCompile(`^[a-zA-Z0-9_-]+$`)

// Model implements model.LLM using the official Anthropic Go SDK.
type Model struct {
	client               *anthropic.Client
	modelName            string
	maxOutputTokens      int
	thinkingBudgetTokens int
	thinkingEffort       string
	thinkingMode         string
	disablePromptCaching bool
}

// Reasoning API mode identifiers. They select which on-the-wire shape
// the adapter uses to ask the model to reason:
//
//   - ThinkingModeEnabled is the CLASSIC API: "thinking":{"type":"enabled",
//     "budget_tokens":N}. Accepted by Claude 3.7 / Sonnet 4 / Opus 4.
//   - ThinkingModeAdaptive is the EFFORT-based API: "thinking":{"type":"adaptive"}
//     plus "output_config":{"effort":"<level>"}. Required by Opus 4.5+, which
//     reject the classic "enabled" form.
const (
	ThinkingModeEnabled  = "enabled"
	ThinkingModeAdaptive = "adaptive"
)

// HTTPOptions holds optional HTTP-level configuration for the Anthropic client.
type HTTPOptions struct {
	Client  *http.Client
	Headers http.Header
}

// Config holds configuration for creating a new Model.
type Config struct {
	// APIKey is the Anthropic API key. If empty, the ANTHROPIC_API_KEY
	// environment variable is used instead.
	APIKey string

	// BaseURL overrides the API base URL. Leave empty for the default
	// Anthropic endpoint; set it only to target a custom or proxy host.
	BaseURL string

	// ModelName is the model identifier to call, e.g.
	// "claude-sonnet-4-5-20250929".
	ModelName string

	// MaxOutputTokens caps how many tokens Claude may generate in its
	// response. This is an output-only limit; it does not affect the input
	// or context window. When zero, the adapter defaults to 4096.
	MaxOutputTokens int

	// ThinkingMode selects which reasoning API the request uses. It exists
	// so the caller, not the adapter, decides the shape: the adapter never
	// inspects the model name to guess.
	//
	// Accepted values:
	//
	//   ""          Auto. The mode is deduced from the fields you set:
	//               ThinkingEffort set        => adaptive
	//               else ThinkingBudgetTokens > 0 => enabled
	//               else                       => no reasoning.
	//
	//   "enabled"   Classic budget-based API: "thinking":{"type":"enabled"}.
	//               Requires ThinkingBudgetTokens > 0.
	//
	//   "adaptive"  Effort-based API: "thinking":{"type":"adaptive"} plus
	//               "output_config":{"effort"}. Requires ThinkingEffort.
	//
	// Rule of thumb: "adaptive" for Opus 4.5 and newer, "enabled" (or auto)
	// for older models.
	ThinkingMode string

	// ThinkingBudgetTokens drives the CLASSIC ("enabled") reasoning API. It
	// is the number of output tokens Claude may spend on internal reasoning
	// before writing its final answer. These thinking tokens count as output
	// tokens, so the value must be >= 1024 and strictly less than
	// MaxOutputTokens. Zero leaves classic extended thinking off.
	//
	// Accepted by Claude 3.7, Sonnet 4 and Opus 4. Newer models (Opus 4.5+)
	// reject this form and need ThinkingEffort instead.
	ThinkingBudgetTokens int

	// ThinkingEffort drives the newer ("adaptive") reasoning API used by
	// Opus 4.5 and later, which reject the classic "enabled" form. It sets
	// the reasoning depth; typical values are "low", "medium" and "high"
	// (some models also accept "xhigh" or "max"). When set, the request
	// sends "thinking":{"type":"adaptive"} plus "output_config":{"effort"}.
	// Empty leaves adaptive reasoning off.
	ThinkingEffort string

	// HTTPOptions carries optional HTTP-level overrides, such as extra
	// request headers.
	HTTPOptions HTTPOptions

	// DisablePromptCaching turns OFF the cache_control breakpoints the
	// adapter stamps on every request (see caching.go). Caching is on
	// by default because it only ever lowers the bill: requests whose
	// prefix is below Anthropic's minimum cacheable length are simply
	// not cached, with no error and no extra cost. Disable it only for
	// proxies/gateways that reject the cache_control field.
	DisablePromptCaching bool
}

// New creates an Anthropic client from config (API key, base URL, model name).
func New(cfg Config) *Model {
	opts := []option.RequestOption{}

	if cfg.APIKey != "" {
		opts = append(opts, option.WithAPIKey(cfg.APIKey))
	}
	if cfg.BaseURL != "" {
		opts = append(opts, option.WithBaseURL(cfg.BaseURL))
	}
	if cfg.HTTPOptions.Client != nil {
		opts = append(opts, option.WithHTTPClient(cfg.HTTPOptions.Client))
	}
	for k, vals := range cfg.HTTPOptions.Headers {
		for _, v := range vals {
			opts = append(opts, option.WithHeaderAdd(k, v))
		}
	}

	client := anthropic.NewClient(opts...)

	return &Model{
		client:               &client,
		modelName:            cfg.ModelName,
		maxOutputTokens:      cfg.MaxOutputTokens,
		thinkingBudgetTokens: cfg.ThinkingBudgetTokens,
		thinkingEffort:       cfg.ThinkingEffort,
		thinkingMode:         cfg.ThinkingMode,
		disablePromptCaching: cfg.DisablePromptCaching,
	}
}

// resolveThinkingMode returns the effective reasoning API mode for this
// model. An explicit ThinkingMode wins; otherwise it is deduced from
// which knob the caller set (effort -> adaptive, budget -> enabled).
// Returns "" when no reasoning was requested at all.
func (m *Model) resolveThinkingMode() string {
	switch m.thinkingMode {
	case ThinkingModeAdaptive, ThinkingModeEnabled:
		return m.thinkingMode
	}
	if m.thinkingEffort != "" {
		return ThinkingModeAdaptive
	}
	if m.thinkingBudgetTokens > 0 {
		return ThinkingModeEnabled
	}
	return ""
}

// Name returns the model name (e.g. "claude-sonnet-4-5-20250929").
func (m *Model) Name() string {
	return m.modelName
}

// GenerateContent sends the request to Anthropic and returns responses (streaming or single).
func (m *Model) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	if stream {
		return m.generateStream(ctx, req)
	}
	return m.generate(ctx, req)
}

// reasoningRequestOptions returns the per-request options needed for
// the adaptive (effort-based) reasoning API. When the resolved mode is
// adaptive it injects "output_config.effort" directly into the JSON
// body (the typed MessageNewParams has no field for it on the stable
// API; it is added raw via sjson). The matching "thinking":{"type":
// "adaptive"} block is set in buildMessageParams. Returns nil for any
// other mode so the classic path is untouched.
func (m *Model) reasoningRequestOptions() []option.RequestOption {
	if m.resolveThinkingMode() != ThinkingModeAdaptive {
		return nil
	}
	return []option.RequestOption{
		option.WithJSONSet("thinking.type", "adaptive"),
		option.WithJSONSet("output_config.effort", m.thinkingEffort),
	}
}

// generate sends a single request and yields one complete response.
func (m *Model) generate(ctx context.Context, req *model.LLMRequest) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		params, err := m.buildMessageParams(req)
		if err != nil {
			yield(nil, err)
			return
		}

		resp, err := m.client.Messages.New(ctx, params, m.reasoningRequestOptions()...)
		if err != nil {
			yield(nil, err)
			return
		}

		llmResp, err := m.convertResponse(resp)
		if err != nil {
			yield(nil, err)
			return
		}

		yield(llmResp, nil)
	}
}

// generateStream sends a request and yields partial responses as they arrive, then a final complete one.
func (m *Model) generateStream(ctx context.Context, req *model.LLMRequest) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		params, err := m.buildMessageParams(req)
		if err != nil {
			yield(nil, err)
			return
		}

		stream := m.client.Messages.NewStreaming(ctx, params, m.reasoningRequestOptions()...)

		message := anthropic.Message{}

		for stream.Next() {
			event := stream.Current()
			if err := message.Accumulate(event); err != nil {
				yield(nil, err)
				return
			}

			// Yield partial text content
			switch eventVariant := event.AsAny().(type) {
			case anthropic.ContentBlockDeltaEvent:
				switch deltaVariant := eventVariant.Delta.AsAny().(type) {
				case anthropic.TextDelta:
					if deltaVariant.Text != "" {
						part := &genai.Part{Text: deltaVariant.Text}
						llmResp := &model.LLMResponse{
							Content:      &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{part}},
							Partial:      true,
							TurnComplete: false,
						}
						if !yield(llmResp, nil) {
							return
						}
					}
				case anthropic.ThinkingDelta:
					// TODO(test): no coverage for the streaming thinking path.
					// Asserting it needs a mock of the SDK's SSE stream; left as
					// a follow-up. The non-streaming wire body is covered by
					// TestWireBody_* and the block round-trip by TestThinkingBlockRoundTrip.
					if deltaVariant.Thinking != "" {
						part := &genai.Part{Text: deltaVariant.Thinking, Thought: true}
						llmResp := &model.LLMResponse{
							Content:      &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{part}},
							Partial:      true,
							TurnComplete: false,
						}
						if !yield(llmResp, nil) {
							return
						}
					}
				}
			}
		}

		if err := stream.Err(); err != nil {
			yield(nil, err)
			return
		}

		// Build final aggregated response
		llmResp, err := m.convertResponse(&message)
		if err != nil {
			yield(nil, err)
			return
		}

		llmResp.Partial = false
		llmResp.TurnComplete = true
		yield(llmResp, nil)
	}
}

// buildMessageParams converts an LLMRequest into Anthropic's API format (system prompt, messages, tools, config).
func (m *Model) buildMessageParams(req *model.LLMRequest) (anthropic.MessageNewParams, error) {
	// Validate the reasoning configuration up front so misconfiguration
	// fails locally with a clear message instead of as a server-side 400.
	switch m.resolveThinkingMode() {
	case ThinkingModeAdaptive:
		if m.thinkingEffort == "" {
			return anthropic.MessageNewParams{}, fmt.Errorf("anthropic: thinking mode %q requires a non-empty ThinkingEffort", ThinkingModeAdaptive)
		}
	case ThinkingModeEnabled:
		if m.thinkingBudgetTokens <= 0 {
			return anthropic.MessageNewParams{}, fmt.Errorf("anthropic: thinking mode %q requires ThinkingBudgetTokens > 0", ThinkingModeEnabled)
		}
	}

	// Default max tokens (required by Anthropic API)
	maxTokens := int64(4096)
	if m.maxOutputTokens > 0 {
		maxTokens = int64(m.maxOutputTokens)
	}
	if req.Config != nil && req.Config.MaxOutputTokens > 0 {
		maxTokens = int64(req.Config.MaxOutputTokens)
	}

	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(m.modelName),
		MaxTokens: maxTokens,
	}

	// Reasoning: the resolved mode decides the on-the-wire shape.
	//   - adaptive: the whole "thinking":{"type":"adaptive"} object plus
	//     "output_config.effort" is injected as raw JSON at call time via
	//     reasoningRequestOptions (the SDK's typed ThinkingConfigParamUnion
	//     has no "adaptive" variant, only enabled/disabled). Opus 4.5+
	//     require this and reject the classic "enabled" form, so we set
	//     nothing on params.Thinking here.
	//   - enabled: set the classic "thinking":{"type":"enabled",
	//     "budget_tokens":N} block (Claude 3.7 / Sonnet 4 / Opus 4).
	if m.resolveThinkingMode() == ThinkingModeEnabled && m.thinkingBudgetTokens > 0 {
		params.Thinking = anthropic.ThinkingConfigParamUnion{
			OfEnabled: &anthropic.ThinkingConfigEnabledParam{
				BudgetTokens: int64(m.thinkingBudgetTokens),
			},
		}
	}

	// Add system instruction if present
	if req.Config != nil && req.Config.SystemInstruction != nil {
		systemText := extractTextFromContent(req.Config.SystemInstruction)
		if systemText != "" {
			params.System = []anthropic.TextBlockParam{
				{Text: systemText},
			}
		}
	}

	// Convert content messages
	messages := []anthropic.MessageParam{}
	for _, content := range req.Contents {
		msg, err := m.convertContentToMessage(content)
		if err != nil {
			return anthropic.MessageNewParams{}, err
		}
		if msg != nil {
			messages = append(messages, *msg)
		}
	}

	// Repair message history to comply with Anthropic's requirements
	// (each tool_use must have a corresponding tool_result immediately after)
	messages = repairMessageHistory(messages)
	messages = trimFinalAssistantWhitespace(messages)

	params.Messages = messages

	// Apply config settings
	if req.Config != nil {
		if req.Config.Temperature != nil {
			params.Temperature = anthropic.Float(float64(*req.Config.Temperature))
		}
		if req.Config.TopP != nil {
			params.TopP = anthropic.Float(float64(*req.Config.TopP))
		}
		if len(req.Config.StopSequences) > 0 {
			params.StopSequences = req.Config.StopSequences
		}

		// Convert tools
		if len(req.Config.Tools) > 0 {
			tools, err := m.convertTools(req.Config.Tools)
			if err != nil {
				return anthropic.MessageNewParams{}, err
			}
			params.Tools = tools
		}

		// ToolConfig → tool_choice
		//
		// Maps genai.FunctionCallingConfig.Mode to Anthropic's tool_choice:
		//   ModeAuto → {type: "auto"} (default behaviour; model may or may not call a tool)
		//   ModeAny  → {type: "any"}  (model MUST call a tool; use for agentic loops
		//                              that can't handle a plain-text reply)
		//   ModeNone → {type: "none"} (tools disabled for this call even if provided)
		//
		// When AllowedFunctionNames holds exactly one name with ModeAny, Anthropic's
		// equivalent is {type: "tool", name: "..."}. For multiple names we fall back
		// to {type: "any"} because Anthropic's tool variant accepts a single name,
		// not a list — same pragmatic choice as the OpenAI adapter. Callers who need
		// a multi-function allowlist should rely on ModeAny plus prompt-level
		// instructions to pick within the allowed set.
		if req.Config.ToolConfig != nil && req.Config.ToolConfig.FunctionCallingConfig != nil {
			fcc := req.Config.ToolConfig.FunctionCallingConfig
			switch fcc.Mode {
			case genai.FunctionCallingConfigModeAuto:
				params.ToolChoice = anthropic.ToolChoiceUnionParam{
					OfAuto: &anthropic.ToolChoiceAutoParam{},
				}
			case genai.FunctionCallingConfigModeNone:
				params.ToolChoice = anthropic.ToolChoiceUnionParam{
					OfNone: &anthropic.ToolChoiceNoneParam{},
				}
			case genai.FunctionCallingConfigModeAny:
				if len(fcc.AllowedFunctionNames) == 1 {
					params.ToolChoice = anthropic.ToolChoiceParamOfTool(fcc.AllowedFunctionNames[0])
				} else {
					params.ToolChoice = anthropic.ToolChoiceUnionParam{
						OfAny: &anthropic.ToolChoiceAnyParam{},
					}
				}
			}
		}
	}

	// Prompt caching: stamp the cache_control breakpoints LAST so every
	// section (system, tools, repaired messages) is in its final shape.
	// See caching.go for the breakpoint strategy.
	if !m.disablePromptCaching {
		applyCacheControl(&params)
	}

	return params, nil
}

// convertContentToMessage transforms a genai.Content (text, images, tool calls/results) into an Anthropic message.
func (m *Model) convertContentToMessage(content *genai.Content) (*anthropic.MessageParam, error) {
	role := convertRoleToAnthropic(content.Role)

	var blocks []anthropic.ContentBlockParamUnion

	for _, part := range content.Parts {
		// Reasoning parts must be reconstructed as their dedicated block
		// types and placed before tool_use, which is the order Anthropic
		// expects when a turn carries both. A thought Part with non-empty
		// Text is a normal thinking block (echo Text + signature); a
		// thought Part with empty Text carries a redacted_thinking blob in
		// ThoughtSignature. See convertResponse for the inverse mapping.
		//
		// Anthropic only accepts thinking/redacted_thinking blocks inside
		// assistant messages and rejects the whole request with a 400
		// otherwise. Thought parts can legitimately appear under a user
		// role: the ADK contents processor rewrites events authored by a
		// DIFFERENT agent as user-role "For context:" content
		// (ConvertForeignEvent) and passes non-text parts — including
		// signature-only thought parts — through verbatim. Those foreign
		// reasoning blocks are useless as context (their signatures belong
		// to another conversation anyway), so drop them instead of letting
		// the API bounce the request.
		if part.Thought {
			if role != anthropic.MessageParamRoleAssistant {
				continue
			}
			if part.Text != "" {
				blocks = append(blocks, anthropic.ContentBlockParamUnion{
					OfThinking: &anthropic.ThinkingBlockParam{
						Thinking:  part.Text,
						Signature: string(part.ThoughtSignature),
					},
				})
			} else if len(part.ThoughtSignature) > 0 {
				blocks = append(blocks, anthropic.ContentBlockParamUnion{
					OfRedactedThinking: &anthropic.RedactedThinkingBlockParam{
						Data: string(part.ThoughtSignature),
					},
				})
			}
			continue
		}

		if part.Text != "" {
			blocks = append(blocks, anthropic.NewTextBlock(part.Text))
		}

		if part.InlineData != nil {
			block, err := convertInlineDataToBlock(part.InlineData)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, *block)
		}

		if part.FunctionCall != nil {
			input, err := common.MarshalToolPayload(part.FunctionCall.Args)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal function call args: %w", err)
			}
			blocks = append(blocks, anthropic.ContentBlockParamUnion{
				OfToolUse: &anthropic.ToolUseBlockParam{
					ID:    sanitizeToolID(part.FunctionCall.ID),
					Name:  part.FunctionCall.Name,
					Input: input,
				},
			})
		}

		if part.FunctionResponse != nil {
			responseJSON, err := common.MarshalToolPayload(part.FunctionResponse.Response)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal function response: %w", err)
			}
			blocks = append(blocks, anthropic.NewToolResultBlock(sanitizeToolID(part.FunctionResponse.ID), string(responseJSON), false))
		}
	}

	if len(blocks) == 0 {
		return nil, nil
	}

	return &anthropic.MessageParam{Role: role, Content: blocks}, nil
}

// convertResponse transforms Anthropic's response (text, tool_use blocks, usage) into the generic LLMResponse.
func (m *Model) convertResponse(resp *anthropic.Message) (*model.LLMResponse, error) {
	content := &genai.Content{
		Role:  genai.RoleModel,
		Parts: []*genai.Part{},
	}

	// Convert content blocks
	for _, block := range resp.Content {
		switch variant := block.AsAny().(type) {
		case anthropic.TextBlock:
			content.Parts = append(content.Parts, &genai.Part{Text: variant.Text})
		case anthropic.ThinkingBlock:
			// Preserve the reasoning block AND its signature. Anthropic
			// requires the thinking block (with signature) to be echoed back
			// in the next request when the turn also contains tool_use, or
			// the follow-up call is rejected. We carry it as a thought Part:
			// Thought=true, Text=the reasoning, ThoughtSignature=signature.
			content.Parts = append(content.Parts, &genai.Part{
				Text:             variant.Thinking,
				Thought:          true,
				ThoughtSignature: []byte(variant.Signature),
			})
		case anthropic.RedactedThinkingBlock:
			// Encrypted reasoning we cannot read but MUST echo back intact.
			// Convention: a thought Part with empty Text and the opaque blob
			// in ThoughtSignature marks a redacted_thinking block (vs. a
			// normal thinking block, which always has non-empty Text).
			content.Parts = append(content.Parts, &genai.Part{
				Thought:          true,
				ThoughtSignature: []byte(variant.Data),
			})
		case anthropic.ToolUseBlock:
			content.Parts = append(content.Parts, &genai.Part{
				FunctionCall: &genai.FunctionCall{
					ID:   variant.ID,
					Name: variant.Name,
					Args: convertToolInput(variant.Input),
				},
			})
		}
	}

	// Convert usage metadata.
	//
	// With prompt caching active (the default, see caching.go) Anthropic
	// reports the prompt in three buckets: InputTokens is only the
	// un-cached suffix; the cached prefix goes in CacheReadInputTokens
	// and CacheCreationInputTokens. The model still processed the full
	// prompt, so PromptTokenCount is the sum of the three. Without
	// caching both cache fields are zero and the sum is plain
	// InputTokens. CachedContentTokenCount carries the read-hit portion
	// for cost-aware consumers.
	var usageMetadata *genai.GenerateContentResponseUsageMetadata
	promptTokens := resp.Usage.InputTokens +
		resp.Usage.CacheReadInputTokens +
		resp.Usage.CacheCreationInputTokens
	if promptTokens > 0 || resp.Usage.OutputTokens > 0 {
		usageMetadata = &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:        int32(promptTokens),
			CachedContentTokenCount: int32(resp.Usage.CacheReadInputTokens),
			CandidatesTokenCount:    int32(resp.Usage.OutputTokens),
			TotalTokenCount:         int32(promptTokens + resp.Usage.OutputTokens),
		}
	}

	return &model.LLMResponse{
		Content:       content,
		UsageMetadata: usageMetadata,
		FinishReason:  convertStopReason(resp.StopReason),
		TurnComplete:  true,
	}, nil
}

// convertTools transforms genai tool definitions into Anthropic's tool format (name, description, JSON schema).
func (m *Model) convertTools(genaiTools []*genai.Tool) ([]anthropic.ToolUnionParam, error) {
	var tools []anthropic.ToolUnionParam

	for _, genaiTool := range genaiTools {
		if genaiTool == nil {
			continue
		}

		for _, funcDecl := range genaiTool.FunctionDeclarations {
			params := funcDecl.ParametersJsonSchema
			if params == nil {
				params = funcDecl.Parameters
			}

			var inputSchema anthropic.ToolInputSchemaParam
			// Type is required by Anthropic API, must be "object"
			inputSchema.Type = "object"
			if params != nil {
				// ParametersJsonSchema is typically *jsonschema.Schema, not map[string]any.
				// Marshal/unmarshal normalises any concrete type into a plain map so we
				// can extract fields generically. If it is already a map (e.g. built by
				// hand in Go) we use it directly to avoid the round-trip.
				var m map[string]any
				if dm, ok := params.(map[string]any); ok {
					m = dm
				} else {
					jsonBytes, err := json.Marshal(params)
					if err == nil {
						json.Unmarshal(jsonBytes, &m) //nolint:errcheck
					}
				}
				if m != nil {
					lowercaseTypes(m)
					if props, ok := m["properties"]; ok {
						inputSchema.Properties = props
					}
					// After json.Unmarshal, string arrays always arrive as []interface{},
					// never []string, regardless of the source type. We handle both to be
					// defensive: []string covers maps built directly in Go without a JSON
					// round-trip; []interface{} covers the normal unmarshal path.
					switch req := m["required"].(type) {
					case []string:
						inputSchema.Required = req
					case []interface{}:
						strs := make([]string, len(req))
						for i, v := range req {
							strs[i] = fmt.Sprint(v)
						}
						inputSchema.Required = strs
					}
				}
			}

			tools = append(tools, anthropic.ToolUnionParam{
				OfTool: &anthropic.ToolParam{
					Name:        funcDecl.Name,
					Description: anthropic.String(funcDecl.Description),
					InputSchema: inputSchema,
				},
			})
		}
	}

	return tools, nil
}

// convertRoleToAnthropic maps "user"/"model" to Anthropic's role enum (user/assistant).
func convertRoleToAnthropic(role string) anthropic.MessageParamRole {
	switch role {
	case "user":
		return anthropic.MessageParamRoleUser
	case "model":
		return anthropic.MessageParamRoleAssistant
	default:
		return anthropic.MessageParamRoleUser
	}
}

// convertStopReason maps Anthropic's stop reasons (end_turn, max_tokens, tool_use) to genai.FinishReason.
func convertStopReason(reason anthropic.StopReason) genai.FinishReason {
	switch reason {
	case anthropic.StopReasonEndTurn:
		return genai.FinishReasonStop
	case anthropic.StopReasonMaxTokens:
		return genai.FinishReasonMaxTokens
	case anthropic.StopReasonStopSequence:
		return genai.FinishReasonStop
	case anthropic.StopReasonToolUse:
		return genai.FinishReasonStop
	default:
		return genai.FinishReasonUnspecified
	}
}

// convertToolInput converts tool input to map[string]any for storing in genai.FunctionCall.Args.
// Used when receiving tool_use blocks from Anthropic responses.
func convertToolInput(input any) map[string]any {
	if input == nil {
		return map[string]any{}
	}
	if m, ok := input.(map[string]any); ok {
		return m
	}

	// Get JSON bytes: use directly if json.RawMessage, otherwise marshal
	var data []byte
	if raw, ok := input.(json.RawMessage); ok {
		data = raw
	} else {
		var err error
		if data, err = json.Marshal(input); err != nil {
			return map[string]any{}
		}
	}

	var result map[string]any
	if err := json.Unmarshal(data, &result); err != nil {
		return map[string]any{}
	}
	return result
}

// extractTextFromContent concatenates all text parts from a genai.Content with newlines.
func extractTextFromContent(content *genai.Content) string {
	if content == nil {
		return ""
	}
	var texts []string
	for _, part := range content.Parts {
		if part.Text != "" {
			texts = append(texts, part.Text)
		}
	}
	return strings.Join(texts, "\n")
}

// sanitizeToolID replaces invalid tool IDs (chars outside [a-zA-Z0-9_-]) with a SHA256-based valid ID.
func sanitizeToolID(id string) string {
	if anthropicToolIDPattern.MatchString(id) {
		return id
	}

	// Generate a valid ID from the original using SHA256
	hash := sha256.Sum256([]byte(id))
	return "toolu_" + hex.EncodeToString(hash[:16])
}

// repairMessageHistory removes orphaned tool_use blocks (those without a matching tool_result in the next message).
func repairMessageHistory(messages []anthropic.MessageParam) []anthropic.MessageParam {
	if len(messages) == 0 {
		return messages
	}

	result := make([]anthropic.MessageParam, 0, len(messages))

	for i := 0; i < len(messages); i++ {
		msg := messages[i]

		// Check if this assistant message has tool_use blocks
		if msg.Role == anthropic.MessageParamRoleAssistant {
			toolUseIDs := extractToolUseIDs(msg)

			if len(toolUseIDs) > 0 {
				// Check if next message is a user message with matching tool_results
				if i+1 < len(messages) && messages[i+1].Role == anthropic.MessageParamRoleUser {
					toolResultIDs := extractToolResultIDs(messages[i+1])

					// Find which tool_use IDs have matching tool_results
					matchedIDs := make(map[string]bool)
					for _, id := range toolResultIDs {
						matchedIDs[id] = true
					}

					// Filter out unmatched tool_use blocks from this message
					filteredMsg := filterToolUse(msg, matchedIDs)
					if hasContent(filteredMsg) {
						result = append(result, filteredMsg)
					}
					continue
				} else {
					// No following user message with tool_results - remove all tool_use blocks
					filteredMsg := filterToolUse(msg, nil)
					if hasContent(filteredMsg) {
						result = append(result, filteredMsg)
					}
					continue
				}
			}
		}

		result = append(result, msg)
	}

	return result
}

// trimFinalAssistantWhitespace right-trims the last text block of a trailing
// assistant message. Anthropic rejects a request whose final assistant content
// ends in whitespace ("final assistant content cannot end with trailing
// whitespace"): the prefill continues from those exact tokens and a trailing
// space is ambiguous to tokenise. A block left empty after trimming is dropped,
// since empty text blocks are also rejected.
func trimFinalAssistantWhitespace(messages []anthropic.MessageParam) []anthropic.MessageParam {
	if len(messages) == 0 {
		return messages
	}
	last := &messages[len(messages)-1]
	if last.Role != anthropic.MessageParamRoleAssistant {
		return messages
	}

	for i := len(last.Content) - 1; i >= 0; i-- {
		text := last.Content[i].OfText
		if text == nil {
			continue
		}
		text.Text = strings.TrimRight(text.Text, " \t\n\r")
		if text.Text == "" {
			last.Content = append(last.Content[:i], last.Content[i+1:]...)
		}
		break
	}
	return messages
}

// extractToolUseIDs returns all tool_use IDs from an assistant message.
func extractToolUseIDs(msg anthropic.MessageParam) []string {
	var ids []string
	for _, block := range msg.Content {
		if block.OfToolUse != nil {
			ids = append(ids, block.OfToolUse.ID)
		}
	}
	return ids
}

// extractToolResultIDs returns all tool_result IDs from a user message.
func extractToolResultIDs(msg anthropic.MessageParam) []string {
	var ids []string
	for _, block := range msg.Content {
		if block.OfToolResult != nil {
			ids = append(ids, block.OfToolResult.ToolUseID)
		}
	}
	return ids
}

// filterToolUse keeps tool_use blocks whose IDs are in allowedIDs. If allowedIDs is nil, removes all tool_use.
func filterToolUse(msg anthropic.MessageParam, allowedIDs map[string]bool) anthropic.MessageParam {
	var filteredBlocks []anthropic.ContentBlockParamUnion
	for _, block := range msg.Content {
		if block.OfToolUse != nil {
			if allowedIDs != nil && allowedIDs[block.OfToolUse.ID] {
				filteredBlocks = append(filteredBlocks, block)
			}
			continue
		}
		filteredBlocks = append(filteredBlocks, block)
	}
	return anthropic.MessageParam{Role: msg.Role, Content: filteredBlocks}
}

// convertInlineDataToBlock converts inline data to the appropriate Anthropic content block.
// Supports images (jpeg, png, gif, webp), PDFs, and plain text documents.
// Returns an error for unsupported MIME types, matching Gemini's behavior of letting
// the request fail rather than silently dropping content.
func convertInlineDataToBlock(data *genai.Blob) (*anthropic.ContentBlockParamUnion, error) {
	if data == nil {
		return nil, fmt.Errorf("inline data is nil")
	}

	mediaType := data.MIMEType
	base64Data := base64.StdEncoding.EncodeToString(data.Data)

	switch {
	case mediaType == "image/jpeg" || mediaType == "image/jpg" || mediaType == "image/png" ||
		mediaType == "image/gif" || mediaType == "image/webp":
		return &anthropic.ContentBlockParamUnion{
			OfImage: &anthropic.ImageBlockParam{
				Source: anthropic.ImageBlockParamSourceUnion{
					OfBase64: &anthropic.Base64ImageSourceParam{
						MediaType: anthropic.Base64ImageSourceMediaType(mediaType),
						Data:      base64Data,
					},
				},
			},
		}, nil

	case mediaType == "application/pdf":
		return &anthropic.ContentBlockParamUnion{
			OfDocument: &anthropic.DocumentBlockParam{
				Source: anthropic.DocumentBlockParamSourceUnion{
					OfBase64: &anthropic.Base64PDFSourceParam{
						Data: base64Data,
					},
				},
			},
		}, nil

	case strings.HasPrefix(mediaType, "text/"):
		return &anthropic.ContentBlockParamUnion{
			OfDocument: &anthropic.DocumentBlockParam{
				Source: anthropic.DocumentBlockParamSourceUnion{
					OfText: &anthropic.PlainTextSourceParam{
						Data: string(data.Data),
					},
				},
			},
		}, nil

	default:
		return nil, fmt.Errorf("unsupported inline data MIME type for Anthropic: %s", mediaType)
	}
}

// hasContent returns true if the message has at least one content block.
func hasContent(msg anthropic.MessageParam) bool {
	return len(msg.Content) > 0
}

// lowercaseTypes recursively traverses a JSON schema map and lowercases all "type" fields
// to comply with Anthropic's JSON schema validation.
func lowercaseTypes(m map[string]any) {
	for k, v := range m {
		if k == "type" {
			if s, ok := v.(string); ok {
				m[k] = strings.ToLower(s)
			}
		} else if vMap, ok := v.(map[string]any); ok {
			lowercaseTypes(vMap)
		} else if vList, ok := v.([]any); ok {
			for _, item := range vList {
				if itemMap, ok := item.(map[string]any); ok {
					lowercaseTypes(itemMap)
				}
			}
		}
	}
}
