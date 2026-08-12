// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

// Package openai provides an OpenAI-compatible LLM implementation for the ADK.
// It supports both native OpenAI API and compatible providers like Ollama.
package openai

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"log/slog"
	"mime"
	"net/http"
	"strings"
	"sync"

	"github.com/achetronic/adk-utils-go/genai/common"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/shared"
	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"
)

var _ model.LLM = &Model{}

var (
	ErrNoChoicesInResponse = errors.New("no choices in OpenAI response")
)

// OpenAI enforces a 40-character limit on tool_call_id fields.
const maxToolCallIDLength = 40

// Model implements model.LLM using the official OpenAI Go SDK.
// Works with OpenAI API and compatible providers (Ollama, vLLM, etc.).
type Model struct {
	client    *openai.Client
	modelName string

	// dialect carries the provider-specific behaviour this model opts into.
	// Nil means OpenAI-pure: nothing is read, sent or reshaped beyond the
	// documented OpenAI shapes. The capability fields below are the dialect
	// asserted once in New, so the hot path is a nil check, not a type
	// assertion.
	dialect          Dialect
	toolIDNormalizer ToolIDNormalizer
	paramsAdjuster   ParamsAdjuster
	reasoningDecoder ReasoningDecoder
	reasoningEncoder ReasoningEncoder
	usageDecoder     UsageDecoder
	thinkingMapper   ThinkingMapper

	// reasoningEgress is normalised in New, so the converters never
	// re-derive a default.
	reasoningEgress ReasoningEgressMode

	// extraBody holds provider extensions merged into the root of every
	// request body. It is a copy of the caller's map, read-only from here on,
	// so concurrent requests can share it.
	extraBody map[string]any

	// toolCallIDMap stores original IDs when the wire shape differs from
	// what ADK uses. Keys are the wire IDs, values the original IDs.
	toolCallIDMap   map[string]string
	toolCallIDMapMu sync.RWMutex
}

// HTTPOptions holds optional HTTP-level configuration for the OpenAI client.
type HTTPOptions struct {
	Client  *http.Client
	Headers http.Header
}

// Config holds the configuration for creating an OpenAI Model.
type Config struct {
	// APIKey for authentication. Falls back to OPENAI_API_KEY env var if empty.
	APIKey string
	// BaseURL for the API endpoint. Use for OpenAI-compatible providers.
	// Example: "http://localhost:11434/v1" for Ollama.
	BaseURL string
	// ModelName specifies which model to use (e.g., "gpt-4o", "qwen3:8b").
	ModelName string
	// HTTPOptions holds optional HTTP-level overrides (e.g. extra headers).
	HTTPOptions HTTPOptions
	// Dialect layers provider-specific wire behaviour onto the adapter:
	// reasoning fields on ingest and egress, tool_call_id shapes, usage
	// buckets outside the standard object, and a last pass over the request
	// params. See the Dialect type for the capability interfaces. Nil (the
	// zero value) keeps the adapter OpenAI-pure, which matches OpenAI's own
	// API: its reasoning models never expose the reasoning text in Chat
	// Completions responses, only a token count, so there is nothing to
	// read or send back. Compatible providers plug a dialect in:
	// NewTextDialect() for a plain-text reasoning field (Kimi, Mistral,
	// vLLM, ...), DeepSeek for the same fields plus the provider's strict
	// replay rule, OpenRouter for the structured reasoning_details shape,
	// or a custom implementation for a wire shape this package does not
	// know yet.
	Dialect Dialect
	// ReasoningEgress selects how thought Parts are sent back as history.
	// Empty (the zero value) means ReasoningEgressNative. Use
	// ReasoningEgressThinkTags for backends that reject an extra reasoning
	// field, or ReasoningEgressOmit to send no reasoning at all. Without
	// reasoning to send the mode changes nothing.
	ReasoningEgress ReasoningEgressMode
	// ExtraBody carries provider extensions that Chat Completions does not
	// define, merged into the root of every request body. OpenRouter's
	// reasoning controls live here, for example:
	//
	//	ExtraBody: map[string]any{
	//		"reasoning": map[string]any{"effort": "high"},
	//	}
	//
	// Values must be JSON-serialisable. A key that collides with a field the
	// adapter sets replaces it on the wire, so this is an extension point, not
	// a way to rewrite messages or model. The map is copied at construction,
	// so mutating the caller's copy afterwards changes nothing.
	ExtraBody map[string]any
}

// New creates a new OpenAI Model with the given configuration.
func New(cfg Config) *Model {
	var opts []option.RequestOption

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

	client := openai.NewClient(opts...)

	// An unrecognised mode degrades to the native shape reasoning providers
	// document rather than to silent data loss.
	reasoningEgress := cfg.ReasoningEgress
	switch reasoningEgress {
	case ReasoningEgressNative, ReasoningEgressThinkTags, ReasoningEgressOmit:
	default:
		reasoningEgress = ReasoningEgressNative
	}
	// A dialect that knows its provider's replay rules vetoes shapes the
	// provider rejects; the override is logged once, at construction.
	if policy, ok := cfg.Dialect.(EgressPolicy); ok {
		if effective := policy.ResolveEgress(reasoningEgress); effective != reasoningEgress {
			slog.Warn("reasoning egress overridden by the dialect",
				"model", cfg.ModelName,
				"dialect", cfg.Dialect.Name(),
				"requested", string(reasoningEgress),
				"effective", string(effective),
			)
			reasoningEgress = effective
		}
	}

	// Copied so a caller mutating its own map later cannot race with a
	// request in flight.
	var extraBody map[string]any
	if len(cfg.ExtraBody) > 0 {
		extraBody = make(map[string]any, len(cfg.ExtraBody))
		for k, v := range cfg.ExtraBody {
			extraBody[k] = v
		}
	}

	m := &Model{
		client:          &client,
		modelName:       cfg.ModelName,
		dialect:         cfg.Dialect,
		reasoningEgress: reasoningEgress,
		extraBody:       extraBody,
		toolCallIDMap:   make(map[string]string),
	}
	// Capabilities are asserted once here so the conversion path only nil
	// checks. A dialect picks exactly the areas it needs.
	if d := cfg.Dialect; d != nil {
		m.toolIDNormalizer, _ = d.(ToolIDNormalizer)
		m.paramsAdjuster, _ = d.(ParamsAdjuster)
		m.reasoningDecoder, _ = d.(ReasoningDecoder)
		m.reasoningEncoder, _ = d.(ReasoningEncoder)
		m.usageDecoder, _ = d.(UsageDecoder)
		m.thinkingMapper, _ = d.(ThinkingMapper)
	}
	return m
}

// Name returns the model name.
func (m *Model) Name() string {
	return m.modelName
}

// GenerateContent sends a request to the LLM and returns responses.
// Set stream=true for streaming responses, false for a single response.
func (m *Model) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	if stream {
		return m.generateStream(ctx, req)
	}
	return m.generate(ctx, req)
}

// generate sends a non-streaming request and yields a single response.
func (m *Model) generate(ctx context.Context, req *model.LLMRequest) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		params, err := m.buildChatCompletionParams(req, false)
		if err != nil {
			yield(nil, err)
			return
		}

		resp, err := m.client.Chat.Completions.New(ctx, params)
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

// generateStream sends a streaming request and yields partial responses
// as they arrive, followed by a final aggregated response.
func (m *Model) generateStream(ctx context.Context, req *model.LLMRequest) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		params, err := m.buildChatCompletionParams(req, true)
		if err != nil {
			yield(nil, err)
			return
		}

		stream := m.client.Chat.Completions.NewStreaming(ctx, params)
		acc := openai.ChatCompletionAccumulator{}
		// Reasoning is accumulated here rather than read back off acc at the
		// end: the SDK accumulator merges chunks field by field and keeps no
		// raw JSON on the aggregated message, so every non-standard field
		// (all reasoning is non-standard) is gone by the time the stream
		// ends. Without a reasoning decoder there is nothing to accumulate,
		// so this stays nil and every decode call below is skipped.
		var reasoningAcc ReasoningAccumulator
		if m.reasoningDecoder != nil {
			reasoningAcc = m.reasoningDecoder.NewAccumulator()
		}

		// Yield partial responses as chunks arrive
		for stream.Next() {
			chunk := stream.Current()
			acc.AddChunk(chunk)

			if len(chunk.Choices) == 0 {
				continue
			}

			delta := chunk.Choices[0].Delta
			// Reasoning arrives in fields the official Chat Completions schema
			// does not define, so the dialect reads it from the raw JSON
			// envelope rather than from a typed field on Delta.
			var reasoningParts []*genai.Part
			if reasoningAcc != nil {
				reasoningParts = m.reasoningDecoder.DecodeDelta(delta.RawJSON())
				reasoningAcc.AddDelta(reasoningParts)
			}

			if delta.Content == "" && len(reasoningParts) == 0 {
				continue
			}

			// Order is significant: reasoning tokens are emitted before the
			// final answer tokens, so the Part order mirrors the temporal
			// order in which the model produced them. Downstream consumers
			// (e.g. ADK's llmagent) iterate parts and filter on Thought, so
			// having reasoning first matches the natural transcript order.
			parts := make([]*genai.Part, 0, len(reasoningParts)+1)
			parts = append(parts, reasoningParts...)
			if delta.Content != "" {
				parts = append(parts, &genai.Part{Text: delta.Content})
			}

			llmResp := &model.LLMResponse{
				Content: &genai.Content{
					Role:  genai.RoleModel,
					Parts: parts,
				},
				Partial:      true,
				TurnComplete: false,
			}
			if !yield(llmResp, nil) {
				return
			}
		}

		if err := stream.Err(); err != nil {
			yield(nil, err)
			return
		}

		// Build and yield final aggregated response
		yield(m.buildStreamFinalResponse(&acc, reasoningAcc), nil)
	}
}

// buildStreamFinalResponse creates the final LLMResponse from accumulated
// stream data. Reasoning comes from the adapter's own accumulator, not from
// acc: the SDK aggregate keeps no raw JSON, so there is nothing to read
// there. The accumulator is nil when no dialect decodes reasoning.
func (m *Model) buildStreamFinalResponse(acc *openai.ChatCompletionAccumulator, reasoning ReasoningAccumulator) *model.LLMResponse {
	content := &genai.Content{
		Role:  genai.RoleModel,
		Parts: []*genai.Part{},
	}

	// Reasoning Parts go before the final-answer Part to preserve the
	// temporal order in which the model produced the tokens.
	if reasoning != nil {
		content.Parts = append(content.Parts, reasoning.Parts()...)
	}

	if len(acc.Choices) > 0 {
		choice := acc.Choices[0]

		if choice.Message.Content != "" {
			content.Parts = append(content.Parts, &genai.Part{Text: choice.Message.Content})
		}

		for _, tc := range choice.Message.ToolCalls {
			content.Parts = append(content.Parts, &genai.Part{
				FunctionCall: &genai.FunctionCall{
					ID:   tc.ID,
					Name: tc.Function.Name,
					Args: parseJSONArgs(tc.Function.Arguments),
				},
			})
		}
	}

	var finishReason genai.FinishReason
	if len(acc.Choices) > 0 {
		finishReason = convertFinishReason(string(acc.Choices[0].FinishReason))
	}

	return &model.LLMResponse{
		Content:       content,
		UsageMetadata: m.decodeUsageMetadata(acc.Usage),
		FinishReason:  finishReason,
		Partial:       false,
		TurnComplete:  true,
	}
}

// buildChatCompletionParams converts an LLMRequest into OpenAI API
// parameters. stream reports whether this call will stream; the dialect's
// ParamsAdjuster sees it, because stream-only incompatibilities exist
// (stream_options on some gateways).
func (m *Model) buildChatCompletionParams(req *model.LLMRequest, stream bool) (openai.ChatCompletionNewParams, error) {
	var messages []openai.ChatCompletionMessageParamUnion

	// Add system instruction
	if req.Config != nil && req.Config.SystemInstruction != nil {
		if text := extractText(req.Config.SystemInstruction); text != "" {
			messages = append(messages, openai.SystemMessage(text))
		}
	}

	// Convert conversation messages
	for _, content := range req.Contents {
		msgs, err := m.convertContentToMessages(content)
		if err != nil {
			return openai.ChatCompletionNewParams{}, err
		}
		messages = append(messages, msgs...)
	}

	params := openai.ChatCompletionNewParams{
		Model:    openai.ChatModel(m.modelName),
		Messages: messages,
	}

	// Apply optional configuration
	if req.Config != nil {
		m.applyGenerationConfig(&params, req.Config)
	}

	// Streaming requests opt into the final usage chunk. Without
	// stream_options.include_usage the server never emits it, the
	// accumulator's Usage stays zero, and the terminal LLMResponse carries
	// empty UsageMetadata - leaving consumers no way to price a streamed
	// turn. Set before the dialect's last pass so a gateway that rejects
	// the field can still strip it.
	if stream {
		params.StreamOptions.IncludeUsage = param.NewOpt(true)
	}

	// Provider extensions go on next so they land at the root of the body.
	// The SDK has no typed field for them, so they travel through the same
	// extra-fields escape hatch the reasoning fields use.
	if len(m.extraBody) > 0 {
		params.SetExtraFields(m.extraBody)
	}

	// The dialect's ParamsAdjuster fires last: it sees the exact body the
	// wire will get, params and extra fields alike, and mutates it in place.
	if m.paramsAdjuster != nil {
		m.paramsAdjuster.AdjustParams(&params, req, stream)
	}

	return params, nil
}

// applyGenerationConfig applies optional generation settings to the request params.
func (m *Model) applyGenerationConfig(params *openai.ChatCompletionNewParams, cfg *genai.GenerateContentConfig) {
	if cfg.Temperature != nil {
		params.Temperature = openai.Float(float64(*cfg.Temperature))
	}
	if cfg.MaxOutputTokens > 0 {
		params.MaxTokens = openai.Int(int64(cfg.MaxOutputTokens))
	}
	if cfg.TopP != nil {
		params.TopP = openai.Float(float64(*cfg.TopP))
	}

	// Stop sequences
	if len(cfg.StopSequences) == 1 {
		params.Stop = openai.ChatCompletionNewParamsStopUnion{
			OfString: openai.String(cfg.StopSequences[0]),
		}
	} else if len(cfg.StopSequences) > 1 {
		params.Stop = openai.ChatCompletionNewParamsStopUnion{
			OfStringArray: cfg.StopSequences,
		}
	}

	// Reasoning effort: the dialect owns the mapping when it implements
	// one. The native knob varies by provider (OpenRouter's effort lives in
	// a reasoning object at the request root, vLLM and Qwen use
	// enable_thinking), so one typed field cannot serve them all. Without
	// a mapper the typed OpenAI field is used.
	if cfg.ThinkingConfig != nil {
		if m.thinkingMapper != nil {
			m.thinkingMapper.ApplyThinkingLevel(params, cfg.ThinkingConfig.ThinkingLevel)
		} else {
			params.ReasoningEffort = convertThinkingLevel(cfg.ThinkingConfig.ThinkingLevel)
		}
	}

	// JSON mode
	if cfg.ResponseMIMEType == "application/json" {
		params.ResponseFormat = openai.ChatCompletionNewParamsResponseFormatUnion{
			OfJSONObject: &openai.ResponseFormatJSONObjectParam{},
		}
	}

	// Structured output with schema
	if cfg.ResponseSchema != nil {
		if schemaMap, err := convertSchema(cfg.ResponseSchema); err == nil {
			params.ResponseFormat = openai.ChatCompletionNewParamsResponseFormatUnion{
				OfJSONSchema: &openai.ResponseFormatJSONSchemaParam{
					JSONSchema: openai.ResponseFormatJSONSchemaJSONSchemaParam{
						Name:        "response",
						Description: openai.String(cfg.ResponseSchema.Description),
						Schema:      schemaMap,
						Strict:      openai.Bool(true),
					},
				},
			}
		}
	}

	// Tools
	if len(cfg.Tools) > 0 {
		if tools, err := m.convertTools(cfg.Tools); err == nil {
			params.Tools = tools
		}
	}

	// ToolConfig -> tool_choice
	//
	// Maps genai.FunctionCallingConfig.Mode to OpenAI's tool_choice:
	//   ModeAuto -> "auto"   (default behaviour; model may or may not call a tool)
	//   ModeAny  -> "required" (model MUST call a tool; use for agentic loops
	//                         that can't handle a plain-text reply)
	//   ModeNone -> "none"   (tools disabled for this call even if provided)
	//
	// When AllowedFunctionNames is set with ModeAny, OpenAI's equivalent is a
	// named function choice - we pick the first name since OpenAI's
	// tool_choice accepts only one specific function, not a list. Callers who
	// need a multi-function allowlist should rely on ModeAny plus prompt-level
	// instructions to pick within the allowed set.
	if cfg.ToolConfig != nil && cfg.ToolConfig.FunctionCallingConfig != nil {
		fcc := cfg.ToolConfig.FunctionCallingConfig
		switch fcc.Mode {
		case genai.FunctionCallingConfigModeAuto:
			params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{
				OfAuto: openai.String("auto"),
			}
		case genai.FunctionCallingConfigModeNone:
			params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{
				OfAuto: openai.String("none"),
			}
		case genai.FunctionCallingConfigModeAny:
			if len(fcc.AllowedFunctionNames) == 1 {
				params.ToolChoice = openai.ToolChoiceOptionFunctionToolChoice(
					openai.ChatCompletionNamedToolChoiceFunctionParam{
						Name: fcc.AllowedFunctionNames[0],
					},
				)
			} else {
				params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{
					OfAuto: openai.String("required"),
				}
			}
		}
	}
}

// convertContentToMessages converts a genai.Content into OpenAI message format.
// Handles text, images, audio, files, function calls, and function responses.
func (m *Model) convertContentToMessages(content *genai.Content) ([]openai.ChatCompletionMessageParamUnion, error) {
	var messages []openai.ChatCompletionMessageParamUnion
	var textParts []string
	var thoughts []*genai.Part
	var toolCalls []openai.ChatCompletionMessageToolCallUnionParam
	var mediaParts []openai.ChatCompletionContentPartUnionParam

	for _, part := range content.Parts {
		switch {
		// Reasoning is collected apart from the reply text: merging the two
		// into content hides the chain of thought inside the answer and
		// leaves out the field the strict thinking providers require to be
		// echoed back. This branch comes first because a thought Part also
		// carries its text in Text.
		case part.Thought:
			thoughts = append(thoughts, part)

		case part.FunctionResponse != nil:
			// Skip ADK internal framework parts (HITL confirmation
			// protocol): upstream providers reject them as undeclared
			// tool messages.
			if common.IsADKInternalCall(part.FunctionResponse.Name) {
				continue
			}
			responseJSON, err := common.MarshalToolPayload(part.FunctionResponse.Response)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal function response: %w", err)
			}
			normalizedID := m.normalizeToolCallID(part.FunctionResponse.ID)
			messages = append(messages, openai.ToolMessage(string(responseJSON), normalizedID))

		case part.FunctionCall != nil:
			if common.IsADKInternalCall(part.FunctionCall.Name) {
				continue
			}
			argsJSON, err := common.MarshalToolPayload(part.FunctionCall.Args)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal function args: %w", err)
			}
			normalizedID := m.normalizeToolCallID(part.FunctionCall.ID)
			toolCalls = append(toolCalls, openai.ChatCompletionMessageToolCallUnionParam{
				OfFunction: &openai.ChatCompletionMessageFunctionToolCallParam{
					ID: normalizedID,
					Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{
						Name:      part.FunctionCall.Name,
						Arguments: string(argsJSON),
					},
				},
			})

		case part.Text != "":
			textParts = append(textParts, part.Text)

		case part.InlineData != nil:
			p, err := convertInlineDataToPart(part.InlineData)
			if err != nil {
				return nil, err
			}
			mediaParts = append(mediaParts, *p)

		case part.FileData != nil:
			p, err := convertFileDataToPart(part.FileData)
			if err != nil {
				return nil, err
			}
			mediaParts = append(mediaParts, *p)
		}
	}

	// Reasoning only belongs to an assistant turn. ADK's contents processor
	// rewrites events authored by a different agent as user-role "For
	// context:" content and passes non-text parts through verbatim, so a
	// thought Part can legitimately arrive under a user role; that reasoning
	// belongs to another conversation, and no provider accepts it on a user
	// message, so it is dropped.
	var reasoningExtra map[string]any
	if convertRole(content.Role) == "assistant" && len(thoughts) > 0 {
		switch m.reasoningEgress {
		case ReasoningEgressThinkTags:
			// The reasoning becomes part of content, ahead of the reply,
			// which is the order the model produced it in. A new slice keeps
			// the converter read-only over its input. A thought Part without
			// readable text, an encrypted block's Part, contributes nothing.
			if text := thoughtTextFor(thoughts); text != "" {
				textParts = append([]string{"<think>\n" + text + "\n</think>"}, textParts...)
			}
		case ReasoningEgressOmit:
			// Nothing to send, in any shape.
		default:
			// Native mode delegates the wire shape to the dialect. A dialect
			// without an encoder degrades to think tags: the trace survives
			// in content instead of being dropped, which a session replayed
			// against a closed-schema backend still tolerates.
			if m.reasoningEncoder != nil {
				reasoningExtra = m.reasoningEncoder.EncodeReasoning(thoughts)
			} else if text := thoughtTextFor(thoughts); text != "" {
				textParts = append([]string{"<think>\n" + text + "\n</think>"}, textParts...)
			}
		}
	}

	// Reasoning alone does not produce a message: an assistant message with
	// neither content nor tool_calls is not a valid Chat Completions message,
	// and there is no other turn to attach the field to.
	if len(textParts) > 0 || len(mediaParts) > 0 || len(toolCalls) > 0 {
		msg := m.buildRoleMessage(content.Role, textParts, mediaParts, toolCalls, reasoningExtra)
		if msg != nil {
			messages = append(messages, *msg)
		}
	}

	return messages, nil
}

// buildRoleMessage creates the appropriate message type based on role.
// reasoningExtra is only meaningful for the assistant role; the other roles
// have no field to carry it.
func (m *Model) buildRoleMessage(role string, texts []string, media []openai.ChatCompletionContentPartUnionParam, toolCalls []openai.ChatCompletionMessageToolCallUnionParam, reasoningExtra map[string]any) *openai.ChatCompletionMessageParamUnion {
	switch convertRole(role) {
	case "user":
		return buildUserMessage(texts, media)
	case "assistant":
		return buildAssistantMessage(texts, toolCalls, reasoningExtra)
	case "system":
		msg := openai.SystemMessage(joinTexts(texts))
		return &msg
	}
	return nil
}

// buildUserMessage creates a user message, with multi-part support for media.
func buildUserMessage(texts []string, media []openai.ChatCompletionContentPartUnionParam) *openai.ChatCompletionMessageParamUnion {
	if len(media) == 0 {
		msg := openai.UserMessage(joinTexts(texts))
		return &msg
	}

	var parts []openai.ChatCompletionContentPartUnionParam
	for _, text := range texts {
		parts = append(parts, openai.ChatCompletionContentPartUnionParam{
			OfText: &openai.ChatCompletionContentPartTextParam{Text: text},
		})
	}
	parts = append(parts, media...)

	return &openai.ChatCompletionMessageParamUnion{
		OfUser: &openai.ChatCompletionUserMessageParam{
			Content: openai.ChatCompletionUserMessageParamContentUnion{
				OfArrayOfContentParts: parts,
			},
		},
	}
}

// buildAssistantMessage creates an assistant message with optional tool calls
// and reasoning. No reasoning field is defined by the Chat Completions
// schema, so reasoning attaches through the SDK's extra-fields escape hatch
// instead of typed fields. The extra map comes from the dialect's encoder:
// OpenRouter writes the reasoning_details array, TextDialect the plain-text
// field, each exactly as the provider expects.
func buildAssistantMessage(texts []string, toolCalls []openai.ChatCompletionMessageToolCallUnionParam, reasoningExtra map[string]any) *openai.ChatCompletionMessageParamUnion {
	msg := openai.ChatCompletionAssistantMessageParam{}

	if len(texts) > 0 {
		msg.Content = openai.ChatCompletionAssistantMessageParamContentUnion{
			OfString: openai.String(joinTexts(texts)),
		}
	}
	if len(toolCalls) > 0 {
		msg.ToolCalls = toolCalls
	}
	if len(reasoningExtra) > 0 {
		msg.SetExtraFields(reasoningExtra)
	}

	return &openai.ChatCompletionMessageParamUnion{OfAssistant: &msg}
}

// convertResponse transforms an OpenAI response into an LLMResponse.
func (m *Model) convertResponse(resp *openai.ChatCompletion) (*model.LLMResponse, error) {
	if len(resp.Choices) == 0 {
		return nil, ErrNoChoicesInResponse
	}

	choice := resp.Choices[0]
	content := &genai.Content{
		Role:  genai.RoleModel,
		Parts: []*genai.Part{},
	}

	// Reasoning lives in fields openai-go does not type, so the dialect
	// reads it from the raw JSON envelope. Without a reasoning decoder
	// nothing is read, which is what OpenAI's own reasoning models need:
	// they never expose the reasoning text on this API.
	if m.reasoningDecoder != nil {
		content.Parts = append(content.Parts, m.reasoningDecoder.DecodeMessage(choice.Message.RawJSON())...)
	}

	if choice.Message.Content != "" {
		content.Parts = append(content.Parts, &genai.Part{Text: choice.Message.Content})
	}

	for _, tc := range choice.Message.ToolCalls {
		content.Parts = append(content.Parts, &genai.Part{
			FunctionCall: &genai.FunctionCall{
				ID:   tc.ID,
				Name: tc.Function.Name,
				Args: parseJSONArgs(tc.Function.Arguments),
			},
		})
	}

	return &model.LLMResponse{
		Content:       content,
		UsageMetadata: m.decodeUsageMetadata(resp.Usage),
		FinishReason:  convertFinishReason(string(choice.FinishReason)),
		TurnComplete:  true,
	}, nil
}

// decodeUsageMetadata builds the standard usage mapping and lets the dialect
// fold in buckets the provider reports outside the standard object. The
// dialect sees the raw usage envelope; the SDK's raw JSON is empty only for
// hand-built structs, never for a live response.
func (m *Model) decodeUsageMetadata(usage openai.CompletionUsage) *genai.GenerateContentResponseUsageMetadata {
	metadata := convertUsageMetadata(usage)
	if metadata == nil || m.usageDecoder == nil {
		return metadata
	}
	m.usageDecoder.DecodeUsage(usage.RawJSON(), metadata)
	return metadata
}

// convertTools transforms genai tools into OpenAI function tool format.
func (m *Model) convertTools(genaiTools []*genai.Tool) ([]openai.ChatCompletionToolUnionParam, error) {
	var tools []openai.ChatCompletionToolUnionParam

	for _, genaiTool := range genaiTools {
		if genaiTool == nil {
			continue
		}

		for _, funcDecl := range genaiTool.FunctionDeclarations {
			params := funcDecl.ParametersJsonSchema
			if params == nil {
				params = funcDecl.Parameters
			}

			tools = append(tools, openai.ChatCompletionFunctionTool(shared.FunctionDefinitionParam{
				Name:        funcDecl.Name,
				Description: openai.String(funcDecl.Description),
				Parameters:  convertToFunctionParams(params),
			}))
		}
	}

	return tools, nil
}

// convertToFunctionParams converts various parameter types to OpenAI format.
// OpenAI requires object schemas to have a "properties" field, even if empty.
func convertToFunctionParams(params any) shared.FunctionParameters {
	if params == nil {
		return nil
	}

	var m map[string]any

	// Direct map
	if dm, ok := params.(map[string]any); ok {
		m = dm
	} else {
		// Convert via JSON for other types (e.g., *jsonschema.Schema)
		jsonBytes, err := json.Marshal(params)
		if err != nil {
			return nil
		}
		if json.Unmarshal(jsonBytes, &m) != nil {
			return nil
		}
	}

	// Standardise types to lowercase for JSON schema compliance
	lowercaseTypes(m)
	// OpenAI requires "properties" for object types
	ensureObjectProperties(m)

	return shared.FunctionParameters(m)
}

// ensureObjectProperties recursively ensures all object schemas have a properties field.
func ensureObjectProperties(schema map[string]any) {
	if schema == nil {
		return
	}

	// If type is "object" and no properties, add empty properties
	if t, ok := schema["type"].(string); ok && t == "object" {
		if _, hasProps := schema["properties"]; !hasProps {
			schema["properties"] = map[string]any{}
		}
	}

	// Recursively process nested properties
	if props, ok := schema["properties"].(map[string]any); ok {
		for _, prop := range props {
			if propMap, ok := prop.(map[string]any); ok {
				ensureObjectProperties(propMap)
			}
		}
	}

	// Process array items
	if items, ok := schema["items"].(map[string]any); ok {
		ensureObjectProperties(items)
	}
}

// lowercaseTypes recursively traverses a JSON schema map and lowercases all "type" fields
// to comply with standard JSON schema validation.
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

// convertSchema recursively converts a genai.Schema to OpenAI JSON schema format.
func convertSchema(schema *genai.Schema) (map[string]any, error) {
	if schema == nil {
		return map[string]any{"type": "object", "properties": map[string]any{}}, nil
	}

	result := make(map[string]any)

	if schema.Type != genai.TypeUnspecified {
		result["type"] = schemaTypeToString(schema.Type)
	}
	if schema.Description != "" {
		result["description"] = schema.Description
	}
	if len(schema.Required) > 0 {
		result["required"] = schema.Required
	}
	if len(schema.Enum) > 0 {
		result["enum"] = schema.Enum
	}

	if len(schema.Properties) > 0 {
		props := make(map[string]any)
		for name, propSchema := range schema.Properties {
			converted, err := convertSchema(propSchema)
			if err != nil {
				return nil, err
			}
			props[name] = converted
		}
		result["properties"] = props
	}

	if schema.Items != nil {
		items, err := convertSchema(schema.Items)
		if err != nil {
			return nil, err
		}
		result["items"] = items
	}

	return result, nil
}

// normalizeToolCallID produces the wire shape the provider accepts for tool
// call IDs. A dialect's ToolIDNormalizer owns it entirely when present; the
// adapter keeps the wire-to-original mapping either way so ADK keeps seeing
// its own IDs. Without a dialect the OpenAI rule applies: IDs over 40
// characters are hashed shorter, the rest pass through.
func (m *Model) normalizeToolCallID(id string) string {
	if m.toolIDNormalizer != nil {
		normalized := m.toolIDNormalizer.NormalizeToolID(id)
		if normalized != id {
			m.toolCallIDMapMu.Lock()
			m.toolCallIDMap[normalized] = id
			m.toolCallIDMapMu.Unlock()
		}
		return normalized
	}

	if len(id) <= maxToolCallIDLength {
		return id
	}

	hash := sha256.Sum256([]byte(id))
	shortID := "tc_" + hex.EncodeToString(hash[:])[:maxToolCallIDLength-3]

	m.toolCallIDMapMu.Lock()
	m.toolCallIDMap[shortID] = id
	m.toolCallIDMapMu.Unlock()

	return shortID
}

// denormalizeToolCallID restores the original ID from a wire one.
func (m *Model) denormalizeToolCallID(shortID string) string {
	m.toolCallIDMapMu.RLock()
	defer m.toolCallIDMapMu.RUnlock()

	if original, exists := m.toolCallIDMap[shortID]; exists {
		return original
	}
	return shortID
}

// --- Helper functions ---

// convertInlineDataToPart converts inline data to the appropriate OpenAI content part.
// Supports images (as data URI), audio (wav, mp3), and generic files (PDF, etc.).
// Returns an error for unsupported MIME types, matching Gemini's behavior of letting
// the request fail rather than silently dropping content.
func convertInlineDataToPart(data *genai.Blob) (*openai.ChatCompletionContentPartUnionParam, error) {
	if data == nil {
		return nil, fmt.Errorf("inline data is nil")
	}

	mediaType := normalizeMIMEType(data.MIMEType)
	base64Data := base64.StdEncoding.EncodeToString(data.Data)

	switch {
	case mediaType == "image/jpeg" || mediaType == "image/jpg" || mediaType == "image/png" ||
		mediaType == "image/gif" || mediaType == "image/webp":
		return &openai.ChatCompletionContentPartUnionParam{
			OfImageURL: &openai.ChatCompletionContentPartImageParam{
				ImageURL: openai.ChatCompletionContentPartImageImageURLParam{
					URL:    fmt.Sprintf("data:%s;base64,%s", mediaType, base64Data),
					Detail: "auto",
				},
			},
		}, nil

	case mediaType == "audio/wav" || mediaType == "audio/mp3" ||
		mediaType == "audio/mpeg" || mediaType == "audio/webm":
		format := "wav"
		if mediaType == "audio/mp3" || mediaType == "audio/mpeg" {
			format = "mp3"
		}
		return &openai.ChatCompletionContentPartUnionParam{
			OfInputAudio: &openai.ChatCompletionContentPartInputAudioParam{
				InputAudio: openai.ChatCompletionContentPartInputAudioInputAudioParam{
					Data:   base64Data,
					Format: format,
				},
			},
		}, nil

	case mediaType == "application/pdf" || strings.HasPrefix(mediaType, "text/"):
		return &openai.ChatCompletionContentPartUnionParam{
			OfFile: &openai.ChatCompletionContentPartFileParam{
				File: openai.ChatCompletionContentPartFileFileParam{
					FileData: openai.String(fmt.Sprintf("data:%s;base64,%s", mediaType, base64Data)),
				},
			},
		}, nil

	default:
		return nil, fmt.Errorf("unsupported inline data MIME type for OpenAI: %s", mediaType)
	}
}

// convertFileDataToPart maps a FileData part to an image_url content part;
// the URI goes through verbatim, nothing is downloaded. Images only: audio
// and files need uploaded bytes (InlineData). Plain http is allowed for
// OpenAI-compatible gateways (Ollama, vLLM, ...). Other schemes error
// instead of being silently dropped; gs://, the scheme genai.FileData
// documents, gets its own message pointing at InlineData. DisplayName is
// dropped: image_url has no field for it.
func convertFileDataToPart(data *genai.FileData) (*openai.ChatCompletionContentPartUnionParam, error) {
	if data == nil {
		return nil, fmt.Errorf("file data is nil")
	}
	if strings.HasPrefix(data.FileURI, "gs://") {
		return nil, fmt.Errorf("file data URI %q is a Google Cloud Storage URI, which OpenAI cannot fetch: download the bytes and use InlineData instead", data.FileURI)
	}
	if !strings.HasPrefix(data.FileURI, "https://") && !strings.HasPrefix(data.FileURI, "http://") {
		return nil, fmt.Errorf("file data URI must be an http(s) URL for OpenAI, got %q", data.FileURI)
	}

	mediaType := normalizeMIMEType(data.MIMEType)
	switch mediaType {
	case "image/jpeg", "image/jpg", "image/png", "image/gif", "image/webp":
		return &openai.ChatCompletionContentPartUnionParam{
			OfImageURL: &openai.ChatCompletionContentPartImageParam{
				ImageURL: openai.ChatCompletionContentPartImageImageURLParam{
					URL:    data.FileURI,
					Detail: "auto",
				},
			},
		}, nil

	default:
		return nil, fmt.Errorf("unsupported file data MIME type for OpenAI: %s", mediaType)
	}
}

// normalizeMIMEType strips parameters from a MIME string ("image/png;
// charset=utf-8" becomes "image/png") so the converters match on the media
// type alone. Malformed strings come back unchanged and fail the match with
// the caller's input in the error.
func normalizeMIMEType(mimeType string) string {
	mediaType, _, err := mime.ParseMediaType(mimeType)
	if err != nil {
		return mimeType
	}
	return mediaType
}

// convertUsageMetadata converts OpenAI usage stats to genai format.
//
// ReasoningTokens (billed as output tokens by o-series, gpt-5.x and
// reasoning-exposing compatible providers) maps to ThoughtsTokenCount;
// CachedTokens (the cache-hit subset of PromptTokens, billed at a distinct
// input rate) maps to CachedContentTokenCount. Both are documented Chat
// Completions fields, so they are always mapped. A provider that omits a
// detail leaves it at zero, and genai's omitempty drops it on the wire.
func convertUsageMetadata(usage openai.CompletionUsage) *genai.GenerateContentResponseUsageMetadata {
	if usage.TotalTokens == 0 {
		return nil
	}
	return &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:        int32(usage.PromptTokens),
		CandidatesTokenCount:    int32(usage.CompletionTokens),
		TotalTokenCount:         int32(usage.TotalTokens),
		ThoughtsTokenCount:      int32(usage.CompletionTokensDetails.ReasoningTokens),
		CachedContentTokenCount: int32(usage.PromptTokensDetails.CachedTokens),
	}
}

// convertRole maps genai roles to OpenAI roles.
func convertRole(role string) string {
	if role == "model" {
		return "assistant"
	}
	return role // "user" and "system" are the same
}

// convertFinishReason maps OpenAI finish reasons to genai format.
func convertFinishReason(reason string) genai.FinishReason {
	switch reason {
	case "stop", "tool_calls", "function_call":
		return genai.FinishReasonStop
	case "length":
		return genai.FinishReasonMaxTokens
	case "content_filter":
		return genai.FinishReasonSafety
	default:
		return genai.FinishReasonUnspecified
	}
}

// convertThinkingLevel maps genai thinking levels to OpenAI reasoning effort.
func convertThinkingLevel(level genai.ThinkingLevel) shared.ReasoningEffort {
	switch level {
	case genai.ThinkingLevelLow:
		return shared.ReasoningEffortLow
	case genai.ThinkingLevelHigh:
		return shared.ReasoningEffortHigh
	default:
		return shared.ReasoningEffortMedium
	}
}

// schemaTypeToString converts genai.Type to JSON schema type string.
func schemaTypeToString(t genai.Type) string {
	types := map[genai.Type]string{
		genai.TypeString:  "string",
		genai.TypeNumber:  "number",
		genai.TypeInteger: "integer",
		genai.TypeBoolean: "boolean",
		genai.TypeArray:   "array",
		genai.TypeObject:  "object",
	}
	if s, ok := types[t]; ok {
		return s
	}
	return "string"
}

// extractText extracts all text parts from a Content and joins them.
func extractText(content *genai.Content) string {
	if content == nil {
		return ""
	}
	var texts []string
	for _, part := range content.Parts {
		if part.Text != "" {
			texts = append(texts, part.Text)
		}
	}
	return joinTexts(texts)
}

// joinTexts joins multiple text strings with newlines.
func joinTexts(texts []string) string {
	return strings.Join(texts, "\n")
}

// parseJSONArgs parses a JSON string into a map. Returns empty map on error.
func parseJSONArgs(argsJSON string) map[string]any {
	if argsJSON == "" {
		return make(map[string]any)
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return make(map[string]any)
	}
	return args
}
