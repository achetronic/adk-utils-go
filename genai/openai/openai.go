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

package openai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"iter"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/shared"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

var _ model.LLM = &Model{}

var (
	ErrNoChoicesInResponse = errors.New("no choices in OpenAI response")
)

// Model implements model.LLM using the official OpenAI Go SDK.
type Model struct {
	client    *openai.Client
	modelName string
}

// Config holds configuration for creating a new Model.
type Config struct {
	// APIKey is the OpenAI API key. If empty, uses OPENAI_API_KEY env var.
	APIKey string
	// BaseURL is the API base URL. Use this for OpenAI-compatible APIs like Ollama.
	// Example: "http://localhost:11434/v1"
	BaseURL string
	// ModelName is the model to use (e.g., "gpt-4o", "qwen3:8b").
	ModelName string
}

// New creates a new Model with the given configuration.
func New(cfg Config) *Model {
	opts := []option.RequestOption{}

	if cfg.APIKey != "" {
		opts = append(opts, option.WithAPIKey(cfg.APIKey))
	}
	if cfg.BaseURL != "" {
		opts = append(opts, option.WithBaseURL(cfg.BaseURL))
	}

	client := openai.NewClient(opts...)

	return &Model{
		client:    &client,
		modelName: cfg.ModelName,
	}
}

// Name implements model.LLM.
func (m *Model) Name() string {
	return m.modelName
}

// GenerateContent implements model.LLM.
func (m *Model) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	if stream {
		return m.generateStream(ctx, req)
	}
	return m.generate(ctx, req)
}

func (m *Model) generate(ctx context.Context, req *model.LLMRequest) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		params, err := m.buildChatCompletionParams(req)
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

func (m *Model) generateStream(ctx context.Context, req *model.LLMRequest) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		params, err := m.buildChatCompletionParams(req)
		if err != nil {
			yield(nil, err)
			return
		}

		stream := m.client.Chat.Completions.NewStreaming(ctx, params)

		acc := openai.ChatCompletionAccumulator{}

		for stream.Next() {
			chunk := stream.Current()
			acc.AddChunk(chunk)

			// Yield partial text content
			if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
				part := &genai.Part{Text: chunk.Choices[0].Delta.Content}
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

		if err := stream.Err(); err != nil {
			yield(nil, err)
			return
		}

		// Build final aggregated response
		finalContent := &genai.Content{
			Role:  genai.RoleModel,
			Parts: []*genai.Part{},
		}

		if len(acc.Choices) > 0 {
			choice := acc.Choices[0]

			// Add text content
			if choice.Message.Content != "" {
				finalContent.Parts = append(finalContent.Parts, &genai.Part{Text: choice.Message.Content})
			}

			// Add tool calls
			for _, toolCall := range choice.Message.ToolCalls {
				finalContent.Parts = append(finalContent.Parts, &genai.Part{
					FunctionCall: &genai.FunctionCall{
						ID:   toolCall.ID,
						Name: toolCall.Function.Name,
						Args: parseJSONArgs(toolCall.Function.Arguments),
					},
				})
			}
		}

		var usageMetadata *genai.GenerateContentResponseUsageMetadata
		if acc.Usage.TotalTokens > 0 {
			usageMetadata = &genai.GenerateContentResponseUsageMetadata{
				PromptTokenCount:     int32(acc.Usage.PromptTokens),
				CandidatesTokenCount: int32(acc.Usage.CompletionTokens),
				TotalTokenCount:      int32(acc.Usage.TotalTokens),
			}
		}

		var finishReason genai.FinishReason
		if len(acc.Choices) > 0 {
			finishReason = convertFinishReason(string(acc.Choices[0].FinishReason))
		}

		finalResp := &model.LLMResponse{
			Content:       finalContent,
			UsageMetadata: usageMetadata,
			FinishReason:  finishReason,
			Partial:       false,
			TurnComplete:  true,
		}
		yield(finalResp, nil)
	}
}

func (m *Model) buildChatCompletionParams(req *model.LLMRequest) (openai.ChatCompletionNewParams, error) {
	messages := []openai.ChatCompletionMessageParamUnion{}

	// Add system instruction if present
	if req.Config != nil && req.Config.SystemInstruction != nil {
		systemText := extractTextFromContent(req.Config.SystemInstruction)
		if systemText != "" {
			messages = append(messages, openai.SystemMessage(systemText))
		}
	}

	// Convert content messages
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

	// Apply config settings
	if req.Config != nil {
		if req.Config.Temperature != nil {
			params.Temperature = openai.Float(float64(*req.Config.Temperature))
		}
		if req.Config.MaxOutputTokens > 0 {
			params.MaxTokens = openai.Int(int64(req.Config.MaxOutputTokens))
		}
		if req.Config.TopP != nil {
			params.TopP = openai.Float(float64(*req.Config.TopP))
		}
		if len(req.Config.StopSequences) > 0 {
			if len(req.Config.StopSequences) == 1 {
				params.Stop = openai.ChatCompletionNewParamsStopUnion{
					OfString: openai.String(req.Config.StopSequences[0]),
				}
			} else {
				params.Stop = openai.ChatCompletionNewParamsStopUnion{
					OfStringArray: req.Config.StopSequences,
				}
			}
		}

		// Handle ThinkingConfig (ReasoningEffort for o-series models)
		if req.Config.ThinkingConfig != nil {
			switch req.Config.ThinkingConfig.ThinkingLevel {
			case genai.ThinkingLevelLow:
				params.ReasoningEffort = shared.ReasoningEffortLow
			case genai.ThinkingLevelHigh:
				params.ReasoningEffort = shared.ReasoningEffortHigh
			default:
				params.ReasoningEffort = shared.ReasoningEffortMedium
			}
		}

		// Handle JSON mode
		if req.Config.ResponseMIMEType == "application/json" {
			params.ResponseFormat = openai.ChatCompletionNewParamsResponseFormatUnion{
				OfJSONObject: &openai.ResponseFormatJSONObjectParam{},
			}
		}

		// Handle response schema
		if req.Config.ResponseSchema != nil {
			schemaMap, err := convertSchema(req.Config.ResponseSchema)
			if err != nil {
				return openai.ChatCompletionNewParams{}, err
			}
			params.ResponseFormat = openai.ChatCompletionNewParamsResponseFormatUnion{
				OfJSONSchema: &openai.ResponseFormatJSONSchemaParam{
					JSONSchema: openai.ResponseFormatJSONSchemaJSONSchemaParam{
						Name:        "response",
						Description: openai.String(req.Config.ResponseSchema.Description),
						Schema:      schemaMap,
						Strict:      openai.Bool(true),
					},
				},
			}
		}

		// Convert tools
		if len(req.Config.Tools) > 0 {
			tools, err := m.convertTools(req.Config.Tools)
			if err != nil {
				return openai.ChatCompletionNewParams{}, err
			}
			params.Tools = tools
		}
	}

	return params, nil
}

func (m *Model) convertContentToMessages(content *genai.Content) ([]openai.ChatCompletionMessageParamUnion, error) {
	var messages []openai.ChatCompletionMessageParamUnion

	// Handle function responses first (they become tool messages)
	for _, part := range content.Parts {
		if part.FunctionResponse != nil {
			responseJSON, err := json.Marshal(part.FunctionResponse.Response)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal function response: %w", err)
			}
			messages = append(messages, openai.ToolMessage(part.FunctionResponse.ID, string(responseJSON)))
		}
	}

	// Collect non-function-response parts
	var textParts []string
	var toolCalls []openai.ChatCompletionMessageToolCallUnionParam
	var imageParts []openai.ChatCompletionContentPartImageParam

	for _, part := range content.Parts {
		if part.FunctionResponse != nil {
			continue // Already handled
		}

		if part.Text != "" {
			textParts = append(textParts, part.Text)
		}

		if part.FunctionCall != nil {
			argsJSON, err := json.Marshal(part.FunctionCall.Args)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal function args: %w", err)
			}
			toolCalls = append(toolCalls, openai.ChatCompletionMessageToolCallUnionParam{
				OfFunction: &openai.ChatCompletionMessageFunctionToolCallParam{
					ID: part.FunctionCall.ID,
					Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{
						Name:      part.FunctionCall.Name,
						Arguments: string(argsJSON),
					},
				},
			})
		}

		if part.InlineData != nil {
			switch part.InlineData.MIMEType {
			case "image/jpg", "image/jpeg", "image/png", "image/gif", "image/webp":
				base64Data := base64.StdEncoding.EncodeToString(part.InlineData.Data)
				dataURL := fmt.Sprintf("data:%s;base64,%s", part.InlineData.MIMEType, base64Data)
				imageParts = append(imageParts, openai.ChatCompletionContentPartImageParam{
					ImageURL: openai.ChatCompletionContentPartImageImageURLParam{
						URL:    dataURL,
						Detail: "auto",
					},
				})
			}
		}
	}

	// Build the message based on role and content
	if len(textParts) > 0 || len(imageParts) > 0 || len(toolCalls) > 0 {
		role := convertRoleToOpenAI(content.Role)

		switch role {
		case "user":
			if len(imageParts) > 0 {
				// Multi-part message with images
				var parts []openai.ChatCompletionContentPartUnionParam
				for _, text := range textParts {
					parts = append(parts, openai.ChatCompletionContentPartUnionParam{
						OfText: &openai.ChatCompletionContentPartTextParam{Text: text},
					})
				}
				for _, img := range imageParts {
					parts = append(parts, openai.ChatCompletionContentPartUnionParam{
						OfImageURL: &img,
					})
				}
				messages = append(messages, openai.ChatCompletionMessageParamUnion{
					OfUser: &openai.ChatCompletionUserMessageParam{
						Content: openai.ChatCompletionUserMessageParamContentUnion{
							OfArrayOfContentParts: parts,
						},
					},
				})
			} else {
				messages = append(messages, openai.UserMessage(joinTexts(textParts)))
			}

		case "assistant":
			msg := openai.ChatCompletionAssistantMessageParam{}
			if len(textParts) > 0 {
				msg.Content = openai.ChatCompletionAssistantMessageParamContentUnion{
					OfString: openai.String(joinTexts(textParts)),
				}
			}
			if len(toolCalls) > 0 {
				msg.ToolCalls = toolCalls
			}
			messages = append(messages, openai.ChatCompletionMessageParamUnion{OfAssistant: &msg})

		case "system":
			messages = append(messages, openai.SystemMessage(joinTexts(textParts)))
		}
	}

	return messages, nil
}

func (m *Model) convertResponse(resp *openai.ChatCompletion) (*model.LLMResponse, error) {
	if len(resp.Choices) == 0 {
		return nil, ErrNoChoicesInResponse
	}

	choice := resp.Choices[0]
	content := &genai.Content{
		Role:  genai.RoleModel,
		Parts: []*genai.Part{},
	}

	// Convert message content
	if choice.Message.Content != "" {
		content.Parts = append(content.Parts, &genai.Part{Text: choice.Message.Content})
	}

	// Convert tool calls
	for _, toolCall := range choice.Message.ToolCalls {
		content.Parts = append(content.Parts, &genai.Part{
			FunctionCall: &genai.FunctionCall{
				ID:   toolCall.ID,
				Name: toolCall.Function.Name,
				Args: parseJSONArgs(toolCall.Function.Arguments),
			},
		})
	}

	// Convert usage metadata
	var usageMetadata *genai.GenerateContentResponseUsageMetadata
	if resp.Usage.TotalTokens > 0 {
		usageMetadata = &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     int32(resp.Usage.PromptTokens),
			CandidatesTokenCount: int32(resp.Usage.CompletionTokens),
			TotalTokenCount:      int32(resp.Usage.TotalTokens),
		}
	}

	return &model.LLMResponse{
		Content:       content,
		UsageMetadata: usageMetadata,
		FinishReason:  convertFinishReason(string(choice.FinishReason)),
		TurnComplete:  true,
	}, nil
}

func (m *Model) convertTools(genaiTools []*genai.Tool) ([]openai.ChatCompletionToolUnionParam, error) {
	var tools []openai.ChatCompletionToolUnionParam

	for _, genaiTool := range genaiTools {
		if genaiTool == nil {
			continue
		}

		// Only function declarations are supported
		for _, funcDecl := range genaiTool.FunctionDeclarations {
			params := funcDecl.ParametersJsonSchema
			if params == nil {
				params = funcDecl.Parameters
			}

			var funcParams shared.FunctionParameters
			if params != nil {
				// Try direct map[string]any first
				if m, ok := params.(map[string]any); ok {
					funcParams = shared.FunctionParameters(m)
				} else {
					// Handle other types (e.g., *jsonschema.Schema from ADK functiontool)
					// by converting via JSON serialization
					jsonBytes, err := json.Marshal(params)
					if err == nil {
						var m map[string]any
						if json.Unmarshal(jsonBytes, &m) == nil {
							funcParams = shared.FunctionParameters(m)
						}
					}
				}
			}

			tools = append(tools, openai.ChatCompletionFunctionTool(shared.FunctionDefinitionParam{
				Name:        funcDecl.Name,
				Description: openai.String(funcDecl.Description),
				Parameters:  funcParams,
			}))
		}
	}

	return tools, nil
}

func convertSchema(schema *genai.Schema) (map[string]any, error) {
	if schema == nil {
		return map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		}, nil
	}

	result := make(map[string]any)

	if schema.Type != genai.TypeUnspecified {
		result["type"] = convertSchemaType(schema.Type)
	}

	if schema.Description != "" {
		result["description"] = schema.Description
	}

	if len(schema.Properties) > 0 {
		properties := make(map[string]any)
		for propName, propSchema := range schema.Properties {
			convertedProp, err := convertSchema(propSchema)
			if err != nil {
				return nil, err
			}
			properties[propName] = convertedProp
		}
		result["properties"] = properties
	}

	if len(schema.Required) > 0 {
		result["required"] = schema.Required
	}

	if schema.Items != nil {
		items, err := convertSchema(schema.Items)
		if err != nil {
			return nil, err
		}
		result["items"] = items
	}

	if len(schema.Enum) > 0 {
		result["enum"] = schema.Enum
	}

	return result, nil
}

func convertSchemaType(t genai.Type) string {
	switch t {
	case genai.TypeString:
		return "string"
	case genai.TypeNumber:
		return "number"
	case genai.TypeInteger:
		return "integer"
	case genai.TypeBoolean:
		return "boolean"
	case genai.TypeArray:
		return "array"
	case genai.TypeObject:
		return "object"
	default:
		return "string"
	}
}

func convertRoleToOpenAI(role string) string {
	switch role {
	case "user":
		return "user"
	case "model":
		return "assistant"
	case "system":
		return "system"
	default:
		return "user"
	}
}

func convertFinishReason(reason string) genai.FinishReason {
	switch reason {
	case "stop":
		return genai.FinishReasonStop
	case "length":
		return genai.FinishReasonMaxTokens
	case "tool_calls", "function_call":
		return genai.FinishReasonStop
	case "content_filter":
		return genai.FinishReasonSafety
	default:
		return genai.FinishReasonUnspecified
	}
}

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
	return joinTexts(texts)
}

func joinTexts(texts []string) string {
	if len(texts) == 0 {
		return ""
	}
	if len(texts) == 1 {
		return texts[0]
	}
	result := ""
	for i, text := range texts {
		if i > 0 {
			result += "\n"
		}
		result += text
	}
	return result
}

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
