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
	client    *anthropic.Client
	modelName string
}

// HTTPOptions holds optional HTTP-level configuration for the Anthropic client.
type HTTPOptions struct {
	Headers http.Header
}

// Config holds configuration for creating a new Model.
type Config struct {
	// APIKey is the Anthropic API key. If empty, uses ANTHROPIC_API_KEY env var.
	APIKey string
	// BaseURL is the API base URL (optional, for custom endpoints).
	BaseURL string
	// ModelName is the model to use (e.g., "claude-sonnet-4-5-20250929").
	ModelName string
	// HTTPOptions holds optional HTTP-level overrides (e.g. extra headers).
	HTTPOptions HTTPOptions
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
	for k, vals := range cfg.HTTPOptions.Headers {
		for _, v := range vals {
			opts = append(opts, option.WithHeaderAdd(k, v))
		}
	}

	client := anthropic.NewClient(opts...)

	return &Model{
		client:    &client,
		modelName: cfg.ModelName,
	}
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

// generate sends a single request and yields one complete response.
func (m *Model) generate(ctx context.Context, req *model.LLMRequest) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		params, err := m.buildMessageParams(req)
		if err != nil {
			yield(nil, err)
			return
		}

		resp, err := m.client.Messages.New(ctx, params)
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

		stream := m.client.Messages.NewStreaming(ctx, params)

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
	// Default max tokens (required by Anthropic API)
	maxTokens := int64(4096)
	if req.Config != nil && req.Config.MaxOutputTokens > 0 {
		maxTokens = int64(req.Config.MaxOutputTokens)
	}

	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(m.modelName),
		MaxTokens: maxTokens,
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
	}

	return params, nil
}

// convertContentToMessage transforms a genai.Content (text, images, tool calls/results) into an Anthropic message.
func (m *Model) convertContentToMessage(content *genai.Content) (*anthropic.MessageParam, error) {
	role := convertRoleToAnthropic(content.Role)

	var blocks []anthropic.ContentBlockParamUnion

	for _, part := range content.Parts {
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
			blocks = append(blocks, anthropic.ContentBlockParamUnion{
				OfToolUse: &anthropic.ToolUseBlockParam{
					ID:    sanitizeToolID(part.FunctionCall.ID),
					Name:  part.FunctionCall.Name,
					Input: convertToolInputToRaw(part.FunctionCall.Args),
				},
			})
		}

		if part.FunctionResponse != nil {
			responseJSON, err := json.Marshal(part.FunctionResponse.Response)
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

	// Convert usage metadata
	var usageMetadata *genai.GenerateContentResponseUsageMetadata
	if resp.Usage.InputTokens > 0 || resp.Usage.OutputTokens > 0 {
		usageMetadata = &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     int32(resp.Usage.InputTokens),
			CandidatesTokenCount: int32(resp.Usage.OutputTokens),
			TotalTokenCount:      int32(resp.Usage.InputTokens + resp.Usage.OutputTokens),
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

// emptyJSONObject is the JSON representation of an empty object.
var emptyJSONObject = json.RawMessage(`{}`)

// convertToolInputToRaw converts tool input to json.RawMessage for sending to Anthropic API.
// Handles nil values and nil maps inside interfaces by returning "{}".
func convertToolInputToRaw(input any) json.RawMessage {
	if input == nil {
		return emptyJSONObject
	}

	// If already json.RawMessage, use directly
	if raw, ok := input.(json.RawMessage); ok && len(raw) > 0 {
		return raw
	}

	// Marshal to JSON (handles nil maps inside interface correctly)
	data, err := json.Marshal(input)
	if err != nil || len(data) == 0 || string(data) == "null" {
		return emptyJSONObject
	}
	return data
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
