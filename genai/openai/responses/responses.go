// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

// Package responses provides an OpenAI Responses API (/v1/responses)
// implementation for the ADK.
//
// The Responses API is the interface OpenAI recommends for new applications,
// with native reasoning, built-in tools, and structured output.
//
// This adapter drives the API statelessly to match ADK's model: ADK owns the
// conversation state and passes the full history on every call, so each request
// replays that history as input items instead of chaining server-side state via
// previous_response_id. Requests are sent with store=false so nothing is
// persisted server-side, and reasoning is requested with encrypted_content so
// reasoning items can be replayed across turns: this keeps the model's chain
// of thought available between tool calls, as reasoning models require.
// Reasoning items without encrypted content (e.g. from gateways that do not
// support the include parameter) are skipped on replay, since their bare IDs
// only resolve in the context of the originating response.
//
// Encrypted reasoning content is bound to the credentials that produced it:
// the API encrypts it with the organization's key, and replaying it through a
// different provider, API key, or model fails with a 400
// invalid_encrypted_content error. Sessions may switch channels mid-flight
// (e.g. OpenAI to Azure fallback), so each reasoning part records the origin
// (a fingerprint of base URL, API key, and model) and is replayed only when
// it matches the requesting model; otherwise the encrypted content is
// dropped and the turn degrades gracefully to fresh reasoning.
//
// For OpenAI-compatible gateways (Ollama, vLLM, DeepSeek, Kimi, etc.) that only
// expose the Chat Completions endpoint, use the parent openai package instead.
package responses

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"maps"
	"mime"
	"net/http"
	"os"
	"path"
	"slices"
	"sort"
	"strings"

	"github.com/achetronic/adk-utils-go/genai/common"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"
)

var _ model.LLM = &Model{}

var (
	ErrNoOutputInResponse = errors.New("no output items in OpenAI Responses API response")
	// ErrNoConsumableOutput reports a response whose output items were all
	// skipped (unknown item types, or known items carrying no content), so
	// converting it would produce an empty completed turn.
	ErrNoConsumableOutput = errors.New("no consumable output items in OpenAI Responses API response")
	// ErrFunctionCallArgs reports function call arguments that do not decode
	// into a JSON object.
	ErrFunctionCallArgs = errors.New("undecodable function call arguments in OpenAI Responses API response")
)

// Model implements model.LLM using the OpenAI Responses API.
type Model struct {
	client    *openai.Client
	modelName string
	// origin fingerprints the channel (base URL, API key, model) this Model
	// talks to. Encrypted reasoning content is only replayed to the channel
	// that produced it; see the package documentation.
	origin string
}

// HTTPOptions holds optional HTTP-level configuration for the OpenAI client.
type HTTPOptions struct {
	Client  *http.Client
	Headers http.Header
}

// Config holds the configuration for creating a Responses API Model.
type Config struct {
	// APIKey for authentication. Falls back to OPENAI_API_KEY env var if empty.
	APIKey string
	// BaseURL for the API endpoint.
	BaseURL string
	// ModelName specifies which model to use (e.g., "gpt-4o", "gpt-5.4").
	ModelName string
	// HTTPOptions holds optional HTTP-level overrides (e.g. extra headers).
	HTTPOptions HTTPOptions
}

// New creates a new Responses API Model with the given configuration.
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

	apiKey := cfg.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("OPENAI_API_KEY")
	}

	return &Model{
		client:    &client,
		modelName: cfg.ModelName,
		origin:    computeOrigin(cfg.BaseURL, apiKey, cfg.ModelName),
	}
}

// computeOrigin fingerprints a channel so encrypted reasoning content is only
// replayed to the channel that produced it. The API encrypts reasoning with
// the organization's key, so any change of provider, API key, or model makes
// previously captured content undecryptable (400 invalid_encrypted_content).
// The fingerprint deliberately covers all three dimensions.
func computeOrigin(baseURL, apiKey, modelName string) string {
	h := sha256.New()
	for _, s := range []string{baseURL, apiKey, modelName} {
		h.Write([]byte(s))
		h.Write([]byte{0})
	}
	return hex.EncodeToString(h.Sum(nil))[:16]
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
		params, err := m.buildResponseParams(req)
		if err != nil {
			yield(nil, err)
			return
		}

		resp, err := m.client.Responses.New(ctx, params)
		if err != nil {
			yield(nil, err)
			return
		}

		llmResp, err := convertResponse(resp, m.origin)
		if err != nil {
			yield(nil, err)
			return
		}

		yield(llmResp, nil)
	}
}

// generateStream sends a streaming request and yields partial responses
// as they arrive, followed by a final aggregated response.
//
// Streamed deltas (Partial=true) are display-only; ADK persists only the final
// non-partial event. The stream must therefore always end with a complete
// aggregated event. Some OpenAI-compatible gateways omit the aggregated output
// from response.completed, or close the connection without any terminal event
// at all. Relying solely on the server-provided output then yields an empty
// final event, so the assistant turn is lost from history on reload even though
// it streamed fine. As a fallback the deltas are accumulated locally,
// mirroring the Chat Completions adapter.
func (m *Model) generateStream(ctx context.Context, req *model.LLMRequest) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		params, err := m.buildResponseParams(req)
		if err != nil {
			yield(nil, err)
			return
		}

		stream := m.client.Responses.NewStreaming(ctx, params)
		// The iterator returns early on terminal events, so the body is
		// closed explicitly; leaving it open leaks the connection.
		defer stream.Close()

		var acc streamAccumulator
		for stream.Next() {
			event := stream.Current()

			switch event.Type {
			case "response.output_text.delta":
				if event.Delta == "" {
					continue
				}
				acc.addText(event.ItemID, event.OutputIndex, event.Delta)
				llmResp := &model.LLMResponse{
					Content: &genai.Content{
						Role:  genai.RoleModel,
						Parts: []*genai.Part{{Text: event.Delta}},
					},
					Partial:      true,
					TurnComplete: false,
				}
				if !yield(llmResp, nil) {
					return
				}

			case "response.refusal.delta":
				if event.Delta == "" {
					continue
				}
				acc.addRefusal(event.ItemID, event.OutputIndex, event.Delta)
				llmResp := &model.LLMResponse{
					Content: &genai.Content{
						Role:  genai.RoleModel,
						Parts: []*genai.Part{{Text: event.Delta}},
					},
					Partial:      true,
					TurnComplete: false,
				}
				if !yield(llmResp, nil) {
					return
				}

			case "response.reasoning_summary_text.delta":
				if event.Delta == "" {
					continue
				}
				acc.reasoning.WriteString(event.Delta)
				llmResp := &model.LLMResponse{
					Content: &genai.Content{
						Role:  genai.RoleModel,
						Parts: []*genai.Part{{Text: event.Delta, Thought: true}},
					},
					Partial:      true,
					TurnComplete: false,
				}
				if !yield(llmResp, nil) {
					return
				}

			case "response.output_item.done":
				// Completed function calls are accumulated so the fallback
				// final event includes them: losing a tool call would
				// silently break the agent loop.
				if event.Item.Type == "function_call" {
					args, err := functionCallArgs(event.Item)
					if err != nil {
						yield(nil, err)
						return
					}
					acc.functionCalls = append(acc.functionCalls, &genai.FunctionCall{
						ID:   event.Item.CallID,
						Name: event.Item.Name,
						Args: args,
					})
				}
				// Complete items are kept as well: when the terminal event
				// lacks aggregated output they rebuild the turn with message
				// IDs, phase, and encrypted reasoning intact, which the bare
				// delta fallback cannot.
				switch event.Item.Type {
				case "message", "reasoning", "function_call":
					acc.items = append(acc.items, doneItem{
						index: event.OutputIndex,
						item:  event.Item,
					})
				}

			case "response.completed", "response.incomplete":
				resp := &event.Response
				llmResp, err := convertResponse(resp, m.origin)
				if err != nil && !errors.Is(err, ErrNoOutputInResponse) && !errors.Is(err, ErrNoConsumableOutput) {
					// A malformed terminal payload (e.g. broken function
					// arguments) is a real error, not a missing-output case;
					// degrading to the delta fallback would swallow it.
					yield(nil, err)
					return
				}
				if err != nil || hasNoContent(llmResp) {
					// The terminal event carried no aggregated output: rebuild the
					// final response from what the stream delivered, otherwise ADK
					// would persist an empty event and the turn would be lost.
					llmResp = acc.fallbackResponse(resp, m.origin)
					if hasNoContent(llmResp) {
						// Nothing streamed either: surface why the turn is
						// empty instead of yielding it as completed.
						if err == nil {
							err = ErrNoConsumableOutput
						}
						yield(nil, err)
						return
					}
				}
				yield(llmResp, nil)
				return

			case "response.failed":
				errMsg := event.Response.Error.Message
				if errMsg == "" {
					errMsg = "response generation failed"
				}
				yield(nil, fmt.Errorf("openai responses api: %s", errMsg))
				return

			case "error":
				yield(nil, fmt.Errorf("openai responses api stream error: %s", event.Message))
				return
			}
		}

		if err := stream.Err(); err != nil {
			yield(nil, err)
			return
		}

		// The stream ended without any terminal event (some gateways just close
		// the connection after the last delta). Synthesize the final event from
		// what was accumulated so the turn is persisted and ADK does not raise
		// "last event is not final". A stream that closed without delivering
		// anything at all is an error, not a silent zero-event iteration:
		// overloaded gateways answer 200 with an empty SSE body.
		if acc.hasContent() || len(acc.items) > 0 {
			shadow := responses.Response{Status: responses.ResponseStatusCompleted}
			yield(acc.fallbackResponse(&shadow, m.origin), nil)
		} else {
			yield(nil, ErrNoConsumableOutput)
		}
	}
}

// streamAccumulator collects streamed deltas so a complete final response
// can be rebuilt when the terminal event lacks the aggregated output.
// reasoning holds the reasoning summary; texts and refusals hold delta text
// per output item, so the fallback can tell which items the done events
// covered; functionCalls holds tool calls completed during the stream;
// items holds the complete output items from output_item.done events.
type streamAccumulator struct {
	reasoning     strings.Builder
	texts         []*deltaBucket
	refusals      []*deltaBucket
	functionCalls []*genai.FunctionCall
	items         []doneItem
}

// deltaBucket accumulates streamed delta text for one output item; index is
// the item's output_index, so uncovered buckets merge back in stream order.
type deltaBucket struct {
	itemID string
	index  int64
	text   strings.Builder
}

// doneItem pairs a complete output item with its output_index so the
// fallback rebuilds the output array in order regardless of event arrival.
type doneItem struct {
	index int64
	item  responses.ResponseOutputItemUnion
}

func appendDelta(buckets []*deltaBucket, itemID string, index int64, delta string) []*deltaBucket {
	for _, b := range buckets {
		if b.itemID == itemID {
			b.text.WriteString(delta)
			return buckets
		}
	}
	b := &deltaBucket{itemID: itemID, index: index}
	b.text.WriteString(delta)
	return append(buckets, b)
}

func (a *streamAccumulator) addText(itemID string, index int64, delta string) {
	a.texts = appendDelta(a.texts, itemID, index, delta)
}

func (a *streamAccumulator) addRefusal(itemID string, index int64, delta string) {
	a.refusals = appendDelta(a.refusals, itemID, index, delta)
}

// joinDeltas concatenates bucket text in arrival order.
func joinDeltas(buckets []*deltaBucket) string {
	var sb strings.Builder
	for _, b := range buckets {
		sb.WriteString(b.text.String())
	}
	return sb.String()
}

// fallbackResponse rebuilds the final turn when the terminal event lacked
// aggregated output. Complete done items are preferred: they keep message
// IDs, phase, status, and encrypted reasoning for the next stateless
// replay. Deltas the done items did not cover are merged in, and bare
// deltas alone are the last resort.
func (a *streamAccumulator) fallbackResponse(resp *responses.Response, origin string) *model.LLMResponse {
	if len(a.items) > 0 {
		merged := append(append([]doneItem{}, a.items...), a.uncoveredBuckets()...)
		sort.SliceStable(merged, func(i, j int) bool {
			return merged[i].index < merged[j].index
		})
		output := make([]responses.ResponseOutputItemUnion, len(merged))
		for i, di := range merged {
			output[i] = di.item
		}
		shadow := *resp
		shadow.Output = output
		if r, err := convertResponse(&shadow, origin); err == nil && !hasNoContent(r) {
			a.fillMissingReasoning(r)
			return r
		}
	}
	return a.finalResponse(
		convertStatus(resp.Status, resp.IncompleteDetails),
		convertUsageMetadata(resp.Usage),
	)
}

// uncoveredBuckets turns delta buckets whose item never produced a done
// event into synthetic message items: a gateway may complete only some
// items while others streamed as bare deltas, and dropping either side
// would lose output the consumer already saw. Sorting the result with the
// done items by output_index restores the stream order. Buckets without an
// item ID (a gateway that omits item_id) fall back to a coarse check
// against any text the done message items carry.
func (a *streamAccumulator) uncoveredBuckets() []doneItem {
	doneMessageIDs := map[string]bool{}
	doneHasText := false
	for _, di := range a.items {
		if di.item.Type != "message" {
			continue
		}
		if di.item.ID != "" {
			doneMessageIDs[di.item.ID] = true
		}
		for _, cp := range di.item.Content {
			if cp.Text != "" || cp.Refusal != "" {
				doneHasText = true
			}
		}
	}
	covered := func(b *deltaBucket) bool {
		if b.itemID == "" {
			return doneHasText
		}
		return doneMessageIDs[b.itemID]
	}

	var synthetic []doneItem
	for _, b := range a.texts {
		if !covered(b) {
			synthetic = append(synthetic, doneItem{index: b.index, item: responses.ResponseOutputItemUnion{
				Type:    "message",
				ID:      b.itemID,
				Role:    "assistant",
				Status:  "completed",
				Content: []responses.ResponseOutputMessageContentUnion{{Type: "output_text", Text: b.text.String()}},
			}})
		}
	}
	for _, b := range a.refusals {
		if !covered(b) {
			synthetic = append(synthetic, doneItem{index: b.index, item: responses.ResponseOutputItemUnion{
				Type:    "message",
				ID:      b.itemID,
				Role:    "assistant",
				Status:  "completed",
				Content: []responses.ResponseOutputMessageContentUnion{{Type: "refusal", Refusal: b.text.String()}},
			}})
		}
	}
	return synthetic
}

// fillMissingReasoning prepends the delta reasoning summary when the done
// items did not include a reasoning item, so streamed thoughts are not
// dropped from the persisted turn.
func (a *streamAccumulator) fillMissingReasoning(r *model.LLMResponse) {
	if a.reasoning.Len() == 0 {
		return
	}
	for _, p := range r.Content.Parts {
		if p.Thought {
			return
		}
	}
	r.Content.Parts = append(
		[]*genai.Part{{Text: a.reasoning.String(), Thought: true}},
		r.Content.Parts...,
	)
}

// hasContent reports whether anything was accumulated that could be used to
// rebuild a final response.
func (a *streamAccumulator) hasContent() bool {
	return a.reasoning.Len() > 0 || len(a.texts) > 0 || len(a.refusals) > 0 ||
		len(a.functionCalls) > 0
}

// finalResponse builds a non-partial final response from the accumulated deltas
// (for ADK to persist). The reasoning part precedes the answer part, which
// precedes the tool calls, matching the temporal order in which they streamed.
func (a *streamAccumulator) finalResponse(
	finishReason genai.FinishReason,
	usage *genai.GenerateContentResponseUsageMetadata,
) *model.LLMResponse {
	content := &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{}}
	if a.reasoning.Len() > 0 {
		content.Parts = append(content.Parts, &genai.Part{Text: a.reasoning.String(), Thought: true})
	}
	if text := joinDeltas(a.texts); text != "" {
		content.Parts = append(content.Parts, &genai.Part{Text: text})
	}
	if refusal := joinDeltas(a.refusals); refusal != "" {
		content.Parts = append(content.Parts, &genai.Part{
			Text:         refusal,
			PartMetadata: map[string]any{"refusal": true},
		})
	}
	for _, fc := range a.functionCalls {
		content.Parts = append(content.Parts, &genai.Part{FunctionCall: fc})
	}
	return &model.LLMResponse{
		Content:       content,
		UsageMetadata: usage,
		FinishReason:  finishReason,
		Partial:       false,
		TurnComplete:  true,
	}
}

// hasNoContent reports whether convertResponse produced no persistable content
// parts, used to decide whether to fall back to the accumulated deltas.
func hasNoContent(resp *model.LLMResponse) bool {
	return resp == nil || resp.Content == nil || len(resp.Content.Parts) == 0
}

// buildResponseParams converts an LLMRequest into Responses API parameters.
func (m *Model) buildResponseParams(req *model.LLMRequest) (responses.ResponseNewParams, error) {
	params := responses.ResponseNewParams{
		Model: shared.ResponsesModel(m.modelName),
		// ADK owns the conversation state, so nothing needs to be stored
		// server-side. store=false also makes the API return encrypted
		// reasoning content (requested via include below), which is the only
		// way to replay reasoning items in a stateless flow.
		Store: param.NewOpt(false),
		Include: []responses.ResponseIncludable{
			responses.ResponseIncludableReasoningEncryptedContent,
		},
	}

	// The system instruction maps to the instructions field.
	if req.Config != nil && req.Config.SystemInstruction != nil {
		if text := extractText(req.Config.SystemInstruction); text != "" {
			params.Instructions = param.NewOpt(text)
		}
	}

	// The conversation history maps to input items. Unpaired calls and
	// outputs are dropped on the way: ADK histories can end mid-tool-call
	// (cancel, compaction, agent switch), and the API rejects a
	// function_call whose output never arrived, and the reverse.
	dangling := danglingCallIDs(req.Contents)
	var input responses.ResponseInputParam
	for _, content := range req.Contents {
		items, err := convertContentToInputItems(content, m.origin, dangling)
		if err != nil {
			return responses.ResponseNewParams{}, err
		}
		input = append(input, items...)
	}
	if len(input) > 0 {
		params.Input.OfInputItemList = input
	}

	// Generation config
	if req.Config != nil {
		if err := applyGenerationConfig(&params, req.Config); err != nil {
			return responses.ResponseNewParams{}, err
		}
	}

	return params, nil
}

// applyGenerationConfig applies optional generation settings to the request
// params. StopSequences is ignored: the Responses API has no stop parameter,
// so unlike the Chat Completions adapter there is nothing to map it to.
func applyGenerationConfig(params *responses.ResponseNewParams, cfg *genai.GenerateContentConfig) error {
	if cfg.Temperature != nil {
		params.Temperature = param.NewOpt(float64(*cfg.Temperature))
	}
	if cfg.MaxOutputTokens > 0 {
		params.MaxOutputTokens = param.NewOpt(int64(cfg.MaxOutputTokens))
	}
	if cfg.TopP != nil {
		params.TopP = param.NewOpt(float64(*cfg.TopP))
	}

	// Reasoning (native support via Responses API)
	if cfg.ThinkingConfig != nil {
		params.Reasoning = shared.ReasoningParam{
			Effort: convertThinkingLevel(cfg.ThinkingConfig.ThinkingLevel),
		}
		// Summaries are requested only when the caller wants thoughts
		// surfaced; encrypted reasoning replay works without them.
		if cfg.ThinkingConfig.IncludeThoughts {
			params.Reasoning.Summary = shared.ReasoningSummaryAuto
		}
	}

	// JSON mode
	if cfg.ResponseMIMEType == "application/json" {
		params.Text = responses.ResponseTextConfigParam{
			Format: responses.ResponseFormatTextConfigUnionParam{
				OfJSONObject: &shared.ResponseFormatJSONObjectParam{},
			},
		}
	}

	// Structured output with schema (also strict-normalised). Conversion
	// errors are returned rather than swallowed: silently dropping the
	// schema or the tools would send the request anyway and surface as
	// inexplicable model behaviour.
	if cfg.ResponseSchema != nil {
		schemaMap, err := convertSchema(cfg.ResponseSchema)
		if err != nil {
			return fmt.Errorf("failed to convert response schema: %w", err)
		}
		// Strict mode requires an object root; a primitive or array root
		// (which the typed schema allows) goes non-strict instead of
		// building a request the API rejects.
		strict := isObjectSchema(schemaMap)
		if strict {
			normalizeStrictSchema(schemaMap)
		}
		params.Text = responses.ResponseTextConfigParam{
			Format: responses.ResponseFormatTextConfigUnionParam{
				OfJSONSchema: &responses.ResponseFormatTextJSONSchemaConfigParam{
					Name:        "response",
					Description: param.NewOpt(cfg.ResponseSchema.Description),
					Schema:      schemaMap,
					Strict:      param.NewOpt(strict),
				},
			},
		}
	}

	// A raw JSON schema (ResponseJsonSchema) goes through the same pipeline
	// as tool parameters: strict when it fits the subset, non-strict
	// otherwise, never silently ignored. The typed ResponseSchema wins when
	// both are set.
	if cfg.ResponseSchema == nil && cfg.ResponseJsonSchema != nil {
		schemaMap, strict := convertFunctionParams(cfg.ResponseJsonSchema)
		if schemaMap == nil {
			return fmt.Errorf("response json schema cannot be converted to a JSON schema object")
		}
		// Strict response formats additionally require an object root.
		if !isObjectSchema(schemaMap) {
			strict = false
		}
		params.Text = responses.ResponseTextConfigParam{
			Format: responses.ResponseFormatTextConfigUnionParam{
				OfJSONSchema: &responses.ResponseFormatTextJSONSchemaConfigParam{
					Name:   "response",
					Schema: schemaMap,
					Strict: param.NewOpt(strict),
				},
			},
		}
	}

	// Tools
	if len(cfg.Tools) > 0 {
		tools, err := convertTools(cfg.Tools)
		if err != nil {
			return fmt.Errorf("failed to convert tools: %w", err)
		}
		params.Tools = tools
	}

	// ToolConfig maps to tool_choice.
	if cfg.ToolConfig != nil && cfg.ToolConfig.FunctionCallingConfig != nil {
		fcc := cfg.ToolConfig.FunctionCallingConfig
		switch fcc.Mode {
		case genai.FunctionCallingConfigModeAuto:
			params.ToolChoice = responses.ResponseNewParamsToolChoiceUnion{
				OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptionsAuto),
			}
		case genai.FunctionCallingConfigModeNone:
			params.ToolChoice = responses.ResponseNewParamsToolChoiceUnion{
				OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptionsNone),
			}
		case genai.FunctionCallingConfigModeAny:
			if len(fcc.AllowedFunctionNames) == 1 {
				params.ToolChoice = responses.ResponseNewParamsToolChoiceUnion{
					OfFunctionTool: &responses.ToolChoiceFunctionParam{
						Name: fcc.AllowedFunctionNames[0],
					},
				}
			} else {
				// A multi-name allowlist is enforced by sending only the
				// allowed function definitions: tool_choice "required" alone
				// would leave every declared tool callable, and the native
				// allowed_tools choice is not implemented across
				// OpenAI-compatible backends (vLLM rejects everything but
				// "auto" on this endpoint).
				if len(fcc.AllowedFunctionNames) > 1 {
					allowed := make(map[string]bool, len(fcc.AllowedFunctionNames))
					for _, name := range fcc.AllowedFunctionNames {
						allowed[name] = true
					}
					kept := make([]responses.ToolUnionParam, 0, len(fcc.AllowedFunctionNames))
					for _, tool := range params.Tools {
						if tool.OfFunction != nil && allowed[tool.OfFunction.Name] {
							kept = append(kept, tool)
						}
					}
					params.Tools = kept
				}
				params.ToolChoice = responses.ResponseNewParamsToolChoiceUnion{
					OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptionsRequired),
				}
			}
		}
	}

	return nil
}

// danglingCallIDs collects the IDs of function calls without a matching
// output and outputs without a matching call across the whole history, so
// the conversion can drop them: the API rejects both shapes.
func danglingCallIDs(contents []*genai.Content) map[string]bool {
	calls := map[string]bool{}
	outputs := map[string]bool{}
	for _, content := range contents {
		if content == nil {
			continue
		}
		for _, part := range content.Parts {
			switch {
			case part.FunctionCall != nil:
				calls[part.FunctionCall.ID] = true
			case part.FunctionResponse != nil:
				outputs[part.FunctionResponse.ID] = true
			}
		}
	}
	dangling := map[string]bool{}
	for id := range calls {
		if !outputs[id] {
			dangling[id] = true
		}
	}
	for id := range outputs {
		if !calls[id] {
			dangling[id] = true
		}
	}
	return dangling
}

// convertContentToInputItems converts a genai.Content into Responses API input items.
// A single Content may produce multiple items: text/media coalesce into a message,
// while FunctionCall and FunctionResponse become separate typed items. origin
// identifies the requesting channel: encrypted reasoning is only replayed when
// it was captured from the same origin. Parts whose call ID is in dangling
// are skipped.
func convertContentToInputItems(content *genai.Content, origin string, dangling map[string]bool) ([]responses.ResponseInputItemUnionParam, error) {
	var items []responses.ResponseInputItemUnionParam
	var textParts []string
	var refusalFlags []bool
	var mediaParts []responses.ResponseInputContentUnionParam
	// orderedContents keeps text and media in their original interleaved
	// order for mixed messages; regrouping them would change the prompt
	// (e.g. which image a "describe the above" refers to).
	var orderedContents responses.ResponseInputMessageContentListParam
	var phase string
	var messageID string
	var msgStatus string
	role := convertRole(content.Role)

	// A replayed reasoning item must be followed by another item from the
	// same turn (message or function call), or the API rejects it as
	// dangling. Track the last part that produces such an item so trailing
	// thoughts (e.g. from an interrupted turn) are not replayed.
	lastFollower := -1
	for i, part := range content.Parts {
		if part.Thought {
			continue
		}
		if part.FunctionCall != nil &&
			(dangling[part.FunctionCall.ID] || common.IsADKInternalCall(part.FunctionCall.Name)) {
			continue
		}
		if part.FunctionResponse != nil &&
			(dangling[part.FunctionResponse.ID] || common.IsADKInternalCall(part.FunctionResponse.Name)) {
			continue
		}
		if part.FunctionCall != nil || part.FunctionResponse != nil ||
			part.Text != "" || part.InlineData != nil || part.FileData != nil {
			lastFollower = i
		}
	}

	flushMessage := func() {
		if len(textParts) == 0 && len(mediaParts) == 0 {
			return
		}

		// Model output with a message ID builds an OutputMessage manually,
		// so the item identity survives the stateless round trip (models
		// like gpt-5.3-codex expect phase back on every assistant message,
		// and the API asks for output items to be replayed as-is). The ID is
		// the gate: OutputMessage requires one, so phase-only content (seen
		// in histories written by other adapters) goes through the input
		// message path, which carries phase itself. Output message content
		// can only carry text and refusals, so a message that also holds
		// media keeps the media and gives up the identity via the
		// content-list path instead of silently dropping parts.
		if role == responses.EasyInputMessageRoleAssistant && messageID != "" &&
			len(mediaParts) == 0 {
			var contentParts []responses.ResponseOutputMessageContentUnionParam
			for i, t := range textParts {
				if refusalFlags[i] {
					contentParts = append(contentParts, responses.ResponseOutputMessageContentUnionParam{
						OfRefusal: &responses.ResponseOutputRefusalParam{Refusal: t},
					})
					continue
				}
				contentParts = append(contentParts, responses.ResponseOutputMessageContentUnionParam{
					OfOutputText: &responses.ResponseOutputTextParam{Text: t},
				})
			}
			status := responses.ResponseOutputMessageStatusCompleted
			if msgStatus != "" {
				status = responses.ResponseOutputMessageStatus(msgStatus)
			}
			msg := responses.ResponseOutputMessageParam{
				ID:      messageID,
				Content: contentParts,
				Status:  status,
				Phase:   responses.ResponseOutputMessagePhase(phase),
			}
			items = append(items, responses.ResponseInputItemUnionParam{
				OfOutputMessage: &msg,
			})
		} else if len(mediaParts) == 0 {
			// Type is set explicitly: the OpenAPI spec discriminates input
			// union items on it, even though the endpoint tolerates its
			// absence. Phase is assistant-only: multi-agent histories
			// relabel other agents' output as user while keeping part
			// metadata, and the API rejects phase on user messages.
			msgPhase := responses.EasyInputMessagePhase("")
			if role == responses.EasyInputMessageRoleAssistant {
				msgPhase = responses.EasyInputMessagePhase(phase)
			}
			items = append(items, responses.ResponseInputItemUnionParam{
				OfMessage: &responses.EasyInputMessageParam{
					Phase: msgPhase,
					Content: responses.EasyInputMessageContentUnionParam{
						OfString: param.NewOpt(joinTexts(textParts)),
					},
					Role: role,
					Type: responses.EasyInputMessageTypeMessage,
				},
			})
		} else {
			// Only the message ID has no field on this path; the phase does
			// (models like gpt-5.3-codex expect it back), so it is kept.
			// Phase is assistant-only: multi-agent histories relabel other
			// agents' output as user while keeping part metadata, and the
			// API rejects phase on user messages.
			msgPhase := responses.EasyInputMessagePhase("")
			if role == responses.EasyInputMessageRoleAssistant {
				msgPhase = responses.EasyInputMessagePhase(phase)
			}
			items = append(items, responses.ResponseInputItemUnionParam{
				OfMessage: &responses.EasyInputMessageParam{
					Phase: msgPhase,
					Content: responses.EasyInputMessageContentUnionParam{
						OfInputItemContentList: orderedContents,
					},
					Role: role,
					Type: responses.EasyInputMessageTypeMessage,
				},
			})
		}

		textParts = nil
		refusalFlags = nil
		mediaParts = nil
		orderedContents = nil
		phase = ""
		messageID = ""
		msgStatus = ""
	}

	for i, part := range content.Parts {
		switch {
		case part.FunctionResponse != nil:
			// Skip ADK internal framework parts (HITL confirmation
			// protocol): they are not tools the model declared, and the
			// API rejects undeclared call/output items.
			if common.IsADKInternalCall(part.FunctionResponse.Name) {
				continue
			}
			if dangling[part.FunctionResponse.ID] {
				continue
			}
			flushMessage()
			responseJSON, err := common.MarshalToolPayload(part.FunctionResponse.Response)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal function response: %w", err)
			}
			items = append(items, responses.ResponseInputItemParamOfFunctionCallOutput(
				part.FunctionResponse.ID, string(responseJSON),
			))

		case part.FunctionCall != nil:
			if common.IsADKInternalCall(part.FunctionCall.Name) {
				continue
			}
			if dangling[part.FunctionCall.ID] {
				continue
			}
			flushMessage()
			argsJSON, err := common.MarshalToolPayload(part.FunctionCall.Args)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal function args: %w", err)
			}
			items = append(items, responses.ResponseInputItemParamOfFunctionCall(
				string(argsJSON), part.FunctionCall.ID, part.FunctionCall.Name,
			))

		case part.Thought:
			// Reasoning items with encrypted content are replayed so the
			// model keeps its chain of thought across turns, as reasoning
			// models require. Items the API would reject are skipped instead
			// (non-assistant turns, missing encrypted content, a foreign
			// origin, no same-turn item following), so the model re-derives
			// its reasoning rather than the request failing with a 400.
			if role != responses.EasyInputMessageRoleAssistant {
				continue
			}
			enc, _ := part.PartMetadata["encrypted_content"].(string)
			id, _ := part.PartMetadata["reasoning_id"].(string)
			if enc == "" || id == "" {
				continue
			}
			if partOrigin, _ := part.PartMetadata["reasoning_origin"].(string); origin == "" || partOrigin != origin {
				continue
			}
			if i > lastFollower {
				continue
			}
			flushMessage()
			reasoning := responses.ResponseReasoningItemParam{
				ID:               id,
				Summary:          []responses.ResponseReasoningItemSummaryParam{},
				EncryptedContent: param.NewOpt(enc),
			}
			if part.Text != "" {
				reasoning.Summary = append(reasoning.Summary,
					responses.ResponseReasoningItemSummaryParam{Text: part.Text})
			}
			items = append(items, responses.ResponseInputItemUnionParam{
				OfReasoning: &reasoning,
			})

		case part.Text != "":
			// Parts from different output messages must not coalesce:
			// flushing on a message_id or phase change keeps each replayed
			// message's identity and phase intact (models like
			// gpt-5.3-codex expect commentary and final answer separate).
			// Only buffered text marks a boundary: identity comes from text
			// parts alone, so a media-only prefix has none yet and adopts
			// the incoming identity instead of splitting off the media.
			partID, _ := part.PartMetadata["message_id"].(string)
			partPhase, _ := part.PartMetadata["phase"].(string)
			if len(textParts) > 0 &&
				(partID != messageID || partPhase != phase) {
				flushMessage()
			}
			messageID = partID
			phase = partPhase
			if s, ok := part.PartMetadata["status"].(string); ok && s != "" {
				msgStatus = s
			}
			refusal, _ := part.PartMetadata["refusal"].(bool)
			refusalFlags = append(refusalFlags, refusal)
			textParts = append(textParts, part.Text)
			orderedContents = append(orderedContents, responses.ResponseInputContentParamOfInputText(part.Text))

		case part.InlineData != nil:
			p, err := convertInlineDataToPart(part.InlineData)
			if err != nil {
				return nil, err
			}
			mediaParts = append(mediaParts, *p)
			orderedContents = append(orderedContents, *p)

		case part.FileData != nil:
			p, err := convertFileDataToPart(part.FileData)
			if err != nil {
				return nil, err
			}
			mediaParts = append(mediaParts, *p)
			orderedContents = append(orderedContents, *p)
		}
	}

	flushMessage()
	return items, nil
}

// convertResponse transforms a Responses API response into an LLMResponse.
// origin identifies the channel that produced the response; it is recorded
// alongside encrypted reasoning content so replay can be restricted to the
// same channel.
func convertResponse(resp *responses.Response, origin string) (*model.LLMResponse, error) {
	// A failed response is an error, mirroring the streaming response.failed
	// handling: without this check the server's message would degrade into
	// "no output items", or partial output would pass as a completed turn.
	if resp.Status == responses.ResponseStatusFailed {
		msg := resp.Error.Message
		if msg == "" {
			msg = string(resp.Error.Code)
		}
		if msg == "" {
			msg = "response generation failed"
		}
		return nil, fmt.Errorf("openai responses api: %s", msg)
	}
	if len(resp.Output) == 0 {
		return nil, ErrNoOutputInResponse
	}

	content := &genai.Content{
		Role:  genai.RoleModel,
		Parts: []*genai.Part{},
	}

	itemTypes := map[string]bool{}
	for _, item := range resp.Output {
		itemTypes[item.Type] = true
		switch item.Type {
		case "reasoning":
			// One part per reasoning item, even when the summary is empty:
			// the encrypted content must survive the round-trip through ADK
			// history so the item can be replayed on the next turn.
			var summaryTexts []string
			for _, summary := range item.Summary {
				if summary.Text != "" {
					summaryTexts = append(summaryTexts, summary.Text)
				}
			}
			if len(summaryTexts) == 0 && item.EncryptedContent == "" {
				continue
			}
			meta := map[string]any{"reasoning_id": item.ID}
			if item.EncryptedContent != "" {
				meta["encrypted_content"] = item.EncryptedContent
				meta["reasoning_origin"] = origin
			}
			content.Parts = append(content.Parts, &genai.Part{
				Text:         joinTexts(summaryTexts),
				Thought:      true,
				PartMetadata: meta,
			})

		case "message":
			// ID, phase, status, and the refusal flag are kept per part so
			// the message replays with its identity intact instead of
			// degrading to plain completed output_text.
			meta := func(refusal bool) map[string]any {
				m := map[string]any{}
				if item.Phase != "" {
					m["phase"] = string(item.Phase)
				}
				if item.ID != "" {
					m["message_id"] = item.ID
				}
				if item.Status != "" {
					m["status"] = string(item.Status)
				}
				if refusal {
					m["refusal"] = true
				}
				if len(m) == 0 {
					return nil
				}
				return m
			}
			for _, cp := range item.Content {
				switch cp.Type {
				case "output_text":
					content.Parts = append(content.Parts, &genai.Part{
						Text: cp.Text, PartMetadata: meta(false),
					})
				case "refusal":
					content.Parts = append(content.Parts, &genai.Part{
						Text: cp.Refusal, PartMetadata: meta(true),
					})
				}
			}

		case "function_call":
			args, err := functionCallArgs(item)
			if err != nil {
				return nil, err
			}
			content.Parts = append(content.Parts, &genai.Part{
				FunctionCall: &genai.FunctionCall{
					ID:   item.CallID,
					Name: item.Name,
					Args: args,
				},
			})
		}
	}

	// Unknown item types are skipped for forward compatibility, but when
	// nothing consumable remains the turn must not pass as completed: ADK
	// would persist an empty event with no clue about what was dropped.
	if len(content.Parts) == 0 {
		return nil, fmt.Errorf("%w (item types: %s)", ErrNoConsumableOutput, joinSortedKeys(itemTypes))
	}

	return &model.LLMResponse{
		Content:       content,
		UsageMetadata: convertUsageMetadata(resp.Usage),
		FinishReason:  convertStatus(resp.Status, resp.IncompleteDetails),
		TurnComplete:  true,
	}, nil
}

// convertTools transforms genai tools into Responses API function tool format.
func convertTools(genaiTools []*genai.Tool) ([]responses.ToolUnionParam, error) {
	var tools []responses.ToolUnionParam

	for _, genaiTool := range genaiTools {
		if genaiTool == nil {
			continue
		}

		// Tool facets other than function declarations (GoogleSearch,
		// CodeExecution, ...) have no Responses API mapping; failing loud
		// beats silently sending no tool at all.
		rest := *genaiTool
		rest.FunctionDeclarations = nil
		if data, err := json.Marshal(rest); err == nil && string(data) != "{}" {
			return nil, fmt.Errorf("unsupported tool fields for the Responses API: %s", data)
		}

		for _, funcDecl := range genaiTool.FunctionDeclarations {
			params := funcDecl.ParametersJsonSchema
			// Assign only a non-nil *genai.Schema: a nil pointer stored in
			// the interface would defeat the nil check that gives
			// parameterless tools a valid empty object schema.
			if params == nil && funcDecl.Parameters != nil {
				params = funcDecl.Parameters
			}

			toolParams, strict := convertFunctionParams(params)
			if toolParams == nil {
				return nil, fmt.Errorf("parameters of tool %q cannot be converted to a JSON schema object", funcDecl.Name)
			}

			tools = append(tools, responses.ToolUnionParam{
				OfFunction: &responses.FunctionToolParam{
					Name:        funcDecl.Name,
					Description: param.NewOpt(funcDecl.Description),
					Parameters:  toolParams,
					Strict:      param.NewOpt(strict),
				},
			})
		}
	}

	return tools, nil
}

// --- Helper functions ---

// convertInlineDataToPart converts inline data to the appropriate Responses
// API content part. Images become input_image data URIs and documents become
// input_file data; audio is rejected because the API's file inputs do not
// accept it.
func convertInlineDataToPart(data *genai.Blob) (*responses.ResponseInputContentUnionParam, error) {
	if data == nil {
		return nil, fmt.Errorf("inline data is nil")
	}

	mediaType := normalizeMIMEType(data.MIMEType)
	base64Data := base64.StdEncoding.EncodeToString(data.Data)
	dataURI := fmt.Sprintf("data:%s;base64,%s", mediaType, base64Data)

	switch {
	case mediaType == "image/jpeg" || mediaType == "image/jpg" || mediaType == "image/png" ||
		mediaType == "image/gif" || mediaType == "image/webp":
		return &responses.ResponseInputContentUnionParam{
			OfInputImage: &responses.ResponseInputImageParam{
				ImageURL: param.NewOpt(dataURI),
				Detail:   responses.ResponseInputImageDetailAuto,
			},
		}, nil

	case isDocumentMIMEType(mediaType):
		// Base64 file uploads need a filename with an extension: the API
		// identifies the document type by it. The caller's DisplayName
		// wins; a missing extension is filled in from the MIME type, and
		// types without an unambiguous extension require a DisplayName.
		filename := data.DisplayName
		if path.Ext(filename) == "" {
			ext := extensionForMIME(mediaType)
			if ext == "" {
				return nil, fmt.Errorf("no filename extension can be derived for MIME type %s: set DisplayName to a filename with an extension", mediaType)
			}
			if filename == "" {
				filename = "input"
			}
			filename += "." + ext
		}
		return &responses.ResponseInputContentUnionParam{
			OfInputFile: &responses.ResponseInputFileParam{
				FileData: param.NewOpt(dataURI),
				Filename: param.NewOpt(filename),
			},
		}, nil

	case strings.HasPrefix(mediaType, "audio/"):
		// The API's file inputs accept no audio and the SDK's input
		// content union has no input_audio member, so audio is rejected
		// instead of being sent as a file the server will refuse.
		return nil, fmt.Errorf("audio input is not supported by the Responses API adapter: %s", mediaType)

	default:
		return nil, fmt.Errorf("unsupported inline data MIME type for Responses API: %s", mediaType)
	}
}

// isDocumentMIMEType reports whether a media type is in the Responses API's
// documented file-input list: text/ wholesale, plus the documented
// application/ and message/ types (documents, spreadsheets, presentations,
// and code carriers).
func isDocumentMIMEType(mediaType string) bool {
	if strings.HasPrefix(mediaType, "text/") {
		return true
	}
	switch mediaType {
	case "application/pdf",
		"message/rfc822",
		// Spreadsheets.
		"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
		"application/vnd.ms-excel",
		"application/csv",
		"application/x-iif",
		"application/vnd.google-apps.spreadsheet",
		// Rich documents.
		"application/vnd.openxmlformats-officedocument.wordprocessingml.document",
		"application/msword",
		"application/rtf",
		"application/vnd.oasis.opendocument.text",
		"application/vnd.apple.pages",
		"application/vnd.google-apps.document",
		"application/vnd.apple.iwork",
		// Presentations.
		"application/vnd.openxmlformats-officedocument.presentationml.presentation",
		"application/vnd.ms-powerpoint",
		"application/vnd.apple.keynote",
		"application/vnd.google-apps.presentation",
		// Code and structured text.
		"application/javascript",
		"application/typescript",
		"application/json",
		"application/json5",
		"application/x-json5",
		"application/x-ndjson",
		"application/yaml",
		"application/x-yaml",
		"application/toml",
		"application/x-toml",
		"application/graphql",
		"application/x-graphql",
		"application/x-protobuf",
		"application/x-sql",
		"application/x-scala",
		"application/x-rust",
		"application/x-php",
		"application/x-httpd-php",
		"application/x-httpd-php-source",
		"application/x-powershell",
		"application/x-bash",
		"application/x-awk",
		"application/x-terraform",
		"application/x-patch",
		"application/x-subrip":
		return true
	}
	return false
}

// convertFileDataToPart maps a FileData part to a content part; the URI goes
// through verbatim, nothing is downloaded. Images become input_image and
// documents become input_file with file_url; audio has no URL transport in
// the API. Plain http is allowed for API-compatible gateways. Other schemes
// error instead of being silently dropped; gs://, the scheme genai.FileData
// documents, gets its own message pointing at InlineData. DisplayName is
// dropped: neither part has a field for it.
func convertFileDataToPart(data *genai.FileData) (*responses.ResponseInputContentUnionParam, error) {
	if data == nil {
		return nil, fmt.Errorf("file data is nil")
	}
	if strings.HasPrefix(data.FileURI, "gs://") {
		return nil, fmt.Errorf("file data URI %q is a Google Cloud Storage URI, which the Responses API cannot fetch: download the bytes and use InlineData instead", data.FileURI)
	}
	if !strings.HasPrefix(data.FileURI, "https://") && !strings.HasPrefix(data.FileURI, "http://") {
		return nil, fmt.Errorf("file data URI must be an http(s) URL for the Responses API, got %q", data.FileURI)
	}

	mediaType := normalizeMIMEType(data.MIMEType)
	switch {
	case mediaType == "image/jpeg" || mediaType == "image/jpg" || mediaType == "image/png" ||
		mediaType == "image/gif" || mediaType == "image/webp":
		return &responses.ResponseInputContentUnionParam{
			OfInputImage: &responses.ResponseInputImageParam{
				ImageURL: param.NewOpt(data.FileURI),
				Detail:   responses.ResponseInputImageDetailAuto,
			},
		}, nil

	case isDocumentMIMEType(mediaType):
		return &responses.ResponseInputContentUnionParam{
			OfInputFile: &responses.ResponseInputFileParam{
				FileURL: param.NewOpt(data.FileURI),
			},
		}, nil

	case strings.HasPrefix(mediaType, "audio/"):
		return nil, fmt.Errorf("audio input is not supported by the Responses API adapter: %s", mediaType)

	default:
		return nil, fmt.Errorf("unsupported file data MIME type for Responses API: %s", mediaType)
	}
}

// extensionForMIME derives a filename extension for base64 file uploads.
// Types whose subtype is not a usable extension are mapped explicitly; the
// rest use the subtype with "x-" and "vnd." prefixes stripped. An empty
// result marks a type with no unambiguous extension (e.g. iWork, which
// covers both Pages and Keynote), where the caller must name the file.
func extensionForMIME(mediaType string) string {
	switch mediaType {
	case "text/plain":
		return "txt"
	case "text/markdown":
		return "md"
	case "application/javascript", "text/javascript":
		return "js"
	case "application/typescript", "text/x-typescript":
		return "ts"
	case "text/x-python", "text/x-script.python":
		return "py"
	case "text/x-c":
		return "c"
	case "text/x-c++":
		return "cpp"
	case "text/x-csharp":
		return "cs"
	case "text/x-golang":
		return "go"
	case "text/x-ruby":
		return "rb"
	case "text/x-rust", "application/x-rust":
		return "rs"
	case "text/x-shellscript", "application/x-bash":
		return "sh"
	case "text/x-perl":
		return "pl"
	case "text/x-kotlin":
		return "kt"
	case "message/rfc822":
		return "eml"
	case "application/msword":
		return "doc"
	case "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
		return "docx"
	case "application/vnd.ms-excel":
		return "xls"
	case "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
		return "xlsx"
	case "application/vnd.ms-powerpoint":
		return "ppt"
	case "application/vnd.openxmlformats-officedocument.presentationml.presentation":
		return "pptx"
	case "application/vnd.oasis.opendocument.text":
		return "odt"
	case "application/vnd.apple.pages":
		return "pages"
	case "application/vnd.apple.keynote":
		return "key"
	case "application/vnd.apple.iwork":
		return ""
	case "application/x-ndjson":
		return "ndjson"
	case "application/x-subrip":
		return "srt"
	case "application/x-httpd-php", "application/x-httpd-php-source":
		return "php"
	case "application/x-protobuf":
		return "proto"
	case "application/x-terraform":
		return "tf"
	case "application/x-powershell":
		return "ps1"
	case "application/graphql", "application/x-graphql":
		return "graphql"
	}
	sub := mediaType[strings.LastIndexByte(mediaType, '/')+1:]
	sub = strings.TrimPrefix(sub, "x-")
	sub = strings.TrimPrefix(sub, "vnd.")
	if strings.ContainsAny(sub, "./+") {
		return ""
	}
	return sub
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

// convertUsageMetadata converts Responses API usage stats to genai format.
func convertUsageMetadata(usage responses.ResponseUsage) *genai.GenerateContentResponseUsageMetadata {
	if usage.TotalTokens == 0 {
		return nil
	}
	return &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:        int32(usage.InputTokens),
		CandidatesTokenCount:    int32(usage.OutputTokens),
		TotalTokenCount:         int32(usage.TotalTokens),
		ThoughtsTokenCount:      int32(usage.OutputTokensDetails.ReasoningTokens),
		CachedContentTokenCount: int32(usage.InputTokensDetails.CachedTokens),
	}
}

// convertRole maps genai roles to Responses API EasyInputMessageRole. The
// literal "assistant" appears in histories written by other adapters and
// must not degrade to a user message.
func convertRole(role string) responses.EasyInputMessageRole {
	switch role {
	case "model", "assistant":
		return responses.EasyInputMessageRoleAssistant
	case "system":
		return responses.EasyInputMessageRoleSystem
	default:
		return responses.EasyInputMessageRoleUser
	}
}

// convertStatus maps Responses API status to genai finish reason.
func convertStatus(status responses.ResponseStatus, details responses.ResponseIncompleteDetails) genai.FinishReason {
	switch status {
	case responses.ResponseStatusCompleted:
		return genai.FinishReasonStop
	case responses.ResponseStatusIncomplete:
		switch details.Reason {
		case "max_output_tokens":
			return genai.FinishReasonMaxTokens
		case "content_filter":
			return genai.FinishReasonSafety
		}
		return genai.FinishReasonUnspecified
	default:
		return genai.FinishReasonUnspecified
	}
}

// convertThinkingLevel maps genai thinking levels to Responses API reasoning effort.
func convertThinkingLevel(level genai.ThinkingLevel) shared.ReasoningEffort {
	switch level {
	case genai.ThinkingLevelMinimal:
		return shared.ReasoningEffortMinimal
	case genai.ThinkingLevelLow:
		return shared.ReasoningEffortLow
	case genai.ThinkingLevelHigh:
		return shared.ReasoningEffortHigh
	default:
		return shared.ReasoningEffortMedium
	}
}

// convertToStrictFunctionParams converts various parameter types to a map
// compliant with OpenAI strict mode. A nil input produces a valid empty
// object schema so parameterless tools are accepted. The input is deep-copied
// via JSON round-trip to avoid mutating the caller's schema.
func convertToStrictFunctionParams(params any) map[string]any {
	if params == nil {
		return map[string]any{
			"type":                 "object",
			"properties":           map[string]any{},
			"required":             []any{},
			"additionalProperties": false,
		}
	}

	m := deepCopySchema(params)
	if m == nil {
		return nil
	}

	lowercaseTypes(m)
	normalizeStrictSchema(m)
	return m
}

// convertFunctionParams prepares tool parameters for the API and decides the
// strict flag. Schemas inside the strict subset are normalized and sent with
// strict=true so the API constrains generation to them. Schemas the subset
// cannot express, such as free-form objects (additionalProperties: true) or
// arrays without an item schema, are sent as authored with strict=false:
// forcing them through strict normalization would silently change their
// meaning (a free-form object would harden into an empty one), and the API
// rejects them with invalid_function_parameters if sent strict anyway.
func convertFunctionParams(params any) (map[string]any, bool) {
	if params == nil {
		return convertToStrictFunctionParams(nil), true
	}
	m := deepCopySchema(params)
	if m == nil {
		return nil, false
	}
	lowercaseTypes(m)
	m = inlineRootRef(m)
	// Strict mode requires the root to be a plain object: a root that is
	// still a $ref, an array, or a union goes non-strict as-is. anyOf and
	// oneOf are both checked because normalization would rename a root
	// oneOf into the anyOf the API rejects at the root.
	if !isObjectSchema(m) {
		return m, false
	}
	for _, key := range []string{"anyOf", "oneOf"} {
		if _, ok := m[key]; ok {
			return m, false
		}
	}
	if !fitsStrictSubset(m) {
		return m, false
	}
	normalizeStrictSchema(m)
	return m, true
}

// inlineRootRef resolves the root-level "$ref plus $defs" shape that Go
// schema generators commonly emit. Strict mode wants the root to be a plain
// object, so the referenced definition's keys are copied onto the root and
// $defs stays for the remaining references. Roots with extra siblings or an
// unresolvable pointer come back unchanged and fail the object check.
func inlineRootRef(m map[string]any) map[string]any {
	ref, ok := m["$ref"].(string)
	if !ok {
		return m
	}
	for key := range m {
		if key != "$ref" && key != "$defs" {
			return m
		}
	}
	name, found := strings.CutPrefix(ref, "#/$defs/")
	if !found || strings.Contains(name, "/") {
		return m
	}
	defs, _ := m["$defs"].(map[string]any)
	target, ok := defs[name].(map[string]any)
	if !ok {
		return m
	}
	if _, hasOwnDefs := target["$defs"]; hasOwnDefs {
		return m
	}
	// The target is copied so the root and the $defs entry do not share
	// maps during normalization.
	root := deepCopySchema(target)
	if root == nil {
		return m
	}
	root["$defs"] = defs
	return root
}

// unsupportedStrictKeywords are schema keywords the strict subset rejects and
// normalization cannot express or rewrite. Their presence anywhere in a tool
// schema sends the tool as non-strict. oneOf is absent on purpose: it is
// rewritten to anyOf during normalization.
var unsupportedStrictKeywords = []string{
	"allOf", "not", "if", "then", "else",
	"dependentRequired", "dependentSchemas", "patternProperties",
	"propertyNames", "prefixItems", "contains", "unevaluatedProperties",
	"minLength", "maxLength",
}

// fitsStrictSubset reports whether a schema can be normalized into the strict
// subset without changing what it accepts. It walks the same nodes as
// normalizeStrictSchema and fails on free-form objects, arrays without an
// item schema, nodes whose type cannot be derived, and keywords the subset
// rejects outright.
func fitsStrictSubset(schema map[string]any) bool {
	for _, key := range unsupportedStrictKeywords {
		if _, ok := schema[key]; ok {
			return false
		}
	}

	if ap, ok := schema["additionalProperties"]; ok && ap != false {
		return false
	}

	// A $ref next to anyOf or oneOf is a conjunction the strict subset
	// cannot express: hoisting the reference into the union would change
	// what the schema accepts, so such nodes go non-strict.
	if _, ok := schema["$ref"]; ok {
		for _, key := range []string{"anyOf", "oneOf"} {
			if _, exists := schema[key]; exists {
				return false
			}
		}
	}

	if isArraySchema(schema) {
		if _, ok := schema["items"]; !ok {
			return false
		}
	}

	if !hasDerivableType(schema) {
		return false
	}

	if props, ok := schema["properties"].(map[string]any); ok {
		for _, prop := range props {
			if propMap, ok := prop.(map[string]any); ok && !fitsStrictSubset(propMap) {
				return false
			}
		}
	}
	if items, ok := schema["items"].(map[string]any); ok && !fitsStrictSubset(items) {
		return false
	}
	if defs, ok := schema["$defs"].(map[string]any); ok {
		for _, def := range defs {
			if defMap, ok := def.(map[string]any); ok && !fitsStrictSubset(defMap) {
				return false
			}
		}
	}
	for _, key := range []string{"anyOf", "oneOf"} {
		branches, _ := schema[key].([]any)
		for _, branch := range branches {
			if branchMap, ok := branch.(map[string]any); ok && !fitsStrictSubset(branchMap) {
				return false
			}
		}
	}
	return true
}

// hasDerivableType reports whether a node either has a type or lets strict
// normalization derive one: $ref and union nodes need no type of their own,
// object and array shapes are recognized from properties/items, and literal
// nodes derive their type from const or enum values.
func hasDerivableType(schema map[string]any) bool {
	if _, ok := schema["type"]; ok {
		return true
	}
	if _, ok := schema["$ref"]; ok {
		return true
	}
	for _, key := range []string{"anyOf", "oneOf"} {
		if _, ok := schema[key]; ok {
			return true
		}
	}
	if isObjectSchema(schema) {
		return true
	}
	if _, ok := schema["items"]; ok {
		return true
	}
	if v, ok := schema["const"]; ok {
		return literalType(v) != ""
	}
	if vals, ok := schema["enum"].([]any); ok && len(vals) > 0 {
		t := literalType(vals[0])
		if t == "" {
			return false
		}
		for _, v := range vals[1:] {
			if literalType(v) != t {
				return false
			}
		}
		return true
	}
	return false
}

// isArraySchema reports whether the schema's type names "array", either as a
// plain string or inside a type union list.
func isArraySchema(schema map[string]any) bool {
	switch t := schema["type"].(type) {
	case string:
		return t == "array"
	case []any:
		for _, v := range t {
			if s, ok := v.(string); ok && s == "array" {
				return true
			}
		}
	}
	return false
}

// normalizeStrictSchema recursively makes a JSON schema compliant with
// OpenAI strict mode. For every object type (whether type is "object" or
// contains "object" in an array like ["object", "null"]) it:
//   - adds "additionalProperties": false
//   - ensures "properties" exists
//   - sets "required" to all property keys
//   - expands originally-optional properties to nullable (["type", "null"])
//
// Recursion covers properties, items, $defs and anyOf branches, oneOf keys
// are renamed to anyOf (the strict subset rejects oneOf), and bare const or
// enum nodes get their literal type filled in. Child schemas
// are normalised before the parent marks optional fields as nullable, so
// nested objects get their own required/additionalProperties regardless of
// whether the parent considers them optional.
func normalizeStrictSchema(schema map[string]any) {
	if schema == nil {
		return
	}

	// Recurse into children first so nested objects are fully normalised
	// before we decide which parent-level fields are optional.
	if props, ok := schema["properties"].(map[string]any); ok {
		for _, prop := range props {
			if propMap, ok := prop.(map[string]any); ok {
				normalizeStrictSchema(propMap)
			}
		}
	}
	if items, ok := schema["items"].(map[string]any); ok {
		normalizeStrictSchema(items)
	}
	ensureTypeForLiteral(schema)
	if _, ok := schema["type"]; !ok {
		if _, hasItems := schema["items"]; hasItems {
			schema["type"] = "array"
		}
	}
	if defs, ok := schema["$defs"].(map[string]any); ok {
		for _, def := range defs {
			if defMap, ok := def.(map[string]any); ok {
				normalizeStrictSchema(defMap)
			}
		}
	}

	// The strict subset rejects "oneOf", so the key is renamed to anyOf.
	// This widens validation from "exactly one branch" to "at least one
	// branch": every payload the original schema accepted stays valid, so
	// generation keeps following the authored branches. In the rare case
	// where anyOf is already present the oneOf branches are dropped, as
	// the subset has no way to express the conjunction of the two.
	if branches, ok := schema["oneOf"].([]any); ok {
		if _, exists := schema["anyOf"]; !exists {
			schema["anyOf"] = branches
		}
		delete(schema, "oneOf")
	}

	// The strict validator rejects a $ref with sibling keys, so the
	// reference moves into a single-branch anyOf while the siblings stay
	// on the node: annotations and constraints keep applying alongside
	// the reference instead of being dropped.
	if ref, ok := schema["$ref"]; ok && len(schema) > 1 {
		if _, exists := schema["anyOf"]; !exists {
			delete(schema, "$ref")
			schema["anyOf"] = []any{map[string]any{"$ref": ref}}
		}
	}

	if branches, ok := schema["anyOf"].([]any); ok {
		for _, branch := range branches {
			if branchMap, ok := branch.(map[string]any); ok {
				normalizeStrictSchema(branchMap)
			}
		}
	}

	// OpenAPI-style "nullable" is not a JSON Schema keyword and the strict
	// validator rejects it; it converts faithfully to a null type union.
	if v, ok := schema["nullable"]; ok {
		if b, _ := v.(bool); b {
			makeNullable(schema)
		}
		delete(schema, "nullable")
	}

	if !isObjectSchema(schema) {
		return
	}

	if _, ok := schema["type"]; !ok {
		schema["type"] = "object"
	}

	if _, hasProps := schema["properties"]; !hasProps {
		schema["properties"] = map[string]any{}
	}
	schema["additionalProperties"] = false

	props, _ := schema["properties"].(map[string]any)
	existing := toStringSet(schema["required"])
	allKeys := make([]any, 0, len(props))
	for key := range props {
		allKeys = append(allKeys, key)
		if !existing[key] {
			makeNullable(props[key])
		}
	}
	// Sort for a deterministic order: map iteration is randomised, and an
	// unstable "required" array changes the serialized tool definition on
	// every request, which breaks OpenAI prompt-cache prefix matching.
	sort.Slice(allKeys, func(i, j int) bool {
		return allKeys[i].(string) < allKeys[j].(string)
	})
	schema["required"] = allKeys
}

// isObjectSchema returns true if the schema represents an object, either
// by explicit type or by having a "properties" field (common in dynamically
// registered tools that omit "type").
func isObjectSchema(schema map[string]any) bool {
	if isObjectType(schema) {
		return true
	}
	_, hasProps := schema["properties"]
	return hasProps
}

// isObjectType returns true if the schema's "type" is "object", either as a
// plain string or inside an array (e.g. ["object", "null"]).
func isObjectType(schema map[string]any) bool {
	switch t := schema["type"].(type) {
	case string:
		return t == "object"
	case []any:
		for _, v := range t {
			if s, ok := v.(string); ok && s == "object" {
				return true
			}
		}
	}
	return false
}

// makeNullable expands a property's type to ["<original>", "null"] so strict
// mode accepts it as an optional (nullable) field. Properties without a
// "type" key ($ref, typeless anyOf unions) get a null branch instead.
// ensureTypeForLiteral fills in a missing "type" on schemas written as a
// bare const or enum. The API rejects such nodes with "schema must have a
// 'type' key", and the type of a literal follows from its value, so it is
// derived from the const or from a same-typed enum value list.
func ensureTypeForLiteral(schema map[string]any) {
	if _, ok := schema["type"]; ok {
		return
	}
	if v, ok := schema["const"]; ok {
		if t := literalType(v); t != "" {
			schema["type"] = t
		}
		return
	}
	if vals, ok := schema["enum"].([]any); ok && len(vals) > 0 {
		t := literalType(vals[0])
		for _, v := range vals[1:] {
			if literalType(v) != t {
				return
			}
		}
		if t != "" {
			schema["type"] = t
		}
	}
}

// literalType maps a decoded JSON literal to its schema type name. An empty
// string means the type cannot be derived (null or nested values).
func literalType(v any) string {
	switch v.(type) {
	case string:
		return "string"
	case bool:
		return "boolean"
	case float64:
		return "number"
	default:
		return ""
	}
}

func makeNullable(prop any) {
	propMap, ok := prop.(map[string]any)
	if !ok {
		return
	}
	// A $ref carries no "type" to extend with "null", so nullability is
	// expressed as an anyOf union of the reference and null instead.
	if ref, ok := propMap["$ref"]; ok {
		delete(propMap, "$ref")
		propMap["anyOf"] = []any{
			map[string]any{"$ref": ref},
			map[string]any{"type": "null"},
		}
		return
	}
	// A const admits exactly one value, so widening "type" alone would
	// leave null failing the const check; the property becomes an anyOf
	// union of the const and null instead.
	if c, ok := propMap["const"]; ok {
		branch := map[string]any{"const": c}
		if t, ok := propMap["type"]; ok {
			branch["type"] = t
		}
		delete(propMap, "const")
		delete(propMap, "type")
		propMap["anyOf"] = []any{branch, map[string]any{"type": "null"}}
		return
	}
	// An enum constrains the value independently of "type", so null must
	// join the allowed values too or the property stays effectively
	// required.
	if values, ok := propMap["enum"].([]any); ok {
		hasNull := false
		for _, v := range values {
			if v == nil {
				hasNull = true
				break
			}
		}
		if !hasNull {
			propMap["enum"] = append(values, nil)
		}
	}
	switch t := propMap["type"].(type) {
	case string:
		propMap["type"] = []any{t, "null"}
	case []any:
		for _, v := range t {
			if v == "null" {
				return
			}
		}
		propMap["type"] = append(t, "null")
	default:
		// No "type" to extend: a typeless anyOf union (oneOf is already
		// renamed by this point) gets a null branch, keeping the property
		// optional instead of silently becoming required.
		if branches, ok := propMap["anyOf"].([]any); ok {
			for _, b := range branches {
				if bm, ok := b.(map[string]any); ok && bm["type"] == "null" {
					return
				}
			}
			propMap["anyOf"] = append(branches, map[string]any{"type": "null"})
		}
	}
}

// toStringSet builds a set from a "required" field ([]string or []any).
func toStringSet(v any) map[string]bool {
	set := map[string]bool{}
	switch r := v.(type) {
	case []string:
		for _, s := range r {
			set[s] = true
		}
	case []any:
		for _, item := range r {
			if s, ok := item.(string); ok {
				set[s] = true
			}
		}
	}
	return set
}

// deepCopySchema converts any schema representation into a fresh
// map[string]any via JSON round-trip, avoiding mutation of the caller's data.
func deepCopySchema(params any) map[string]any {
	jsonBytes, err := json.Marshal(params)
	if err != nil {
		return nil
	}
	var m map[string]any
	if json.Unmarshal(jsonBytes, &m) != nil {
		return nil
	}
	return m
}

// lowercaseTypes recursively lowercases all "type" fields in a JSON schema map.
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

// convertSchema recursively converts a genai.Schema to JSON schema format.
// Only Type, Description, Required, Enum, Properties, and Items are carried
// over; constraints outside this subset (Format, Minimum, Nullable, AnyOf,
// ...) are dropped. Callers needing full fidelity should pass a raw schema
// via ResponseJsonSchema instead, which is sent through unmodified.
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

// joinSortedKeys renders a set of item types as a stable comma-separated
// list for error messages.
func joinSortedKeys(set map[string]bool) string {
	keys := slices.Sorted(maps.Keys(set))
	return strings.Join(keys, ", ")
}

// functionCallArgs decodes a function call item's arguments, which arrive
// either as a JSON string (the OpenAI form) or as a bare JSON object (some
// OpenAI-compatible gateways). Reading only the string arm would silently
// drop the object form's arguments.
func functionCallArgs(item responses.ResponseOutputItemUnion) (map[string]any, error) {
	raw := item.Arguments.OfString
	switch v := item.Arguments.OfResponseToolSearchCallArguments.(type) {
	case nil, string:
		// String arm, or no arguments at all: OfString holds the payload.
	default:
		// Bare arm: decode from the wire bytes, so the SDK's first-key-wins
		// union decode cannot disagree with the last-key-wins semantics of
		// the unmarshal below.
		raw = item.Arguments.JSON.OfResponseToolSearchCallArguments.Raw()
		if raw == "" {
			// A hand-built value has no wire bytes for a re-encode to
			// disagree with.
			b, err := json.Marshal(v)
			if err != nil {
				return nil, fmt.Errorf("%w (name %q, call_id %q): %w", ErrFunctionCallArgs, item.Name, item.CallID, err)
			}
			raw = string(b)
		}
	}
	args, err := parseJSONArgs(raw)
	if err != nil {
		return nil, fmt.Errorf("%w (name %q, call_id %q): %w", ErrFunctionCallArgs, item.Name, item.CallID, err)
	}
	return args, nil
}

// parseJSONArgs parses a function-call arguments string into a map. Malformed
// JSON is an error, not an empty map: a tool with side effects must not run
// with silently emptied arguments. A literal "null" is well-formed JSON some
// gateways emit for parameterless calls; like the empty string it means no
// arguments.
func parseJSONArgs(argsJSON string) (map[string]any, error) {
	if argsJSON == "" {
		return make(map[string]any), nil
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return nil, fmt.Errorf("malformed function call arguments %q: %w", argsJSON, err)
	}
	if args == nil {
		return make(map[string]any), nil
	}
	return args, nil
}
