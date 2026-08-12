// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package openai

import (
	"github.com/openai/openai-go/v3"
	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"
)

// Dialect layers provider-specific wire behaviour onto the adapter. The
// adapter speaks OpenAI-pure Chat Completions by default: with a nil
// Dialect it reads no provider field, sends no provider field, and keeps
// the documented OpenAI shapes. A provider that diverges plugs a Dialect
// in through Config and opts into the areas it needs by implementing the
// capability interfaces below:
//
//   - ToolIDNormalizer: the tool_call_id shape on the wire. OpenAI allows
//     up to 40 characters from [a-zA-Z0-9_-]; Mistral rejects anything
//     that is not exactly 9 alphanumeric characters.
//   - ParamsAdjuster: a last pass over the outgoing request params. xAI's
//     reasoning models reject presence_penalty, frequency_penalty and
//     stop even though the schema defines them.
//   - ReasoningDecoder: reasoning fields the schema does not define, on
//     ingest (reasoning_content, reasoning, reasoning_details, ...).
//   - ReasoningEncoder: the same on egress, as assistant-message fields.
//   - UsageDecoder: usage buckets reported outside the standard object.
//     DeepSeek puts prompt_cache_hit_tokens at the usage root, not in
//     prompt_tokens_details.
//   - ThinkingMapper: the provider-native reasoning-effort knob. OpenAI
//     uses the typed reasoning_effort field, OpenRouter a reasoning
//     object at the request root, vLLM and Qwen enable_thinking.
//   - EgressPolicy: the replay shapes the provider tolerates, resolved
//     once at construction; a vetoed mode falls back to an accepted one
//     and is logged.
//
// Pipeline touch points, in request order:
//
//	buildChatCompletionParams:
//	    tool_call ids            -> ToolIDNormalizer.NormalizeToolID
//	    thinking level           -> ThinkingMapper.ApplyThinkingLevel, else
//	                                the typed reasoning_effort field
//	    params built, ExtraBody  -> ParamsAdjuster.AdjustParams (fires last,
//	                                so it sees everything the body will carry)
//	convertResponse / generateStream:
//	    message or delta         -> ReasoningDecoder.DecodeMessage / DecodeDelta
//	    streamed deltas          -> ReasoningAccumulator from NewAccumulator
//	    usage object             -> UsageDecoder.DecodeUsage
//	convertContentToMessages:
//	    assistant reasoning      -> ReasoningEncoder.EncodeReasoning, in the
//	                                native egress mode only
//
//	EgressPolicy resolves at construction, before the pipeline runs.
//
// The pipeline rules a dialect cannot change stay in the adapter: thought
// Parts never merge into the reply text, reasoning attaches to assistant
// turns only, a reasoning-only turn sends no message, and the egress mode
// in Config is the policy applied on top of whatever a dialect encodes.
//
// Capabilities are asserted once in New and held as fields, so the hot
// path is a nil check, not a type assertion.
type Dialect interface {
	// Name identifies the dialect in logs and errors, for example
	// "mistral" or "openrouter".
	Name() string
}

// ToolIDNormalizer rewrites tool_call IDs to the shape the provider
// accepts, applied to the assistant message's tool_calls and to the tool
// messages that refer back to them. The adapter keeps the reverse mapping
// so ADK keeps seeing its original IDs; the dialect only decides the wire
// shape. Not implemented means the OpenAI rule: IDs over 40 characters are
// hashed shorter, the rest pass through.
type ToolIDNormalizer interface {
	NormalizeToolID(id string) string
}

// ParamsAdjuster takes the last look at the outgoing request params, after
// the adapter finished building them and merging ExtraBody, right before
// they hit the wire, and mutates them in place. For providers that reject
// combinations the OpenAI schema accepts: xAI's reasoning models refuse
// presence_penalty, frequency_penalty and stop, and some gateways refuse
// stream_options. stream reports whether this call streams. Not
// implemented means the params go out as built.
type ParamsAdjuster interface {
	AdjustParams(params *openai.ChatCompletionNewParams, req *model.LLMRequest, stream bool)
}

// ReasoningDecoder reads the reasoning fields a provider adds to
// responses, in wire shapes the official Chat Completions schema does not
// define, and turns them into thought Parts. The SDK does not type any of
// these fields; they live only in the raw JSON envelope the decoder
// receives. Not implemented means nothing is read, which is what OpenAI's
// own API needs: its reasoning models expose a token count only.
type ReasoningDecoder interface {
	// DecodeMessage reads the reasoning one complete response message
	// carries, from the message's raw JSON envelope. Returns nil when it
	// carries none.
	DecodeMessage(rawJSON string) []*genai.Part
	// DecodeDelta reads the reasoning one stream chunk's delta carries,
	// from the delta's raw JSON envelope. Returns nil when it carries none.
	DecodeDelta(rawJSON string) []*genai.Part
	// NewAccumulator returns a fresh accumulator for one streamed turn.
	NewAccumulator() ReasoningAccumulator
}

// ReasoningEncoder renders a turn's thought Parts as extra fields for the
// assistant message, used by the native egress mode. The map goes through
// the SDK's extra-fields escape hatch, so any JSON-serialisable value
// under any key the provider expects is fine: OpenRouter's
// reasoning_details array travels here verbatim. Returns nil to send
// nothing. Not implemented means native mode degrades to think tags: the
// trace survives in content instead of being dropped, which is what
// closed-schema providers (Mistral rejects unknown message fields with a
// 400) want without any extra config.
type ReasoningEncoder interface {
	EncodeReasoning(thoughts []*genai.Part) map[string]any
}

// UsageDecoder folds usage buckets the provider reports outside the
// standard object into the metadata the adapter already built. rawJSON is
// the SDK's raw envelope of the usage object. Called only when the
// response reported tokens, so usage is never nil.
type UsageDecoder interface {
	DecodeUsage(rawJSON string, usage *genai.GenerateContentResponseUsageMetadata)
}

// ThinkingMapper translates genai's thinking level into the provider's
// native reasoning-effort knob. Implemented, the dialect owns the mapping
// entirely; not implemented means the typed OpenAI field reasoning_effort
// is used. The native knob varies by provider: OpenRouter's effort lives
// in a reasoning object at the request root, vLLM and Qwen use
// enable_thinking, so one typed field cannot serve them all.
type ThinkingMapper interface {
	ApplyThinkingLevel(params *openai.ChatCompletionNewParams, level genai.ThinkingLevel)
}

// EgressPolicy declares the replay shapes the provider tolerates. A
// requested egress mode the provider rejects is replaced at construction
// by one it accepts, and the override is logged. Without this capability
// the requested mode stands, with the adapter's own fallbacks.
type EgressPolicy interface {
	ResolveEgress(asked ReasoningEgressMode) ReasoningEgressMode
}
