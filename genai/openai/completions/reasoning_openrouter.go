// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package completions

import (
	"encoding/json"

	"github.com/openai/openai-go/v3"
	"google.golang.org/genai"
)

// OpenRouter is the dialect for OpenRouter. It carries reasoning in
// OpenRouter's structured shape: a reasoning_details array of typed blocks
// (reasoning.text with an optional signature, reasoning.summary,
// reasoning.encrypted) alongside a plain-text copy. The structured shape is
// what preserves signatures and encrypted blocks across turns, which models
// that hide readable reasoning need to keep going. Plain-text-only
// providers use TextDialect instead.
//
// OpenRouter requires the replayed block sequence to match what the model
// produced, so blocks travel as decoded JSON objects, never re-typed,
// reordered, filtered or rewritten: an unknown block type or vendor key
// must survive. Where both the array and the plain-text copy arrive, the
// array wins; it describes the same reasoning and carries strictly more.
var OpenRouter OpenRouterDialect

// OpenRouterDialect implements Dialect for OpenRouter: a reasoning decoder,
// encoder and stream accumulator. It takes no configuration.
type OpenRouterDialect struct{}

// reasoningDetailsField is OpenRouter's name for the structured reasoning
// array. OpenRouter uses it for every model it fronts, so it is a constant.
const reasoningDetailsField = "reasoning_details"

// ReasoningDetailMetadataKey is the genai.Part.PartMetadata key under which a
// single reasoning_details block is preserved, exactly as the provider sent
// it. The adapter only ever passes the value straight back; consumers that
// inspect or filter reasoning can look for this key.
const ReasoningDetailMetadataKey = "openai.reasoning_detail"

// Name identifies the dialect.
func (OpenRouterDialect) Name() string { return "openrouter" }

// DecodeMessage reads the reasoning one complete response message carries.
func (OpenRouterDialect) DecodeMessage(rawJSON string) []*genai.Part {
	return decodeOpenRouterReasoning(probeRawJSON(rawJSON))
}

// DecodeDelta reads the reasoning one stream chunk's delta carries.
func (OpenRouterDialect) DecodeDelta(rawJSON string) []*genai.Part {
	return decodeOpenRouterReasoning(probeRawJSON(rawJSON))
}

// NewAccumulator returns the accumulator that merges OpenRouter's streamed
// blocks by their reported index.
func (OpenRouterDialect) NewAccumulator() ReasoningAccumulator {
	return &openRouterReasoningAccumulator{}
}

// EncodeReasoning sends blocks back exactly as they arrived, under
// reasoning_details, and plain thoughts under the reasoning field; a Part
// feeds one of the two, never both.
func (OpenRouterDialect) EncodeReasoning(thoughts []*genai.Part) map[string]any {
	blocks := make([]map[string]any, 0, len(thoughts))
	var texts []string
	for _, part := range thoughts {
		if part == nil || !part.Thought {
			continue
		}
		// A Part carrying a block goes back as that block, which holds
		// strictly more than its text (signature, encrypted data, id). The
		// block feeds the array alone, so the reasoning is never sent twice.
		if block, ok := reasoningDetailOf(part); ok {
			blocks = append(blocks, block)
			continue
		}
		if part.Text != "" {
			texts = append(texts, part.Text)
		}
	}

	extra := make(map[string]any, 2)
	if len(blocks) > 0 {
		extra[reasoningDetailsField] = blocks
	}
	if len(texts) > 0 {
		extra[reasoningWriteField] = joinTexts(texts)
	}
	if len(extra) == 0 {
		return nil
	}
	return extra
}

// ApplyThinkingLevel writes OpenRouter's reasoning effort knob. OpenRouter
// does not read the typed reasoning_effort field; its unified interface is
// a reasoning object at the request root. The dialect's knob wins over a
// reasoning key a caller set in ExtraBody, because this area belongs to
// the dialect.
func (OpenRouterDialect) ApplyThinkingLevel(params *openai.ChatCompletionNewParams, level genai.ThinkingLevel) {
	extra := params.ExtraFields()
	if extra == nil {
		extra = make(map[string]any)
	}
	extra["reasoning"] = map[string]any{"effort": string(convertThinkingLevel(level))}
	params.SetExtraFields(extra)
}

// decodeOpenRouterReasoning reads the reasoning one envelope carries. Blocks
// win over the plain-text copy when both are present; an envelope carrying
// only the text field still yields a thought Part with it.
func decodeOpenRouterReasoning(probe map[string]json.RawMessage) []*genai.Part {
	if probe == nil {
		return nil
	}
	if blocks := extractReasoningDetails(probe[reasoningDetailsField]); len(blocks) > 0 {
		return reasoningDetailsToParts(blocks)
	}
	text := probeString(probe, []string{reasoningWriteField, "reasoning"})
	if text == "" {
		return nil
	}
	return []*genai.Part{thoughtPart(text)}
}

// extractReasoningDetails decodes the reasoning_details array. Returns nil
// when the field is absent, empty, or not an array of objects. Blocks are
// kept as decoded JSON objects rather than typed structs on purpose: the
// schema is open, the documented format list keeps growing, and anything
// this package does not understand has to survive untouched, nulls included.
func extractReasoningDetails(raw json.RawMessage) []map[string]any {
	if len(raw) == 0 {
		return nil
	}
	var blocks []map[string]any
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return nil
	}
	out := make([]map[string]any, 0, len(blocks))
	for _, block := range blocks {
		if len(block) > 0 {
			out = append(out, block)
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// readableReasoningText returns the human-readable text of a block: the text
// of a reasoning.text block, the summary of a reasoning.summary block. An
// encrypted block is opaque and has none, and neither does a block of a type
// this package does not know.
func readableReasoningText(block map[string]any) string {
	switch block["type"] {
	case "reasoning.text":
		text, _ := block["text"].(string)
		return text
	case "reasoning.summary":
		summary, _ := block["summary"].(string)
		return summary
	default:
		return ""
	}
}

// reasoningDetailsToParts maps reasoning blocks to thought Parts, one per
// block and in wire order, so the order OpenRouter requires on replay is the
// Part order. The block travels verbatim in PartMetadata; the readable text
// is mirrored into Text so consumers filtering on Thought still see the
// reasoning. An encrypted block yields a Part with empty Text and metadata
// only, the same convention the Anthropic adapter uses for a redacted
// thinking block.
func reasoningDetailsToParts(blocks []map[string]any) []*genai.Part {
	parts := make([]*genai.Part, 0, len(blocks))
	for _, block := range blocks {
		parts = append(parts, &genai.Part{
			Text:         readableReasoningText(block),
			Thought:      true,
			PartMetadata: map[string]any{ReasoningDetailMetadataKey: block},
		})
	}
	return parts
}

// reasoningDetailOf returns the reasoning block a Part carries, if any.
func reasoningDetailOf(part *genai.Part) (map[string]any, bool) {
	if part == nil || part.PartMetadata == nil {
		return nil, false
	}
	block, ok := part.PartMetadata[ReasoningDetailMetadataKey].(map[string]any)
	if !ok || len(block) == 0 {
		return nil, false
	}
	return block, true
}

// openRouterReasoningAccumulator rebuilds a turn's reasoning from stream
// chunks: the plain text concatenates, and blocks merge by their reported
// index into the complete sequence OpenRouter builds by concatenating chunks
// in order. Blocks without an index append in arrival order.
type openRouterReasoningAccumulator struct {
	text string
	// blocks holds merged reasoning_details blocks in first-seen order;
	// byIndex maps a block's reported index to its slot in blocks.
	blocks  []map[string]any
	byIndex map[float64]int
}

func (a *openRouterReasoningAccumulator) AddDelta(parts []*genai.Part) {
	for _, part := range parts {
		if block, ok := reasoningDetailOf(part); ok {
			a.addBlock(block)
			continue
		}
		if part != nil {
			a.text += part.Text
		}
	}
}

func (a *openRouterReasoningAccumulator) addBlock(block map[string]any) {
	index, hasIndex := block["index"].(float64)
	if !hasIndex {
		a.blocks = append(a.blocks, cloneBlock(block))
		return
	}
	if a.byIndex == nil {
		a.byIndex = map[float64]int{}
	}
	slot, seen := a.byIndex[index]
	if !seen {
		a.byIndex[index] = len(a.blocks)
		a.blocks = append(a.blocks, cloneBlock(block))
		return
	}
	mergeBlock(a.blocks[slot], block)
}

// Parts renders the accumulated reasoning as thought Parts. Blocks win over
// the plain text when both arrived: OpenRouter populates both fields with the
// same reasoning, and the blocks carry strictly more (signatures, encrypted
// data, ids), so using both would duplicate the reasoning.
func (a *openRouterReasoningAccumulator) Parts() []*genai.Part {
	if len(a.blocks) > 0 {
		return reasoningDetailsToParts(a.blocks)
	}
	if a.text != "" {
		return []*genai.Part{thoughtPart(a.text)}
	}
	return nil
}

// cloneBlock copies a block one level deep so merging chunk deltas never
// writes into the JSON the provider handed us.
func cloneBlock(block map[string]any) map[string]any {
	out := make(map[string]any, len(block))
	for k, v := range block {
		out[k] = v
	}
	return out
}

// mergeBlock folds a chunk's block into the one already accumulated at the
// same index: the streamed string fields concatenate, everything else takes
// the newest non-empty value.
func mergeBlock(into, from map[string]any) {
	for key, value := range from {
		switch key {
		case "text", "summary", "data":
			str, ok := value.(string)
			if !ok || str == "" {
				continue
			}
			existing, _ := into[key].(string)
			into[key] = existing + str
		default:
			// A null in a later chunk must not erase a value an earlier
			// chunk provided: signature in particular arrives late and is
			// null until it does.
			if value == nil {
				continue
			}
			if str, ok := value.(string); ok && str == "" {
				continue
			}
			into[key] = value
		}
	}
}
