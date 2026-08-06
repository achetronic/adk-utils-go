// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package openai

import (
	"encoding/json"

	"google.golang.org/genai"
)

// reasoningDetailsField is OpenRouter's normalised name for the structured
// reasoning array. Unlike the plain-text field (Config.ReasoningField) this
// name does not vary by provider: OpenRouter uses it for every model it
// fronts, so it is a constant rather than an option.
const reasoningDetailsField = "reasoning_details"

// ReasoningDetailMetadataKey is the genai.Part.PartMetadata key under which a
// single reasoning_details block is preserved, exactly as the provider sent
// it. Consumers that inspect or filter reasoning can look for this key; the
// adapter itself only ever passes the value straight back.
const ReasoningDetailMetadataKey = "openai.reasoning_detail"

// Block types OpenRouter defines. Only used to find the readable text of a
// block: an unknown type still round-trips, it just has no text to show.
const (
	reasoningDetailText      = "reasoning.text"
	reasoningDetailSummary   = "reasoning.summary"
	reasoningDetailEncrypted = "reasoning.encrypted"
)

// extractReasoningDetails reads OpenRouter's reasoning_details array from the
// SDK's raw JSON envelope, on choices[].message for a whole response and on
// choices[].delta for a stream chunk.
//
// Blocks are kept as decoded JSON objects rather than typed structs on
// purpose. The schema is open (providers may add keys), the documented format
// list keeps growing, and OpenRouter requires the sequence of blocks replayed
// on the next turn to match what the model produced. Anything this adapter
// does not understand therefore has to survive untouched, nulls included.
//
// Returns nil when the field is absent, empty, or not an array of objects.
func extractReasoningDetails(rawJSON string) []map[string]any {
	if rawJSON == "" {
		return nil
	}
	var probe map[string]json.RawMessage
	if err := json.Unmarshal([]byte(rawJSON), &probe); err != nil {
		return nil
	}
	raw, ok := probe[reasoningDetailsField]
	if !ok {
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

// reasoningDetailText returns the human-readable text of a block: the text of
// a reasoning.text block, the summary of a reasoning.summary block. An
// encrypted block is opaque and has none, and neither does a block of a type
// we do not know.
func readableReasoningDetail(block map[string]any) string {
	switch block["type"] {
	case reasoningDetailText:
		text, _ := block["text"].(string)
		return text
	case reasoningDetailSummary:
		summary, _ := block["summary"].(string)
		return summary
	default:
		return ""
	}
}

// reasoningDetailsToParts maps reasoning blocks to thought Parts, one per
// block and in wire order, so the order OpenRouter requires on replay is the
// Part order. The block travels verbatim in PartMetadata; the readable text
// is duplicated into Text so consumers that filter on Thought still see the
// reasoning. An encrypted block yields a Part with empty Text and metadata
// only, the same convention the Anthropic adapter uses for a redacted
// thinking block.
func reasoningDetailsToParts(blocks []map[string]any) []*genai.Part {
	parts := make([]*genai.Part, 0, len(blocks))
	for _, block := range blocks {
		parts = append(parts, &genai.Part{
			Text:         readableReasoningDetail(block),
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

// reasoningAccumulator rebuilds a turn's reasoning from stream chunks.
//
// It exists because the SDK's ChatCompletionAccumulator cannot help here: it
// merges chunks field by field and drops everything it has no typed field
// for, leaving the aggregated message's raw JSON empty. Reading reasoning off
// that aggregate silently yields nothing, so the terminal response of a
// streamed turn would carry no reasoning at all.
type reasoningAccumulator struct {
	// text is the concatenation of the plain-text reasoning deltas.
	text string
	// blocks holds merged reasoning_details blocks in first-seen order.
	blocks []map[string]any
	// byIndex maps a block's reported index to its slot in blocks.
	byIndex map[float64]int
}

// addText appends a plain-text reasoning delta.
func (r *reasoningAccumulator) addText(delta string) {
	r.text += delta
}

// addBlocks merges reasoning_details blocks from one chunk. OpenRouter builds
// the complete sequence by concatenating chunks in order, keyed by the
// block's index; blocks without one are appended in arrival order.
func (r *reasoningAccumulator) addBlocks(blocks []map[string]any) {
	for _, block := range blocks {
		index, hasIndex := block["index"].(float64)
		if !hasIndex {
			r.blocks = append(r.blocks, cloneBlock(block))
			continue
		}
		if r.byIndex == nil {
			r.byIndex = map[float64]int{}
		}
		slot, seen := r.byIndex[index]
		if !seen {
			r.byIndex[index] = len(r.blocks)
			r.blocks = append(r.blocks, cloneBlock(block))
			continue
		}
		mergeBlock(r.blocks[slot], block)
	}
}

// hasReasoning reports whether anything was accumulated.
func (r *reasoningAccumulator) hasReasoning() bool {
	return r.text != "" || len(r.blocks) > 0
}

// parts renders the accumulated reasoning as thought Parts. Blocks win over
// the plain text when both arrived: OpenRouter populates both fields with the
// same reasoning, and the blocks carry strictly more (signatures, encrypted
// data, ids), so using both would duplicate the reasoning.
func (r *reasoningAccumulator) parts() []*genai.Part {
	if len(r.blocks) > 0 {
		return reasoningDetailsToParts(r.blocks)
	}
	if r.text != "" {
		return []*genai.Part{{Text: r.text, Thought: true}}
	}
	return nil
}

// cloneBlock copies a block one level deep so merging chunk deltas never
// writes into the JSON the caller handed us.
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
