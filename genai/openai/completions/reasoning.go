// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package completions

import (
	"encoding/json"

	"google.golang.org/genai"
)

// ReasoningEgressMode selects the wire shape used to send thought Parts back
// to the provider as conversation history. Without reasoning to send there
// is nothing to shape, so the mode changes nothing.
type ReasoningEgressMode string

const (
	// ReasoningEgressNative sends the reasoning in the fields the dialect's
	// encoder chooses, attached to the assistant message next to content and
	// tool_calls. The strict thinking providers check for the field itself,
	// so this is the zero value; a dialect without an encoder degrades to
	// think tags, so the trace survives instead of being dropped.
	ReasoningEgressNative ReasoningEgressMode = "native"
	// ReasoningEgressThinkTags folds the reasoning into the assistant
	// message's content inside a <think> block and sends no extra field.
	// For backends that validate messages against a closed schema and
	// answer an unknown field with a 400.
	ReasoningEgressThinkTags ReasoningEgressMode = "think_tags"
	// ReasoningEgressOmit sends no reasoning in any shape. For backends
	// that discard reasoning history server-side, or callers who would
	// rather not spend the prompt tokens.
	ReasoningEgressOmit ReasoningEgressMode = "omit"
)

// ReasoningAccumulator rebuilds one streamed turn's reasoning from the
// Parts a ReasoningDecoder decoded off the deltas.
type ReasoningAccumulator interface {
	// AddDelta folds one delta's decoded Parts into the accumulated state.
	AddDelta(parts []*genai.Part)
	// Parts renders the accumulated reasoning as thought Parts, in the order
	// the model produced it. Returns nil when nothing arrived.
	Parts() []*genai.Part
}

// probeRawJSON decodes the SDK's raw JSON envelope into a field map. Returns
// nil for an empty or unparseable envelope; callers treat that as "the field
// is not there". Chat Completions leaves provider fields untyped, so raw
// JSON is the only place they live.
func probeRawJSON(rawJSON string) map[string]json.RawMessage {
	if rawJSON == "" {
		return nil
	}
	var probe map[string]json.RawMessage
	if err := json.Unmarshal([]byte(rawJSON), &probe); err != nil {
		return nil
	}
	return probe
}

// probeString returns the first of fields present in probe as a non-empty
// string. The fields are tried in order.
func probeString(probe map[string]json.RawMessage, fields []string) string {
	for _, field := range fields {
		raw, ok := probe[field]
		if !ok {
			continue
		}
		var s string
		if err := json.Unmarshal(raw, &s); err == nil && s != "" {
			return s
		}
	}
	return ""
}

// thoughtPart builds one thought Part carrying the given text.
func thoughtPart(text string) *genai.Part {
	return &genai.Part{Text: text, Thought: true}
}

// readableThoughtTexts returns the texts of the thought Parts that have one,
// in order.
func readableThoughtTexts(thoughts []*genai.Part) []string {
	texts := make([]string, 0, len(thoughts))
	for _, part := range thoughts {
		if part == nil || part.Text == "" {
			continue
		}
		texts = append(texts, part.Text)
	}
	return texts
}

// thoughtTextFor joins the readable texts of thought Parts in order,
// skipping the ones without text.
func thoughtTextFor(thoughts []*genai.Part) string {
	return joinTexts(readableThoughtTexts(thoughts))
}
