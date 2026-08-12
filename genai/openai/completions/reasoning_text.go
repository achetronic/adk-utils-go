// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package completions

import (
	"google.golang.org/genai"
)

// TextDialect carries reasoning as a single plain-text field on the
// assistant message, which is the shape DeepSeek in thinking mode and
// Kimi's preserved-thinking models require. It also covers every
// compatible provider that emits reasoning as one string under a name of
// its own: point ReadFields at that name. Mistral's newer models emit the
// same shape.
type TextDialect struct {
	// WriteField is the assistant-message field the reasoning goes out in.
	// Empty means reasoningWriteField.
	WriteField string
	// ReadFields is the ordered list of fields probed on ingest; the first
	// present non-empty string wins. Empty means DefaultTextReadFields.
	// Read and write are split on purpose: what varies between providers is
	// the name they emit, while they all accept reasoning_content back.
	ReadFields []string
}

// NewTextDialect returns a TextDialect normalised to its defaults.
func NewTextDialect() *TextDialect {
	return &TextDialect{}
}

// DefaultTextReadFields are the plain-text reasoning fields TextDialect
// probes when ReadFields is empty: reasoning_content is the DeepSeek, Kimi
// and Mistral name, reasoning the OpenRouter and newer vLLM name. The
// first present non-empty string wins.
var DefaultTextReadFields = []string{"reasoning_content", "reasoning"}

// reasoningWriteField is the default assistant-message field TextDialect
// sends reasoning back in: DeepSeek, Kimi and vLLM read it on input, and
// OpenRouter accepts it as an alias for its own reasoning field.
const reasoningWriteField = "reasoning_content"

// Name identifies the dialect.
func (d TextDialect) Name() string { return "text" }

func (d TextDialect) writeField() string {
	if d.WriteField != "" {
		return d.WriteField
	}
	return reasoningWriteField
}

func (d TextDialect) readFields() []string {
	if len(d.ReadFields) > 0 {
		return d.ReadFields
	}
	return DefaultTextReadFields
}

func (d TextDialect) decode(rawJSON string) []*genai.Part {
	text := probeString(probeRawJSON(rawJSON), d.readFields())
	if text == "" {
		return nil
	}
	return []*genai.Part{thoughtPart(text)}
}

// DecodeMessage reads the reasoning one complete response message carries.
func (d TextDialect) DecodeMessage(rawJSON string) []*genai.Part { return d.decode(rawJSON) }

// DecodeDelta reads the reasoning one stream chunk's delta carries.
func (d TextDialect) DecodeDelta(rawJSON string) []*genai.Part { return d.decode(rawJSON) }

// NewAccumulator returns an accumulator that concatenates the deltas' texts.
func (d TextDialect) NewAccumulator() ReasoningAccumulator { return &textReasoningAccumulator{} }

// EncodeReasoning writes the joined reasoning as the dialect's write field.
func (d TextDialect) EncodeReasoning(thoughts []*genai.Part) map[string]any {
	texts := readableThoughtTexts(thoughts)
	if len(texts) == 0 {
		return nil
	}
	return map[string]any{d.writeField(): joinTexts(texts)}
}

// textReasoningAccumulator concatenates the deltas' reasoning texts.
type textReasoningAccumulator struct {
	text string
}

func (a *textReasoningAccumulator) AddDelta(parts []*genai.Part) {
	for _, part := range parts {
		if part == nil {
			continue
		}
		a.text += part.Text
	}
}

func (a *textReasoningAccumulator) Parts() []*genai.Part {
	if a.text == "" {
		return nil
	}
	return []*genai.Part{thoughtPart(a.text)}
}
