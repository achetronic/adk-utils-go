// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package completions

import (
	"testing"

	"google.golang.org/genai"
)

// Beam rejects any request whose earlier assistant turn lacks
// reasoning_content, so the replay shape is pinned to native even when the
// caller asks to omit it.
func TestReflectionDialect_ResolveEgressNative(t *testing.T) {
	for _, asked := range []ReasoningEgressMode{ReasoningEgressThinkTags, ReasoningEgressOmit} {
		m := New(Config{ModelName: "beam", Dialect: Reflection, ReasoningEgress: asked})
		if m.reasoningEgress != ReasoningEgressNative {
			t.Errorf("egress = %q, want native for asked %q", m.reasoningEgress, asked)
		}

		msgs, err := m.convertContentToMessages(&genai.Content{
			Role: "model",
			Parts: []*genai.Part{
				{Text: "let me think", Thought: true},
				{Text: "the answer"},
			},
		})
		if err != nil {
			t.Fatalf("convertContentToMessages: %v", err)
		}
		extra := msgs[0].OfAssistant.ExtraFields()
		if extra[reasoningWriteField] != "let me think" {
			t.Errorf("extra[%q] = %v, want the reasoning echoed", reasoningWriteField, extra[reasoningWriteField])
		}
	}
}

// Reflexion decodes reasoning_content on ingest and re-encodes it on egress.
func TestReflectionDialect_RoundTrip(t *testing.T) {
	parts := Reflection.DecodeMessage(`{"reasoning_content":"step one"}`)
	if len(parts) != 1 || parts[0] == nil || !parts[0].Thought || parts[0].Text != "step one" {
		t.Fatalf("DecodeMessage parts = %#v, want one thought part %q", parts, "step one")
	}

	extra := Reflection.EncodeReasoning([]*genai.Part{{Text: "step one", Thought: true}})
	if extra[reasoningWriteField] != "step one" {
		t.Errorf("EncodeReasoning[%q] = %v, want %q", reasoningWriteField, extra[reasoningWriteField], "step one")
	}
}
