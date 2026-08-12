// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package openai

// DeepSeek is the dialect for DeepSeek: reasoning travels as the plain-text
// field, and the provider requires it back on assistant turns.
var DeepSeek DeepSeekDialect

// DeepSeekDialect layers DeepSeek's strict replay rule on the text dialect:
// thinking mode rejects a tool-call history whose assistant turns lack the
// reasoning field, so the replay shape is fixed to native.
type DeepSeekDialect struct {
	TextDialect
}

// Name identifies the dialect.
func (DeepSeekDialect) Name() string { return "deepseek" }

// ResolveEgress accepts the native shape only: omitting or folding the
// reasoning leaves the assistant turn without the key the provider checks
// for, and it rejects the turn with a 400.
func (DeepSeekDialect) ResolveEgress(ReasoningEgressMode) ReasoningEgressMode {
	return ReasoningEgressNative
}
