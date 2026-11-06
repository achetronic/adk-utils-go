// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package completions

// Reflection is the dialect for Reflection AI's Beam models. Reasoning
// travels as the plain-text reasoning_content field, and the provider
// requires it back on every earlier assistant turn.
var Reflection ReflectionDialect

// ReflectionDialect layers Beam's strict replay rule on the text dialect:
// a request whose assistant turns lack reasoning_content is rejected with a
// 400, so the replay shape is fixed to native.
type ReflectionDialect struct {
	TextDialect
}

// Name identifies the dialect.
func (ReflectionDialect) Name() string { return "reflection" }

// ResolveEgress accepts the native shape only, mirroring DeepSeek: Beam
// needs reasoning_content back on every assistant turn.
func (ReflectionDialect) ResolveEgress(ReasoningEgressMode) ReasoningEgressMode {
	return ReasoningEgressNative
}
