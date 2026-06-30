// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package contextguard

// ModelRegistry provides model metadata needed by the ContextGuard plugin.
// Implementations can fetch data from a remote source, a local config, or
// a static map — the plugin only depends on this interface.
type ModelRegistry interface {
	// ContextWindow returns the maximum context window size (in tokens) for
	// the given model ID. If the model is unknown, a reasonable default
	// should be returned (e.g. 128000).
	ContextWindow(modelID string) int

	// DefaultMaxTokens returns the default maximum output tokens for the
	// given model ID. If the model is unknown, a reasonable default should
	// be returned (e.g. 4096).
	DefaultMaxTokens(modelID string) int
}
