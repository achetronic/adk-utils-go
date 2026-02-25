// Copyright 2025 achetronic
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
