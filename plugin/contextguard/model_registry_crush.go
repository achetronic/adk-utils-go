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

import (
	"log/slog"

	"charm.land/catwalk/pkg/catwalk"
	"charm.land/catwalk/pkg/embedded"
)

const (
	crushDefaultCtxWindow = 128000
	crushDefaultMaxTokens = 4096
)

// CrushRegistry implements ModelRegistry using catwalk's embedded model
// database. All model metadata (context windows, max tokens, costs) is
// compiled into the binary — no network calls, no background goroutines.
//
// Usage:
//
//	registry := contextguard.NewCrushRegistry()
//	guard := contextguard.New(registry)
type CrushRegistry struct {
	models map[string]catwalk.Model
}

// NewCrushRegistry creates a registry pre-loaded with every model from
// catwalk's embedded provider database.
func NewCrushRegistry() *CrushRegistry {
	models := make(map[string]catwalk.Model)
	for _, provider := range embedded.GetAll() {
		for _, m := range provider.Models {
			models[m.ID] = m
		}
	}

	slog.Info("CrushRegistry: loaded models from catwalk", "count", len(models))

	return &CrushRegistry{models: models}
}

// ContextWindow returns the context window size (in tokens) for the given
// model ID. Returns 128000 if the model is not found.
func (r *CrushRegistry) ContextWindow(modelID string) int {
	if m, ok := r.models[modelID]; ok && m.ContextWindow > 0 {
		return int(m.ContextWindow)
	}
	return crushDefaultCtxWindow
}

// DefaultMaxTokens returns the default max output tokens for the given
// model ID. Returns 4096 if the model is not found.
func (r *CrushRegistry) DefaultMaxTokens(modelID string) int {
	if m, ok := r.models[modelID]; ok && m.DefaultMaxTokens > 0 {
		return int(m.DefaultMaxTokens)
	}
	return crushDefaultMaxTokens
}
