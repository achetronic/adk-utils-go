// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

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
