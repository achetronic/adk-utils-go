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
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"sync"
	"time"
)

const (
	crushSourceURL         = "https://raw.githubusercontent.com/charmbracelet/crush/main/internal/agent/hyper/provider.json"
	crushRefreshInterval   = 6 * time.Hour
	crushFetchTimeout      = 15 * time.Second
	crushDefaultCtxWindow  = 128000
	crushDefaultMaxTokens  = 4096
	crushMaxResponseBytes  = 2 << 20
)

// crushModelInfo holds the metadata for a single LLM model as read from
// the remote provider.json file.
type crushModelInfo struct {
	ID               string  `json:"id"`
	Name             string  `json:"name"`
	ContextWindow    int     `json:"context_window"`
	DefaultMaxTokens int     `json:"default_max_tokens"`
	CostPerMIn       float64 `json:"cost_per_1m_in"`
	CostPerMOut      float64 `json:"cost_per_1m_out"`
}

// crushProviderJSON mirrors the top-level structure of the Crush
// provider.json so it can be unmarshalled directly.
type crushProviderJSON struct {
	Models []crushModelInfo `json:"models"`
}

// CrushRegistry implements ModelRegistry by fetching and caching model
// metadata from Crush's provider.json. It refreshes in the background
// every 6 hours.
//
// Usage:
//
//	registry := contextguard.NewCrushRegistry()
//	registry.Start(ctx)
//	defer registry.Stop()
//
//	guard := contextguard.New(registry)
type CrushRegistry struct {
	mu     sync.RWMutex
	models map[string]crushModelInfo
	cancel context.CancelFunc
}

// NewCrushRegistry creates an empty registry. Call Start to populate it
// and begin periodic refresh.
func NewCrushRegistry() *CrushRegistry {
	return &CrushRegistry{
		models: make(map[string]crushModelInfo),
	}
}

// Start performs the initial fetch and spawns a background goroutine
// that refreshes every 6 hours.
func (r *CrushRegistry) Start(ctx context.Context) {
	ctx, cancel := context.WithCancel(ctx)
	r.cancel = cancel

	r.fetch()

	go func() {
		ticker := time.NewTicker(crushRefreshInterval)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				r.fetch()
			}
		}
	}()
}

// Stop cancels the background refresh goroutine.
func (r *CrushRegistry) Stop() {
	if r.cancel != nil {
		r.cancel()
	}
}

// ContextWindow returns the context window size (in tokens) for the given
// model ID. Returns 128000 if the model is not found.
func (r *CrushRegistry) ContextWindow(modelID string) int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if info, ok := r.models[modelID]; ok && info.ContextWindow > 0 {
		return info.ContextWindow
	}
	return crushDefaultCtxWindow
}

// DefaultMaxTokens returns the default max output tokens for the given
// model ID. Returns 4096 if the model is not found.
func (r *CrushRegistry) DefaultMaxTokens(modelID string) int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if info, ok := r.models[modelID]; ok && info.DefaultMaxTokens > 0 {
		return info.DefaultMaxTokens
	}
	return crushDefaultMaxTokens
}

// fetch downloads the provider.json, parses it, and atomically replaces
// the in-memory model map. Errors are logged and silently ignored so the
// registry keeps serving stale data rather than failing.
func (r *CrushRegistry) fetch() {
	ctx, cancel := context.WithTimeout(context.Background(), crushFetchTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, crushSourceURL, nil)
	if err != nil {
		slog.Warn("CrushRegistry: failed to create request", "error", err)
		return
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		slog.Warn("CrushRegistry: fetch failed", "error", err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		slog.Warn("CrushRegistry: unexpected status", "status", resp.StatusCode)
		return
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, crushMaxResponseBytes))
	if err != nil {
		slog.Warn("CrushRegistry: read failed", "error", err)
		return
	}

	var provider crushProviderJSON
	if err := json.Unmarshal(body, &provider); err != nil {
		slog.Warn("CrushRegistry: parse failed", "error", err)
		return
	}

	models := make(map[string]crushModelInfo, len(provider.Models))
	for _, m := range provider.Models {
		models[m.ID] = m
	}

	r.mu.Lock()
	r.models = models
	r.mu.Unlock()

	slog.Info(fmt.Sprintf("CrushRegistry: loaded %d models", len(models)))
}
