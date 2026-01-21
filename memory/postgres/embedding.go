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

package postgres

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// OpenAICompatibleEmbedding implements EmbeddingModel using the OpenAI embeddings API format.
// This is the de facto standard supported by: OpenAI, Ollama (/v1), Azure OpenAI, vLLM, LocalAI, LiteLLM, etc.
type OpenAICompatibleEmbedding struct {
	BaseURL string // e.g., "https://api.openai.com/v1", "http://localhost:11434/v1"
	APIKey  string // optional, not required for local models
	Model   string // e.g., "text-embedding-3-small", "nomic-embed-text"
	dim     int    // embedding dimension, auto-detected if 0

	// HTTPClient allows customizing the HTTP client used for requests.
	// If nil, http.DefaultClient is used.
	HTTPClient *http.Client
}

// OpenAICompatibleEmbeddingConfig holds configuration for the embedding model.
type OpenAICompatibleEmbeddingConfig struct {
	BaseURL   string
	APIKey    string
	Model     string
	Dimension int // optional, will be auto-detected on first call if 0

	// HTTPClient allows customizing the HTTP client used for requests.
	// Useful for testing with mock servers.
	HTTPClient *http.Client
}

// NewOpenAICompatibleEmbedding creates a new embedding model using OpenAI-compatible API.
// Works with OpenAI, Ollama, vLLM, LocalAI, LiteLLM, Azure OpenAI, etc.
func NewOpenAICompatibleEmbedding(cfg OpenAICompatibleEmbeddingConfig) *OpenAICompatibleEmbedding {
	httpClient := cfg.HTTPClient
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	return &OpenAICompatibleEmbedding{
		BaseURL:    strings.TrimSuffix(cfg.BaseURL, "/"),
		APIKey:     cfg.APIKey,
		Model:      cfg.Model,
		dim:        cfg.Dimension,
		HTTPClient: httpClient,
	}
}

// Dimension returns the embedding dimension.
// Returns 0 if not yet known (will be auto-detected on first Embed call).
func (e *OpenAICompatibleEmbedding) Dimension() int {
	return e.dim
}

// Embed generates an embedding vector for the given text.
func (e *OpenAICompatibleEmbedding) Embed(ctx context.Context, text string) ([]float32, error) {
	reqBody := map[string]any{
		"model": e.Model,
		"input": text,
	}
	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", e.BaseURL+"/embeddings", bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if e.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+e.APIKey)
	}

	resp, err := e.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to call embedding API: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("embedding API returned status %d: %s", resp.StatusCode, string(body))
	}

	var result embeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if len(result.Data) == 0 {
		return nil, fmt.Errorf("no embedding returned")
	}

	embedding := result.Data[0].Embedding

	// Auto-detect dimension on first successful call
	if e.dim == 0 && len(embedding) > 0 {
		e.dim = len(embedding)
	}

	return embedding, nil
}

// embeddingResponse represents the OpenAI embeddings API response format.
type embeddingResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
	Model string `json:"model"`
	Usage struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
}

// Ensure interface is implemented
var _ EmbeddingModel = (*OpenAICompatibleEmbedding)(nil)
