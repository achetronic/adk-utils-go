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
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestEmbedSuccess(t *testing.T) {
	// Mock server that returns a valid embedding
	mockEmbedding := []float32{0.1, 0.2, 0.3, 0.4, 0.5}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify request
		if r.Method != "POST" {
			t.Errorf("Expected POST, got %s", r.Method)
		}
		if r.URL.Path != "/embeddings" {
			t.Errorf("Expected /embeddings, got %s", r.URL.Path)
		}
		if r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("Expected application/json content type")
		}

		// Verify request body
		var reqBody map[string]any
		if err := json.NewDecoder(r.Body).Decode(&reqBody); err != nil {
			t.Errorf("Failed to decode request body: %v", err)
		}
		if reqBody["model"] != "test-model" {
			t.Errorf("Expected model 'test-model', got %v", reqBody["model"])
		}
		if reqBody["input"] != "test input text" {
			t.Errorf("Expected input 'test input text', got %v", reqBody["input"])
		}

		// Return mock response
		resp := map[string]any{
			"data": []map[string]any{
				{"embedding": mockEmbedding, "index": 0},
			},
			"model": "test-model",
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	emb := NewOpenAICompatibleEmbedding(OpenAICompatibleEmbeddingConfig{
		BaseURL:    server.URL,
		Model:      "test-model",
		HTTPClient: server.Client(),
	})

	result, err := emb.Embed(context.Background(), "test input text")
	if err != nil {
		t.Fatalf("Embed failed: %v", err)
	}

	if len(result) != len(mockEmbedding) {
		t.Errorf("Expected %d dimensions, got %d", len(mockEmbedding), len(result))
	}

	for i, v := range result {
		if v != mockEmbedding[i] {
			t.Errorf("Embedding[%d]: expected %f, got %f", i, mockEmbedding[i], v)
		}
	}

	t.Logf("✓ Embed success: returned %d-dimensional embedding", len(result))
}

func TestEmbedWithAPIKey(t *testing.T) {
	expectedAPIKey := "test-api-key-12345"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify Authorization header
		authHeader := r.Header.Get("Authorization")
		expectedAuth := "Bearer " + expectedAPIKey
		if authHeader != expectedAuth {
			t.Errorf("Expected Authorization '%s', got '%s'", expectedAuth, authHeader)
		}

		// Return minimal valid response
		resp := map[string]any{
			"data": []map[string]any{
				{"embedding": []float32{0.1}, "index": 0},
			},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	emb := NewOpenAICompatibleEmbedding(OpenAICompatibleEmbeddingConfig{
		BaseURL:    server.URL,
		APIKey:     expectedAPIKey,
		Model:      "test-model",
		HTTPClient: server.Client(),
	})

	_, err := emb.Embed(context.Background(), "test")
	if err != nil {
		t.Fatalf("Embed failed: %v", err)
	}

	t.Logf("✓ Embed with API key: Authorization header correctly set")
}

func TestEmbedWithoutAPIKey(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify no Authorization header
		authHeader := r.Header.Get("Authorization")
		if authHeader != "" {
			t.Errorf("Expected no Authorization header, got '%s'", authHeader)
		}

		resp := map[string]any{
			"data": []map[string]any{
				{"embedding": []float32{0.1}, "index": 0},
			},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	emb := NewOpenAICompatibleEmbedding(OpenAICompatibleEmbeddingConfig{
		BaseURL:    server.URL,
		Model:      "test-model",
		HTTPClient: server.Client(),
	})

	_, err := emb.Embed(context.Background(), "test")
	if err != nil {
		t.Fatalf("Embed failed: %v", err)
	}

	t.Logf("✓ Embed without API key: no Authorization header sent")
}

func TestEmbedAutoDimension(t *testing.T) {
	mockEmbedding := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := map[string]any{
			"data": []map[string]any{
				{"embedding": mockEmbedding, "index": 0},
			},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	emb := NewOpenAICompatibleEmbedding(OpenAICompatibleEmbeddingConfig{
		BaseURL:    server.URL,
		Model:      "test-model",
		HTTPClient: server.Client(),
		// Dimension not set - should be auto-detected
	})

	// Before first call, dimension should be 0
	if emb.Dimension() != 0 {
		t.Errorf("Expected dimension 0 before first call, got %d", emb.Dimension())
	}

	_, err := emb.Embed(context.Background(), "test")
	if err != nil {
		t.Fatalf("Embed failed: %v", err)
	}

	// After first call, dimension should be auto-detected
	if emb.Dimension() != len(mockEmbedding) {
		t.Errorf("Expected dimension %d after first call, got %d", len(mockEmbedding), emb.Dimension())
	}

	t.Logf("✓ Embed auto dimension: correctly detected %d dimensions", emb.Dimension())
}

func TestEmbedPresetDimension(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := map[string]any{
			"data": []map[string]any{
				{"embedding": []float32{0.1, 0.2, 0.3}, "index": 0},
			},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	emb := NewOpenAICompatibleEmbedding(OpenAICompatibleEmbeddingConfig{
		BaseURL:    server.URL,
		Model:      "test-model",
		Dimension:  1536, // Preset dimension
		HTTPClient: server.Client(),
	})

	// Dimension should be preset value
	if emb.Dimension() != 1536 {
		t.Errorf("Expected preset dimension 1536, got %d", emb.Dimension())
	}

	t.Logf("✓ Embed preset dimension: correctly set to %d", emb.Dimension())
}

func TestEmbedServerError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte("Internal Server Error"))
	}))
	defer server.Close()

	emb := NewOpenAICompatibleEmbedding(OpenAICompatibleEmbeddingConfig{
		BaseURL:    server.URL,
		Model:      "test-model",
		HTTPClient: server.Client(),
	})

	_, err := emb.Embed(context.Background(), "test")
	if err == nil {
		t.Fatal("Expected error for server error response")
	}

	t.Logf("✓ Embed server error: correctly returned error: %v", err)
}

func TestEmbedEmptyResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := map[string]any{
			"data": []map[string]any{}, // Empty data array
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	emb := NewOpenAICompatibleEmbedding(OpenAICompatibleEmbeddingConfig{
		BaseURL:    server.URL,
		Model:      "test-model",
		HTTPClient: server.Client(),
	})

	_, err := emb.Embed(context.Background(), "test")
	if err == nil {
		t.Fatal("Expected error for empty embedding response")
	}

	t.Logf("✓ Embed empty response: correctly returned error: %v", err)
}

func TestEmbedInvalidJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("not valid json"))
	}))
	defer server.Close()

	emb := NewOpenAICompatibleEmbedding(OpenAICompatibleEmbeddingConfig{
		BaseURL:    server.URL,
		Model:      "test-model",
		HTTPClient: server.Client(),
	})

	_, err := emb.Embed(context.Background(), "test")
	if err == nil {
		t.Fatal("Expected error for invalid JSON response")
	}

	t.Logf("✓ Embed invalid JSON: correctly returned error: %v", err)
}

func TestEmbedContextCancellation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Simulate slow response - this should be cancelled
		select {
		case <-r.Context().Done():
			return
		}
	}))
	defer server.Close()

	emb := NewOpenAICompatibleEmbedding(OpenAICompatibleEmbeddingConfig{
		BaseURL:    server.URL,
		Model:      "test-model",
		HTTPClient: server.Client(),
	})

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	_, err := emb.Embed(ctx, "test")
	if err == nil {
		t.Fatal("Expected error for cancelled context")
	}

	t.Logf("✓ Embed context cancellation: correctly returned error: %v", err)
}

func TestEmbedBaseURLTrailingSlash(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify path doesn't have double slashes
		if r.URL.Path != "/embeddings" {
			t.Errorf("Expected /embeddings, got %s", r.URL.Path)
		}
		resp := map[string]any{
			"data": []map[string]any{
				{"embedding": []float32{0.1}, "index": 0},
			},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	// Test with trailing slash in BaseURL
	emb := NewOpenAICompatibleEmbedding(OpenAICompatibleEmbeddingConfig{
		BaseURL:    server.URL + "/", // Trailing slash
		Model:      "test-model",
		HTTPClient: server.Client(),
	})

	_, err := emb.Embed(context.Background(), "test")
	if err != nil {
		t.Fatalf("Embed failed: %v", err)
	}

	t.Logf("✓ Embed trailing slash: correctly handled")
}
