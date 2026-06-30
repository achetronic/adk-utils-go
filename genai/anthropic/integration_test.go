//go:build integration

// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

// Excluded from the default `go test ./...`. Build with `-tags=integration`.
// Two escalating steps, same as the OpenAI adapter:
//
//	Step A (free, offline): validate the request body the SDK would send
//	  against the pinned Anthropic OpenAPI spec.
//	Step B (paid, opt-in): only if A passed AND ANTHROPIC_API_KEY is set, send
//	  to the real API and require a non-4xx response.
//
// Run: go test -tags=integration ./genai/anthropic/...
// With a real call: ANTHROPIC_API_KEY=sk-ant-... go test -tags=integration ./genai/anthropic/...
package anthropic

import (
	"bytes"
	"encoding/json"
	"net/http"
	"os"
	"testing"

	"github.com/pb33f/libopenapi"
	validator "github.com/pb33f/libopenapi-validator"
	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"
)

// newSchemaValidator loads the pinned spec once. libopenapi does real 3.1
// validation, which Anthropic's spec is (unlike OpenAI's defective one).
func newSchemaValidator(t *testing.T) validator.Validator {
	t.Helper()
	spec, err := os.ReadFile("../testdata/openapi/anthropic.json")
	if err != nil {
		t.Fatalf("read spec: %v", err)
	}
	doc, err := libopenapi.NewDocument(spec)
	if err != nil {
		t.Fatalf("parse spec: %v", err)
	}
	v, errs := validator.NewValidator(doc)
	if len(errs) > 0 {
		t.Fatalf("build validator: %v", errs)
	}
	return v
}

// validateBody runs step A.
func validateBody(t *testing.T, v validator.Validator, body []byte) {
	t.Helper()
	req, _ := http.NewRequest("POST", "https://api.anthropic.com/v1/messages", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", "test")
	req.Header.Set("anthropic-version", "2023-06-01")

	ok, valErrs := v.ValidateHttpRequest(req)
	if !ok {
		for _, e := range valErrs {
			t.Errorf("schema: %s", e.Message)
			for _, sv := range e.SchemaValidationErrors {
				t.Errorf("  - %s", sv.Reason)
			}
		}
		t.Fatalf("request body fails Anthropic schema\n\nbody: %s", body)
	}
}

// sendReal runs step B: dispatch to the real API, require non-4xx. Skipped
// unless ANTHROPIC_API_KEY is set.
func sendReal(t *testing.T, body []byte) {
	t.Helper()
	key := os.Getenv("ANTHROPIC_API_KEY")
	if key == "" {
		t.Skip("ANTHROPIC_API_KEY not set; skipping real-API step")
	}
	baseURL := os.Getenv("ANTHROPIC_BASE_URL")
	if baseURL == "" {
		baseURL = "https://api.anthropic.com"
	}
	req, _ := http.NewRequest("POST", baseURL+"/v1/messages", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", key)
	req.Header.Set("anthropic-version", "2023-06-01")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("real API call: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 && resp.StatusCode < 500 {
		buf := new(bytes.Buffer)
		buf.ReadFrom(resp.Body)
		t.Fatalf("real API rejected the request: %d\n\nbody sent: %s\n\nresponse: %s", resp.StatusCode, body, buf.String())
	}
}

func captureWireBody(t *testing.T, cfg Config, req *model.LLMRequest) []byte {
	t.Helper()
	body := captureBodyFor(t, cfg, req)
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("re-marshal captured body: %v", err)
	}
	return raw
}

// testModel is the real model used by step B; override with ANTHROPIC_TEST_MODEL.
func testModel() string {
	if m := os.Getenv("ANTHROPIC_TEST_MODEL"); m != "" {
		return m
	}
	return "claude-sonnet-4-6"
}

// A tool round-trip (call with no args, then its result) exercises both the
// nil-payload normalisation and the tool_use/tool_result pairing Anthropic
// enforces.
func TestIntegration_Anthropic_ToolRoundTrip(t *testing.T) {
	v := newSchemaValidator(t)
	req := &model.LLMRequest{
		Config: &genai.GenerateContentConfig{MaxOutputTokens: 1024},
		Contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: "stop"}}},
			{Role: "model", Parts: []*genai.Part{
				{FunctionCall: &genai.FunctionCall{ID: "toolu_1", Name: "exit_loop"}},
			}},
			{Role: "user", Parts: []*genai.Part{
				{FunctionResponse: &genai.FunctionResponse{ID: "toolu_1"}},
			}},
		},
	}
	body := captureWireBody(t, Config{ModelName: testModel()}, req)
	validateBody(t, v, body)
	sendReal(t, body)
}
