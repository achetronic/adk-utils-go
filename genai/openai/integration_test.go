//go:build integration

// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

// These tests are excluded from the default `go test ./...` run. Build them
// with `-tags=integration`. They escalate cost on purpose:
//
//	Step A (free, offline): capture the request body the SDK would send and
//	  validate it against the pinned OpenAI OpenAPI spec.
//	Step B (paid, opt-in): only if A passed AND OPENAI_API_KEY is set, send the
//	  request to the real API and require a non-4xx response.
//
// Run: go test -tags=integration ./genai/openai/...
// With a real call: OPENAI_API_KEY=sk-... go test -tags=integration ./genai/openai/...
package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"os"
	"testing"

	"github.com/getkin/kin-openapi/openapi3"
	"github.com/getkin/kin-openapi/openapi3filter"
	"github.com/getkin/kin-openapi/routers"
	"github.com/getkin/kin-openapi/routers/gorillamux"
	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"
)

// newSchemaRouter loads the pinned spec once. kin-openapi is used (not
// libopenapi) because OpenAI's spec declares 3.1.0 yet uses the 3.0 `nullable`
// keyword, which strict 3.1 validators refuse to compile.
func newSchemaRouter(t *testing.T) routers.Router {
	t.Helper()
	loader := openapi3.NewLoader()
	doc, err := loader.LoadFromFile("../testdata/openapi/openai.yaml")
	if err != nil {
		t.Fatalf("load spec: %v", err)
	}
	router, err := gorillamux.NewRouter(doc)
	if err != nil {
		t.Fatalf("build router: %v", err)
	}
	return router
}

// validateBody runs step A: the captured body must satisfy the request schema.
func validateBody(t *testing.T, router routers.Router, body []byte) {
	t.Helper()
	req, _ := http.NewRequest("POST", "https://api.openai.com/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer test")

	route, pathParams, err := router.FindRoute(req)
	if err != nil {
		t.Fatalf("find route: %v", err)
	}
	err = openapi3filter.ValidateRequest(context.Background(), &openapi3filter.RequestValidationInput{
		Request:    req,
		PathParams: pathParams,
		Route:      route,
		Options:    &openapi3filter.Options{AuthenticationFunc: openapi3filter.NoopAuthenticationFunc},
	})
	if err != nil {
		t.Fatalf("request body fails OpenAI schema: %v\n\nbody: %s", err, body)
	}
}

// sendReal runs step B: dispatch the same request to the real API and require
// a non-4xx status. Skipped unless OPENAI_API_KEY is set. A captured body is
// passed so we assert exactly what step A validated reaches the API.
func sendReal(t *testing.T, body []byte) {
	t.Helper()
	key := os.Getenv("OPENAI_API_KEY")
	if key == "" {
		t.Skip("OPENAI_API_KEY not set; skipping real-API step")
	}
	baseURL := os.Getenv("OPENAI_BASE_URL")
	if baseURL == "" {
		baseURL = "https://api.openai.com/v1"
	}
	req, _ := http.NewRequest("POST", baseURL+"/chat/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+key)

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

// captureWireBody fires one request through the SDK at a local server and
// returns the raw bytes it put on the wire, so steps A and B validate/send the
// exact same body the adapter produces.
func captureWireBody(t *testing.T, req *model.LLMRequest) []byte {
	t.Helper()
	body := captureBody(t, req)
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("re-marshal captured body: %v", err)
	}
	return raw
}

func TestIntegration_OpenAI_ToolCallNoArgs(t *testing.T) {
	router := newSchemaRouter(t)
	req := &model.LLMRequest{
		Config: &genai.GenerateContentConfig{},
		Contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: "stop"}}},
			{Role: "model", Parts: []*genai.Part{
				{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "exit_loop"}},
			}},
			{Role: "user", Parts: []*genai.Part{
				{FunctionResponse: &genai.FunctionResponse{ID: "call_1"}},
			}},
		},
	}
	body := captureWireBody(t, req)
	validateBody(t, router, body)
	sendReal(t, body)
}
