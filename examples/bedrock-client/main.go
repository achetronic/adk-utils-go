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

// Bedrock Client Example
//
// This example shows how to use the Bedrock Converse API client with ADK,
// optionally with a Bedrock Guardrail attached. Unlike the Anthropic or
// OpenAI clients, this adapter works against any Bedrock model that
// supports Converse - swap MODEL_ID and nothing else in this file changes.
//
// AWS credentials are resolved via the standard SDK default chain (env
// vars, shared config/credentials files, IAM role, SSO, etc.) - there is no
// API key to pass in.
//
// Environment variables:
//
//	MODEL_ID            - Bedrock model/inference-profile ID (required), e.g.
//	                      "anthropic.claude-sonnet-4-5-20250929-v1:0" or
//	                      "us.meta.llama3-3-70b-instruct-v1:0"
//	AWS_REGION           - AWS region (falls back to the default config chain)
//	AWS_PROFILE          - Named AWS profile (optional)
//	MAX_OUTPUT_TOKENS     - Max output tokens (default: model default)
//	GUARDRAIL_IDENTIFIER  - Bedrock Guardrail ID/ARN (optional; omit to disable)
//	GUARDRAIL_VERSION     - Guardrail version, or "DRAFT" (required if GUARDRAIL_IDENTIFIER is set)
//	GUARDRAIL_TRACE       - enabled | enabled_full (default: disabled)
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strconv"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/genai"

	genaibedrock "github.com/achetronic/adk-utils-go/genai/bedrock"
)

func main() {
	ctx := context.Background()

	modelID := os.Getenv("MODEL_ID")
	if modelID == "" {
		log.Fatal("MODEL_ID environment variable is required")
	}

	// 1. Create the Bedrock client.
	//    This is all you need to switch from Gemini/Anthropic/OpenAI to any
	//    model Bedrock offers through Converse.
	var guardrail *genaibedrock.GuardrailConfig
	if id := os.Getenv("GUARDRAIL_IDENTIFIER"); id != "" {
		guardrail = &genaibedrock.GuardrailConfig{
			Identifier: id,
			Version:    os.Getenv("GUARDRAIL_VERSION"),
			Trace:      os.Getenv("GUARDRAIL_TRACE"),
		}
	}

	llmModel, err := genaibedrock.New(ctx, genaibedrock.Config{
		ModelID:         modelID,
		Region:          os.Getenv("AWS_REGION"),
		Profile:         os.Getenv("AWS_PROFILE"),
		MaxOutputTokens: getEnvInt("MAX_OUTPUT_TOKENS", 0),
		Guardrail:       guardrail,
	})
	if err != nil {
		log.Fatalf("Failed to create Bedrock client: %v", err)
	}

	// 2. Create an agent using the Bedrock model.
	myAgent, err := llmagent.New(llmagent.Config{
		Name:        "assistant",
		Model:       llmModel,
		Description: "A helpful assistant powered by Amazon Bedrock",
		Instruction: "You are a helpful assistant. Be concise.",
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	// 3. Standard ADK setup: session service + runner.
	sessionService := session.InMemoryService()

	sessResp, err := sessionService.Create(ctx, &session.CreateRequest{
		AppName: "example",
		UserID:  "user1",
	})
	if err != nil {
		log.Fatalf("Failed to create session: %v", err)
	}

	runnr, err := runner.New(runner.Config{
		AppName:        "example",
		Agent:          myAgent,
		SessionService: sessionService,
	})
	if err != nil {
		log.Fatalf("Failed to create runner: %v", err)
	}

	// 4. Send a message and get a response.
	userMsg := genai.NewContentFromText("What is the capital of France?", genai.RoleUser)

	fmt.Println("User: What is the capital of France?")
	fmt.Print("Agent: ")

	for event, err := range runnr.Run(ctx, "user1", sessResp.Session.ID(), userMsg, agent.RunConfig{}) {
		if err != nil {
			log.Fatalf("Error: %v", err)
		}
		if event.Content != nil && len(event.Content.Parts) > 0 {
			fmt.Print(event.Content.Parts[0].Text)
		}
	}
	fmt.Println()

	// 5. If a guardrail is attached, check whether it intervened.
	//
	//    LLMResponse.CustomMetadata (where the adapter stamps the guardrail
	//    trace; see GuardrailConfig in guardrail.go) isn't surfaced through
	//    runner.Run's session.Event today, so this calls the model directly
	//    rather than going through the agent/runner loop above.
	if guardrail != nil {
		checkGuardrail(ctx, llmModel, userMsg)
	}
}

func checkGuardrail(ctx context.Context, llmModel *genaibedrock.Model, userMsg *genai.Content) {
	req := &model.LLMRequest{
		Model:    llmModel.Name(),
		Contents: []*genai.Content{userMsg},
	}
	for resp, err := range llmModel.GenerateContent(ctx, req, false) {
		if err != nil {
			log.Printf("guardrail check call failed: %v", err)
			return
		}
		if genaibedrock.Intervened(resp.CustomMetadata) {
			fmt.Println("(guardrail intervened on this turn)")
		}
	}
}

func getEnvInt(key string, defaultValue int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return defaultValue
}
