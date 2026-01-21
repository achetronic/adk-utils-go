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

// Anthropic Client Example
//
// This example shows how to use the Anthropic client with ADK.
//
// Environment variables:
//   ANTHROPIC_API_KEY - Anthropic API key (required)
//   MODEL_NAME        - Model to use (default: claude-sonnet-4-5-20250929)

package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/genai"

	genaianthropic "adk-utils-go/genai/anthropic"
)

func main() {
	ctx := context.Background()

	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		log.Fatal("ANTHROPIC_API_KEY environment variable is required")
	}

	// 1. Create the Anthropic client
	//    This is all you need to switch from Gemini to Anthropic
	llmModel := genaianthropic.New(genaianthropic.Config{
		APIKey:    os.Getenv("ANTHROPIC_API_KEY"),
		ModelName: getEnvOrDefault("MODEL_NAME", "claude-sonnet-4-5-20250929"),
	})

	// 2. Create an agent using the Anthropic model
	myAgent, err := llmagent.New(llmagent.Config{
		Name:        "assistant",
		Model:       llmModel,
		Description: "A helpful assistant powered by Claude",
		Instruction: "You are a helpful assistant. Be concise.",
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	// 3. Standard ADK setup: session service + runner
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

	// 4. Send a message and get response
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
}

func getEnvOrDefault(key, defaultValue string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return defaultValue
}
