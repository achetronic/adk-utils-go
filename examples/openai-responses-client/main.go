// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

// OpenAI Responses API Client Example
//
// This example shows how to use the OpenAI Responses API client with ADK.
// The Responses API is OpenAI's recommended interface for new applications,
// with native reasoning, built-in tools, and structured output. This adapter
// runs it statelessly: ADK owns the conversation state and replays the full
// history on each call.
//
// Environment variables:
//   OPENAI_API_KEY - OpenAI API key
//   MODEL_NAME     - Model to use (default: gpt-5.5)

package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/genai"

	genairesponses "github.com/achetronic/adk-utils-go/genai/openai/responses"
)

func main() {
	ctx := context.Background()

	// 1. Create the Responses API client
	llmModel := genairesponses.New(genairesponses.Config{
		APIKey:    os.Getenv("OPENAI_API_KEY"),
		ModelName: getEnvOrDefault("MODEL_NAME", "gpt-5.5"),
	})

	// 2. Create an agent using the Responses API model
	myAgent, err := llmagent.New(llmagent.Config{
		Name:        "assistant",
		Model:       llmModel,
		Description: "A helpful assistant",
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
