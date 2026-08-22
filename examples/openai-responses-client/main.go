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
// The agent answers a question that benefits from reasoning, with thinking
// enabled and streaming on: the reasoning arrives as partial events flagged
// Thought, then the answer follows.
//
// Environment variables:
//   OPENAI_API_KEY  - OpenAI API key
//   OPENAI_BASE_URL - API base URL (default: https://api.openai.com/v1)
//   MODEL_NAME      - Model to use (default: gpt-5.5)
//
// For an OpenAI-compatible gateway exposing /v1/responses:
//   OPENAI_BASE_URL=https://hyper.charm.land/v1 MODEL_NAME=glm-5.2 go run main.go

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
		BaseURL:   os.Getenv("OPENAI_BASE_URL"), // empty = OpenAI's API
		ModelName: getEnvOrDefault("MODEL_NAME", "gpt-5.5"),
	})

	// 2. Create an agent using the Responses API model. The thinking config
	//    maps to the Responses reasoning.effort level, and IncludeThoughts
	//    asks for the reasoning summaries on top.
	myAgent, err := llmagent.New(llmagent.Config{
		Name:        "assistant",
		Model:       llmModel,
		Description: "A helpful assistant",
		Instruction: "You are a helpful assistant. Be concise.",
		GenerateContentConfig: &genai.GenerateContentConfig{
			ThinkingConfig: &genai.ThinkingConfig{
				ThinkingLevel:   genai.ThinkingLevelLow,
				IncludeThoughts: true,
			},
		},
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

	// 4. Send a question and stream the response. Reasoning events are
	//    partial and flagged Thought; the answer is the final event.
	userMsg := genai.NewContentFromText("What is 17 * 23?", genai.RoleUser)

	fmt.Println("User: What is 17 * 23?")
	fmt.Print("Agent: ")

	for event, err := range runnr.Run(ctx, "user1", sessResp.Session.ID(), userMsg, agent.RunConfig{StreamingMode: agent.StreamingModeSSE}) {
		if err != nil {
			log.Fatalf("Error: %v", err)
		}
		if event.Content == nil {
			continue
		}
		for _, part := range event.Content.Parts {
			if part.Text == "" {
				continue
			}
			if part.Thought {
				fmt.Printf("[thinking] %s", part.Text)
				continue
			}
			fmt.Print(part.Text)
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
