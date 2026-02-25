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

// Context Guard Example
//
// This example shows the three ways to configure the ContextGuard plugin:
//
//  1. CrushRegistry — automatic context window detection from Crush's
//     provider.json (refreshes every 6h, zero config).
//  2. WithMaxTokens — manual context window override, useful when using
//     beta features like Anthropic's 1M context window.
//  3. WithSlidingWindow — turn-count-based compaction instead of
//     token-threshold-based.
//
// The example uses the CrushRegistry with the threshold strategy by
// default. Set STRATEGY=sliding_window to try the sliding window, or
// set MAX_TOKENS to override the context window manually.
//
// Environment variables:
//
//	ANTHROPIC_API_KEY - Anthropic API key (required)
//	MODEL_NAME        - Model to use (default: claude-sonnet-4-5-20250929)
//	STRATEGY          - "threshold" (default) or "sliding_window"
//	MAX_TOKENS        - Manual context window override in tokens (threshold only)
//	MAX_TURNS         - Max turns before compaction (sliding_window only, default: 20)
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strconv"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/genai"

	genaianthropic "github.com/achetronic/adk-utils-go/genai/anthropic"
	"github.com/achetronic/adk-utils-go/plugin/contextguard"
)

func main() {
	ctx := context.Background()

	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		log.Fatal("ANTHROPIC_API_KEY environment variable is required")
	}

	modelName := getEnvOrDefault("MODEL_NAME", "claude-sonnet-4-5-20250929")

	// 1. Start the built-in Crush registry — fetches model metadata
	//    (context windows, costs, max tokens) and refreshes every 6 hours.
	registry := contextguard.NewCrushRegistry()
	registry.Start(ctx)
	defer registry.Stop()

	fmt.Printf("Context window for %s: %d tokens\n", modelName, registry.ContextWindow(modelName))

	// 2. Create the LLM client
	llmModel := genaianthropic.New(genaianthropic.Config{
		APIKey:    os.Getenv("ANTHROPIC_API_KEY"),
		ModelName: modelName,
	})

	// 3. Create an agent
	myAgent, err := llmagent.New(llmagent.Config{
		Name:        "assistant",
		Model:       llmModel,
		Description: "A helpful assistant with context guard",
		Instruction: "You are a helpful assistant. Be concise.",
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	// 4. Configure ContextGuard — pick strategy based on env vars
	guard := contextguard.New(registry)

	strategy := getEnvOrDefault("STRATEGY", "threshold")
	switch strategy {
	case "sliding_window":
		maxTurns := getEnvInt("MAX_TURNS", 20)
		guard.Add("assistant", llmModel, contextguard.WithSlidingWindow(maxTurns))
		fmt.Printf("Strategy: sliding_window (max %d turns)\n", maxTurns)

	default:
		if maxTokens := getEnvInt("MAX_TOKENS", 0); maxTokens > 0 {
			guard.Add("assistant", llmModel, contextguard.WithMaxTokens(maxTokens))
			fmt.Printf("Strategy: threshold (manual override: %d tokens)\n", maxTokens)
		} else {
			guard.Add("assistant", llmModel)
			fmt.Printf("Strategy: threshold (auto-detect from registry)\n")
		}
	}

	// 5. Standard ADK setup
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
		PluginConfig:   guard.PluginConfig(),
	})
	if err != nil {
		log.Fatalf("Failed to create runner: %v", err)
	}

	// 6. Send a message
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

func getEnvInt(key string, defaultValue int) int {
	v := os.Getenv(key)
	if v == "" {
		return defaultValue
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return defaultValue
	}
	return n
}
