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

// Session Memory Example
//
// This example demonstrates how to use Redis for session-based memory (short-term)
// in an ADK agent. Session state persists across conversation turns within a session.
//
// Requirements:
// - Redis running locally
// - Ollama running locally
//
// Run Redis:
//   docker run -d --name redis -p 6379:6379 redis:alpine
//
// Run Ollama:
//   ollama pull qwen3:8b

package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/genai"

	genaiopenai "adk-utils-go/genai/openai"
	sessionredis "adk-utils-go/session/redis"
)

const (
	appName = "session_memory_example"
	userID  = "demo_user"
)

func main() {
	ctx := context.Background()

	// Configure OpenAI-compatible model (Ollama)
	llmModel := getOpenAIModel()

	// Initialize Redis session service
	redisSessionService, err := sessionredis.NewRedisSessionService(sessionredis.RedisSessionServiceConfig{
		Addr:     getEnvOrDefault("REDIS_ADDR", "localhost:6379"),
		Password: os.Getenv("REDIS_PASSWORD"),
		DB:       0,
		TTL:      24 * time.Hour, // Sessions expire after 24 hours
	})
	if err != nil {
		log.Fatalf("Failed to create Redis session service: %v", err)
	}
	defer redisSessionService.Close()

	// Create a new session
	sessResp, err := redisSessionService.Create(ctx, &session.CreateRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: fmt.Sprintf("session-%d", time.Now().UnixNano()),
		State: map[string]any{
			"conversation_started": time.Now().Format(time.RFC3339),
		},
	})
	if err != nil {
		log.Fatalf("Failed to create session: %v", err)
	}

	fmt.Printf("Created session: %s\n", sessResp.Session.ID())

	// Create a simple agent without long-term memory
	rootAgent, err := llmagent.New(llmagent.Config{
		Name:        "session_agent",
		Model:       llmModel,
		Description: "An agent with session-based memory.",
		Instruction: `You are a helpful assistant. You remember everything discussed in the current conversation.
The conversation history is maintained automatically through the session.
Be conversational and reference previous parts of the conversation when relevant.`,
		Toolsets: []tool.Toolset{},
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	// Create runner with Redis session service
	runnr, err := runner.New(runner.Config{
		AppName:        appName,
		Agent:          rootAgent,
		SessionService: redisSessionService,
	})
	if err != nil {
		log.Fatalf("Failed to create runner: %v", err)
	}

	// Example multi-turn conversation
	conversations := []string{
		"Hello! My name is Alice and I'm working on a Go project.",
		"What's my name?",
		"What programming language am I using?",
	}

	for i, userInput := range conversations {
		fmt.Printf("\n=== Turn %d ===\n", i+1)
		fmt.Printf("User: %s\n", userInput)

		response := runAgent(ctx, runnr, sessResp.Session.ID(), userInput)
		fmt.Printf("Agent: %s\n", response)
	}

	// Demonstrate session state access
	fmt.Println("\n=== Session State ===")
	updatedSess, err := redisSessionService.Get(ctx, &session.GetRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessResp.Session.ID(),
	})
	if err != nil {
		log.Printf("Failed to get session: %v", err)
	} else {
		fmt.Printf("Session ID: %s\n", updatedSess.Session.ID())
		fmt.Printf("Events count: %d\n", updatedSess.Session.Events().Len())
		for k, v := range updatedSess.Session.State().All() {
			fmt.Printf("State[%s] = %v\n", k, v)
		}
	}

	// List all sessions for this user
	fmt.Println("\n=== User Sessions ===")
	listResp, err := redisSessionService.List(ctx, &session.ListRequest{
		AppName: appName,
		UserID:  userID,
	})
	if err != nil {
		log.Printf("Failed to list sessions: %v", err)
	} else {
		for _, s := range listResp.Sessions {
			fmt.Printf("- Session: %s (last updated: %s)\n", s.ID(), s.LastUpdateTime().Format(time.RFC3339))
		}
	}
}

func runAgent(ctx context.Context, runnr *runner.Runner, sessionID string, input string) string {
	userMsg := genai.NewContentFromText(input, genai.RoleUser)

	var responseText string
	for event, err := range runnr.Run(ctx, userID, sessionID, userMsg, agent.RunConfig{}) {
		if err != nil {
			log.Printf("Error: %v", err)
			break
		}
		if event.ErrorCode != "" {
			log.Printf("Event error: %s - %s", event.ErrorCode, event.ErrorMessage)
			break
		}
		if event.Content != nil && len(event.Content.Parts) > 0 {
			responseText += event.Content.Parts[0].Text
		}
	}

	return responseText
}

func getOpenAIModel() *genaiopenai.Model {
	return genaiopenai.New(genaiopenai.Config{
		APIKey:    os.Getenv("OPENAI_API_KEY"),
		BaseURL:   getEnvOrDefault("OPENAI_BASE_URL", "http://localhost:11434/v1"),
		ModelName: getEnvOrDefault("MODEL_NAME", "qwen3:8b"),
	})
}

func getEnvOrDefault(key, defaultValue string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return defaultValue
}
