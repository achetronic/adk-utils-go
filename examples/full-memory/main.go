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

// Full Memory Example
//
// This example demonstrates combining Redis for session memory (short-term)
// with PostgreSQL for long-term memory in an ADK agent. This provides a complete
// memory solution with both conversation context and persistent recall.
//
// Requirements:
// - PostgreSQL with pgvector extension
// - Redis running locally
// - Ollama running locally with embedding model
//
// Run PostgreSQL:
//   docker run -d --name postgres -e POSTGRES_PASSWORD=postgres -p 5432:5432 pgvector/pgvector:pg16
//
// Run Redis:
//   docker run -d --name redis -p 6379:6379 redis:alpine
//
// Run Ollama:
//   ollama pull qwen3:8b
//   ollama pull nomic-embed-text

package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"
	
	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/genai"

	genaiopenai "adk-utils-go/genai/openai"
	memorypostgres "adk-utils-go/memory/postgres"
	sessionredis "adk-utils-go/session/redis"
	memorytools "adk-utils-go/tools/memory"
)

const (
	appName = "full_memory_example"
	userID  = "demo_user"
)

func main() {
	ctx := context.Background()

	// Configure OpenAI-compatible model (Ollama)
	llmModel := getOpenAIModel()

	///////////////////////////////////////////////////////////////////
	// INITIALIZE SERVICES
	///////////////////////////////////////////////////////////////////

	// Redis for session memory (short-term, conversation context)
	redisSessionService, err := sessionredis.NewRedisSessionService(sessionredis.RedisSessionServiceConfig{
		Addr:     getEnvOrDefault("REDIS_ADDR", "localhost:6379"),
		Password: os.Getenv("REDIS_PASSWORD"),
		DB:       0,
		TTL:      24 * time.Hour,
	})
	if err != nil {
		log.Fatalf("Failed to create Redis session service: %v", err)
	}
	defer redisSessionService.Close()

	// PostgreSQL with pgvector for long-term memory (semantic search)
	pgMemoryService, err := memorypostgres.NewPostgresMemoryService(ctx, memorypostgres.PostgresMemoryServiceConfig{
		ConnString: getEnvOrDefault("POSTGRES_URL", "postgres://postgres:postgres@localhost:5432/postgres?sslmode=disable"),
		EmbeddingModel: memorypostgres.NewOpenAICompatibleEmbedding(memorypostgres.OpenAICompatibleEmbeddingConfig{
			BaseURL: getEnvOrDefault("EMBEDDING_BASE_URL", "http://localhost:11434/v1"),
			Model:   getEnvOrDefault("EMBEDDING_MODEL", "nomic-embed-text"),
		}),
	})
	if err != nil {
		log.Fatalf("Failed to create Postgres memory service: %v", err)
	}
	defer pgMemoryService.Close()

	///////////////////////////////////////////////////////////////////
	// CREATE SESSION
	///////////////////////////////////////////////////////////////////

	sessResp, err := redisSessionService.Create(ctx, &session.CreateRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: fmt.Sprintf("session-%d", time.Now().UnixNano()),
	})
	if err != nil {
		log.Fatalf("Failed to create session: %v", err)
	}

	fmt.Printf("Created session: %s\n", sessResp.Session.ID())

	///////////////////////////////////////////////////////////////////
	// CREATE AGENT WITH FULL MEMORY
	///////////////////////////////////////////////////////////////////

	// Memory toolset gives the agent explicit control over long-term memory
	memoryToolset, err := memorytools.NewToolset(memorytools.ToolsetConfig{
		MemoryService: pgMemoryService,
		AppName:       appName,
	})
	if err != nil {
		log.Fatalf("Failed to create memory toolset: %v", err)
	}

	rootAgent, err := llmagent.New(llmagent.Config{
		Name:        "full_memory_agent",
		Model:       llmModel,
		Description: "An agent with complete memory capabilities - both session and long-term.",
		Instruction: `You are a helpful assistant with two types of memory:

1. **Session Memory** (automatic): The current conversation history is maintained automatically.
   Use this context to maintain coherent conversations.

2. **Long-Term Memory** (tools): You have explicit tools to manage persistent memory:
   - search_memory: Search for information from past conversations and sessions
   - save_to_memory: Save important information for future recall

Guidelines:
- Use session context for immediate conversation coherence
- Use search_memory when users ask about things from previous sessions or days ago
- Use save_to_memory for:
  * User preferences (e.g., "I prefer dark mode")
  * Important facts about the user
  * Explicit requests to remember something
  * Key decisions or commitments made
- Be explicit about what you're saving or retrieving from long-term memory`,
		Toolsets: []tool.Toolset{
			memoryToolset,
		},
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	// Create runner with both services
	runnr, err := runner.New(runner.Config{
		AppName:        appName,
		Agent:          rootAgent,
		SessionService: redisSessionService,
		MemoryService:  pgMemoryService, // Enables automatic memory persistence
	})
	if err != nil {
		log.Fatalf("Failed to create runner: %v", err)
	}

	///////////////////////////////////////////////////////////////////
	// DEMONSTRATE FULL MEMORY CAPABILITIES
	///////////////////////////////////////////////////////////////////

	// Simulate multiple interactions showing both memory types
	interactions := []string{
		// First interaction - establish facts and preferences
		"Hi! I'm Carlos, a data engineer from Spain. I work with Apache Spark and love working with streaming data. Please remember this.",

		// Second interaction - reference within same session (session memory)
		"What's my name and where am I from?",

		// Third interaction - explicit long-term memory recall
		"Search your long-term memory for information about my work preferences.",

		// Fourth interaction - add more information
		"Also note that I prefer using Go for backend services and I'm learning about AI agents.",

		// Fifth interaction - combined query
		"Based on everything you know about me, what technologies should I explore next?",
	}

	for i, userInput := range interactions {
		fmt.Printf("\n=== Interaction %d ===\n", i+1)
		fmt.Printf("User: %s\n", userInput)

		response := runAgent(ctx, runnr, sessResp.Session.ID(), userInput)
		fmt.Printf("Agent: %s\n", response)
	}

	///////////////////////////////////////////////////////////////////
	// SHOW MEMORY STATE
	///////////////////////////////////////////////////////////////////

	fmt.Println("\n=== Final State ===")

	// Session events
	updatedSess, err := redisSessionService.Get(ctx, &session.GetRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessResp.Session.ID(),
	})
	if err != nil {
		log.Printf("Failed to get session: %v", err)
	} else {
		fmt.Printf("Session events: %d\n", updatedSess.Session.Events().Len())
	}

	// Long-term memories
	fmt.Println("\nLong-term memories about user:")
	searchResult, err := pgMemoryService.Search(ctx, &memory.SearchRequest{
		AppName: appName,
		UserID:  userID,
		Query:   "user preferences work",
	})
	if err != nil {
		log.Printf("Failed to search memory: %v", err)
	} else {
		for i, mem := range searchResult.Memories {
			if mem.Content != nil && len(mem.Content.Parts) > 0 {
				fmt.Printf("%d. %s\n", i+1, truncate(mem.Content.Parts[0].Text, 100))
			}
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

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
