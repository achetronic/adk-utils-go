# ADK Utils Go

Utilities and implementations for [Google's Agent Development Kit (ADK)](https://google.github.io/adk-docs/) in Go.

This repository provides production-ready implementations for:
- **LLM Clients**: OpenAI and Anthropic clients compatible with ADK
- **Session Management**: Redis-based session persistence
- **Long-term Memory**: PostgreSQL + pgvector for semantic search
- **Memory Tools**: Toolsets for agent-controlled memory operations

## Structure

```
├── genai/            # LLM client implementations
│   ├── openai/       # OpenAI client (works with Ollama, OpenRouter, etc.)
│   └── anthropic/    # Anthropic Claude client
├── session/          # Session service implementations
│   └── redis/        # Redis session service
├── memory/           # Memory service implementations
│   └── postgres/     # PostgreSQL + pgvector memory service
├── tools/            # Tool and toolset implementations
│   └── memory/       # Memory toolset for agents
└── examples/         # Working examples
```

## Installation

```bash
go get github.com/achetronic/adk-utils-go
```

## LLM Clients

### OpenAI Client

Works with OpenAI API and any OpenAI-compatible API (Ollama, OpenRouter, Azure OpenAI, etc.):

```go
import genaiopenai "adk-utils-go/genai/openai"

// Create client
llmModel := genaiopenai.New(genaiopenai.Config{
    APIKey:    os.Getenv("OPENAI_API_KEY"),
    BaseURL:   "http://localhost:11434/v1", // For Ollama
    ModelName: "gpt-4o",                     // Or "qwen3:8b" for Ollama
})

// Use with ADK agent
agent, _ := llmagent.New(llmagent.Config{
    Name:  "assistant",
    Model: llmModel,
    // ...
})
```

### Anthropic Client

Native Anthropic Claude support:

```go
import genaianthropic "adk-utils-go/genai/anthropic"

llmModel := genaianthropic.New(genaianthropic.Config{
    APIKey:    os.Getenv("ANTHROPIC_API_KEY"),
    ModelName: "claude-sonnet-4-5-20250929",
})

agent, _ := llmagent.New(llmagent.Config{
    Name:  "assistant",
    Model: llmModel,
    // ...
})
```

### Supported Features

Both clients support:
- Streaming and non-streaming responses
- System instructions
- Tool/function calling
- Image inputs (base64)
- Temperature, TopP, MaxTokens, StopSequences
- Usage metadata

## Session Service (Redis)

Persistent session storage with Redis:

```go
import sessionredis "adk-utils-go/session/redis"

sessionService, _ := sessionredis.NewRedisSessionService(sessionredis.RedisSessionServiceConfig{
    Addr:     "localhost:6379",
    Password: "",
    DB:       0,
    TTL:      24 * time.Hour,
})
defer sessionService.Close()

// Use with ADK runner
runner, _ := runner.New(runner.Config{
    SessionService: sessionService,
    // ...
})
```

## Memory Service (PostgreSQL + pgvector)

Long-term memory with semantic search:

```go
import memorypostgres "adk-utils-go/memory/postgres"

memoryService, _ := memorypostgres.NewPostgresMemoryService(ctx, memorypostgres.PostgresMemoryServiceConfig{
    ConnString: "postgres://postgres:postgres@localhost:5432/postgres?sslmode=disable",
    EmbeddingModel: memorypostgres.NewOpenAICompatibleEmbedding(memorypostgres.OpenAICompatibleEmbeddingConfig{
        BaseURL: "http://localhost:11434/v1",
        Model:   "nomic-embed-text",
    }),
})
defer memoryService.Close()

// Use with ADK runner
runner, _ := runner.New(runner.Config{
    MemoryService: memoryService,
    // ...
})
```

## Memory Toolset

Give agents explicit control over long-term memory:

```go
import memorytools "adk-utils-go/tools/memory"

memoryToolset, _ := memorytools.NewToolset(memorytools.ToolsetConfig{
    MemoryService: memoryService,
    AppName:       "my_app",
})

agent, _ := llmagent.New(llmagent.Config{
    Toolsets: []tool.Toolset{memoryToolset},
    // ...
})
```

The toolset provides:
- `search_memory`: Semantic search across stored memories
- `save_to_memory`: Save information for future recall

## Examples

Complete working examples in the `examples/` directory:

| Example | Description |
|---------|-------------|
| [openai-client](examples/openai-client) | OpenAI/Ollama client usage |
| [anthropic-client](examples/anthropic-client) | Anthropic Claude client usage |
| [ollama-agents](examples/ollama-agents) | Basic ADK agent with Ollama |
| [session-memory](examples/session-memory) | Session management with Redis |
| [long-term-memory](examples/long-term-memory) | Long-term memory with PostgreSQL + pgvector |
| [full-memory](examples/full-memory) | Combined session + long-term memory |

### Quick Start

```bash
# Start services
docker run -d --name postgres -e POSTGRES_PASSWORD=postgres -p 5432:5432 pgvector/pgvector:pg16
docker run -d --name redis -p 6379:6379 redis:alpine
ollama pull qwen3:8b
ollama pull nomic-embed-text

# Run an example
go run ./examples/openai-client
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | OpenAI API key (not needed for Ollama) |
| `OPENAI_BASE_URL` | - | OpenAI-compatible API endpoint |
| `ANTHROPIC_API_KEY` | - | Anthropic API key |
| `MODEL_NAME` | `gpt-4o` / `claude-sonnet-4-5-20250929` | Model name |
| `EMBEDDING_BASE_URL` | `http://localhost:11434/v1` | Embedding API endpoint |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `POSTGRES_URL` | `postgres://postgres:postgres@localhost:5432/postgres?sslmode=disable` | PostgreSQL connection |
| `REDIS_ADDR` | `localhost:6379` | Redis address |

## Requirements

- Go 1.24+
- [Google ADK](https://google.github.io/adk-docs/) v0.3.0+

## License

Apache 2.0
