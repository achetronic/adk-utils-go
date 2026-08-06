<img src="docs/images/header.svg" alt="adk-utils-go" width="600">

# ADK Utils Go

Utilities and implementations for [Google's Agent Development Kit (ADK)](https://google.github.io/adk-docs/) in Go.

This repository provides production-ready implementations for:

- **LLM Clients**: OpenAI and Anthropic clients compatible with ADK
- **Session Management**: Redis-based session persistence
- **Long-term Memory**: PostgreSQL + pgvector for semantic search
- **Memory Tools**: Toolsets for agent-controlled memory operations
- **Artifact Storage**: Filesystem-based artifact persistence with versioning
- **Context Guard**: Automatic context window management with LLM-powered summarization
- **Langfuse**: Observability plugin: traces every LLM call to [Langfuse](https://langfuse.com) with full prompt/response payloads and token usage

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
├── artifact/         # Artifact service implementations
│   └── filesystem/   # Filesystem artifact service (versioned, user-scoped)
├── plugin/           # ADK plugin implementations
│   ├── contextguard/ # Context window management plugin + CrushRegistry
│   └── langfuse/     # Langfuse observability plugin (OTLP/HTTP traces)
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
import genaiopenai "github.com/achetronic/adk-utils-go/genai/openai"

llmModel := genaiopenai.New(genaiopenai.Config{
    APIKey:    os.Getenv("OPENAI_API_KEY"),
    BaseURL:   "http://localhost:11434/v1", // For Ollama
    ModelName: "gpt-4o",                     // Or "qwen3:8b" for Ollama
})

agent, _ := llmagent.New(llmagent.Config{
    Name:  "assistant",
    Model: llmModel,
})
```

#### Reasoning

Many OpenAI-compatible reasoning models (DeepSeek, Kimi K2 thinking, Qwen3-Thinking, ...) return their chain of thought in a non-standard field. The client reads it into a content part flagged `Thought: true`, and sends those parts back as reasoning rather than merging them into `content`. That round-trip is mandatory for some backends: DeepSeek in thinking mode and Kimi K2 thinking reject a conversation whose assistant messages have lost their reasoning.

Other backends do the opposite and reject any field they do not know, so the shape is configurable:

```go
llmModel := genaiopenai.New(genaiopenai.Config{
    APIKey:    os.Getenv("DEEPSEEK_API_KEY"),
    BaseURL:   "https://api.deepseek.com/v1",
    ModelName: "deepseek-chat",

    // Default: reasoning travels as its own field on the assistant message.
    ReasoningEgress: genaiopenai.ReasoningEgressNative,

    // Names the field. Used when reading responses too.
    ReasoningField: "reasoning_content",
})

// For a backend that rejects unknown message fields: the reasoning is kept,
// wrapped in a <think> block at the start of the assistant content.
llmModel = genaiopenai.New(genaiopenai.Config{
    BaseURL:         "http://localhost:8000/v1",
    ModelName:       "some-strict-backend",
    ReasoningEgress: genaiopenai.ReasoningEgressThinkTags,
})

// For a backend that ignores reasoning history anyway, or to save the tokens:
// send none of it back.
llmModel = genaiopenai.New(genaiopenai.Config{
    BaseURL:         "http://localhost:11434/v1",
    ModelName:       "qwen3:8b",
    ReasoningEgress: genaiopenai.ReasoningEgressOmit,
})
```

Every field is optional. Reasoning is only ever attached to assistant messages: a thought part arriving under another role (ADK rewrites events from other agents as user content) is dropped, since no provider accepts it there.

#### Reasoning through OpenRouter

OpenRouter needs two settings. It returns its plain-text reasoning under `reasoning`, not `reasoning_content` (which it accepts only as an input alias), so point `ReasoningField` at `reasoning` or nothing is read back. It also returns a structured `reasoning_details` array for models whose reasoning is summarised or encrypted, which is the only way to preserve reasoning for models that do not hand back readable text. Sending that array back is opt-in, since backends that do not know the field reject it:

```go
llmModel := genaiopenai.New(genaiopenai.Config{
    APIKey:    os.Getenv("OPENROUTER_API_KEY"),
    BaseURL:   "https://openrouter.ai/api/v1",
    ModelName: "anthropic/claude-sonnet-4.6",

    ReasoningField:           "reasoning",
    SupportsReasoningDetails: true,
})
```

Reasoning blocks are preserved on the content part they came from (in `Part.PartMetadata`, under `genaiopenai.ReasoningDetailMetadataKey`) and replayed byte for byte, in order, because OpenRouter requires the sequence to match what the model produced. This matters most for tool calling: a model that pauses mid-reasoning to call a tool resumes from those blocks once the result comes back. Blocks are captured whether or not you enable sending them, so turning the option on later still replays what earlier turns recorded.

#### Provider-specific request fields

Providers add top-level request fields that the OpenAI schema does not define. `ExtraBody` is merged into the root of every request, streaming or not, which is how you turn reasoning on in the first place for OpenRouter and how you reach its routing controls:

```go
llmModel := genaiopenai.New(genaiopenai.Config{
    APIKey:    os.Getenv("OPENROUTER_API_KEY"),
    BaseURL:   "https://openrouter.ai/api/v1",
    ModelName: "anthropic/claude-sonnet-4.6",

    ExtraBody: map[string]any{
        "reasoning": map[string]any{"effort": "high"},
        "provider":  map[string]any{"order": []any{"anthropic"}},
    },
})
```

Values just need to be JSON-serialisable. The map is copied when the client is built, so changing your copy afterwards does not affect requests. A key that collides with something the client sets (`model`, `messages`, ...) replaces it on the wire, so treat it as an extension point rather than a way to rewrite the request.

### Anthropic Client

Native Anthropic Claude support:

```go
import genaianthropic "github.com/achetronic/adk-utils-go/genai/anthropic"

llmModel := genaianthropic.New(genaianthropic.Config{
    APIKey:    os.Getenv("ANTHROPIC_API_KEY"),
    ModelName: "claude-sonnet-4-5-20250929",
})

agent, _ := llmagent.New(llmagent.Config{
    Name:  "assistant",
    Model: llmModel,
})
```

#### Extended Thinking (reasoning)

Claude can produce an internal reasoning chain before its final answer. There are **two reasoning APIs**, and Anthropic rejects the wrong one with an HTTP 400, so you pick per model with `ThinkingMode`:

- **Classic (`"enabled"`)**: budget-based. Reasoning tokens count as output tokens, so `ThinkingBudgetTokens` must be `>= 1024` and strictly less than `MaxOutputTokens`. Accepted by Claude 3.7, Sonnet 4 and Opus 4.
- **Adaptive (`"adaptive"`)**: effort-based. Set `ThinkingEffort` to `"low"`, `"medium"` or `"high"` (some models also accept `"xhigh"` / `"max"`). Required by Opus 4.5 and newer, which reject the classic form.

`ThinkingMode` is optional. Leave it empty and the client deduces the API from the field you set: `ThinkingEffort` set means adaptive; otherwise a `ThinkingBudgetTokens > 0` means enabled.

```go
// Classic budget-based API (Claude 3.7 / Sonnet 4 / Opus 4)
llmModel := genaianthropic.New(genaianthropic.Config{
    APIKey:               os.Getenv("ANTHROPIC_API_KEY"),
    ModelName:            "claude-sonnet-4-20250514",
    ThinkingMode:         genaianthropic.ThinkingModeEnabled, // optional; deduced from the budget
    MaxOutputTokens:      16000,
    ThinkingBudgetTokens: 10000, // must be >= 1024 and < MaxOutputTokens
})

// Effort-based adaptive API (Opus 4.5+)
llmModel = genaianthropic.New(genaianthropic.Config{
    APIKey:         os.Getenv("ANTHROPIC_API_KEY"),
    ModelName:      "claude-opus-4-8",
    ThinkingMode:   genaianthropic.ThinkingModeAdaptive, // optional; deduced from the effort
    ThinkingEffort: "high",
})
```

When streaming, the reasoning is emitted as partial content parts flagged `Thought: true`, and the thinking block (with its signature) is preserved across turns so tool-use loops keep working.

### Custom HTTP Headers

Both clients support custom HTTP headers via `HTTPOptions`, useful for beta features, auth proxies, or provider-specific flags:

```go
import "net/http"

llmModel := genaianthropic.New(genaianthropic.Config{
    APIKey:    os.Getenv("ANTHROPIC_API_KEY"),
    ModelName: "claude-sonnet-4-6-20250929",
    HTTPOptions: genaianthropic.HTTPOptions{
        Headers: http.Header{
            "anthropic-beta": []string{"context-1m-2025-08-07"},
        },
    },
})
```

### Supported Features

Both clients support:

- Streaming and non-streaming responses
- System instructions
- Tool/function calling
- Image inputs: inline bytes (`InlineData`, sent as base64) or remote URLs (`FileData`, passed through to the provider's image-URL field; nothing is downloaded or re-encoded)
- Temperature, TopP, MaxOutputTokens, StopSequences
- Extended thinking: classic budget API (`ThinkingBudgetTokens`) and adaptive effort API (`ThinkingEffort` + `ThinkingMode`)
- Reasoning round-trip: reasoning is read into `Thought` parts and sent back as a separate field (OpenAI) or as thinking blocks (Anthropic), never merged into the reply text
- Usage metadata
- Custom HTTP headers (multi-value)

Remote image URLs (`genai.Part.FileData`) work for image MIME types only; audio and documents still need uploaded bytes via `InlineData`. The URI must be `http(s)`. Plain `http` is allowed because both clients also serve API-compatible gateways (Ollama, vLLM, LiteLLM, ...) that commonly fetch from local http endpoints; which URLs a hosted provider actually fetches is decided on its side. Note that `gs://` URIs are rejected even though `genai.FileData` documents them: neither provider can read from Google Cloud Storage, so for GCS-hosted files fetch the bytes and use `InlineData`. Invalid schemes and non-image MIME types fail with a clear error instead of being silently dropped.

## Session Service (Redis)

Persistent session storage with Redis:

```go
import sessionredis "github.com/achetronic/adk-utils-go/session/redis"

sessionService, _ := sessionredis.NewRedisSessionService(sessionredis.RedisSessionServiceConfig{
    Addr:     "localhost:6379",
    Password: "",
    DB:       0,
    TTL:      24 * time.Hour,
})
defer sessionService.Close()

runner, _ := runner.New(runner.Config{
    SessionService: sessionService,
})
```

## Memory Service (PostgreSQL + pgvector)

Long-term memory with semantic search:

```go
import memorypostgres "github.com/achetronic/adk-utils-go/memory/postgres"

memoryService, _ := memorypostgres.NewPostgresMemoryService(ctx, memorypostgres.PostgresMemoryServiceConfig{
    ConnString: "postgres://postgres:postgres@localhost:5432/postgres?sslmode=disable",
    EmbeddingModel: memorypostgres.NewOpenAICompatibleEmbedding(memorypostgres.OpenAICompatibleEmbeddingConfig{
        BaseURL: "http://localhost:11434/v1",
        Model:   "nomic-embed-text",
    }),
})
defer memoryService.Close()

runner, _ := runner.New(runner.Config{
    MemoryService: memoryService,
})
```

## Memory Toolset

Give agents explicit control over long-term memory:

```go
import memorytools "github.com/achetronic/adk-utils-go/tools/memory"

memoryToolset, _ := memorytools.NewToolset(memorytools.ToolsetConfig{
    MemoryService: memoryService,
    AppName:       "my_app",
})

agent, _ := llmagent.New(llmagent.Config{
    Toolsets: []tool.Toolset{memoryToolset},
})
```

The toolset provides:

- `search_memory`: Semantic search across stored memories
- `save_to_memory`: Save information for future recall

## Artifact Service (Filesystem)

Versioned artifact storage backed by the local filesystem. Agents can save, load, list, and delete files (code, documents, data) that are delivered to the user as downloadable content.

```go
import artifactfs "github.com/achetronic/adk-utils-go/artifact/filesystem"

artifactService, _ := artifactfs.NewFilesystemService(artifactfs.FilesystemServiceConfig{
    BasePath: "data/artifacts",
})

// Use with ADK launcher
launcherCfg := &launcher.Config{
    SessionService:  sessionService,
    AgentLoader:     agentLoader,
    ArtifactService: artifactService,
}
```

Artifacts are stored at `{BasePath}/{appName}/{userID}/{sessionID}/{fileName}/{version}.json`. Filenames prefixed with `user:` are scoped to the user across all sessions, making them accessible from any conversation.

## Langfuse Plugin

Traces every agent invocation and LLM call to [Langfuse](https://langfuse.com) via OTLP/HTTP. Enriches `generate_content` spans with full request/response payloads and token usage so Langfuse can display costs, latency, and prompt/completion content.

Supports all ADK agent topologies: single agents, sequential delegation, SequentialAgent, LoopAgent, and ParallelAgent.

### Setup

```go
import "github.com/achetronic/adk-utils-go/plugin/langfuse"

pluginCfg, shutdown, err := langfuse.Setup(&langfuse.Config{
    PublicKey:   os.Getenv("LANGFUSE_PUBLIC_KEY"),
    SecretKey:   os.Getenv("LANGFUSE_SECRET_KEY"),
    Host:        "https://cloud.langfuse.com", // or self-hosted URL
    Environment: "production",
    ServiceName: "my-agent",
})
if err != nil { log.Fatal(err) }
defer shutdown(context.Background())

runnr, _ := runner.New(runner.Config{
    Agent:        myAgent,
    PluginConfig: pluginCfg,
})
```

### Custom TracerProvider Options

`Config.TracerProviderOptions` passes extra options to the trace provider
that `Setup` builds: a custom ID generator, a sampler, span limits. The
exporter and resource that `Setup` wires itself stay in place. The field is
programmatic only; it cannot be set from YAML/JSON.

The main use for this is pinning trace IDs to your own run IDs, so a paused
agent that resumes in another HTTP request stays in one Langfuse trace.
Without it, every request starts a fresh random trace ID and one logical run
gets split across several traces. Langfuse recommends the same external-ID
correlation pattern in its docs (`create_trace_id(seed=...)` in the Python
SDK):

```go
pluginCfg, shutdown, err := langfuse.Setup(&langfuse.Config{
    PublicKey: os.Getenv("LANGFUSE_PUBLIC_KEY"),
    SecretKey: os.Getenv("LANGFUSE_SECRET_KEY"),
    TracerProviderOptions: []sdktrace.TracerProviderOption{
        // myIDGenerator derives the trace ID from the run ID in ctx
        // (random when absent), so every execution of the same logical
        // run lands in the same Langfuse trace. Keep span IDs random:
        // resumed executions would otherwise collide inside the shared
        // trace.
        sdktrace.WithIDGenerator(myIDGenerator),
        // Optional: cap ingestion volume on high-traffic services.
        // sdktrace.WithSampler(sdktrace.ParentBased(sdktrace.TraceIDRatioBased(0.1))),
    },
})
```

The options run after the resource and span processor `Setup` wires, so an
option with replace semantics (`sdktrace.WithResource`, for example) wins
over what `Setup` set. Note that on this path the ADK receives a finished
provider and skips the extra OTLP trace exporters it would otherwise wire
from `OTEL_EXPORTER_OTLP_ENDPOINT` / `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`.

When the field is empty, `Setup` behaves exactly as before.

### Combining with ContextGuard

```go
langfuseCfg, shutdown, _ := langfuse.Setup(langfuseCfg)
guardCfg := guard.PluginConfig()

combined := runner.PluginConfig{
    Plugins: append(langfuseCfg.Plugins, guardCfg.Plugins...),
}
```

### Per-Request Context

Inject per-request attributes via context (typically in HTTP middleware):

```go
ctx = langfuse.WithUserID(ctx, "user-123")
ctx = langfuse.WithTags(ctx, []string{"beta", "internal"})
ctx = langfuse.WithTraceName(ctx, "customer-support")
ctx = langfuse.WithTraceMetadata(ctx, map[string]string{"tenant": "acme"})
```

### Config

| Field | Required | Default | Description |
|---|---|---|---|
| `PublicKey` | Yes |: | Langfuse project public key (Basic Auth user) |
| `SecretKey` | Yes |: | Langfuse project secret key (Basic Auth pass) |
| `Host` | No | `https://cloud.langfuse.com` | Langfuse server URL |
| `Environment` | No |: | Deployment environment tag |
| `Release` | No |: | Application version tag |
| `ServiceName` | No | `langfuse-adk` | OTel `service.name` resource attribute |
| `Insecure` | No | `false` | Disable TLS for the OTLP/HTTP exporter (for self-hosted plain-HTTP instances) |

Use `cfg.IsEnabled()` to conditionally skip setup when credentials are absent.

## Context Guard Plugin

Automatic context window management that prevents conversations from exceeding the LLM's token limit. It works as an ADK `BeforeModelCallback` plugin: before every LLM call, it checks whether the conversation is approaching the limit and summarizes older messages to make room.

### Strategies

| Strategy | Trigger | Best for |
|----------|---------|----------|
| `threshold` | Token count approaches context window limit | Maximizing context usage, models with known limits |
| `sliding_window` | Turn count exceeds a configured maximum | Predictable compaction, long-running conversations |

### Setup

The plugin requires a `ModelRegistry` to look up context window sizes. The built-in `CrushRegistry` ships catwalk's embedded model database, compiled into the binary: no network calls and no lifecycle to manage.

```go
import "github.com/achetronic/adk-utils-go/plugin/contextguard"

// 1. Create the registry (built-in, embedded model database)
registry := contextguard.NewCrushRegistry()

// 2. Create the guard and add agents
guard := contextguard.New(registry)
guard.Add("assistant", llmModel)

// 3. Pass to ADK runner
runnr, _ := runner.New(runner.Config{
    Agent:        myAgent,
    PluginConfig: guard.PluginConfig(),
})
```

Per-agent options are available via functional options:

```go
guard := contextguard.New(registry)

// Threshold strategy (default): summarizes when tokens approach the limit
guard.Add("assistant", llmModel)

// Sliding window: summarizes after N turns regardless of token count
guard.Add("researcher", llmResearcher, contextguard.WithSlidingWindow(30))

// Manual context window override: bypasses the registry for this agent
guard.Add("writer", llmWriter, contextguard.WithMaxTokens(1_000_000))

// Custom compaction retry limit (default: 3): applies to both strategies
guard.Add("analyst", llmAnalyst, contextguard.WithMaxCompactionAttempts(5))
```

Multi-agent setup is the same API: just call `Add` multiple times:

```go
guard := contextguard.New(registry)
for _, agentDef := range agents {
    guard.Add(agentDef.ID, llmMap[agentDef.ID], optsFromDef(agentDef)...)
}
```

### Custom Model Registry

You can implement your own `ModelRegistry` instead of using `CrushRegistry`:

```go
type myRegistry struct{}

func (r *myRegistry) ContextWindow(modelID string) int {
    windows := map[string]int{
        "claude-sonnet-4-5-20250929": 200000,
        "gpt-4o":                     128000,
    }
    if w, ok := windows[modelID]; ok {
        return w
    }
    return 128000
}

func (r *myRegistry) DefaultMaxTokens(modelID string) int {
    return 4096
}

guard := contextguard.New(&myRegistry{})
guard.Add("assistant", llmModel)
```

### How it works

1. Before every LLM call, the plugin checks the configured strategy for the agent
2. **Threshold**: estimates total tokens and triggers summarization when remaining capacity drops below a safety buffer (fixed 20k for windows >200k, 20% for smaller ones)
3. **Sliding window**: counts Content entries since the last compaction and triggers when the limit is exceeded
4. When triggered, the conversation is split into "old" (summarized by the agent's own LLM) and "recent" (kept verbatim)
5. Both strategies retry compaction up to 3 times (`maxCompactionAttempts`) if the resulting summary still exceeds the threshold. After exhausting all attempts the request is sent as-is (best-effort)
6. The summary is persisted in session state and injected on subsequent requests until the next compaction
7. Tool call chains (`tool_use` + `tool_result`) are never split mid-chain to prevent provider errors

## Examples

Complete working examples in the `examples/` directory:

| Example                                       | Description                                 |
| --------------------------------------------- | ------------------------------------------- |
| [openai-client](examples/openai-client)       | OpenAI/Ollama client usage                                |
| [anthropic-client](examples/anthropic-client) | Anthropic Claude client usage                             |
| [session-memory](examples/session-memory)     | Session management with Redis                             |
| [long-term-memory](examples/long-term-memory) | Long-term memory with PostgreSQL + pgvector               |
| [full-memory](examples/full-memory)           | Combined session + long-term memory                       |
| [context-guard](examples/context-guard)       | ContextGuard plugin with CrushRegistry, manual thresholds, and sliding window |

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

| Variable             | Default                                                                | Description                            |
| -------------------- | ---------------------------------------------------------------------- | -------------------------------------- |
| `OPENAI_API_KEY`     | -                                                                      | OpenAI API key (not needed for Ollama) |
| `OPENAI_BASE_URL`    | -                                                                      | OpenAI-compatible API endpoint         |
| `ANTHROPIC_API_KEY`  | -                                                                      | Anthropic API key                      |
| `MODEL_NAME`         | `gpt-4o` / `claude-sonnet-4-5-20250929`                                | Model name                             |
| `EMBEDDING_BASE_URL` | `http://localhost:11434/v1`                                            | Embedding API endpoint                 |
| `EMBEDDING_MODEL`    | `nomic-embed-text`                                                     | Embedding model                        |
| `POSTGRES_URL`       | `postgres://postgres:postgres@localhost:5432/postgres?sslmode=disable` | PostgreSQL connection                  |
| `REDIS_ADDR`         | `localhost:6379`                                                       | Redis address                          |

## Requirements

- Go 1.24+
- [Google ADK](https://google.github.io/adk-docs/) v0.5.0+

## License

Apache 2.0
