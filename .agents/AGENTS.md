# AGENTS.md

Agent guidelines for working in the `adk-utils-go` repository.

## Project Overview

A Go library providing utilities for Google's Agent Development Kit (ADK). This library extends ADK with additional backend implementations for topics like session management or memory services.

**Module**: `github.com/achetronic/adk-utils-go` (see `go.mod`)  
**Go Version**: 1.24.9+ (toolchain 1.25.5)  
**ADK Version**: v0.4.0

### Key Dependencies

| Package | Purpose |
|---------|---------|
| `google.golang.org/adk` | Google ADK core framework |
| `google.golang.org/genai` | Google GenAI types |
| `github.com/redis/go-redis/v9` | Redis client for session storage |
| `github.com/lib/pq` | PostgreSQL driver for memory storage |

---

## Commands

### Build & Test

```bash
# Run all tests
go test ./...

# Run tests with verbose output
go test -v ./...

# Run tests for a specific package
go test -v ./memory/postgres/...
go test -v ./session/redis/...

# Run with race detection
go test -race ./...
```

### Module Management

```bash
# Download dependencies
go mod download

# Tidy dependencies
go mod tidy

# Verify dependencies
go mod verify
```

---

## Code Organization

```
adk-utils-go/
├── genai/
│   ├── openai/
│   │   └── openai.go            # OpenAI/Ollama-compatible LLM adapter
│   └── anthropic/
│       └── anthropic.go         # Anthropic Claude LLM adapter
├── session/
│   └── redis/
│       ├── session.go           # Redis-backed session.Service implementation
│       └── session_test.go      # Session tests (requires Redis)
├── memory/
│   ├── memorytypes/
│   │   └── types.go             # Shared types and interfaces (EntryWithID, ExtendedMemoryService)
│   └── postgres/
│       ├── memory.go            # PostgreSQL-backed memory.Service implementation
│       ├── memory_test.go       # Memory service tests (requires PostgreSQL)
│       ├── embedding.go         # OpenAI-compatible embedding model
│       └── embedding_test.go    # Embedding tests (uses httptest mocks)
├── artifact/
│   └── filesystem/
│       ├── artifact.go          # Filesystem-backed artifact.Service implementation
│       └── artifact_test.go     # Artifact tests
├── tools/
│   └── memory/
│       └── toolset.go           # Memory toolset for agent tools
├── plugin/
│   └── contextguard/
│       ├── contextguard.go           # Public API: New(), Add(), PluginConfig(), BeforeModel/AfterModel callbacks
│       ├── contextguard_test.go      # 93 tests covering all functions + timing gap proofs
│       ├── compaction_simulation_test.go # Realistic simulation: kube-agent, mixed-debug, pure-tool-storm, timing gap
│       ├── model_registry.go         # ModelRegistry interface (ContextWindow, DefaultMaxTokens)
│       ├── model_registry_crush.go   # CrushRegistry: fetches from Crush provider.json, 6h refresh
│       ├── compaction_utils.go       # Internal helpers: state, summarization, tokens, calibration, splitting, continuation, todos
│       ├── compaction_threshold.go   # Token-threshold strategy (Crush-style full summary)
│       └── compaction_sliding_window.go # Sliding-window strategy (turn-count, with recent tail + retry)
├── examples/
│   ├── openai-client/main.go
│   ├── anthropic-client/main.go
│   ├── session-memory/main.go
│   ├── long-term-memory/main.go
│   ├── full-memory/main.go
│   └── context-guard/main.go    # All 3 modes: CrushRegistry, WithMaxTokens, WithSlidingWindow
├── go.mod
└── go.sum
```

### Package Purposes

| Package | Description |
|---------|-------------|
| `genai/openai` | OpenAI/Ollama-compatible `model.LLM` adapter |
| `genai/anthropic` | Anthropic Claude `model.LLM` adapter |
| `session/redis` | Redis-backed implementation of `session.Service` |
| `memory/memorytypes` | Shared types (`EntryWithID`) and interfaces (`MemoryService`, `ExtendedMemoryService`) |
| `memory/postgres` | PostgreSQL+pgvector implementation of `memory.Service` and `ExtendedMemoryService` |
| `artifact/filesystem` | Filesystem-backed `artifact.Service` implementation |
| `tools/memory` | ADK toolset providing `search_memory`, `save_to_memory`, `update_memory`, and `delete_memory` tools |
| `plugin/contextguard` | ADK plugin for context window management (threshold + sliding window strategies) |

---

## Patterns & Conventions

### Interface Implementation Pattern

All service implementations follow this pattern:

```go
// Config struct for constructor
type ServiceConfig struct {
    // Required and optional fields
}

// Constructor returns concrete type
func NewService(ctx context.Context, cfg ServiceConfig) (*Service, error) {
    // Validate config, establish connections, init schema
}

// Interface compliance check at end of file
var _ some.Interface = (*Service)(nil)
```

### Error Handling

- Use `fmt.Errorf("context: %w", err)` for wrapping errors
- Return early on errors
- Continue processing loops (don't fail entire operation for single item failures)

### Redis Key Naming

```
session:{appName}:{userID}:{sessionID}   # Session data (session-scoped state only)
sessions:{appName}:{userID}              # Session index (SET)
events:{appName}:{userID}:{sessionID}    # Event list (LIST)
appstate:{appName}                       # App-wide state (HASH, shared across all users/sessions)
userstate:{appName}:{userID}             # Per-user state (HASH, shared across all sessions for that user)
```

### PostgreSQL Schema

The `memory_entries` table uses:
- Composite unique constraint: `(app_name, user_id, session_id, event_id)`
- Full-text search via `tsvector` on `content_text`
- Vector similarity search via `pgvector` extension on `embedding` column (optional)

### Embedding Model Interface

```go
type EmbeddingModel interface {
    Embed(ctx context.Context, text string) ([]float32, error)
    Dimension() int
}
```

The `OpenAICompatibleEmbedding` implementation works with any OpenAI-compatible API (OpenAI, Ollama, vLLM, LocalAI, etc.).

---

## Testing

### Unit Tests (No External Dependencies)

- `embedding_test.go` - Uses `httptest` mock servers
- `contextguard_test.go` - 93 unit tests with mock LLM, state, and registry (no network)

### Integration Tests (Require External Services)

- `memory_test.go` - Requires PostgreSQL at `localhost:5432`
  - Default connection: `postgres://postgres:postgres@localhost:5432/postgres?sslmode=disable`
  - Tests clean up after themselves (delete `test_%` prefixed data)

### Test Data Patterns

- Test app names use `test_` prefix for isolation
- Mock session helper: `createTestSession(id, appName, userID, messages)`
- Mock event helper in tests implements `session.Events` interface

---

## Important Gotchas

### Redis Session Service

1. **TTL Management**: Default TTL is 24 hours. TTL is refreshed on session updates.
2. **State Persistence**: State changes via `State().Set()` immediately persist to Redis.
3. **Event Loading**: Events are loaded fresh from Redis on each `Events().All()` call.
4. **Session ID Generation**: If not provided, uses `time.Now().UnixNano()`.
5. **State Tiers**: State keys are routed to separate Redis stores based on prefix, matching the canonical ADK behaviour:
   - `app:` keys → `appstate:{appName}` HASH (shared across all users and sessions)
   - `user:` keys → `userstate:{appName}:{userID}` HASH (shared across all sessions for that user)
   - `temp:` keys → discarded (never persisted)
   - Unprefixed keys → stored in the session JSON (per-session)
6. **State Tier Functions**: `extractStateDeltas` and `mergeStates` mirror `google.golang.org/adk/internal/sessionutils` (which is not importable from outside the ADK module).

### PostgreSQL Memory Service

1. **pgvector Extension**: Required for semantic search. Schema auto-creates the extension if embedding model is configured.
2. **Dimension Auto-Detection**: If `EmbeddingModel.Dimension()` returns 0, the service probes the model on init.
3. **Search Fallback**: Vector search → full-text search → recent entries.
4. **Upsert Behavior**: `AddSession` uses `ON CONFLICT ... DO UPDATE`.

### Memory Toolset

1. **Tool Names**: `search_memory`, `save_to_memory`, `update_memory`, and `delete_memory`
2. **Extended Tools**: `update_memory` and `delete_memory` are only available when the `MemoryService` also implements `memorytypes.ExtendedMemoryService` (e.g., `PostgresMemoryService`).
3. **ID-Aware Search**: When extended service is available, `search_memory` returns entry IDs that can be used with `update_memory` and `delete_memory`.
4. **User Scoping**: Tools automatically use `ctx.UserID()` for isolation.
5. **DisableExtendedTools**: `ToolsetConfig.DisableExtendedTools` allows disabling `update_memory` and `delete_memory` even when the backend supports them.
6. **Single Entry Session**: `save_to_memory` creates a minimal session wrapper around the content.

### Go 1.24 Iterator Pattern

This codebase uses Go 1.24's `iter.Seq` and `iter.Seq2` for iteration:

```go
// State iteration
func (s *State) All() iter.Seq2[string, any]

// Events iteration  
func (e *Events) All() iter.Seq[*session.Event]
```

---

## Adding New Components

### New Session Backend

1. Create package under `session/{backend}/`
2. Implement `session.Service` interface
3. Implement `session.Session`, `session.State`, `session.Events` interfaces
4. Add interface compliance check: `var _ session.Service = (*YourService)(nil)`

### Shared Types (`memory/memorytypes`)

To avoid import cycles between `memory/postgres` and `tools/memory`, shared types and interfaces live in `memory/memorytypes`:

- `EntryWithID` — memory entry with database row ID
- `MemoryService` — base interface (mirrors ADK's `memory.Service`)
- `ExtendedMemoryService` — adds `SearchWithID`, `UpdateMemory`, `DeleteMemory`

Both `memory/postgres` and `tools/memory` import this package; neither imports the other.

### New Memory Backend

1. Create package under `memory/{backend}/`
2. Implement `memory.Service` interface (`AddSession`, `Search`)
3. Optionally implement `memorytypes.ExtendedMemoryService` (`SearchWithID`, `UpdateMemory`, `DeleteMemory`) to enable update/delete tools
4. Consider supporting the `EmbeddingModel` interface for semantic search

### New Toolset

1. Create package under `tools/{purpose}/`
2. Implement `tool.Toolset` interface
3. Use `functiontool.New()` to create tools from functions
4. Define typed args/result structs with JSON tags

---

## ContextGuard Plugin

The `plugin/contextguard` package provides an ADK plugin that prevents conversations from exceeding the LLM's context window. Uses Crush-style full-summary compaction with calibrated real token counts.

### API

```go
guard := contextguard.New(registry)  // ModelRegistry or CrushRegistry
guard.Add("assistant", llm)           // threshold (auto-detect from registry)
guard.Add("agent2", llm, contextguard.WithMaxTokens(500_000))  // threshold (manual)
guard.Add("agent3", llm, contextguard.WithSlidingWindow(30))   // sliding window
cfg := guard.PluginConfig()           // runner.PluginConfig with BeforeModelCallback + AfterModelCallback
```

### Key Design

- **Builder pattern**: `New()` + `Add()` + `PluginConfig()` — single-agent and multi-agent look identical
- **Functional options**: `AgentOption` keeps common case zero-config, overrides via `WithMaxTokens`/`WithSlidingWindow`
- **Per-agent strategies**: each agent gets its own strategy and summarizes with its own LLM
- **Two callbacks**: `BeforeModelCallback` checks tokens and compacts; `AfterModelCallback` persists real token counts from the provider
- **Calibrated heuristic**: bridges the timing gap between callbacks using a correction factor derived from real vs heuristic tokens of the previous call
- **Full summary**: threshold strategy always summarizes the entire conversation (no recent tail). Post-compaction result is `[summary] + [continuation]` — always small and predictable
- **ADK limitation**: `launcher.Config.PluginConfig` is a single field — the plugin internally dispatches by `ctx.AgentName()`
- **State keys**: all keys suffixed with `{agentName}` to prevent cross-agent contamination in shared sessions

### State Keys

| Key | Set by | Read by | Purpose |
|---|---|---|---|
| `__context_guard_summary_{agent}` | `BeforeModelCallback` | `BeforeModelCallback` | Running conversation summary |
| `__context_guard_summarized_at_{agent}` | `BeforeModelCallback` | (diagnostic) | Token count at last compaction |
| `__context_guard_real_tokens_{agent}` | `AfterModelCallback` | `BeforeModelCallback` | Real `PromptTokenCount` from provider |
| `__context_guard_last_heuristic_{agent}` | `BeforeModelCallback` | `BeforeModelCallback` | Heuristic estimate for calibration |
| `__context_guard_contents_at_compaction_{agent}` | Sliding window | Sliding window | Watermark for turn counting |

### File Naming Convention

- `contextguard.go` — public API, `BeforeModelCallback`, `AfterModelCallback`
- `model_registry*.go` — ModelRegistry interface and implementations
- `compaction_threshold.go` — token-threshold strategy (full summary, calibrated tokens)
- `compaction_sliding_window.go` — sliding-window strategy (turn-count based, with recent tail)
- `compaction_utils.go` — shared helpers: state persistence, summarization, token estimation, continuation injection, todo loading
- `compaction_simulation_test.go` — realistic simulation tests and timing gap proofs

### CrushRegistry

Built-in `ModelRegistry` that fetches model metadata from `https://raw.githubusercontent.com/charmbracelet/crush/main/internal/agent/hyper/provider.json`. Background refresh every 6h. Defaults to 128k context / 4096 max tokens for unknown models.

### Strategies

| Strategy | Trigger | Compaction mode | Config |
|---|---|---|---|
| `threshold` | calibrated tokens > (contextWindow - buffer) | Full summary (entire conversation) | `WithMaxTokens(n)` or auto from registry |
| `sliding_window` | turns since last compaction > maxTurns | Split: summarize old, keep recent tail | `WithSlidingWindow(n)` |

Buffer: fixed 20k for windows >200k, 20% for smaller ones.

### Compaction Flow (threshold)

See [INVESTIGATION_RESULTS.md](./INVESTIGATION_RESULTS.md) for the full architecture diagram and timing analysis.

```
BeforeModelCallback:
  tokenCount() → calibrated estimate using real tokens + heuristic correction
  if < threshold → pass through
  if ≥ threshold → summarize everything → [summary] + [continuation]

AfterModelCallback:
  persist PromptTokenCount for next step's calibration
```


