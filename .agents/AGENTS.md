# AGENTS.md

Agent guidelines for working in the `adk-utils-go` repository.

## Project Overview

A Go library providing utilities for Google's Agent Development Kit (ADK). This library extends ADK with additional backend implementations for topics like session management or memory services.

**Module**: `adk` (see `go.mod`)  
**Go Version**: 1.24.9+

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
├── session/
│   └── redis/
│       └── session.go      # Redis-backed session.Service implementation
├── memory/
│   └── postgres/
│       ├── memory.go       # PostgreSQL-backed memory.Service implementation
│       ├── memory_test.go  # Memory service tests (requires PostgreSQL)
│       ├── embedding.go    # OpenAI-compatible embedding model
│       └── embedding_test.go # Embedding tests (uses httptest mocks)
├── tools/
│   └── memory/
│       └── toolset.go      # Memory toolset for agent tools
├── go.mod
└── go.sum
```

### Package Purposes

| Package | Description |
|---------|-------------|
| `session/redis` | Redis-backed implementation of `session.Service` |
| `memory/postgres` | PostgreSQL+pgvector implementation of `memory.Service` |
| `tools/memory` | ADK toolset providing `search_memory` and `save_to_memory` tools |

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
session:{appName}:{userID}:{sessionID}   # Session data
sessions:{appName}:{userID}              # Session index (SET)
events:{appName}:{userID}:{sessionID}    # Event list (LIST)
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

### PostgreSQL Memory Service

1. **pgvector Extension**: Required for semantic search. Schema auto-creates the extension if embedding model is configured.
2. **Dimension Auto-Detection**: If `EmbeddingModel.Dimension()` returns 0, the service probes the model on init.
3. **Search Fallback**: Vector search → full-text search → recent entries.
4. **Upsert Behavior**: `AddSession` uses `ON CONFLICT ... DO UPDATE`.

### Memory Toolset

1. **Tool Names**: `search_memory` and `save_to_memory`
2. **User Scoping**: Tools automatically use `ctx.UserID()` for isolation.
3. **Single Entry Session**: `save_to_memory` creates a minimal session wrapper around the content.

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

### New Memory Backend

1. Create package under `memory/{backend}/`
2. Implement `memory.Service` interface (`AddSession`, `Search`)
3. Consider supporting the `EmbeddingModel` interface for semantic search

### New Toolset

1. Create package under `tools/{purpose}/`
2. Implement `tool.Toolset` interface
3. Use `functiontool.New()` to create tools from functions
4. Define typed args/result structs with JSON tags
