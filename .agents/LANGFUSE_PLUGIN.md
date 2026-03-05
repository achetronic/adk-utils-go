# Langfuse Plugin

ADK plugin that exports traces to [Langfuse](https://langfuse.com) via OTLP/HTTP, enriching generate_content spans with full LLM request/response payloads and token usage so that Langfuse can display costs, latency, and prompt/completion content.

## Package

`github.com/achetronic/adk-utils-go/plugin/langfuse`

## Files

| File | Purpose |
|---|---|
| `langfuse.go` | Public `Setup()` API, spanEnricher (callbacks), enrichingExporter, enrichedSpan wrapper, helpers |
| `types.go` | `Config` struct with yaml/json tags, `IsEnabled()` method |
| `context.go` | Context helpers: `WithUserID`, `WithTags`, `WithTraceMetadata`, `WithEnvironment`, `WithRelease`, `WithTraceName` + corresponding `*FromContext` readers |

## Architecture

The plugin has two cooperating parts that share a single `spanEnricher` instance:

### 1. ADK Plugin Callbacks (spanEnricher)

Registered as `BeforeAgentCallback`, `AfterAgentCallback`, `BeforeModelCallback`, and `AfterModelCallback` via `plugin.New()`.

| Callback | What it does |
|---|---|
| `beforeAgent` | Pushes the current `invoke_agent` span onto a per-branch stack. Decorates it with Langfuse attributes (user ID, session ID, tags, metadata, environment, release, trace name, user input). |
| `afterAgent` | Pops the span from the per-branch stack. Cleans up when the last span for a branch is removed. |
| `beforeModel` | Serialises the full LLM prompt (system instruction + message history + tool calls) as JSON. Enqueues it as a pending `llmCall` keyed by the `invoke_agent` span ID. |
| `afterModel` | Captures the model's response text (or error), token usage (`PromptTokenCount`, `CandidatesTokenCount`, `TotalTokenCount`), and attaches them to the pending `llmCall`. For non-partial final text responses (no function calls), propagates the output to all ancestor `invoke_agent` spans in the same branch. |

### 2. Enriching Exporter (enrichingExporter)

Wraps the real OTLP `SpanExporter`. At export time, for every span named `generate_content*`, it:

1. Looks up the span's parent ID (which is the `invoke_agent` span ID)
2. Pops the oldest pending `llmCall` for that ID (FIFO queue)
3. Injects extra attributes into the span:
   - `gcp.vertex.agent.llm_request` — full serialised prompt
   - `gcp.vertex.agent.llm_response` — model output text
   - `gen_ai.request.model` — model identifier
   - `gen_ai.usage.input_tokens` — prompt token count
   - `gen_ai.usage.output_tokens` — completion token count

The injection is done via `enrichedSpan`, a thin wrapper around `sdktrace.ReadOnlySpan` that appends extra attributes without mutating the original span.

### Why this two-part design?

ADK plugin callbacks see the `invoke_agent` span in their context, but the `generate_content` span is created **inside the ADK** after the `BeforeModelCallback` returns. Plugin callbacks never see it. The enrichingExporter bridges this gap by intercepting spans at export time and matching them via parent span ID.

```
                    ┌─────────────────────────────┐
                    │       ADK Runner             │
                    │                              │
 beforeModel ──────►│  invoke_agent span (visible) │
                    │         │                    │
                    │         ▼                    │
                    │  generate_content span       │──── created by ADK internally
                    │  (NOT visible to callbacks)  │
                    │         │                    │
 afterModel ◄──────│         │                    │
                    └─────────┼────────────────────┘
                              │
                              ▼
                    ┌─────────────────────────────┐
                    │  BatchSpanProcessor          │
                    │         │                    │
                    │         ▼                    │
                    │  enrichingExporter           │
                    │  (matches parent ID →        │
                    │   pops pending llmCall →     │
                    │   injects attributes)        │
                    │         │                    │
                    │         ▼                    │
                    │  OTLP HTTP → Langfuse        │
                    └─────────────────────────────┘
```

## Multi-Agent Safety

The plugin is safe for all ADK agent topologies:

| Topology | How it's handled |
|---|---|
| Single agent | Keys are `invocationID` (branch is empty) |
| Sequential delegation (`transfer_to_agent`) | Each agent gets its own `invoke_agent` span with a unique span ID. The stack tracks nesting. |
| `SequentialAgent` / `LoopAgent` | Same as delegation — sequential execution, stack-based tracking |
| `ParallelAgent` | Each sub-agent gets a distinct `Branch()` value. The `branchKey` function combines `invocationID + ":" + branch` so concurrent sub-agents never collide. |

## Concurrency

All mutable state lives in `spanEnricher` behind a single `sync.Mutex`. The state is request-scoped and ephemeral:

- `agentSpans`: per-branch stack of active `invoke_agent` spans — pushed in `beforeAgent`, popped in `afterAgent`
- `pending`: FIFO queue of `llmCall` per span ID — enqueued in `beforeModel`, updated in `afterModel`, dequeued in `ExportSpans`

Each HTTP request's lifecycle is handled entirely by one server instance (spans don't hop across replicas), so this is safe for multi-replica deployments.

## Config

```go
type Config struct {
    PublicKey   string `yaml:"publicKey" json:"publicKey"`     // Required: Basic Auth username
    SecretKey   string `yaml:"secretKey" json:"secretKey"`     // Required: Basic Auth password
    Host        string `yaml:"host" json:"host"`               // Langfuse server URL (default: https://cloud.langfuse.com)
    Environment string `yaml:"environment" json:"environment"` // Optional: deployment environment tag
    Release     string `yaml:"release" json:"release"`         // Optional: version tag
    ServiceName string `yaml:"serviceName,omitempty"`          // Optional: OTel service.name (default: "langfuse-adk")
}
```

`IsEnabled()` returns `true` only when both `PublicKey` and `SecretKey` are non-empty.

## Context Helpers

Per-request attributes can be injected via context values before the ADK runner executes. These are read by `beforeAgent` and set as span attributes.

| Function | Span Attribute | Fallback |
|---|---|---|
| `WithUserID(ctx, id)` | `langfuse.user.id` | `ctx.UserID()` from ADK |
| `WithTags(ctx, []string)` | `langfuse.trace.tags` | none |
| `WithTraceMetadata(ctx, map)` | `langfuse.trace.metadata.<key>` | none |
| `WithEnvironment(ctx, env)` | `langfuse.environment` | `Config.Environment` (static) |
| `WithRelease(ctx, rel)` | `langfuse.release` | `Config.Release` (static) |
| `WithTraceName(ctx, name)` | `langfuse.trace.name` | auto-generated by Langfuse |

Typical usage: HTTP middleware sets these before passing the context to the ADK handler.

## Setup() API

`Setup` is the single entry point. It:

1. Creates an OTLP/HTTP exporter pointed at `{host}/api/public/otel/v1/traces` with Basic Auth
2. Wraps it in `enrichingExporter`
3. Creates ADK telemetry providers with a `BatchSpanProcessor`
4. Sets the global OTel `TracerProvider`
5. Creates the ADK plugin with all four callbacks
6. Returns `(runner.PluginConfig, shutdown, error)`

```go
pluginCfg, shutdown, err := langfuse.Setup(&langfuse.Config{
    PublicKey: os.Getenv("LANGFUSE_PUBLIC_KEY"),
    SecretKey: os.Getenv("LANGFUSE_SECRET_KEY"),
    Host:      "https://cloud.langfuse.com",
})
if err != nil { log.Fatal(err) }
defer shutdown(context.Background())
```

### Combining with other plugins (e.g. ContextGuard)

```go
langfuseCfg, shutdown, _ := langfuse.Setup(cfg)
guardCfg := guard.PluginConfig()

combined := runner.PluginConfig{
    Plugins: append(langfuseCfg.Plugins, guardCfg.Plugins...),
}

runnr, _ := runner.New(runner.Config{
    Agent:        myAgent,
    PluginConfig: combined,
})
```

## Span Attributes Reference

### On `invoke_agent` spans (set by beforeAgent)

| Attribute | Source |
|---|---|
| `langfuse.user.id` | `WithUserID(ctx)` or `ctx.UserID()` |
| `langfuse.session.id` | `ctx.SessionID()` |
| `langfuse.trace.tags` | `WithTags(ctx)` |
| `langfuse.trace.metadata.<key>` | `WithTraceMetadata(ctx)` |
| `langfuse.environment` | `WithEnvironment(ctx)` |
| `langfuse.release` | `WithRelease(ctx)` |
| `langfuse.trace.name` | `WithTraceName(ctx)` |
| `langfuse.trace.input` | `ctx.UserContent()` (JSON-serialised) |
| `langfuse.observation.input` | `ctx.UserContent()` (JSON-serialised) |
| `langfuse.trace.output` | Final text response (set by afterModel) |
| `langfuse.observation.output` | Final text response (set by afterModel) |

### On `generate_content` spans (injected by enrichingExporter)

| Attribute | Source |
|---|---|
| `gcp.vertex.agent.llm_request` | JSON-serialised prompt (system + messages + tool calls) |
| `gcp.vertex.agent.llm_response` | Plain-text model output |
| `gen_ai.request.model` | Model identifier from `LLMRequest.Model` |
| `gen_ai.usage.input_tokens` | `UsageMetadata.PromptTokenCount` |
| `gen_ai.usage.output_tokens` | `UsageMetadata.CandidatesTokenCount` |

## LLM Request Serialisation

`marshalLLMRequest` converts the ADK `LLMRequest` to a JSON map with:

```json
{
  "system": "system instruction text",
  "messages": [
    {"role": "user", "content": "hello"},
    {"role": "model", "tool_call": {"name": "search", "args": "{\"q\":\"foo\"}"}},
    {"role": "tool", "tool_response": {"name": "search", "result": "{\"results\":[]}"}}
  ]
}
```

## Cost Tracking

Langfuse calculates costs when it receives `gen_ai.usage.input_tokens` and `gen_ai.usage.output_tokens` **and** recognises the model name in `gen_ai.request.model`. If the model name (e.g. `claude-sonnet-4-6`) is not in Langfuse's pricing table, tokens will appear but costs will show as $0. Fix by adding a custom model definition in Langfuse UI (Settings > Models).

## Dependencies

The plugin adds these direct dependencies beyond what adk-utils-go already has:

- `go.opentelemetry.io/otel` — OTel API (attributes, trace)
- `go.opentelemetry.io/otel/sdk` — OTel SDK (TracerProvider, BatchSpanProcessor, ReadOnlySpan)
- `go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp` — OTLP/HTTP exporter
- `go.opentelemetry.io/otel/semconv/v1.36.0` — semantic conventions (service.name, deployment.environment)

All of these were already indirect dependencies via ADK; the plugin promotes them to direct.
