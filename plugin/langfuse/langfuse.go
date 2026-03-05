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

// Package langfuse wires ADK Go telemetry to a Langfuse instance via OTLP/HTTP.
//
// It is fully self-contained — it has zero imports from any host application.
// Any project using the ADK can import this package as a library, call
// Setup to configure the exporter and get back the plugin config and a
// shutdown function.
//
// The plugin is safe for multi-agent flows: single agents, sequential
// delegation (transfer_to_agent), SequentialAgent, LoopAgent and ParallelAgent
// are all supported. Parallel branches are isolated via the ADK Branch()
// identifier so that concurrent sub-agents never mix their spans or LLM
// payloads.
package langfuse

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/sdk/instrumentation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.36.0"
	oteltrace "go.opentelemetry.io/otel/trace"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/plugin"
	"google.golang.org/adk/runner"
	adktelemetry "google.golang.org/adk/telemetry"
	"google.golang.org/genai"
)

// defaultHost is the Langfuse Cloud US endpoint used when Config.Host is empty.
const defaultHost = "https://cloud.langfuse.com"

// defaultServiceName is used as the OTel service name when Config.ServiceName
// is not provided.
const defaultServiceName = "langfuse-adk"

// Setup initialises the full Langfuse integration: an OTLP/HTTP trace exporter
// pointed at the Langfuse ingestion endpoint, an enriching span exporter that
// injects LLM request/response payloads, and an ADK plugin that captures them.
//
// It returns a runner.PluginConfig ready to pass to the ADK launcher/runner
// and a shutdown function that flushes pending spans. The caller must defer
// the shutdown function.
//
// Usage:
//
//	pluginCfg, shutdown, err := langfuse.Setup(&langfuse.Config{
//	    PublicKey: os.Getenv("LANGFUSE_PUBLIC_KEY"),
//	    SecretKey: os.Getenv("LANGFUSE_SECRET_KEY"),
//	    Host:      "https://cloud.langfuse.com",
//	})
//	if err != nil { log.Fatal(err) }
//	defer shutdown(context.Background())
//
//	runnr, _ := runner.New(runner.Config{
//	    Agent:        myAgent,
//	    PluginConfig: pluginCfg,
//	})
func Setup(cfg *Config) (runner.PluginConfig, func(context.Context) error, error) {
	auth := base64.StdEncoding.EncodeToString(
		[]byte(cfg.PublicKey + ":" + cfg.SecretKey),
	)

	host := cfg.Host
	if host == "" {
		host = defaultHost
	}

	exporter, err := otlptracehttp.New(context.Background(),
		otlptracehttp.WithEndpointURL(fmt.Sprintf("%s/api/public/otel/v1/traces", host)),
		otlptracehttp.WithHeaders(map[string]string{
			"Authorization": "Basic " + auth,
		}),
		otlptracehttp.WithInsecure(),
	)
	if err != nil {
		return runner.PluginConfig{}, nil, fmt.Errorf("langfuse: create OTLP exporter: %w", err)
	}

	serviceName := cfg.ServiceName
	if serviceName == "" {
		serviceName = defaultServiceName
	}
	attrs := []attribute.KeyValue{
		semconv.ServiceNameKey.String(serviceName),
	}
	if cfg.Environment != "" {
		attrs = append(attrs, semconv.DeploymentEnvironmentNameKey.String(cfg.Environment))
	}
	res, err := resource.New(context.Background(), resource.WithAttributes(attrs...))
	if err != nil {
		return runner.PluginConfig{}, nil, fmt.Errorf("langfuse: create OTel resource: %w", err)
	}

	enricher := newSpanEnricher()
	wrapped := &enrichingExporter{inner: exporter, enricher: enricher}

	providers, err := adktelemetry.New(context.Background(),
		adktelemetry.WithSpanProcessors(sdktrace.NewBatchSpanProcessor(wrapped)),
		adktelemetry.WithResource(res),
	)
	if err != nil {
		return runner.PluginConfig{}, nil, fmt.Errorf("langfuse: create ADK telemetry providers: %w", err)
	}
	providers.SetGlobalOtelProviders()

	plug, _ := plugin.New(plugin.Config{
		Name:                "langfuse-enrichment",
		BeforeAgentCallback: agent.BeforeAgentCallback(enricher.beforeAgent),
		AfterAgentCallback:  agent.AfterAgentCallback(enricher.afterAgent),
		BeforeModelCallback: llmagent.BeforeModelCallback(enricher.beforeModel),
		AfterModelCallback:  llmagent.AfterModelCallback(enricher.afterModel),
	})

	pluginCfg := runner.PluginConfig{Plugins: []*plugin.Plugin{plug}}
	return pluginCfg, providers.Shutdown, nil
}

// ---------------------------------------------------------------------------
// llmCall
// ---------------------------------------------------------------------------

// llmCall captures the serialised request and response text for a single
// generate_content LLM invocation. The enrichingExporter consumes these to
// inject gcp.vertex.agent.llm_request / llm_response span attributes as
// well as gen_ai.usage.* token-count attributes so that Langfuse can
// display costs.
type llmCall struct {
	request      string // JSON-encoded prompt (system + messages + tool calls)
	response     string // plain-text model output or error message
	model        string // model identifier (e.g. "gemini-2.0-flash")
	inputTokens  int32
	outputTokens int32
	totalTokens  int32
}

// ---------------------------------------------------------------------------
// spanEnricher
// ---------------------------------------------------------------------------

// spanEnricher is the core state holder for the Langfuse ADK plugin. It
// tracks in-flight agent spans and pending LLM calls so that the
// enrichingExporter can attach request/response payloads to the correct
// generate_content spans at export time.
//
// Keys are built from invocationID + branch (via branchKey) so that
// ParallelAgent sub-agents running concurrently under the same invocation
// never collide. For sequential flows the branch is empty and the key
// degrades to the invocationID alone.
//
// Pending LLM calls are keyed by the invoke_agent span ID visible to the
// callbacks. The enrichingExporter matches them by looking up the parent
// span ID of each generate_content span (which is its invoke_agent ancestor).
//
// All callbacks return (nil, nil) — the plugin is a pure observer that never
// alters agent behaviour or short-circuits the ADK callback chain.
type spanEnricher struct {
	mu         sync.Mutex
	agentSpans map[string][]oteltrace.Span // branchKey → stack of invoke_agent spans
	pending    map[string][]llmCall        // invoke_agent spanID → FIFO queue of LLM calls awaiting export
}

// branchKey builds the map key that isolates parallel branches. In
// sequential flows Branch() returns "" and the key equals the invocationID.
func branchKey(ctx agent.CallbackContext) string {
	if b := ctx.Branch(); b != "" {
		return ctx.InvocationID() + ":" + b
	}
	return ctx.InvocationID()
}

// newSpanEnricher creates a fresh spanEnricher.
func newSpanEnricher() *spanEnricher {
	return &spanEnricher{
		agentSpans: make(map[string][]oteltrace.Span),
		pending:    make(map[string][]llmCall),
	}
}

// beforeAgent is the BeforeAgentCallback. It pushes the current
// invoke_agent span onto a per-branch stack and decorates it with Langfuse
// trace attributes sourced from the request context (user ID, session ID,
// tags, metadata, environment, release, trace name) and the user's input
// content.
//
// In a ParallelAgent flow each sub-agent gets a distinct branch, so their
// spans are tracked independently and never interleave.
func (e *spanEnricher) beforeAgent(ctx agent.CallbackContext) (*genai.Content, error) {
	span := oteltrace.SpanFromContext(ctx)
	if !span.IsRecording() {
		return nil, nil
	}
	key := branchKey(ctx)
	e.mu.Lock()
	e.agentSpans[key] = append(e.agentSpans[key], span)
	e.mu.Unlock()
	if userID := UserIDFromContext(ctx); userID != "" {
		span.SetAttributes(attribute.String("langfuse.user.id", userID))
	} else if userID := ctx.UserID(); userID != "" {
		span.SetAttributes(attribute.String("langfuse.user.id", userID))
	}
	if sessionID := ctx.SessionID(); sessionID != "" {
		span.SetAttributes(attribute.String("langfuse.session.id", sessionID))
	}
	if tags := TagsFromContext(ctx); len(tags) > 0 {
		span.SetAttributes(attribute.StringSlice("langfuse.trace.tags", tags))
	}
	for k, v := range TraceMetadataFromContext(ctx) {
		span.SetAttributes(attribute.String("langfuse.trace.metadata."+k, v))
	}
	if env := EnvironmentFromContext(ctx); env != "" {
		span.SetAttributes(attribute.String("langfuse.environment", env))
	}
	if rel := ReleaseFromContext(ctx); rel != "" {
		span.SetAttributes(attribute.String("langfuse.release", rel))
	}
	if name := TraceNameFromContext(ctx); name != "" {
		span.SetAttributes(attribute.String("langfuse.trace.name", name))
	}
	if uc := ctx.UserContent(); uc != nil {
		if s, err := json.Marshal(contentToText(uc)); err == nil {
			span.SetAttributes(
				attribute.String("langfuse.trace.input", string(s)),
				attribute.String("langfuse.observation.input", string(s)),
			)
		}
	}
	return nil, nil
}

// afterAgent is the AfterAgentCallback. It pops the invoke_agent span from
// the per-branch stack, cleaning up when the last span is removed.
func (e *spanEnricher) afterAgent(ctx agent.CallbackContext) (*genai.Content, error) {
	key := branchKey(ctx)
	e.mu.Lock()
	stack := e.agentSpans[key]
	if len(stack) > 1 {
		e.agentSpans[key] = stack[:len(stack)-1]
	} else {
		delete(e.agentSpans, key)
	}
	e.mu.Unlock()
	return nil, nil
}

// beforeModel is the BeforeModelCallback. It serialises the full LLM prompt
// (system instruction, conversation history, tool calls/responses) and
// enqueues it as a pending llmCall keyed by the invoke_agent span ID.
//
// ADK callbacks see the invoke_agent span in their context (the
// generate_content span is created later, inside the ADK, and is never
// visible to plugin callbacks). The enrichingExporter bridges the gap by
// looking up pending calls via the parent span ID of each generate_content
// span, which is the invoke_agent span ID stored here.
func (e *spanEnricher) beforeModel(ctx agent.CallbackContext, req *model.LLMRequest) (*model.LLMResponse, error) {
	span := oteltrace.SpanFromContext(ctx)
	if !span.IsRecording() {
		return nil, nil
	}
	spanID := span.SpanContext().SpanID().String()
	reqJSON, _ := json.Marshal(marshalLLMRequest(req))

	e.mu.Lock()
	e.pending[spanID] = append(e.pending[spanID], llmCall{
		request: string(reqJSON),
		model:   req.Model,
	})
	e.mu.Unlock()
	return nil, nil
}

// afterModel is the AfterModelCallback. It captures the model's response
// text (or the error message on failure), attaches it to the pending
// llmCall, and — when the response is a non-partial final text answer (no
// function calls) — propagates the output to the ancestor invoke_agent
// spans in the same branch so that Langfuse shows the final answer at the
// trace level. The check avoids resp.TurnComplete because that field is
// only populated in streaming mode; instead it uses !Partial +
// !hasFunctionCalls which works in both streaming and non-streaming flows.
func (e *spanEnricher) afterModel(ctx agent.CallbackContext, resp *model.LLMResponse, llmErr error) (*model.LLMResponse, error) {
	span := oteltrace.SpanFromContext(ctx)
	if !span.IsRecording() {
		return nil, nil
	}
	spanID := span.SpanContext().SpanID().String()

	var text string
	if llmErr != nil {
		text = llmErr.Error()
	} else if resp != nil && resp.Content != nil {
		text = contentToText(resp.Content)
	}

	e.mu.Lock()
	queue := e.pending[spanID]
	if len(queue) > 0 {
		queue[len(queue)-1].response = text
		if resp != nil && resp.UsageMetadata != nil {
			queue[len(queue)-1].inputTokens = resp.UsageMetadata.PromptTokenCount
			queue[len(queue)-1].outputTokens = resp.UsageMetadata.CandidatesTokenCount
			queue[len(queue)-1].totalTokens = resp.UsageMetadata.TotalTokenCount
		}
	}
	e.mu.Unlock()

	if resp != nil && !resp.Partial && text != "" && !hasFunctionCalls(resp.Content) {
		key := branchKey(ctx)
		e.mu.Lock()
		stack := e.agentSpans[key]
		e.mu.Unlock()
		outputJSON, _ := json.Marshal(text)
		for _, s := range stack {
			if s != nil && s.IsRecording() {
				s.SetAttributes(
					attribute.String("langfuse.trace.output", string(outputJSON)),
					attribute.String("langfuse.observation.output", string(outputJSON)),
				)
			}
		}
	}
	return nil, nil
}

// popCall dequeues the oldest pending llmCall for the given span ID. It
// returns false when no calls are pending.
func (e *spanEnricher) popCall(spanID string) (llmCall, bool) {
	e.mu.Lock()
	defer e.mu.Unlock()
	queue := e.pending[spanID]
	if len(queue) == 0 {
		return llmCall{}, false
	}
	call := queue[0]
	if len(queue) == 1 {
		delete(e.pending, spanID)
	} else {
		e.pending[spanID] = queue[1:]
	}
	return call, true
}

// ---------------------------------------------------------------------------
// enrichingExporter
// ---------------------------------------------------------------------------

// enrichingExporter wraps a real OTLP SpanExporter and injects
// gcp.vertex.agent.llm_request / llm_response attributes into
// generate_content spans just before they are exported. This is necessary
// because the ADK does not natively attach prompt/response payloads to its
// OTel spans.
//
// Pending calls are looked up by the parent span ID of each generate_content
// span — that parent is the invoke_agent span whose ID was used as key by
// the beforeModel/afterModel callbacks.
type enrichingExporter struct {
	inner    sdktrace.SpanExporter
	enricher *spanEnricher
}

// ExportSpans enriches generate_content spans with the pending LLM
// request/response payloads and delegates to the inner exporter.
func (ex *enrichingExporter) ExportSpans(ctx context.Context, spans []sdktrace.ReadOnlySpan) error {
	enriched := make([]sdktrace.ReadOnlySpan, len(spans))
	for i, s := range spans {
		var extra []attribute.KeyValue

		if strings.HasPrefix(s.Name(), "generate_content") {
			parentID := s.Parent().SpanID().String()
			if call, ok := ex.enricher.popCall(parentID); ok {
				if call.request != "" {
					extra = append(extra, attribute.String("gcp.vertex.agent.llm_request", call.request))
				}
				if call.response != "" {
					extra = append(extra, attribute.String("gcp.vertex.agent.llm_response", call.response))
				}
				if call.model != "" {
					extra = append(extra, attribute.String("gen_ai.request.model", call.model))
				}
				if call.inputTokens > 0 {
					extra = append(extra, attribute.Int64("gen_ai.usage.input_tokens", int64(call.inputTokens)))
				}
				if call.outputTokens > 0 {
					extra = append(extra, attribute.Int64("gen_ai.usage.output_tokens", int64(call.outputTokens)))
				}
			}
		}

		if len(extra) > 0 {
			enriched[i] = &enrichedSpan{ReadOnlySpan: s, extra: extra}
		} else {
			enriched[i] = s
		}
	}
	return ex.inner.ExportSpans(ctx, enriched)
}

// Shutdown delegates to the inner exporter, flushing any buffered spans.
func (ex *enrichingExporter) Shutdown(ctx context.Context) error {
	return ex.inner.Shutdown(ctx)
}

// ---------------------------------------------------------------------------
// enrichedSpan
// ---------------------------------------------------------------------------

// enrichedSpan wraps an sdktrace.ReadOnlySpan and appends extra attributes
// without modifying the original. Every method of the ReadOnlySpan interface
// is explicitly forwarded so that the wrapper satisfies the full contract.
type enrichedSpan struct {
	sdktrace.ReadOnlySpan
	extra []attribute.KeyValue
}

// Attributes returns the original span attributes plus the extra ones
// injected by the enrichingExporter.
func (s *enrichedSpan) Attributes() []attribute.KeyValue {
	return append(s.ReadOnlySpan.Attributes(), s.extra...)
}

func (s *enrichedSpan) Name() string                               { return s.ReadOnlySpan.Name() }
func (s *enrichedSpan) SpanContext() oteltrace.SpanContext          { return s.ReadOnlySpan.SpanContext() }
func (s *enrichedSpan) Parent() oteltrace.SpanContext               { return s.ReadOnlySpan.Parent() }
func (s *enrichedSpan) SpanKind() oteltrace.SpanKind                { return s.ReadOnlySpan.SpanKind() }
func (s *enrichedSpan) StartTime() time.Time                        { return s.ReadOnlySpan.StartTime() }
func (s *enrichedSpan) EndTime() time.Time                          { return s.ReadOnlySpan.EndTime() }
func (s *enrichedSpan) Events() []sdktrace.Event                    { return s.ReadOnlySpan.Events() }
func (s *enrichedSpan) Links() []sdktrace.Link                      { return s.ReadOnlySpan.Links() }
func (s *enrichedSpan) Status() sdktrace.Status                     { return s.ReadOnlySpan.Status() }
func (s *enrichedSpan) Resource() *resource.Resource                { return s.ReadOnlySpan.Resource() }
func (s *enrichedSpan) DroppedAttributes() int                      { return s.ReadOnlySpan.DroppedAttributes() }
func (s *enrichedSpan) DroppedEvents() int                          { return s.ReadOnlySpan.DroppedEvents() }
func (s *enrichedSpan) DroppedLinks() int                           { return s.ReadOnlySpan.DroppedLinks() }
func (s *enrichedSpan) ChildSpanCount() int                         { return s.ReadOnlySpan.ChildSpanCount() }
func (s *enrichedSpan) InstrumentationScope() instrumentation.Scope { return s.ReadOnlySpan.InstrumentationScope() }
func (s *enrichedSpan) InstrumentationLibrary() instrumentation.Scope {
	return s.ReadOnlySpan.InstrumentationLibrary()
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

// marshalLLMRequest converts an ADK LLMRequest into a JSON-friendly map
// containing the system instruction and the full message history (text,
// tool calls, and tool responses). The result is intended for the
// gcp.vertex.agent.llm_request span attribute.
func marshalLLMRequest(req *model.LLMRequest) map[string]any {
	msgs := make([]map[string]any, 0, len(req.Contents))
	for _, c := range req.Contents {
		if c == nil {
			continue
		}
		for _, p := range c.Parts {
			switch {
			case p.Text != "":
				msgs = append(msgs, map[string]any{"role": c.Role, "content": p.Text})
			case p.FunctionCall != nil:
				args, _ := json.Marshal(p.FunctionCall.Args)
				msgs = append(msgs, map[string]any{
					"role":      c.Role,
					"tool_call": map[string]any{"name": p.FunctionCall.Name, "args": string(args)},
				})
			case p.FunctionResponse != nil:
				r, _ := json.Marshal(p.FunctionResponse.Response)
				msgs = append(msgs, map[string]any{
					"role":          "tool",
					"tool_response": map[string]any{"name": p.FunctionResponse.Name, "result": string(r)},
				})
			}
		}
	}
	result := map[string]any{"messages": msgs}
	if req.Config != nil && req.Config.SystemInstruction != nil {
		result["system"] = contentToText(req.Config.SystemInstruction)
	}
	return result
}

// contentToText flattens a genai.Content into a single human-readable string.
// Text parts are joined with newlines; function calls and responses are
// rendered as bracketed annotations (e.g. "[tool_call: name(args)]").
func contentToText(c *genai.Content) string {
	if c == nil {
		return ""
	}
	var parts []string
	for _, p := range c.Parts {
		switch {
		case p.Text != "":
			parts = append(parts, p.Text)
		case p.FunctionCall != nil:
			args, _ := json.Marshal(p.FunctionCall.Args)
			parts = append(parts, fmt.Sprintf("[tool_call: %s(%s)]", p.FunctionCall.Name, string(args)))
		case p.FunctionResponse != nil:
			resp, _ := json.Marshal(p.FunctionResponse.Response)
			parts = append(parts, fmt.Sprintf("[tool_response: %s → %s]", p.FunctionResponse.Name, string(resp)))
		}
	}
	return strings.Join(parts, "\n")
}

// hasFunctionCalls reports whether c contains at least one FunctionCall part.
// It is used to distinguish intermediate tool-call responses from final text
// answers so that only the latter are propagated as trace output.
func hasFunctionCalls(c *genai.Content) bool {
	if c == nil {
		return false
	}
	for _, p := range c.Parts {
		if p.FunctionCall != nil {
			return true
		}
	}
	return false
}
