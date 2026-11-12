// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package langfuse

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	oteltrace "go.opentelemetry.io/otel/trace"
)

// fixedIDGenerator is an sdktrace.IDGenerator that always returns the same
// trace ID, standing in for real-world generators that derive trace IDs
// deterministically from an external request/run ID.
type fixedIDGenerator struct {
	traceID oteltrace.TraceID
}

func (g *fixedIDGenerator) NewIDs(_ context.Context) (oteltrace.TraceID, oteltrace.SpanID) {
	return g.traceID, oteltrace.SpanID{1, 2, 3, 4, 5, 6, 7, 8}
}

func (g *fixedIDGenerator) NewSpanID(_ context.Context, _ oteltrace.TraceID) oteltrace.SpanID {
	return oteltrace.SpanID{8, 7, 6, 5, 4, 3, 2, 1}
}

// TestSetupAppliesTracerProviderOptions verifies that options supplied via
// Config.TracerProviderOptions reach the trace provider Setup installs:
// a custom ID generator controls the trace ID of new root spans, and an
// additional span processor receives the finished spans.
func TestSetupAppliesTracerProviderOptions(t *testing.T) {
	prev := otel.GetTracerProvider()
	defer otel.SetTracerProvider(prev)

	wantTraceID := oteltrace.TraceID{0xde, 0xad, 0xbe, 0xef, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	recorder := tracetest.NewInMemoryExporter()

	cfg := &Config{
		PublicKey:   "pk-test",
		SecretKey:   "sk-test",
		Host:        "http://127.0.0.1:1", // never dialled: the batch exporter only flushes on shutdown
		ServiceName: "setup-test-service",
		TracerProviderOptions: []sdktrace.TracerProviderOption{
			sdktrace.WithIDGenerator(&fixedIDGenerator{traceID: wantTraceID}),
			sdktrace.WithSpanProcessor(sdktrace.NewSimpleSpanProcessor(recorder)),
		},
	}

	pluginCfg, shutdown, err := Setup(cfg)
	if err != nil {
		t.Fatalf("Setup returned error: %v", err)
	}
	if len(pluginCfg.Plugins) == 0 {
		t.Fatalf("Setup returned no plugins")
	}

	_, span := otel.Tracer("setup-test").Start(context.Background(), "probe")
	gotTraceID := span.SpanContext().TraceID()
	span.End()

	if gotTraceID != wantTraceID {
		t.Errorf("root span trace ID = %s, want %s (custom IDGenerator not applied)", gotTraceID, wantTraceID)
	}

	spans := recorder.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("additional span processor recorded %d spans, want 1", len(spans))
	}
	if spans[0].SpanContext.TraceID() != wantTraceID {
		t.Errorf("recorded span trace ID = %s, want %s", spans[0].SpanContext.TraceID(), wantTraceID)
	}

	// The provider's resource must carry Setup's own wiring (service.name)
	// merged over resource.Default() (telemetry.sdk.*).
	resAttrs := make(map[attribute.Key]string)
	for _, kv := range spans[0].Resource.Attributes() {
		resAttrs[kv.Key] = kv.Value.Emit()
	}
	if got := resAttrs["service.name"]; got != "setup-test-service" {
		t.Errorf("resource service.name = %q, want %q (Setup resource not applied)", got, "setup-test-service")
	}
	if _, ok := resAttrs["telemetry.sdk.name"]; !ok {
		t.Errorf("resource is missing telemetry.sdk.name (resource.Default() not merged)")
	}

	// Shut down with an already-cancelled context so the OTLP batch exporter
	// does not attempt to flush to the fake endpoint.
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	_ = shutdown(cancelled)
}

// TestSetupSendsV4IngestionHeader verifies the OTLP exporter declares the
// Langfuse v4 ingestion version. Without the header a v4 deployment running
// the v3 -> v4 migration can route the spans through the delayed dual-write
// staging pipeline instead of writing them to the observations table.
func TestSetupSendsV4IngestionHeader(t *testing.T) {
	prev := otel.GetTracerProvider()
	defer otel.SetTracerProvider(prev)

	var (
		mu       sync.Mutex
		received bool
		path     string
		auth     string
		version  string
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		received = true
		path = r.URL.Path
		auth = r.Header.Get("Authorization")
		version = r.Header.Get(ingestionVersionHeader)
		mu.Unlock()
		w.Header().Set("Content-Type", "application/x-protobuf")
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	cfg := &Config{
		PublicKey: "pk-test",
		SecretKey: "sk-test",
		Host:      srv.URL,
		Insecure:  true,
	}
	_, shutdown, err := Setup(cfg)
	if err != nil {
		t.Fatalf("Setup returned error: %v", err)
	}

	_, span := otel.Tracer("setup-test").Start(context.Background(), "probe")
	span.End()
	if err := shutdown(context.Background()); err != nil {
		t.Fatalf("shutdown: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if !received {
		t.Fatal("no spans reached the Langfuse endpoint")
	}
	if path != "/api/public/otel/v1/traces" {
		t.Errorf("export path = %q, want %q", path, "/api/public/otel/v1/traces")
	}
	if !strings.HasPrefix(auth, "Basic ") {
		t.Errorf("Authorization header = %q, want a Basic credential", auth)
	}
	if version != ingestionVersionV4 {
		t.Errorf("%s header = %q, want %q", ingestionVersionHeader, version, ingestionVersionV4)
	}
}

// TestSetupWithoutTracerProviderOptions guards the default path: an empty
// option list must keep the pre-existing behaviour and still install a
// working global tracer provider.
func TestSetupWithoutTracerProviderOptions(t *testing.T) {
	prev := otel.GetTracerProvider()
	defer otel.SetTracerProvider(prev)

	cfg := &Config{
		PublicKey: "pk-test",
		SecretKey: "sk-test",
		Host:      "http://127.0.0.1:1",
	}

	pluginCfg, shutdown, err := Setup(cfg)
	if err != nil {
		t.Fatalf("Setup returned error: %v", err)
	}
	if len(pluginCfg.Plugins) == 0 {
		t.Fatalf("Setup returned no plugins")
	}

	_, span := otel.Tracer("setup-test").Start(context.Background(), "probe")
	if !span.SpanContext().IsValid() {
		t.Errorf("default path produced an invalid span context")
	}
	span.End()

	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	_ = shutdown(cancelled)
}
