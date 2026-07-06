// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package langfuse

import (
	"context"
	"testing"

	"go.opentelemetry.io/otel"
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
		PublicKey: "pk-test",
		SecretKey: "sk-test",
		Host:      "http://127.0.0.1:1", // never dialled: the batch exporter only flushes on shutdown
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

	// Shut down with an already-cancelled context so the OTLP batch exporter
	// does not attempt to flush to the fake endpoint.
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	_ = shutdown(cancelled)
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
