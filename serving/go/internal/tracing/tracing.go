// Package tracing wires OpenTelemetry, and decides what gets sampled.
package tracing

import (
	"context"
	"fmt"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.26.0"
	"go.opentelemetry.io/otel/trace"
)

// Name is the instrumentation scope every span in this binary is created
// under, so a span can be traced back to the code that made it.
const Name = "github.com/MattyChoi/Recommender-System-Design/serving/go"

const (
	AttrDegraded   = "recsys.degraded"
	AttrFallback   = "recsys.used_fallback"
	AttrIncomplete = "recsys.incomplete_rows"
	AttrVariant    = "recsys.experiment_variant"
	AttrSource     = "recsys.source"
	AttrBudget     = "recsys.budget_ms"
	AttrCandidates = "recsys.candidates"
)

func Tracer() trace.Tracer { return otel.Tracer(Name) }

func Setup(ctx context.Context, serviceName, endpoint, version string, ratio float64) (func(context.Context) error, error) {
	if endpoint == "" {
		return func(context.Context) error { return nil }, nil
	}

	exporter, err := otlptracegrpc.New(ctx, otlptracegrpc.WithEndpoint(endpoint), otlptracegrpc.WithInsecure())
	if err != nil {
		return nil, fmt.Errorf("otlp exporter at %s: %w", endpoint, err)
	}

	attributes, err := resource.Merge(resource.Default(), resource.NewWithAttributes(
		semconv.SchemaURL,
		semconv.ServiceName(serviceName),
		semconv.ServiceVersion(version),
	))
	if err != nil {
		return nil, fmt.Errorf("building resource: %w", err)
	}

	provider := sdktrace.NewTracerProvider(
		// Batched, not synchronous: a synchronous exporter puts the collector
		// on the request path, so a slow collector becomes a slow recommender.
		sdktrace.WithBatcher(exporter, sdktrace.WithBatchTimeout(5*time.Second)),
		sdktrace.WithResource(attributes),
		// ParentBased, so an upstream decision survives every hop. A
		// per-service sampler produces traces with holes, which are worse than
		// no trace: the gap looks like a missing service rather than a
		// sampling artefact.
		sdktrace.WithSampler(sdktrace.ParentBased(sdktrace.TraceIDRatioBased(ratio))),
	)
	otel.SetTracerProvider(provider)

	// W3C tracecontext and baggage, so the trace survives crossing into the
	// Python sidecars -- which is the whole reason to trace a fan-out.
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
		propagation.TraceContext{}, propagation.Baggage{},
	))

	return provider.Shutdown, nil
}

// Degraded marks a span so a tail sampler can keep it.
func Degraded(span trace.Span, sources []string) {
	if len(sources) == 0 {
		return
	}
	span.SetAttributes(attribute.StringSlice(AttrDegraded, sources))
}

// Source annotates one retriever's span with its budget, so a span that ends
// exactly at its deadline is legible as a DROPPED source rather than a fast
// one that happened to return nothing.
func Source(span trace.Span, name string, budget time.Duration) {
	span.SetAttributes(
		attribute.String(AttrSource, name),
		attribute.Int64(AttrBudget, budget.Milliseconds()),
	)
}
