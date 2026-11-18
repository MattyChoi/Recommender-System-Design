// Package tracing wires OpenTelemetry, and decides what gets sampled.
package tracing

import (
	"context"
	"fmt"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	otelcodes "go.opentelemetry.io/otel/codes"
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
	AttrDegraded     = "recsys.degraded"
	AttrFallback     = "recsys.used_fallback"
	AttrIncomplete   = "recsys.incomplete_rows"
	AttrVariant      = "recsys.experiment_variant"
	AttrSource       = "recsys.source"
	AttrBudget       = "recsys.budget_ms"
	AttrCandidates   = "recsys.candidates"
	AttrModelVersion = "recsys.model_version"
	AttrIndexVersion = "recsys.index_version"
	AttrRows         = "recsys.feature_rows"
	AttrScores       = "recsys.scores"
	AttrServed       = "recsys.served"
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

	// NewSchemaless, NOT NewWithAttributes(semconv.SchemaURL, ...). Merge
	// REFUSES two resources whose schema URLs differ, returning Empty and an
	// error -- and resource.Default() carries whichever semconv version the SDK
	// was built against (1.43.0 under sdk v1.47.0), against the 1.26.0 this
	// file imports. Declaring no schema lets the default's stand and survives
	// the next SDK bump
	attributes, err := resource.Merge(resource.Default(), resource.NewSchemaless(
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

// Served tags a span with what answered the request.
func Served(span trace.Span, model, index, variant string) {
	attributes := make([]attribute.KeyValue, 0, 3)
	if model != "" {
		attributes = append(attributes, attribute.String(AttrModelVersion, model))
	}
	if index != "" {
		attributes = append(attributes, attribute.String(AttrIndexVersion, index))
	}
	if variant != "" {
		attributes = append(attributes, attribute.String(AttrVariant, variant))
	}
	if len(attributes) > 0 {
		span.SetAttributes(attributes...)
	}
}

// Built records the rank stage's feature matrix -- the half of a "ranker"
// degradation that belongs to the gateway rather than to Triton.
func Built(span trace.Span, rows, incomplete int, err error) {
	span.SetAttributes(
		attribute.Int(AttrRows, rows),
		attribute.Int(AttrIncomplete, incomplete),
	)
	if err != nil {
		span.RecordError(err)
		span.SetStatus(otelcodes.Error, "feature build failed")
	}
}

// Inferred records the ranker's answer.
func Inferred(span trace.Span, got, want int, err error) {
	span.SetAttributes(attribute.Int(AttrScores, got))
	switch {
	case err != nil:
		span.RecordError(err)
		span.SetStatus(otelcodes.Error, "ranker failed")
	case got != want:
		span.SetStatus(otelcodes.Error, "score width mismatch")
	}
}

// Fallback marks the popularity path.
func Fallback(span trace.Span, served int, err error) {
	span.SetAttributes(
		attribute.Bool(AttrFallback, true),
		attribute.Int(AttrServed, served),
	)
	if err != nil {
		span.RecordError(err)
		span.SetStatus(otelcodes.Error, "fallback failed")
	}
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
