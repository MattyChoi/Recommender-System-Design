// Package tracing wires OpenTelemetry, and decides what gets sampled.
//
// # Why a trace and not just the metrics
//
// The metrics say a stage was slow and which one. They cannot say that the
// slow stage was slow BECAUSE it waited on something, and a five-source
// fan-out with per-source deadlines is exactly the shape where that is the
// question. A waterfall shows the sources side by side, each ending at its own
// budget, and makes the architecture legible in one picture -- which is also
// why §16.1 asks for a screenshot of it.
//
// # The sampling problem, stated rather than defaulted
//
// Head sampling decides at the root, before anything interesting has happened.
// At any ratio below 1.0 that means **the degraded requests -- the ones worth
// looking at -- are dropped at the same rate as the healthy ones**, and they
// are by definition rare. A 1% head sample of a service with a 2% ranker
// timeout rate captures almost nothing of the 2%.
//
// Two things follow, and both are implemented here:
//
//   - The sampler is parent-based, so a caller that decided to sample a
//     request keeps that decision through every hop. A per-service sampler
//     produces traces with holes in them.
//   - Degradation is recorded as span ATTRIBUTES and events, not just as
//     metrics, so a collector can TAIL-sample on them: keep every trace where
//     `recsys.degraded` is non-empty, ratio-sample the rest. That moves the
//     decision to after the facts are known, which is the only place it can
//     be made correctly. The collector config is deployment, not code, but
//     the attributes it would key on have to exist here or the option is not
//     available later.
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

// Tracer returns the shared tracer.
//
// Safe to call before -- or without -- Setup: the OpenTelemetry API is a no-op
// when no provider is registered. That is what makes it reasonable to
// instrument the pipeline directly instead of threading an interface through
// it: with tracing off, every span call is a couple of nil checks, and the
// tests need no configuration at all.
func Tracer() trace.Tracer { return otel.Tracer(Name) }

// Setup installs a provider exporting over OTLP, and returns its shutdown.
//
// An empty endpoint leaves the global no-op provider in place and returns a
// shutdown that does nothing, so "tracing is off" is a configuration rather
// than a code path.
func Setup(
	ctx context.Context, serviceName, endpoint, version string, ratio float64,
) (func(context.Context) error, error) {
	if endpoint == "" {
		return func(context.Context) error { return nil }, nil
	}

	exporter, err := otlptracegrpc.New(ctx,
		otlptracegrpc.WithEndpoint(endpoint),
		// Insecure because the collector is a sidecar or a cluster-local
		// service. If it ever stops being either, this is the line to change.
		otlptracegrpc.WithInsecure(),
	)
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

// Attributes recorded on the request span. Named constants because a
// collector's tail-sampling policy keys on these strings, and a typo there
// silently stops keeping the traces that matter.
const (
	AttrDegraded   = "recsys.degraded"
	AttrFallback   = "recsys.used_fallback"
	AttrIncomplete = "recsys.incomplete_rows"
	AttrVariant    = "recsys.experiment_variant"
	AttrSource     = "recsys.source"
	AttrBudget     = "recsys.budget_ms"
	AttrCandidates = "recsys.candidates"
)

// Degraded marks a span so a tail sampler can keep it.
//
// Recorded as an ATTRIBUTE rather than only counted, because the counter
// answers "how often" and the trace answers "what else was happening at the
// time" -- and the second question is the one that gets asked at 3am.
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
