package metrics

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/service"
)

func newPipeline(t *testing.T) (*Pipeline, *prometheus.Registry) {
	t.Helper()
	registry := prometheus.NewRegistry()
	return New(registry), registry
}

func degradedResult() service.Result {
	return service.Result{
		Scores: []float64{1.5, -2.5},
		StageLatency: map[string]time.Duration{
			"retrieval": 12 * time.Millisecond,
			"rank":      30 * time.Millisecond,
		},
		StageCounts: map[string]int{
			"retrieved":       400,
			"after_filter":    0,
			"served":          10,
			"incomplete_rows": 7,
		},
		DegradedSources: []string{"ranker", "features"},
		UsedFallback:    true,
	}
}

// --- What the alerts need ----------------------------------------------------

// TestADegradedRequestIsCountedEvenThoughItSucceeded is the reason this
// package exists.
//
// Every stage here degrades rather than fails, so a request served from the
// retrieval order by a cold user with a saturated filter returns 200 and a
// well-formed slate. Without these counters the service looks healthy while
// serving nobody anything good.
func TestADegradedRequestIsCountedEvenThoughItSucceeded(t *testing.T) {
	pipeline, registry := newPipeline(t)

	pipeline.Request(codes.OK.String(), 50*time.Millisecond)
	pipeline.Observed(degradedResult(), "control")

	if got := testutil.ToFloat64(pipeline.requests.WithLabelValues("OK")); got != 1 {
		t.Errorf("requests_total{code=OK} = %v", got)
	}
	for _, source := range []string{"ranker", "features"} {
		if got := testutil.ToFloat64(pipeline.degraded.WithLabelValues(source)); got != 1 {
			t.Errorf("degraded_total{source=%s} = %v", source, got)
		}
	}
	if got := testutil.ToFloat64(pipeline.fallbacks); got != 1 {
		t.Errorf("fallback_total = %v", got)
	}
	// The gathered output is what Prometheus actually sees; ToFloat64 alone
	// would pass on a collector that was never registered.
	if count := testutil.CollectAndCount(registry); count == 0 {
		t.Error("nothing was registered")
	}
}

func TestIncompleteRowsAreCounted(t *testing.T) {
	// Rung 2 serves candidates whose model columns are zero-filled. The slate
	// is real and the ordering is worse, and nothing else in the response
	// distinguishes it from a healthy one.
	pipeline, _ := newPipeline(t)

	pipeline.Observed(degradedResult(), "control")

	if got := testutil.ToFloat64(pipeline.incomplete); got != 7 {
		t.Errorf("incomplete_rows_total = %v, want 7", got)
	}
}

func TestTheHoldbackIsCountedAsAnArm(t *testing.T) {
	// §17.2 checks the ratio first. An uncounted holdback makes the SRM check
	// compare arms against a denominator it cannot see.
	pipeline, _ := newPipeline(t)

	pipeline.Observed(service.Result{}, "")

	if got := testutil.ToFloat64(pipeline.assignments.WithLabelValues("none")); got != 1 {
		t.Errorf("experiment_assignments_total{variant=none} = %v", got)
	}
}

func TestStageLatencyAndCountsAreLabelledPerStage(t *testing.T) {
	pipeline, registry := newPipeline(t)

	pipeline.Observed(degradedResult(), "control")

	gathered, err := registry.Gather()
	if err != nil {
		t.Fatalf("Gather: %v", err)
	}
	stages := map[string]bool{}
	for _, family := range gathered {
		if family.GetName() != Namespace+"_stage_duration_seconds" {
			continue
		}
		for _, metric := range family.GetMetric() {
			for _, label := range metric.GetLabel() {
				stages[label.GetValue()] = true
			}
		}
	}
	for _, want := range []string{"retrieval", "rank"} {
		if !stages[want] {
			t.Errorf("no stage_duration_seconds for %q", want)
		}
	}
}

// --- Bucket placement --------------------------------------------------------

// TestTheBudgetFallsInsideTheBuckets is a test for a DESIGN choice.
//
// Prometheus's defaults run 5ms to 10s. Against a 90ms budget that puts nearly
// every request in two buckets, and a p99 interpolated from two buckets is a
// number with no information in it. The alert threshold (§16.1: p99 > 100ms)
// must sit ON a boundary, or the quantile either side of it is an
// interpolation rather than a measurement.
func TestTheBudgetFallsInsideTheBuckets(t *testing.T) {
	for _, want := range []float64{0.090, 0.100} {
		found := false
		for _, bucket := range requestBuckets {
			if bucket == want {
				found = true
			}
		}
		if !found {
			t.Errorf("no request bucket boundary at %vs; the alert threshold interpolates", want)
		}
	}

	// The stage budgets from docs/design.md: 8ms features, 25ms retrieval,
	// 35ms rank. Each needs a boundary for the same reason.
	for _, want := range []float64{0.008, 0.025, 0.035} {
		found := false
		for _, bucket := range stageBuckets {
			if bucket == want {
				found = true
			}
		}
		if !found {
			t.Errorf("no stage bucket boundary at %vs", want)
		}
	}
}

// --- The interceptor ---------------------------------------------------------

func TestTheInterceptorRecordsTheCode(t *testing.T) {
	pipeline, _ := newPipeline(t)
	intercept := pipeline.Interceptor()

	failing := func(context.Context, any) (any, error) {
		return nil, status.Error(codes.InvalidArgument, "no")
	}
	if _, err := intercept(context.Background(), nil, nil, failing); err == nil {
		t.Fatal("the interceptor must not swallow the error")
	}

	if got := testutil.ToFloat64(
		pipeline.requests.WithLabelValues("InvalidArgument"),
	); got != 1 {
		t.Errorf("requests_total{code=InvalidArgument} = %v", got)
	}
}

// TestAnUnknownErrorStillGetsAClosedLabel guards the cardinality rule.
//
// A plain error is not a status; mapping it to its message would create one
// time series per distinct error string, which is unbounded by construction.
func TestAnUnknownErrorStillGetsAClosedLabel(t *testing.T) {
	pipeline, _ := newPipeline(t)
	intercept := pipeline.Interceptor()

	failing := func(context.Context, any) (any, error) {
		return nil, errors.New("something with a unique message 12345")
	}
	_, _ = intercept(context.Background(), nil, nil, failing)

	if got := testutil.ToFloat64(pipeline.requests.WithLabelValues("Unknown")); got != 1 {
		t.Errorf("an unmapped error should land on Unknown, got %v", got)
	}
}

func TestASuccessfulCallIsCountedOK(t *testing.T) {
	pipeline, _ := newPipeline(t)
	intercept := pipeline.Interceptor()

	ok := func(context.Context, any) (any, error) { return "slate", nil }
	got, err := intercept(context.Background(), nil, nil, ok)

	if err != nil || got != "slate" {
		t.Fatalf("the interceptor must pass the response through: %v, %v", got, err)
	}
	if count := testutil.ToFloat64(pipeline.requests.WithLabelValues("OK")); count != 1 {
		t.Errorf("requests_total{code=OK} = %v", count)
	}
}

// --- The scrape endpoint -----------------------------------------------------

func TestTheHandlerExposesTheNamespacedSeries(t *testing.T) {
	pipeline, registry := newPipeline(t)
	pipeline.Request(codes.OK.String(), 10*time.Millisecond)

	body, err := testutil.GatherAndLint(registry)
	if err != nil {
		t.Fatalf("lint: %v", err)
	}
	// GatherAndLint returns the problems it found, not the payload.
	for _, problem := range body {
		t.Errorf("%s: %s", problem.Metric, problem.Text)
	}

	families, err := registry.Gather()
	if err != nil {
		t.Fatalf("Gather: %v", err)
	}
	for _, family := range families {
		if !strings.HasPrefix(family.GetName(), Namespace+"_") {
			t.Errorf("series %q is outside the namespace", family.GetName())
		}
	}
}
