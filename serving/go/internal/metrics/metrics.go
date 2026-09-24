// Package metrics is what makes the pipeline's degradation visible.
//
// Every stage of this orchestrator is built to degrade rather than fail, which
// is the right behaviour and has one consequence: **a degraded service returns
// 200s.** The retriever that timed out, the ranker that missed its deadline,
// the feature gateway that could not be reached, the filter that blocked every
// candidate -- all of them produce a well-formed response. Without these
// counters the service looks healthy while serving the retrieval order to
// everybody.
//
// # Bucket choice is the whole game
//
// Prometheus's default histogram buckets run from 5ms to 10s. Against a 90ms
// budget that puts essentially every request in two buckets, and a p99
// interpolated from two buckets is a number with no information in it. The
// buckets here are placed around the budgets `docs/design.md` sets, so the
// quantiles say something about the thing being alerted on.
//
// # Cardinality
//
// No user id, no item id, no request id -- ever. A label with unbounded values
// creates one time series per value, and a recommender has as many values as
// it has traffic. The labels here are all closed sets: stage names, source
// names, gRPC codes, experiment arms.
package metrics

import (
	"time"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/service"
)

// Namespace prefixes every series.
const Namespace = "recsys"

// requestBuckets bracket docs/design.md's 90ms end-to-end budget.
//
// Dense either side of 90ms, because that is where the alert threshold sits
// (§16.1: p99 > 100ms for 5 minutes) and a quantile is only as precise as the
// bucket it lands in. Sparse in the tail, where the question is no longer "how
// slow" but "how many".
var requestBuckets = []float64{
	0.005, 0.010, 0.020, 0.030, 0.040, 0.050, 0.060, 0.070,
	0.080, 0.090, 0.100, 0.120, 0.150, 0.200, 0.500, 1.0,
}

// stageBuckets are finer and shorter: the stage budgets are 8ms (features),
// 25ms (retrieval), 35ms (rank) and 5ms (re-rank), so the interesting range is
// an order of magnitude below the request's.
var stageBuckets = []float64{
	0.001, 0.002, 0.004, 0.006, 0.008, 0.012, 0.016, 0.020,
	0.025, 0.035, 0.050, 0.075, 0.100, 0.250,
}

// candidateBuckets span the funnel: ~400 candidates in, ~10 out.
var candidateBuckets = []float64{0, 1, 5, 10, 25, 50, 100, 200, 400, 600, 1000}

// Pipeline holds the collectors the orchestrator feeds.
type Pipeline struct {
	requests    *prometheus.CounterVec
	duration    prometheus.Histogram
	stage       *prometheus.HistogramVec
	candidates  *prometheus.HistogramVec
	degraded    *prometheus.CounterVec
	fallbacks   prometheus.Counter
	incomplete  prometheus.Counter
	assignments *prometheus.CounterVec
	scores      prometheus.Histogram
}

// New registers the collectors and returns the recorder.
func New(registry prometheus.Registerer) *Pipeline {
	p := &Pipeline{
		requests: prometheus.NewCounterVec(prometheus.CounterOpts{
			Namespace: Namespace,
			Name:      "requests_total",
			Help:      "Recommendation requests by gRPC status code.",
		}, []string{"code"}),

		duration: prometheus.NewHistogram(prometheus.HistogramOpts{
			Namespace: Namespace,
			Name:      "request_duration_seconds",
			Help:      "End-to-end latency, server-measured.",
			Buckets:   requestBuckets,
		}),

		stage: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Namespace: Namespace,
			Name:      "stage_duration_seconds",
			Help:      "Per-stage latency. A single end-to-end number says a request was slow; this says which stage was.",
			Buckets:   stageBuckets,
		}, []string{"stage"}),

		candidates: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Namespace: Namespace,
			Name:      "stage_candidates",
			Help:      "Candidates surviving each stage. after_filter collapsing to zero is how a saturated seen-list is caught in production rather than in a postmortem.",
			Buckets:   candidateBuckets,
		}, []string{"stage"}),

		degraded: prometheus.NewCounterVec(prometheus.CounterOpts{
			Namespace: Namespace,
			Name:      "degraded_total",
			Help:      "Requests in which a named dependency degraded. Every one of these returned 200.",
		}, []string{"source"}),

		fallbacks: prometheus.NewCounter(prometheus.CounterOpts{
			Namespace: Namespace,
			Name:      "fallback_total",
			Help:      "Requests served by the popularity fallback. §16.1 alerts past 5%.",
		}),

		incomplete: prometheus.NewCounter(prometheus.CounterOpts{
			Namespace: Namespace,
			Name:      "incomplete_rows_total",
			Help:      "Candidate rows scored with zero-filled model columns, i.e. retrieval on rung 2.",
		}),

		assignments: prometheus.NewCounterVec(prometheus.CounterOpts{
			Namespace: Namespace,
			Name:      "experiment_assignments_total",
			Help:      "Assignments per arm. §17.2: check the ratio FIRST -- an SRM invalidates the experiment whatever the primary metric did.",
		}, []string{"variant"}),

		// The prediction distribution, for PSI. §16.1 alerts past 0.25.
		//
		// A histogram rather than a summary, because PSI is computed BETWEEN
		// two distributions and needs the same bucket edges on both sides.
		// Quantiles cannot be re-bucketed after the fact; counts can.
		scores: prometheus.NewHistogram(prometheus.HistogramOpts{
			Namespace: Namespace,
			Name:      "served_score",
			Help:      "Ranker score of served items. The input to the PSI drift check.",
			Buckets:   prometheus.LinearBuckets(-10, 1, 21),
		}),
	}

	registry.MustRegister(
		p.requests, p.duration, p.stage, p.candidates,
		p.degraded, p.fallbacks, p.incomplete, p.assignments, p.scores,
	)
	return p
}

// Request records a completed call by gRPC status code and total latency.
//
// Counted by CODE rather than as a separate error counter, so the error rate
// §16.1 alerts on has a denominator that cannot drift from its numerator.
func (p *Pipeline) Request(code string, elapsed time.Duration) {
	p.requests.WithLabelValues(code).Inc()
	p.duration.Observe(elapsed.Seconds())
}

// Observed records everything the pipeline reported about one result.
//
// Takes the Result rather than the protobuf: the response carries stage
// latencies truncated to whole milliseconds, and a 5ms re-rank budget measured
// in whole milliseconds has four useful values.
func (p *Pipeline) Observed(result service.Result, variant string) {
	for stage, elapsed := range result.StageLatency {
		p.stage.WithLabelValues(stage).Observe(elapsed.Seconds())
	}
	for stage, count := range result.StageCounts {
		p.candidates.WithLabelValues(stage).Observe(float64(count))
	}
	// One increment per REQUEST per source, not per occurrence: the alert is
	// "what share of traffic is degraded", and a source that degrades twice in
	// one request is still one degraded request.
	for _, source := range result.DegradedSources {
		p.degraded.WithLabelValues(source).Inc()
	}
	if result.UsedFallback {
		p.fallbacks.Inc()
	}
	if count := result.StageCounts["incomplete_rows"]; count > 0 {
		p.incomplete.Add(float64(count))
	}
	// The empty variant is counted too, under "none". A holdback that is not
	// counted makes the SRM check compare arms against a denominator it
	// cannot see.
	arm := variant
	if arm == "" {
		arm = "none"
	}
	p.assignments.WithLabelValues(arm).Inc()

	for _, score := range result.Scores {
		p.scores.Observe(score)
	}
}
