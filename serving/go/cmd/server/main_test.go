package main

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/grpc/health"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"
)

// pollPeriod is deliberately tiny: these tests are about transitions, not about
// the production cadence.
const pollPeriod = 5 * time.Millisecond

// newHealth mirrors what run() publishes before starting the poller, including
// both initial NOT_SERVINGs. watchUpstreams only writes on a TRANSITION, so a
// probe that fails from the very first poll produces no write -- the caller's
// initial status is what makes "not up yet" correct.
func newHealth() *health.Server {
	server := health.NewServer()
	server.SetServingStatus(readinessService, healthpb.HealthCheckResponse_NOT_SERVING)
	server.SetServingStatus(upstreamsService, healthpb.HealthCheckResponse_NOT_SERVING)
	return server
}

func statusOf(
	t *testing.T, server *health.Server, service string,
) healthpb.HealthCheckResponse_ServingStatus {
	t.Helper()
	got, err := server.Check(context.Background(), &healthpb.HealthCheckRequest{Service: service})
	if err != nil {
		t.Fatalf("check %q: %v", service, err)
	}
	return got.Status
}

func awaitStatus(
	t *testing.T, server *health.Server, service string,
	want healthpb.HealthCheckResponse_ServingStatus,
) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for {
		if got := statusOf(t, server, service); got == want {
			return
		} else if time.Now().After(deadline) {
			t.Fatalf("%q stuck at %v, want %v", service, got, want)
		}
		time.Sleep(pollPeriod)
	}
}

func watch(t *testing.T, server *health.Server, ready func(context.Context) error) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go watchUpstreams(ctx, server, upstreamsService, ready, pollPeriod)
}

func TestUpstreamsTurnServingOnceTheBackendsAnswer(t *testing.T) {
	server := newHealth()

	var up atomic.Bool
	watch(t, server, func(context.Context) error {
		if up.Load() {
			return nil
		}
		return errors.New("still starting")
	})

	awaitStatus(t, server, upstreamsService, healthpb.HealthCheckResponse_NOT_SERVING)
	up.Store(true)
	awaitStatus(t, server, upstreamsService, healthpb.HealthCheckResponse_SERVING)
}

func TestUpstreamsTurnNotServingWhenTheBackendsGoAway(t *testing.T) {
	server := newHealth()

	var down atomic.Bool
	watch(t, server, func(context.Context) error {
		if down.Load() {
			return errors.New("gateway unreachable")
		}
		return nil
	})

	awaitStatus(t, server, upstreamsService, healthpb.HealthCheckResponse_SERVING)
	down.Store(true)
	awaitStatus(t, server, upstreamsService, healthpb.HealthCheckResponse_NOT_SERVING)
}

func TestADeadBackendChangesNothingThatRoutes(t *testing.T) {
	// **The property the whole split exists for.**
	//
	// A pod with every backend down still answers from popular.json. If a dead
	// Triton flipped readiness, kubelet would pull every pod from the Service's
	// endpoints at once and a service that could still answer would return
	// nothing -- a degraded dependency promoted to a total outage. And if it
	// flipped liveness, the same outage would also restart every pod.
	//
	// So the upstream verdict must move and the two routing signals must not.
	server := newHealth()
	server.SetServingStatus(readinessService, healthpb.HealthCheckResponse_SERVING)

	watch(t, server, func(context.Context) error { return errors.New("everything is down") })

	awaitStatus(t, server, upstreamsService, healthpb.HealthCheckResponse_NOT_SERVING)

	// Several more periods to do the wrong thing in.
	time.Sleep(10 * pollPeriod)

	if got := statusOf(t, server, readinessService); got != healthpb.HealthCheckResponse_SERVING {
		t.Errorf("readiness %v with the backends down; it must not route on their health", got)
	}
	if got := statusOf(t, server, ""); got != healthpb.HealthCheckResponse_SERVING {
		t.Errorf("overall service %v with the backends down; liveness must not track them", got)
	}
}

func TestCancellingStopsThePoller(t *testing.T) {
	// The deferred stopUpstreams in run() has to actually end the goroutine, or
	// every restart of the server in one process leaks one.
	server := newHealth()

	var polls atomic.Int64
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		defer close(done)
		watchUpstreams(ctx, server, upstreamsService, func(context.Context) error {
			polls.Add(1)
			return nil
		}, pollPeriod)
	}()

	deadline := time.Now().Add(2 * time.Second)
	for polls.Load() < 2 {
		if time.Now().After(deadline) {
			t.Fatalf("the poller ran %d times; it is not polling", polls.Load())
		}
		time.Sleep(pollPeriod)
	}

	cancel()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("the poller did not return when its context was cancelled")
	}
}

// --- the flags reaching the pipeline ----------------------------------------

// wired sets every rerank knob to something that is NOT its zero value, so a
// field that stops being copied shows up as a zero rather than passing by
// coincidence.
func wired() options {
	return options{
		candidates:     400,
		deadline:       90 * time.Millisecond,
		rankDeadline:   35 * time.Millisecond,
		featDeadline:   8 * time.Millisecond,
		lambda:         0.7,
		capPerCategory: 2,
		epsilon:        0.25,
		exploreSlots:   3,
		modelVersion:   "model-abc",
		indexVersion:   "index-def",
	}
}

// TestFlagsReachTheConfig is the regression test for the defect it was written
// after: -epsilon, -explore-slots, -cap-per-category and -lambda were parsed,
// advertised in -h, and never assigned. The reranker ran with exploration off
// and no category cap, nothing failed, no metric moved, and every logged
// propensity was 1.0 -- which makes a log that CANNOT support off-policy
// evaluation look exactly like one that can.
func TestFlagsReachTheConfig(t *testing.T) {
	o := wired()
	config, err := pipelineConfig(o)
	if err != nil {
		t.Fatalf("pipelineConfig: %v", err)
	}

	if config.Lambda != o.lambda {
		t.Errorf("Lambda = %v, want %v (the flag, not a literal)", config.Lambda, o.lambda)
	}
	if config.Epsilon != o.epsilon {
		t.Errorf("Epsilon = %v, want %v; exploration would never fire", config.Epsilon, o.epsilon)
	}
	if config.ExploreSlots != o.exploreSlots {
		t.Errorf("ExploreSlots = %d, want %d", config.ExploreSlots, o.exploreSlots)
	}
	if config.CapPerCategory == nil {
		t.Fatal("CapPerCategory is nil; the per-subcategory cap would be off")
	}
	if *config.CapPerCategory != o.capPerCategory {
		t.Errorf("CapPerCategory = %d, want %d", *config.CapPerCategory, o.capPerCategory)
	}
	if config.MaxCandidates != o.candidates || config.RankDeadline != o.rankDeadline {
		t.Errorf("the budgets did not survive either: %+v", config)
	}
	if config.ModelVersion != o.modelVersion || config.IndexVersion != o.indexVersion {
		t.Errorf("versions = %q/%q, want %q/%q",
			config.ModelVersion, config.IndexVersion, o.modelVersion, o.indexVersion)
	}
}

// -1 disables the cap; 0 does not. Zero means "no slots for any subcategory",
// which is catastrophic and must stay expressible rather than being folded
// into the disabled case by a truthiness check.
func TestCapPerCategoryDistinguishesDisabledFromZero(t *testing.T) {
	for _, testCase := range []struct {
		name  string
		flag  int
		want  int
		isNil bool
	}{
		{name: "disabled", flag: -1, isNil: true},
		{name: "zero is a real setting", flag: 0, want: 0},
		{name: "ordinary", flag: 3, want: 3},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			o := wired()
			o.capPerCategory = testCase.flag
			config, err := pipelineConfig(o)
			if err != nil {
				t.Fatalf("pipelineConfig: %v", err)
			}
			if testCase.isNil {
				if config.CapPerCategory != nil {
					t.Fatalf("cap = %d, want nil", *config.CapPerCategory)
				}
				return
			}
			if config.CapPerCategory == nil {
				t.Fatal("cap is nil, want a value")
			}
			if *config.CapPerCategory != testCase.want {
				t.Errorf("cap = %d, want %d", *config.CapPerCategory, testCase.want)
			}
		})
	}
}

// Refused at STARTUP rather than once per request. rerank.Select returns this
// error too, but by then it is a 100% error rate on a process that started
// cleanly and whose every other signal looks healthy.
func TestExplorationWithNoSlotsIsRefusedAtStartup(t *testing.T) {
	o := wired()
	o.exploreSlots = 0
	if _, err := pipelineConfig(o); err == nil {
		t.Fatal("a positive epsilon with no explore slots should be refused")
	}

	o.epsilon = 0
	if _, err := pipelineConfig(o); err != nil {
		t.Fatalf("epsilon 0 with no slots is the OFF setting, not an error: %v", err)
	}
}
