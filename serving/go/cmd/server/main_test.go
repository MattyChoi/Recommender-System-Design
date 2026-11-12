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
// the initial NOT_SERVING. watchReadiness only writes on a TRANSITION, so a
// probe that fails from the very first poll produces no write -- the caller's
// initial status is what makes "not ready yet" correct.
func newHealth() *health.Server {
	server := health.NewServer()
	server.SetServingStatus(readinessService, healthpb.HealthCheckResponse_NOT_SERVING)
	return server
}

func statusOf(
	t *testing.T, server *health.Server, service string,
) healthpb.HealthCheckResponse_ServingStatus {
	t.Helper()
	got, err := server.Check(
		context.Background(), &healthpb.HealthCheckRequest{Service: service})
	if err != nil {
		t.Fatalf("check %q: %v", service, err)
	}
	return got.Status
}

func awaitStatus(
	t *testing.T, server *health.Server, want healthpb.HealthCheckResponse_ServingStatus,
) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for {
		if got := statusOf(t, server, readinessService); got == want {
			return
		} else if time.Now().After(deadline) {
			t.Fatalf("readiness stuck at %v, want %v", got, want)
		}
		time.Sleep(pollPeriod)
	}
}

func TestReadinessTurnsServingOnceTheBackendsAnswer(t *testing.T) {
	server := newHealth()

	var up atomic.Bool
	ready := func(context.Context) error {
		if up.Load() {
			return nil
		}
		return errors.New("still starting")
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go watchReadiness(ctx, server, ready, pollPeriod)

	// A pod whose backends have not answered yet must not be routed to.
	awaitStatus(t, server, healthpb.HealthCheckResponse_NOT_SERVING)

	up.Store(true)
	awaitStatus(t, server, healthpb.HealthCheckResponse_SERVING)
}

func TestReadinessTurnsNotServingWhenTheBackendsGoAway(t *testing.T) {
	server := newHealth()

	var down atomic.Bool
	ready := func(context.Context) error {
		if down.Load() {
			return errors.New("gateway unreachable")
		}
		return nil
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go watchReadiness(ctx, server, ready, pollPeriod)

	awaitStatus(t, server, healthpb.HealthCheckResponse_SERVING)

	down.Store(true)
	awaitStatus(t, server, healthpb.HealthCheckResponse_NOT_SERVING)
}

func TestTheOverallServiceIsNeverFlipped(t *testing.T) {
	// "" is what a LIVENESS probe asks for. It must stay SERVING however badly
	// the backends are doing: a dead Triton that restarted every pod would turn
	// a degraded dependency into a restart loop, which is strictly worse than
	// serving the fallback slate.
	server := newHealth()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go watchReadiness(
		ctx, server,
		func(context.Context) error { return errors.New("everything is down") },
		pollPeriod,
	)

	// Give the poller several periods to do the wrong thing.
	time.Sleep(10 * pollPeriod)

	if got := statusOf(t, server, ""); got != healthpb.HealthCheckResponse_SERVING {
		t.Errorf("overall service %v with the backends down; liveness must not track them", got)
	}
	if got := statusOf(t, server, readinessService); got != healthpb.HealthCheckResponse_NOT_SERVING {
		t.Errorf("readiness %v with the backends down, want NOT_SERVING", got)
	}
}

func TestCancellingStopsThePoller(t *testing.T) {
	// The deferred stopReadiness in run() has to actually end the goroutine, or
	// every restart of the server in one process leaks one.
	server := newHealth()

	var polls atomic.Int64
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		defer close(done)
		watchReadiness(ctx, server, func(context.Context) error {
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
