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
