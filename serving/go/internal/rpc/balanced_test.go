package rpc

import (
	"context"
	"net"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials/insecure"
)

func targetsOf(t *testing.T, target string, count int) []string {
	t.Helper()
	// NewClient does not connect, so nothing needs to be listening.
	balanced, err := Dial(target, count, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	t.Cleanup(func() { _ = balanced.Close() })

	got := make([]string, len(balanced.conns))
	for index, conn := range balanced.conns {
		got[index] = conn.Target()
	}
	return got
}

func TestOneTargetKeepsTheOldBehaviour(t *testing.T) {
	got := targetsOf(t, "localhost:15052", 3)
	if len(got) != 3 {
		t.Fatalf("%d connections, want 3", len(got))
	}
	for _, target := range got {
		if target != "localhost:15052" {
			t.Errorf("connection to %q", target)
		}
	}
}

// TestEveryTargetGetsTheSameShare is the point of the multi-target form:
// SO_REUSEPORT on one port left a 6/2/0/0 split across four processes.
func TestEveryTargetGetsTheSameShare(t *testing.T) {
	// 4 over 3 does not divide; rounding DOWN would starve one backend.
	got := targetsOf(t, "a:1, b:2,c:3,", 4)

	want := []string{"a:1", "b:2", "c:3", "a:1", "b:2", "c:3"}
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	// Interleaved, so consecutive round-robin picks alternate backends.
	for index := range want {
		if got[index] != want[index] {
			t.Fatalf("got %v, want %v", got, want)
		}
	}
}

func TestMoreTargetsThanConnections(t *testing.T) {
	if got := targetsOf(t, "a:1,b:2,c:3", 1); len(got) != 3 {
		t.Errorf("got %v: every target needs at least one connection", got)
	}
}

func TestAnEmptyTargetIsRefused(t *testing.T) {
	if _, err := Dial(" , ", 4); err == nil {
		t.Error("expected an error for a target list with no addresses")
	}
}

// TestConnectAllOpensEveryConnection: Each touches one connection per address,
// so a 2-per-address pool would leave half its connections unopened.
func TestConnectAllOpensEveryConnection(t *testing.T) {
	var targets []string
	for range 2 {
		listener, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			t.Fatalf("listen: %v", err)
		}
		server := grpc.NewServer()
		go func() { _ = server.Serve(listener) }()
		t.Cleanup(server.Stop)
		targets = append(targets, listener.Addr().String())
	}

	balanced, err := Dial(targets[0]+","+targets[1], 4,
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	t.Cleanup(func() { _ = balanced.Close() })

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := balanced.ConnectAll(ctx); err != nil {
		t.Fatalf("ConnectAll: %v", err)
	}
	for index, conn := range balanced.conns {
		if state := conn.GetState(); state != connectivity.Ready {
			t.Errorf("connection %d to %s is %v, want READY", index, conn.Target(), state)
		}
	}
}

func TestConnectAllGivesUpWithTheContext(t *testing.T) {
	// Bound then closed, so nothing is listening on it.
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	dead := listener.Addr().String()
	_ = listener.Close()

	balanced, err := Dial(dead, 2, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	t.Cleanup(func() { _ = balanced.Close() })

	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()
	if err := balanced.ConnectAll(ctx); err == nil {
		t.Fatal("ConnectAll to a dead address returned nil")
	}
}
