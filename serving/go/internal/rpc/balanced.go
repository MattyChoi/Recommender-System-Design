// Package rpc holds one thing: several HTTP/2 connections to one or more
// copies of a backend, behind the interface a generated gRPC client already
// takes.
//
// Against ONE Python sidecar process, more connections bought nothing; the
// process was GIL-bound, not connection-bound. What fixed it was more
// processes, each on its own port (serving/processes.py)
package rpc

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"

	"google.golang.org/grpc"
)

// The 500 rps rows are identical within noise
const DefaultConnections = 1

// Balanced spreads calls across several connections, to one or more targets.
type Balanced struct {
	conns []*grpc.ClientConn
	next  atomic.Uint64

	// One per backend, in dial order. conns[i] goes to addresses[i%len], so
	// conns[:len(addresses)] is exactly one connection to each.
	addresses []string
}

// Use Round-Robin scheduling
func (b *Balanced) pick() *grpc.ClientConn {
	if len(b.conns) == 1 {
		return b.conns[0]
	}
	return b.conns[int(b.next.Add(1)%uint64(len(b.conns)))]
}

func (b *Balanced) Invoke(
	ctx context.Context, method string, args, reply any, opts ...grpc.CallOption,
) error {
	return b.pick().Invoke(ctx, method, args, reply, opts...)
}

func (b *Balanced) NewStream(
	ctx context.Context, desc *grpc.StreamDesc, method string, opts ...grpc.CallOption,
) (grpc.ClientStream, error) {
	return b.pick().NewStream(ctx, desc, method, opts...)
}

func (b *Balanced) Close() error {
	var failures []error
	for _, conn := range b.conns {
		if err := conn.Close(); err != nil {
			failures = append(failures, err)
		}
	}
	return errors.Join(failures...)
}

func (b *Balanced) Connections() int { return len(b.conns) }

// Addresses is every backend this pool spreads over, in dial order.
func (b *Balanced) Addresses() []string { return append([]string(nil), b.addresses...) }

// Each runs check once per ADDRESS, concurrently, and joins every failure
// with the address it came from.
//
// A health probe sent through Invoke reaches whichever backend pick lands on,
// so with four sidecar processes it vouches for one of them and says nothing
// about the other three. Concurrent so a probe of N backends costs one
// timeout, not N.
func (b *Balanced) Each(
	ctx context.Context,
	check func(ctx context.Context, address string, conn grpc.ClientConnInterface) error,
) error {
	failures := make([]error, len(b.addresses))
	var wait sync.WaitGroup
	for index, address := range b.addresses {
		wait.Add(1)
		go func() {
			defer wait.Done()
			if err := check(ctx, address, b.conns[index]); err != nil {
				failures[index] = fmt.Errorf("%s: %w", address, err)
			}
		}()
	}
	wait.Wait()
	return errors.Join(failures...)
}

// Dial opens count connections spread EVENLY over the comma-separated
// addresses in target, rounding up so every address gets the same number.
func Dial(target string, count int, opts ...grpc.DialOption) (*Balanced, error) {
	targets := Targets(target)
	if len(targets) == 0 {
		return nil, fmt.Errorf("no target in %q", target)
	}
	perTarget := max((count+len(targets)-1)/len(targets), 1)
	total := perTarget * len(targets)

	balanced := &Balanced{conns: make([]*grpc.ClientConn, 0, total), addresses: targets}
	for index := range total {
		address := targets[index%len(targets)]
		conn, err := grpc.NewClient(address, opts...)
		if err != nil {
			_ = balanced.Close()
			return nil, fmt.Errorf("dialling %s (connection %d of %d): %w",
				address, index+1, total, err)
		}
		balanced.conns = append(balanced.conns, conn)
	}
	return balanced, nil
}

// Targets splits a comma-separated target list, dropping blanks and spaces.
func Targets(target string) []string {
	var out []string
	for _, part := range strings.Split(target, ",") {
		if part = strings.TrimSpace(part); part != "" {
			out = append(out, part)
		}
	}
	return out
}

// Compile-time proof
var _ grpc.ClientConnInterface = (*Balanced)(nil)
