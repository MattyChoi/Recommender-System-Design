// Package rpc holds one thing: several HTTP/2 connections to one backend,
// behind the interface a generated gRPC client already takes.
//
// **Why this exists, measured rather than assumed.** A CPU profile of the
// orchestrator at 500 rps put `transport.(*loopyWriter).run` at 30.5% of all
// samples, with everything beneath it being `bufWriter.Flush` ->
// `net.conn.Write` -> `syscall.write`. Another ~21% sat in `runtime.futex`
// under `notewakeup`/`wakep`, and `runtime.findRunnable` took 16.8%. Almost
// nothing was application code: no marshalling in the top 25, GC under 5%, and
// none of the blend, feature-building or re-ranking paths visible at all.
//
// `loopyWriter` is gRPC-Go's SINGLE WRITER GOROUTINE PER HTTP/2 CONNECTION.
// One `grpc.ClientConn` therefore funnels every outbound frame for every
// concurrent request through one goroutine issuing one write syscall at a
// time. The process was using well under half a core while the pipeline missed
// 25 ms stage deadlines and the machine sat 76% idle -- a serialisation limit,
// not a capacity one.
//
// **Why a ClientConnInterface wrapper and not a pool of typed clients.**
// `pb.NewRetrievalClient` and friends take a `grpc.ClientConnInterface`, which
// is two methods. Implementing it means every generated client, every call
// site and every existing fake keeps working unchanged -- the alternative, a
// generic pool handing out typed clients, would have rewritten three Dial
// signatures and the tests that construct those structs directly.
//
// ⚠️ **AND IT DID NOT WORK.** The A/B is in DefaultConnections below: four
// connections changed the degraded share by less than run-to-run variance. The
// writer was a symptom of the wakeup churn, not its cause. This package is kept
// because the flag makes that negative result reproducible and because the
// remedy is real on a topology with genuine network latency -- but on loopback
// it is a no-op, and it defaults to off.
//
// What the measurement leaves standing: the ceiling is per-request RPC
// round-trip latency, four hops deep. Only fewer hops, cheaper hops (unix
// sockets) or less per-hop work can move it.
package rpc

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"

	"google.golang.org/grpc"
)

// DefaultConnections is how many HTTP/2 connections a backend gets.
//
// **ONE, because four was measured and bought nothing.** The A/B, same box,
// same session, degraded share of requests:
//
//	rps    -conns 1    -conns 4
//	300      0.02%       0.02%
//	400      0.68%       0.08%
//	500     50.10%      49.60%
//
// The 500 rps rows are identical within noise; at 400 both are clean, and
// run-to-run variance there is around half a percent, which covers the whole
// difference. So this package does not ship enabled: four sockets, four
// goroutines and four flow-control windows per backend for no measured gain is
// the kind of complexity the rest of this codebase argues against.
//
// **Why it did not work, which is the useful part.** Parallelising a writer
// helps when the writer is a THROUGHPUT bottleneck. It was not. Four writers
// issue the same number of write syscalls and trigger the same number of
// goroutine wakeups as one -- the work is spread across more goroutines on a
// box that already had 23 idle cores. What each request waits on is unchanged:
// per-hop syscall plus scheduler wakeup latency, four hops deep. loopyWriter
// dominated the profile because that is where the syscalls are ISSUED, not
// because anything was queueing behind it.
//
// The code stays because the flag makes the negative result reproducible, and
// because a topology with real network latency and TLS -- where a connection's
// writer genuinely can saturate -- would want this. On loopback it is a no-op.
const DefaultConnections = 1

// Balanced spreads calls across several connections to the SAME target.
//
// Not load balancing in the usual sense -- there is one server. It is writer
// balancing: N connections mean N loopyWriter goroutines, so concurrent calls
// are no longer queued behind one of them.
type Balanced struct {
	conns []*grpc.ClientConn
	next  atomic.Uint64
}

// Dial opens `count` connections to `target`.
//
// Like grpc.NewClient, it does NOT block on the server being up: gRPC
// reconnects on its own, and a serving process that refuses to start because a
// dependency is briefly down turns a blip into an outage. See
// cmd/server's warmConnections for why the first call still costs setup, and
// what is done about it.
func Dial(target string, count int, opts ...grpc.DialOption) (*Balanced, error) {
	if count < 1 {
		count = 1
	}
	balanced := &Balanced{conns: make([]*grpc.ClientConn, 0, count)}
	for index := range count {
		conn, err := grpc.NewClient(target, opts...)
		if err != nil {
			// Close what opened, so a partial failure does not leak sockets.
			_ = balanced.Close()
			return nil, fmt.Errorf("dialling %s (connection %d of %d): %w",
				target, index+1, count, err)
		}
		balanced.conns = append(balanced.conns, conn)
	}
	return balanced, nil
}

// pick is round-robin, which is the right policy precisely BECAUSE the
// connections are interchangeable: same target, same server, no state on
// either side that would make one a better choice than another. Anything
// cleverer would be measuring something that does not vary.
//
// Atomic rather than a mutex: this is on the hot path of every RPC, and the
// only shared state is one counter. Wrapping at 2^64 is not a concern, and an
// occasional duplicate index under contention costs nothing -- the pick does
// not have to be a perfect rotation, only spread.
func (b *Balanced) pick() *grpc.ClientConn {
	if len(b.conns) == 1 {
		return b.conns[0]
	}
	return b.conns[int(b.next.Add(1)%uint64(len(b.conns)))]
}

// Invoke satisfies grpc.ClientConnInterface for unary calls.
func (b *Balanced) Invoke(
	ctx context.Context, method string, args, reply any, opts ...grpc.CallOption,
) error {
	return b.pick().Invoke(ctx, method, args, reply, opts...)
}

// NewStream satisfies grpc.ClientConnInterface for streaming calls.
//
// Unused today -- every RPC in this system is unary -- but the interface
// requires it, and silently failing here would be a trap for whoever adds the
// first stream.
func (b *Balanced) NewStream(
	ctx context.Context, desc *grpc.StreamDesc, method string, opts ...grpc.CallOption,
) (grpc.ClientStream, error) {
	return b.pick().NewStream(ctx, desc, method, opts...)
}

// Close shuts every connection, reporting all failures rather than the first.
func (b *Balanced) Close() error {
	var failures []error
	for _, conn := range b.conns {
		if err := conn.Close(); err != nil {
			failures = append(failures, err)
		}
	}
	return errors.Join(failures...)
}

// Connections reports how many are open, for the startup log.
func (b *Balanced) Connections() int { return len(b.conns) }

// Compile-time proof this is what a generated client accepts.
var _ grpc.ClientConnInterface = (*Balanced)(nil)
