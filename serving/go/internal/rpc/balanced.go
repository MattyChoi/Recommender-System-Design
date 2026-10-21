// Package rpc holds one thing: several HTTP/2 connections to one backend,
// behind the interface a generated gRPC client already takes.
//
// ⚠️ AND IT DID NOT WORK.
package rpc

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"

	"google.golang.org/grpc"
)

// DefaultConnections is ONE, because four was measured and bought nothing.
//
//	rps    -conns 1    -conns 4
//	300      0.02%       0.02%
//	400      0.68%       0.08%
//	500     50.10%      49.60%
//
// The 500 rps rows are identical within noise
const DefaultConnections = 1

// Balanced spreads calls across several connections to the SAME target.
type Balanced struct {
	conns []*grpc.ClientConn
	next  atomic.Uint64
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

func Dial(target string, count int, opts ...grpc.DialOption) (*Balanced, error) {
	if count < 1 {
		count = 1
	}

	balanced := &Balanced{conns: make([]*grpc.ClientConn, 0, count)}
	for index := range count {
		conn, err := grpc.NewClient(target, opts...)
		if err != nil {
			_ = balanced.Close()
			return nil, fmt.Errorf("dialling %s (connection %d of %d): %w",
				target, index+1, count, err)
		}
		balanced.conns = append(balanced.conns, conn)
	}
	return balanced, nil
}

// Compile-time proof
var _ grpc.ClientConnInterface = (*Balanced)(nil)
