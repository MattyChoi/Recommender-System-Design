// Package sources holds the concrete retrievers the orchestrator fans out to.
//
// Separate from internal/retrieval on purpose, and not only to break an import
// cycle. That package is a pure algorithm -- the blend, mirrored by
// models/ranking/dataset.py:blend_one and pinned by generated parity fixtures.
// These are the I/O: gRPC to the retrieval sidecar, Redis for the precomputed
// lists. Keeping a function that must agree with Python byte for byte in the
// same package as a network client makes neither easier to reason about.
package sources

import (
	"context"
	"fmt"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "github.com/MattyChoi/Recommender-System-Design/serving/go/internal/pb"
	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/service"
)

// Sidecar is the two-tower source: a gRPC client for the Python process that
// holds the FAISS index and the user encoder.
//
// ADR 0013 records why it is a process and not a library. The short version is
// that a FAISS index is a C++ artifact this binary cannot read and `encode_user`
// is a PyTorch forward pass it cannot run -- and that making the strongest
// retriever a REMOTE call is what gives it a budget it can miss, which is the
// behaviour the whole fan-out is designed around. An in-process matmul cannot
// time out.
type Sidecar struct {
	client pb.RetrievalClient
	conn   *grpc.ClientConn

	// SourceName is what this source is called in quotas, degraded_sources and
	// the per-item attribution. Configurable because Config.Quotas is
	// POSITIONAL and the blend's source order has to line up with it.
	SourceName string

	// Deadline is this source's own budget.
	Deadline time.Duration

	// K is how many candidates to ask for.
	K int

	// EFSearch overrides the server's default when non-zero. ADR 0002 ships
	// 512 over the throughput-optimal 128 because that is where the recall
	// cost stopped reproducing across checkpoints; left at 0, the server's
	// configured value applies and the response says which.
	EFSearch int32

	// Fallback runs when the sidecar cannot be reached. Nil means this source
	// simply degrades. See ADR 0013's ladder: rung 2 is exact search in this
	// process against a cached user embedding.
	Fallback Searcher

	// What the sidecar last said answered, for the health endpoint -- so a
	// server that quietly failed over to exact search is one call away from
	// being visible.
	//
	// Behind a mutex, because ONE Sidecar serves every concurrent request: the
	// orchestrator holds it as a singleton and the fan-out calls it from a
	// goroutine per request. Written as plain fields these are a data race
	// that `go test -race` reports and a release build simply loses.
	mu          sync.RWMutex
	lastKind    string
	lastVersion string
}

// LastIndex reports what most recently answered: "hnsw", "flat", or empty
// before the first call.
func (s *Sidecar) LastIndex() (kind string, version string) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.lastKind, s.lastVersion
}

func (s *Sidecar) setLastIndex(kind, version string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lastKind, s.lastVersion = kind, version
}

// Searcher is rung 2: exact search in this process. Satisfied by
// internal/index.Flat paired with a cached user embedding.
type Searcher interface {
	// Search returns candidates for a user whose embedding is cached. A
	// cache miss is not an error -- it means rung 2 is unavailable for this
	// user and the pipeline falls to popularity.
	Search(ctx context.Context, userID string, k int) ([]int32, bool, error)
}

// DialSidecar opens a connection without blocking on the server being up.
// gRPC reconnects on its own, and a serving process that refuses to start
// because a dependency is briefly down turns a blip into an outage.
func DialSidecar(
	target, name string, k int, deadline time.Duration, opts ...grpc.DialOption,
) (*Sidecar, error) {
	// Caller options LAST, so a caller can override the transport credentials
	// rather than silently having them re-set beneath it.
	opts = append([]grpc.DialOption{
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	}, opts...)
	conn, err := grpc.NewClient(target, opts...)
	if err != nil {
		return nil, fmt.Errorf("dialling retrieval sidecar at %s: %w", target, err)
	}
	return &Sidecar{
		client:     pb.NewRetrievalClient(conn),
		conn:       conn,
		SourceName: name,
		Deadline:   deadline,
		K:          k,
	}, nil
}

func (s *Sidecar) Close() error { return s.conn.Close() }

func (s *Sidecar) Name() string { return s.SourceName }

func (s *Sidecar) Budget() time.Duration { return s.Deadline }

// Retrieve asks the sidecar, falling back to in-process exact search.
func (s *Sidecar) Retrieve(
	ctx context.Context, user service.User,
) (service.Candidates, error) {
	response, err := s.client.Retrieve(ctx, &pb.RetrieveRequest{
		UserId: user.ID,
		K:      int32(s.K),
		// Sent, not fetched here. The orchestrator's pre-fan-out call already
		// paid for these, and a second read could see a different snapshot
		// than the one the ranker's own features came from.
		UserFeats: user.Feats,
		History:   user.History,
		EfSearch:  s.EFSearch,
	})
	if err != nil {
		return s.degrade(ctx, user, err)
	}

	// Recorded even on success, because the interesting case IS a success: a
	// sidecar that fell back to exact search internally is correct and much
	// slower, and says so only here.
	s.setLastIndex(response.GetIndexKind(), response.GetIndexVersion())

	items := response.GetItems()
	scores := response.GetScores()
	similarity := response.GetContentSimilarity()
	// Three arrays the sidecar builds together and the wire carries
	// separately. Checked rather than trusted: a short array pairs each item
	// with the NEXT item's score from that point on, and produces a complete,
	// plausible, wrong slate. The sidecar filters its OOV rows before
	// computing these for exactly this reason, so a mismatch here means the
	// two sides are out of step and the request should degrade rather than
	// carry on with silently shifted columns.
	if len(scores) != len(items) || len(similarity) != len(items) {
		return s.degrade(ctx, user, fmt.Errorf(
			"%d items, %d scores, %d similarities", len(items), len(scores), len(similarity),
		))
	}

	return service.Candidates{Items: items, Scores: scores, Similarity: similarity}, nil
}

// degrade is rung 2 and rung 3 of ADR 0013's ladder.
//
// The original error is returned when the fallback cannot answer, rather than
// the fallback's own. Whoever reads the log needs to know the SIDECAR failed;
// "no cached embedding" is a consequence, and reporting it as the cause sends
// them to Redis to debug a gRPC outage.
func (s *Sidecar) degrade(
	ctx context.Context, user service.User, cause error,
) (service.Candidates, error) {
	if s.Fallback == nil {
		return service.Candidates{}, fmt.Errorf("retrieval sidecar: %w", cause)
	}

	items, ok, err := s.Fallback.Search(ctx, user.ID, s.K)
	if err != nil || !ok {
		return service.Candidates{}, fmt.Errorf("retrieval sidecar: %w", cause)
	}
	// Flat, and said so: the health endpoint reports index_kind, and a box
	// serving correct results without its index is invisible in every other
	// signal. This is also a REAL degradation even though it returns
	// candidates -- benchmarks.md measures it at 3.9ms of CPU against 500 QPS
	// on a 2-core pod, so it is survivable for a request and not for a peak.
	// Version deliberately blanked, not carried over: the last version the
	// SIDECAR reported says nothing about what this process just searched, and
	// a health endpoint claiming an index version it did not use is worse than
	// one admitting it does not know.
	s.setLastIndex("flat", "")
	// Ids only. Rung 2 has the item embeddings but neither the content table
	// nor the tower's view of this user, so it cannot produce the two
	// model-derived columns -- and inventing them would be worse than their
	// absence. The builder sees them missing and says so.
	return service.Candidates{Items: items}, nil
}

// Compile-time proof this satisfies the port the orchestrator fans out over.
var _ service.Retriever = (*Sidecar)(nil)
