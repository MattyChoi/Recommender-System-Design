package sources

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "github.com/MattyChoi/Recommender-System-Design/serving/go/internal/pb"
	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/rpc"
	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/service"
)

type Searcher interface {
	Search(ctx context.Context, userID string, k int) ([]int32, bool, error)
}

// DefaultFallbackDeadline bounds the exact search fallback: one Redis GET plus
// an in-process exact search, both well under a millisecond when healthy.
const DefaultFallbackDeadline = 5 * time.Millisecond

type ANNIndex struct {
	client pb.RetrievalClient
	conn   *rpc.Balanced

	SourceName string
	Deadline   time.Duration
	K          int
	EFSearch   int32

	Fallback Searcher

	// FallbackDeadline is the exact search's OWN deadline, detached from the
	// call's. Zero means DefaultRescueBudget.
	FallbackDeadline time.Duration

	// What the ANNIndex last said answered. Behind a mutex because ONE instance
	// serves every concurrent request
	mu          sync.RWMutex
	lastKind    string
	lastVersion string
}

func DialIndex(target, name string, k, conns int, efSearch int32, deadline time.Duration, opts ...grpc.DialOption) (*ANNIndex, error) {
	opts = append([]grpc.DialOption{
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	}, opts...)
	pool, err := rpc.Dial(target, conns, opts...)
	if err != nil {
		return nil, fmt.Errorf("dialling FAISS index sidecar at %s: %w", target, err)
	}
	return &ANNIndex{
		client:     pb.NewRetrievalClient(pool),
		conn:       pool,
		SourceName: name,
		Deadline:   deadline,
		K:          k,
		EFSearch:   efSearch,
	}, nil
}

func (s *ANNIndex) Name() string          { return s.SourceName }
func (s *ANNIndex) Budget() time.Duration { return s.Deadline }

// LastIndex reports what most recently answered: "hnsw", "flat", or empty
// before the first call.
func (s *ANNIndex) LastIndex() (kind string, version string) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.lastKind, s.lastVersion
}

func (s *ANNIndex) setLastIndex(kind, version string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lastKind, s.lastVersion = kind, version
}

func (s *ANNIndex) Retrieve(
	ctx context.Context, user service.User,
) (service.Candidates, error) {
	response, err := s.client.Retrieve(ctx, &pb.RetrieveRequest{
		UserId:          user.ID,
		K:               int32(s.K),
		UserFeats:       user.Feats,
		History:         user.History,
		HasUserFeatures: user.Found,
		EfSearch:        s.EFSearch,
	})
	if err != nil {
		return s.degrade(ctx, user, err)
	}
	s.setLastIndex(response.GetIndexKind(), response.GetIndexVersion())

	items := response.GetItems()
	scores := response.GetScores()
	similarity := response.GetContentSimilarity()

	if len(scores) != len(items) || len(similarity) != len(items) {
		return s.degrade(ctx, user, fmt.Errorf(
			"%d items, %d scores, %d similarities", len(items), len(scores), len(similarity),
		))
	}

	return service.Candidates{Items: items, Scores: scores, Similarity: similarity}, nil
}

func (s *ANNIndex) degrade(
	ctx context.Context, user service.User, cause error,
) (service.Candidates, error) {
	if s.Fallback == nil {
		return service.Candidates{}, fmt.Errorf("FAISS index retrieval: %w", cause)
	}

	// NOT the caller's ctx. The commonest reason to be here is that the
	// sidecar ran out the source deadline, which leaves that ctx already
	// expired -- and rung 2 under it fails its Redis GET before trying, so the
	// rescue only ever worked for the failures that did not need it.
	budget := s.FallbackDeadline
	if budget <= 0 {
		budget = DefaultFallbackDeadline
	}
	rescueCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), budget)
	defer cancel()

	items, ok, err := s.Fallback.Search(rescueCtx, user.ID, s.K)
	if err != nil {
		return service.Candidates{}, fmt.Errorf("FAISS index retrieval: %w",
			errors.Join(cause, fmt.Errorf("exact search fallback: %w", err)))
	}
	if !ok {
		return service.Candidates{}, fmt.Errorf("FAISS index retrieval: %w", cause)
	}

	s.setLastIndex("flat", "")

	return service.Candidates{Items: items}, nil

}

// Ready reports whether EVERY index process is serving a non-empty index, and
// the SAME one.
func (s *ANNIndex) Ready(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, readyTimeout)
	defer cancel()

	if s.conn == nil {
		response, err := indexHealth(ctx, s.client)
		if err != nil {
			return err
		}
		s.setLastIndex(response.GetIndexKind(), response.GetIndexVersion())
		return nil
	}

	var mu sync.Mutex
	versions := map[string][]string{}
	var last *pb.RetrievalHealthResponse
	err := s.conn.Each(ctx, func(ctx context.Context, address string, conn grpc.ClientConnInterface) error {
		response, err := indexHealth(ctx, pb.NewRetrievalClient(conn))
		if err != nil {
			return err
		}
		mu.Lock()
		defer mu.Unlock()
		versions[response.GetIndexVersion()] = append(versions[response.GetIndexVersion()], address)
		last = response
		return nil
	})
	if err != nil {
		return err
	}
	// Every process healthy is not enough. Processes restarted across an index
	// promotion can be on different versions, and the round-robin then serves
	// each user's requests from whichever index the call happens to land on.
	if len(versions) > 1 {
		return fmt.Errorf("FAISS index sidecar processes disagree on the index version: %v", versions)
	}
	s.setLastIndex(last.GetIndexKind(), last.GetIndexVersion())
	return nil
}

// indexHealth is one index process's answer, refused unless it is usable.
func indexHealth(ctx context.Context, client pb.RetrievalClient) (*pb.RetrievalHealthResponse, error) {
	response, err := client.Health(ctx, &pb.RetrievalHealthRequest{})
	if err != nil {
		return nil, fmt.Errorf("FAISS index sidecar Health: %w", err)
	}
	if !response.GetReady() {
		return nil, fmt.Errorf("FAISS index sidecar not ready: %s", response.GetDetail())
	}
	if count := response.GetItemCount(); count == 0 {
		return nil, fmt.Errorf("FAISS index sidecar reports an empty index (version %q)",
			response.GetIndexVersion())
	}
	return response, nil
}

func (s *ANNIndex) Close() error {
	if s.conn == nil {
		return nil
	}
	return s.conn.Close()
}

func (s *ANNIndex) Connections() int {
	if s.conn == nil {
		return 0
	}
	return s.conn.Connections()
}

var _ service.Retriever = (*ANNIndex)(nil)
