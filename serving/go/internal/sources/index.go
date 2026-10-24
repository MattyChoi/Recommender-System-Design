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

type ANNIndex struct {
	client pb.RetrievalClient
	conn   *rpc.Balanced

	SourceName string
	Deadline   time.Duration
	K          int
	EFSearch   int32

	Fallback Searcher

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

	items, ok, err := s.Fallback.Search(ctx, user.ID, s.K)
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

func (s *ANNIndex) Ready(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, readyTimeout)
	defer cancel()

	response, err := s.client.Health(ctx, &pb.RetrievalHealthRequest{})
	if err != nil {
		return fmt.Errorf("FAISS index sidecar Health: %w", err)
	}
	if !response.GetReady() {
		return fmt.Errorf("FAISS index sidecar not ready: %s", response.GetDetail())
	}
	if count := response.GetItemCount(); count == 0 {
		return fmt.Errorf("FAISS index sidecar reports an empty index (version %q)",
			response.GetIndexVersion())
	}
	s.setLastIndex(response.GetIndexKind(), response.GetIndexVersion())
	return nil

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
