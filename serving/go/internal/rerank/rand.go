package rerank

import (
	"math/rand"
	"sync"
)

// lockedSource makes one seeded stream safe to share across requests.
//
// rand.New's own source is NOT safe for concurrent use, and the orchestrator
// reranks every request on its own goroutine. The global rand functions are
// safe but cannot be seeded reproducibly, which is the one property Select
// demands -- see the Rand field's doc comment.
//
// Two draws per exploring slot, so the mutex is held for nanoseconds a few
// times per request. Contention here is not measurable next to a 35ms ranker.
type lockedSource struct {
	mu  sync.Mutex
	src rand.Source
}

func (s *lockedSource) Int63() int64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.src.Int63()
}

func (s *lockedSource) Seed(seed int64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.src.Seed(seed)
}

// NewLockedRand returns a seeded *rand.Rand that many goroutines may share.
//
// Lives here rather than in cmd/server because this package is the one that
// refuses to explore without a reproducible source; the requirement and the
// means to satisfy it belong together.
func NewLockedRand(seed int64) *rand.Rand {
	return rand.New(&lockedSource{src: rand.NewSource(seed)})
}
