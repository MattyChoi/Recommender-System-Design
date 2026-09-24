package sources

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"

	"github.com/redis/go-redis/v9"

	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/index"
)

// CachedSearch is rung 2 of ADR 0013's degradation ladder: exact search in
// this process, against the user embedding the sidecar cached.
//
// It exists because the two halves of retrieval fail independently. The
// sidecar holds the index AND the tower, so losing it normally loses both --
// but §14.4's Redis cache already keeps a recent user embedding off the hot
// path, and that embedding is the only thing this process cannot compute for
// itself. Given it, exact search over the item table is arithmetic.
//
// **Correct results, more CPU, no ANN.** benchmarks.md measures the scan at
// 3.9ms per query on MIND-small, which fits the 25ms retrieval budget and does
// NOT fit the capacity budget: ~2.0 cores at 500 QPS against the `cpu: "2"`
// pod limit. A sidecar outage at peak needs load-shedding, not transparent
// failover, and that is written down rather than discovered.
type CachedSearch struct {
	client redis.UniversalClient
	index  *index.Flat

	// KeyPrefix must match serving/retrieval/cache.py:KEY_PREFIX. The sidecar
	// writes these keys and this reads them; they are a cross-process format
	// with no header to negotiate.
	KeyPrefix string
}

// EmbeddingPrefix mirrors serving/retrieval/cache.py.
const EmbeddingPrefix = "uemb"

// NewCachedSearch wires the cache reader to a loaded index.
func NewCachedSearch(client redis.UniversalClient, flat *index.Flat) *CachedSearch {
	return &CachedSearch{client: client, index: flat, KeyPrefix: EmbeddingPrefix}
}

// Search returns candidates for a user whose embedding is cached.
//
// The bool is "was there an embedding", not "did it work". A cache miss is the
// ordinary case for a user the sidecar has not served recently, and it means
// rung 2 is unavailable for this request -- not that anything is broken. The
// caller falls to popularity and reports the SIDECAR's failure, because that
// is the cause; "no cached embedding" is a consequence, and reporting it as
// the cause sends whoever is on call to Redis to debug a gRPC outage.
func (c *CachedSearch) Search(
	ctx context.Context, userID string, k int,
) ([]int32, bool, error) {
	raw, err := c.client.Get(ctx, fmt.Sprintf("%s:%s", c.KeyPrefix, userID)).Bytes()
	if err != nil {
		if err == redis.Nil {
			return nil, false, nil
		}
		return nil, false, fmt.Errorf("embedding cache GET: %w", err)
	}

	query, err := c.decode(raw)
	if err != nil {
		// Not a miss: a malformed entry means the cache holds something this
		// build cannot read, which is a different problem from an absent one
		// and would otherwise be invisible behind a miss count.
		return nil, false, err
	}

	items, _, err := c.index.Search(query, k)
	if err != nil {
		return nil, false, fmt.Errorf("exact search: %w", err)
	}
	return items, true, nil
}

// decode reads the raw little-endian float32 the sidecar wrote.
//
// No header, by agreement: a pickle would be unreadable from here and JSON
// would cost a text round trip and lose the last bit of every float. Length is
// the only thing to check, and it is checked -- a vector of the wrong width is
// a DIFFERENT MODEL's embedding left behind by a deploy, and searching the
// index with it would return neighbours in a space the query is not in.
func (c *CachedSearch) decode(raw []byte) ([]float32, error) {
	width := c.index.Dim()
	if len(raw) != width*4 {
		return nil, fmt.Errorf(
			"cached embedding is %d bytes, the index expects %d floats (%d bytes)",
			len(raw), width, width*4,
		)
	}
	out := make([]float32, width)
	for position := range out {
		// Explicit little-endian rather than a cast, matching the writer. A
		// cast is faster and silently wrong on a big-endian host, and produces
		// numbers rather than an error.
		out[position] = math.Float32frombits(binary.LittleEndian.Uint32(raw[position*4:]))
	}
	return out, nil
}

// Compile-time proof this is the rung the sidecar falls back to.
var _ Searcher = (*CachedSearch)(nil)
