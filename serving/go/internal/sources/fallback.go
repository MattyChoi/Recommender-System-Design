// This package handles the fallback the service utilizes when the FAISS index service is down.
//
// Two rungs of the degradation ladder
//		* No FAISS index but user embedding is cached
//		* No FAISS index and no cached user embedding

package sources

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"

	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/index"
	"github.com/redis/go-redis/v9"
)

const (
	EmbeddingPrefix = "uemb"
	// EmbeddingDB is 2: Feast owns 0 and the seen-list owns 1. Mirrors
	// serving/retrieval/cache.py, which writes the entries this reads.
	EmbeddingDB = 2
)

type CachedSearch struct {
	client    redis.UniversalClient
	index     *index.Flat
	KeyPrefix string
}

func NewCachedSearch(client redis.UniversalClient, flat *index.Flat) *CachedSearch {
	return &CachedSearch{client: client, index: flat, KeyPrefix: EmbeddingPrefix}
}

func (c *CachedSearch) Search(ctx context.Context, userID string, k int) ([]int32, bool, error) {
	raw, err := c.client.Get(ctx, fmt.Sprintf("%s:%s", c.KeyPrefix, userID)).Bytes()
	if err != nil {
		if err == redis.Nil {
			return nil, false, nil
		}
		return nil, false, fmt.Errorf("embedding cache GET: %w", err)
	}

	query, err := c.decode(raw)
	if err != nil {
		// not a miss, this is a malformed cache entry
		return nil, false, err
	}

	items, _, err := c.index.Search(query, k)
	if err != nil {
		return nil, false, fmt.Errorf("exact search: %w", err)
	}
	return items, true, nil
}

func (c *CachedSearch) decode(raw []byte) ([]float32, error) {
	width := c.index.Dim()
	if len(raw) != width*4 {
		return nil, fmt.Errorf(
			"cached embedding is %d bytes, the index expects %d floats (%d bytes)",
			len(raw),
			width,
			width*4,
		)
	}
	out := make([]float32, width)
	for position := range out {
		out[position] = math.Float32frombits(binary.LittleEndian.Uint32(raw[position*4:]))
	}
	return out, nil
}

var _ Searcher = (*CachedSearch)(nil)
