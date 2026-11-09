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
	"time"

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
	// now is the clock the hour bucket is read from. A field so tests can pin
	// the hour; time.Now in production.
	now func() time.Time
}

func NewCachedSearch(client redis.UniversalClient, flat *index.Flat) *CachedSearch {
	return &CachedSearch{client: client, index: flat, KeyPrefix: EmbeddingPrefix, now: time.Now}
}

// hourBucket is whole hours since the Unix epoch, in UTC. It mirrors
// hour_bucket in serving/retrieval/cache.py
func hourBucket(moment time.Time) int64 {
	return moment.Unix() / 3600
}

// EmbeddingKey is the cache key for one user in one hour bucket.
func EmbeddingKey(prefix, userID string, bucket int64) string {
	return fmt.Sprintf("%s:%s:%d", prefix, userID, bucket)
}

func (c *CachedSearch) Search(ctx context.Context, userID string, k int) ([]int32, bool, error) {
	// raw, err := c.client.Get(ctx, fmt.Sprintf("%s:%s", c.KeyPrefix, userID)).Bytes()
	bucket := hourBucket(c.now())

	// fetch the current hour and the previous hour, in case the current hour's embedding hasn't been written yet
	values, err := c.client.MGet(ctx,
		EmbeddingKey(c.KeyPrefix, userID, bucket),
		EmbeddingKey(c.KeyPrefix, userID, bucket-1),
	).Result()
	if err != nil {
		return nil, false, fmt.Errorf("embedding cache MGET: %w", err)
	}

	var raw []byte
	for _, value := range values {
		// MGET returns nil for a missing key and a string for a present one.
		if text, ok := value.(string); ok {
			raw = []byte(text)
			break
		}
	}
	if raw == nil {
		return nil, false, nil
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
