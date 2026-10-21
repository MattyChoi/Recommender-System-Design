package sources

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"time"

	"github.com/redis/go-redis/v9"
	"golang.org/x/crypto/blake2b"
)

type SeenList struct {
	client redis.UniversalClient

	// Bits and Hashes must match what wrote the filter. Sized by
	// models.reranking.seen.sizing
	Bits   uint64
	Hashes int

	// TTL is how long "recently shown" lasts.
	TTL time.Duration

	// Namespace keeps the keyspace legible under SCAN and separable from
	// anything else in the database.
	Namespace string
}

// SeenDefaults mirror models/reranking/seen_redis.py.
const (
	SeenNamespace = "seen"
	SeenTTL       = 7 * 24 * time.Hour
	// SeenDB is 1: Feast owns 0 on the same instance
	SeenDB = 1
)

// Sizing mirrors models/reranking/seen.py:sizing.
//
//	bits   = ceil(-capacity * ln(rate) / ln(2)^2)
//	hashes = max(1, round(bits / capacity * ln 2))
func Sizing(capacity int, falsePositiveRate float64) (bits uint64, hashes int) {
	raw := -float64(capacity) * math.Log(falsePositiveRate) / (math.Ln2 * math.Ln2)
	bits = uint64(math.Ceil(raw))
	hashes = int(math.RoundToEven(float64(bits) / float64(capacity) * math.Ln2))
	if hashes < 1 {
		hashes = 1
	}
	return bits, hashes
}

// NewSeenList wraps a connected client.
func NewSeenList(client redis.UniversalClient, bits uint64, hashes int) *SeenList {
	return &SeenList{
		client:    client,
		Bits:      bits,
		Hashes:    hashes,
		TTL:       SeenTTL,
		Namespace: SeenNamespace,
	}
}

func (s *SeenList) key(userID string) string {
	return fmt.Sprintf("%s:%s", s.Namespace, userID)
}

// Positions is the k bit positions one item owns.
func (s *SeenList) Positions(item int32) []uint64 {
	var buffer [8]byte
	// Eight bytes, big-endian, matching `item.to_bytes(8, "big")`. A different
	// width or byte order agrees with Python on no item at all.
	binary.BigEndian.PutUint64(buffer[:], uint64(item))

	hasher, err := blake2b.New(16, nil)
	if err != nil {
		// Unreachable: New only errors on an out-of-range size or an oversized
		// key, and both are constants here.
		panic(fmt.Sprintf("blake2b.New(16): %v", err))
	}
	_, _ = hasher.Write(buffer[:])
	digest := hasher.Sum(nil)

	base := binary.BigEndian.Uint64(digest[:8])
	// Odd, so repeated addition generates the whole ring rather than a subset.
	step := binary.BigEndian.Uint64(digest[8:16]) | 1

	reducedBase := base % s.Bits
	reducedStep := step % s.Bits

	positions := make([]uint64, s.Hashes)
	for index := range positions {
		positions[index] = (reducedBase + uint64(index)*reducedStep) % s.Bits
	}
	return positions
}

// Blocked reports, per candidate, whether the filter says it was already shown.
func (s *SeenList) Blocked(
	ctx context.Context, userID string, items []int32,
) ([]bool, error) {
	if len(items) == 0 {
		return nil, nil
	}

	raw, err := s.client.Get(ctx, s.key(userID)).Bytes()
	if err != nil {
		if err == redis.Nil {
			// No filter yet is not an error: a new user has been shown
			// nothing, and "nothing is blocked" is the correct answer.
			return make([]bool, len(items)), nil
		}
		return nil, fmt.Errorf("seen-list GET: %w", err)
	}

	mask := make([]bool, len(items))
	for index, item := range items {
		blocked := true
		for _, position := range s.Positions(item) {
			if !bitSet(raw, position) {
				blocked = false
				break
			}
		}
		mask[index] = blocked
	}
	return mask, nil
}

// Record marks items as shown and refreshes the window, atomically.
func (s *SeenList) Record(ctx context.Context, userID string, items []int32) error {
	if len(items) == 0 {
		return nil
	}
	key := s.key(userID)
	pipe := s.client.TxPipeline()
	for _, item := range items {
		for _, position := range s.Positions(item) {
			pipe.SetBit(ctx, key, int64(position), 1)
		}
	}
	pipe.Expire(ctx, key, s.TTL)
	if _, err := pipe.Exec(ctx); err != nil {
		return fmt.Errorf("seen-list write: %w", err)
	}
	return nil
}

// Load is the share of bits set, which is the alarm worth watching.
func (s *SeenList) Load(ctx context.Context, userID string) (float64, error) {
	raw, err := s.client.Get(ctx, s.key(userID)).Bytes()
	if err != nil {
		if err == redis.Nil {
			return 0, nil
		}
		return 0, fmt.Errorf("seen-list GET: %w", err)
	}
	set := 0
	for position := uint64(0); position < s.Bits; position++ {
		if bitSet(raw, position) {
			set++
		}
	}
	return float64(set) / float64(s.Bits), nil
}

// bitSet reads one Redis bit position out of the raw string.
//
// Redis grows the string only as far as the highest bit ever SET, so a sparse
// filter comes back short. A position past the end is unset, not an error --
// the Python side pads for the same reason.
func bitSet(raw []byte, position uint64) bool {
	index := position / 8
	if index >= uint64(len(raw)) {
		return false
	}
	return raw[index]&(1<<(7-position%8)) != 0
}
