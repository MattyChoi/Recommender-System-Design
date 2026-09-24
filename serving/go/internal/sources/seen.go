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

// SeenList is the serving half of models/reranking/seen_redis.py: per-user
// Bloom filters held as Redis bitmaps.
//
// **This is the fourth cross-language parity surface.** The offline harness
// wrote these bits to measure what the filter costs the funnel (-0.0227 NDCG,
// Part L); this reads them in the request path. If the two disagree about
// which bits an item owns, the reader answers plausibly and wrongly in BOTH
// directions -- hiding fresh items, and re-showing seen ones, which is the
// failure the one-sided guarantee exists to make impossible.
//
// # Two things that are not obvious
//
// **One GET, not k*n GETBITs.** A request tests ~100 candidates against a
// 4-hash filter: 400 round trips would consume the whole re-rank budget. The
// bitmap is 60-1,200 bytes, so fetching it whole and testing locally is one
// round trip. At serving time the cost is round trips, not bit arithmetic, and
// that inverts the obvious implementation.
//
// **Writes are one transaction.** Two concurrent requests for the same user
// can otherwise interleave and leave an item's bits partially written -- which
// is the one way this structure CAN produce a false negative and re-show
// something.
type SeenList struct {
	client redis.UniversalClient

	// Bits and Hashes must match what wrote the filter. Sized by
	// models.reranking.seen.sizing; Part L ships capacity 200 at a 1% target.
	Bits   uint64
	Hashes int

	// TTL is how long "recently shown" lasts.
	//
	// ⚠️ It contradicts the one-sided guarantee and both are load-bearing. The
	// window is what stops the filter saturating -- past capacity it answers
	// "seen" to everything and returns a blank page rather than a degraded one
	// -- but an EXPIRE is a scheduled reset, and never-a-false-negative holds
	// only while the filter is never cleared. The trade is unavoidable; this
	// is where it is written down.
	TTL time.Duration

	// Namespace keeps the keyspace legible under SCAN and separable from
	// anything else in the database.
	Namespace string
}

// SeenDefaults mirror models/reranking/seen_redis.py.
const (
	SeenNamespace = "seen"
	SeenTTL       = 7 * 24 * time.Hour
	// SeenDB is 1: Feast owns 0 on the same instance, and sharing a keyspace
	// with a feature store means one FLUSHDB during a materialisation takes
	// every seen-list with it.
	SeenDB = 1
)

// Sizing mirrors models/reranking/seen.py:sizing.
//
//	bits   = ceil(-capacity * ln(rate) / ln(2)^2)
//	hashes = max(1, round(bits / capacity * ln 2))
//
// Ported rather than configured, because a filter read with different bits
// than it was written with agrees on nothing: every position is taken modulo
// `bits`, so one wrong number moves every bit of every item.
//
// ⚠️ **RoundToEven, not Round.** Python's `round()` is banker's rounding --
// halves go to the nearest EVEN integer -- and Go's math.Round goes away from
// zero. They differ on exactly the inputs where `bits / capacity * ln 2` lands
// on a half, which is rare enough to survive every test anyone writes by hand
// and produces a filter with one extra hash position, i.e. a filter that
// agrees with the writer nowhere.
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
//
// ⚠️ **The modulus is applied to each term, not to the sum, and that is the
// whole reason this function is not two lines.** Python computes
// `(base + i*step) % bits` on arbitrary-precision integers, so the sum is
// exact. `base` and `step` are both full 64-bit values read out of the digest,
// so holding the sum in a uint64 wraps at 2**64 first -- and
// `(x mod 2**64) mod bits` is not `x mod bits` unless bits is a power of two,
// which the sizing formula never produces.
//
// Reducing each term first gives the same answer as the exact arithmetic,
// because modular reduction distributes over addition and multiplication. The
// products stay small: hashes is single digits and bits is ~10^4, so
// `i * (step mod bits)` is far from overflowing.
func (s *SeenList) Positions(item int32) []uint64 {
	var buffer [8]byte
	// Eight bytes, big-endian, matching `item.to_bytes(8, "big")`. A different
	// width or byte order agrees with Python on no item at all.
	binary.BigEndian.PutUint64(buffer[:], uint64(item))

	// SIXTEEN bytes of BLAKE2b, matching `digest_size=16` -- and that is a
	// different hash from the first 16 bytes of BLAKE2b-256, because the
	// output length goes into the parameter block that seeds the state.
	// blake2b.Sum256(...)[:16] compiles, runs, and agrees with Python on
	// nothing.
	//
	// A hasher per call rather than a shared one: hash.Hash carries mutable
	// state and this runs once per candidate across concurrent requests. The
	// allocation is a few hundred bytes against a blake2b compression; if a
	// profile ever says otherwise, a sync.Pool is the fix, not a shared hasher.
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
//
// Part L measured that the filter stops being USEFUL well before it stops
// being correct: sized for 100 items and given 1,000 it reaches 100% load and
// answers "seen" to everything. Bit load moves long before the false-positive
// rate does, and unlike the rate it is observable per user at serving time.
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
// Redis numbers bits BIG-ENDIAN within each byte: position 0 is the HIGH bit
// of byte 0. That matches numpy's `unpackbits(..., bitorder="big")` on the
// Python side. Getting it backwards yields a filter that answers plausibly and
// wrongly, with no error anywhere.
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
