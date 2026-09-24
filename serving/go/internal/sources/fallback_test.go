package sources

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"

	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/index"
)

const fallbackDim = 4

// axisIndex writes a tiny index whose rows are the unit axes, so the best
// match for a one-hot query is unambiguous and no tie-break can affect it.
func axisIndex(t *testing.T) *index.Flat {
	t.Helper()
	stem := filepath.Join(t.TempDir(), "items")

	body := make([]byte, 0, fallbackDim*fallbackDim*4)
	for row := 0; row < fallbackDim; row++ {
		for column := 0; column < fallbackDim; column++ {
			value := float32(0)
			if row == column {
				value = 1
			}
			word := make([]byte, 4)
			binary.LittleEndian.PutUint32(word, math.Float32bits(value))
			body = append(body, word...)
		}
	}
	if err := os.WriteFile(stem+".bin", body, 0o600); err != nil {
		t.Fatalf("writing vectors: %v", err)
	}

	meta, err := json.Marshal(index.Meta{
		Rows: fallbackDim, Dim: fallbackDim, BaseIndex: 1, Version: "v=test",
	})
	if err != nil {
		t.Fatalf("marshalling meta: %v", err)
	}
	if err := os.WriteFile(stem+".json", meta, 0o600); err != nil {
		t.Fatalf("writing meta: %v", err)
	}

	flat, err := index.Load(stem)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	return flat
}

func newCached(t *testing.T) (*CachedSearch, *miniredis.Miniredis) {
	t.Helper()
	server := miniredis.RunT(t)
	client := redis.NewClient(&redis.Options{Addr: server.Addr()})
	t.Cleanup(func() { _ = client.Close() })
	return NewCachedSearch(client, axisIndex(t)), server
}

// encode writes a vector the way serving/retrieval/cache.py does: raw
// little-endian float32, no header.
func encode(values ...float32) string {
	raw := make([]byte, len(values)*4)
	for index, value := range values {
		binary.LittleEndian.PutUint32(raw[index*4:], math.Float32bits(value))
	}
	return string(raw)
}

func TestRungTwoSearchesWithTheCachedEmbedding(t *testing.T) {
	search, server := newCached(t)
	// Row 0 is the unit vector on axis 0, and BaseIndex 1 makes it item 1.
	if err := server.Set(EmbeddingPrefix+":U1", encode(1, 0, 0, 0)); err != nil {
		t.Fatalf("seeding: %v", err)
	}

	items, ok, err := search.Search(context.Background(), "U1", 1)

	if err != nil || !ok {
		t.Fatalf("got %v, %v", ok, err)
	}
	if len(items) != 1 || items[0] != 1 {
		t.Errorf("items %v, want [1]", items)
	}
}

// TestACacheMissIsNotAnError is the distinction the whole ladder rests on.
//
// A user the sidecar has not served recently has no cached embedding. That is
// ordinary, not broken: rung 2 is simply unavailable for this request and the
// caller falls to popularity, reporting the SIDECAR's failure as the cause.
func TestACacheMissIsNotAnError(t *testing.T) {
	search, _ := newCached(t)

	items, ok, err := search.Search(context.Background(), "U-unknown", 5)

	if err != nil {
		t.Fatalf("a miss must not be an error: %v", err)
	}
	if ok || items != nil {
		t.Errorf("got %v, %v; want nil, false", items, ok)
	}
}

// TestAWrongWidthEmbeddingIsAnErrorNotAMiss separates two things that look
// identical from the outside.
//
// A vector of the wrong width is a DIFFERENT MODEL's embedding left behind by
// a deploy. Searching with it would return neighbours in a space the query is
// not in -- real item ids, plausible ordering, wrong answers. Reported as an
// error so it does not disappear into a miss count.
func TestAWrongWidthEmbeddingIsAnErrorNotAMiss(t *testing.T) {
	search, server := newCached(t)
	if err := server.Set(EmbeddingPrefix+":U1", encode(1, 0)); err != nil {
		t.Fatalf("seeding: %v", err)
	}

	_, ok, err := search.Search(context.Background(), "U1", 1)

	if err == nil {
		t.Fatal("a wrong-width embedding must be reported, not treated as absent")
	}
	if ok {
		t.Error("ok should be false when nothing usable was found")
	}
	if !strings.Contains(err.Error(), "bytes") {
		t.Errorf("the error should show the arithmetic, got %q", err)
	}
}

// TestTheDecoderMatchesThePythonWriterByteForByte pins the format.
//
// There is no header to negotiate: the sidecar writes raw little-endian
// float32 and this reads it. The byte string below is what numpy's
// `astype("<f4").tobytes()` produces for [1.0, 2.0, 0.5, -1.0].
func TestTheDecoderMatchesThePythonWriterByteForByte(t *testing.T) {
	search, _ := newCached(t)
	raw := []byte{
		0x00, 0x00, 0x80, 0x3f, // 1.0
		0x00, 0x00, 0x00, 0x40, // 2.0
		0x00, 0x00, 0x00, 0x3f, // 0.5
		0x00, 0x00, 0x80, 0xbf, // -1.0
	}

	got, err := search.decode(raw)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}

	want := []float32{1.0, 2.0, 0.5, -1.0}
	for index := range want {
		if got[index] != want[index] {
			t.Fatalf("decoded %v, want %v", got, want)
		}
	}
}

func TestAnUnreachableCacheIsAnError(t *testing.T) {
	search, server := newCached(t)
	server.Close()

	if _, _, err := search.Search(context.Background(), "U1", 1); err == nil {
		t.Fatal("an unreachable cache must be reported")
	}
}
