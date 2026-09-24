package index

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// write lays down a stem.json / stem.bin pair. Rows are normalised here so a
// test that is not about the norm check does not have to think about it.
func write(t *testing.T, rows [][]float32, meta Meta) string {
	t.Helper()
	stem := filepath.Join(t.TempDir(), "items")

	body := make([]byte, 0, len(rows)*len(rows[0])*4)
	for _, row := range rows {
		for _, value := range row {
			word := make([]byte, 4)
			binary.LittleEndian.PutUint32(word, math.Float32bits(value))
			body = append(body, word...)
		}
	}
	if err := os.WriteFile(stem+".bin", body, 0o600); err != nil {
		t.Fatalf("writing vectors: %v", err)
	}
	raw, err := json.Marshal(meta)
	if err != nil {
		t.Fatalf("marshalling meta: %v", err)
	}
	if err := os.WriteFile(stem+".json", raw, 0o600); err != nil {
		t.Fatalf("writing meta: %v", err)
	}
	return stem
}

// unit vectors on the axes, so every inner product is exactly one coordinate
// of the query and the expected ranking can be read off by eye.
func axes(n int) [][]float32 {
	rows := make([][]float32, n)
	for row := range rows {
		rows[row] = make([]float32, n)
		rows[row][row] = 1
	}
	return rows
}

func standardMeta(rows, dim int) Meta {
	return Meta{Rows: rows, Dim: dim, BaseIndex: 1, Version: "v=test"}
}

func TestSearchReturnsTheBestFirst(t *testing.T) {
	stem := write(t, axes(4), standardMeta(4, 4))
	flat, err := Load(stem)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	// Row 2 scores 0.9, row 0 scores 0.5, row 3 scores 0.1, row 1 scores 0.
	items, scores, err := flat.Search([]float32{0.5, 0, 0.9, 0.1}, 3)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}

	// BaseIndex 1: row r holds item r+1. Index 0 is the reserved OOV row and
	// never enters an index, so an off-by-one here returns real ids for the
	// wrong articles -- which searches fine and reports nothing.
	want := []int32{3, 1, 4}
	for position := range want {
		if items[position] != want[position] {
			t.Fatalf("items %v, want %v", items, want)
		}
	}
	if !(scores[0] > scores[1] && scores[1] > scores[2]) {
		t.Errorf("scores %v are not descending", scores)
	}
}

// TestTiesKeepTheEarlierRow pins this file's rule: the first maximum wins,
// which is numpy's convention and also rerank.Select's, so the codebase has
// one tie-break rather than two.
//
// It is NOT FAISS's rule, and FAISS does not have one -- measured on identical
// rows, IndexFlatIP keeps the earlier row at k=1 and the later at k=5 and
// k=100, because the k=1 path and the heap path are different code. The parity
// test compares tied groups as sets for that reason; this test is what pins
// the behaviour on our side, where it IS stable.
func TestTiesKeepTheEarlierRow(t *testing.T) {
	stem := write(t, axes(3), standardMeta(3, 3))
	flat, err := Load(stem)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	// Every row scores 1 against this query, so the tie-break alone decides.
	items, _, err := flat.Search([]float32{1, 1, 1}, 2)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if items[0] != 1 || items[1] != 2 {
		t.Errorf("items %v, want [1 2]: an all-ties query keeps the earlier rows", items)
	}
}

// TestATieDoesNotDisplaceAtTheCutoff is the half of the rule the insertion
// loop alone does not cover: a row merely TYING the k-th best is rejected, so
// the earlier row keeps the slot. Using `<` instead of `<=` in the skip passes
// the test above and fails here.
func TestATieDoesNotDisplaceAtTheCutoff(t *testing.T) {
	stem := write(t, axes(3), standardMeta(3, 3))
	flat, err := Load(stem)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	items, _, err := flat.Search([]float32{1, 1, 1}, 1)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(items) != 1 || items[0] != 1 {
		t.Errorf("items %v, want [1]: the first tied row keeps the only slot", items)
	}
}

func TestKIsClampedToTheCatalogue(t *testing.T) {
	stem := write(t, axes(3), standardMeta(3, 3))
	flat, err := Load(stem)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	items, _, err := flat.Search([]float32{1, 0, 0}, 50)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	// Short, not padded. Padding would put the reserved index in front of a
	// user; the blend already knows how to spend an under-filled quota.
	if len(items) != 3 {
		t.Errorf("got %d items from a 3-row index", len(items))
	}
}

func TestAQueryOfTheWrongWidthIsRefused(t *testing.T) {
	stem := write(t, axes(4), standardMeta(4, 4))
	flat, err := Load(stem)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if _, _, err := flat.Search([]float32{1, 0}, 2); err == nil {
		t.Fatal("a 2-wide query against a 4-wide index must be refused")
	}
}

// --- What the loader has to refuse -------------------------------------------

// TestATruncatedMatrixIsRefused: the failure this guards is not a crash. A
// short file whose rows are offset from the rows they claim to be searches
// perfectly and returns the wrong article for every query.
func TestATruncatedMatrixIsRefused(t *testing.T) {
	stem := write(t, axes(4), standardMeta(4, 4))
	body, err := os.ReadFile(stem + ".bin")
	if err != nil {
		t.Fatalf("reading back: %v", err)
	}
	if err := os.WriteFile(stem+".bin", body[:len(body)-8], 0o600); err != nil {
		t.Fatalf("truncating: %v", err)
	}

	if _, err := Load(stem); err == nil {
		t.Fatal("a matrix shorter than its meta declares must be refused")
	}
}

// TestAnUnnormalisedTableIsRefused mirrors assert_unit_norm on the Python side.
// Inner product is cosine ONLY because the towers normalise; without it the
// index ranks by |u||v|cos, so long vectors win and nothing in the output says
// so. The artifact can be replaced without the builder running again, so the
// check has to exist at both ends.
func TestAnUnnormalisedTableIsRefused(t *testing.T) {
	rows := axes(3)
	rows[1][1] = 4
	stem := write(t, rows, standardMeta(3, 3))

	_, err := Load(stem)
	if err == nil {
		t.Fatal("a table that is not unit-norm must be refused")
	}
	if !strings.Contains(err.Error(), "magnitude") {
		t.Errorf("the error should say what goes wrong, got %q", err)
	}
}

func TestMissingArtifactsAreNamed(t *testing.T) {
	if _, err := Load(filepath.Join(t.TempDir(), "absent")); err == nil {
		t.Fatal("a missing index must be refused at load, not at the first request")
	}
}

// --- The measurement ---------------------------------------------------------
//
// What rung 2 of ADR 0013's ladder costs. Not an argument for skipping the
// index -- the sidecar is the retrieval path -- but a degradation whose price
// is unmeasured is a degradation nobody can size. If this lands inside the
// design doc's 25ms retrieval budget, a sidecar outage is a CPU problem; if it
// does not, it is an outage, and the ladder needs another rung.
//
//	go test ./internal/index -bench BenchmarkSearch -benchtime 200x

// benchDim is the tower width, from models/retrieval/two_tower.py (out_dim=128).
// Not a round number picked for the benchmark: at 64 this measures half the
// work the real artifact does, and reports a fallback as twice as affordable as
// it is.
const (
	benchDim = 128
	benchK   = 100
)

// The two shapes worth knowing, because they answer different questions.
//
//	shipped -- MIND-small's catalogue, what this build actually serves.
//	target  -- docs/design.md's stated scale assumption. The fallback has to
//	           still be a fallback there, or the ladder only works at the size
//	           the project happens to be today.
var benchShapes = []struct {
	name string
	rows int
}{
	{"shipped_65k", 65_000},
	{"design_target_160k", 160_000},
}

func benchIndex(rows int) *Flat {
	source := rand.New(rand.NewSource(7))
	vectors := make([]float32, rows*benchDim)
	for row := 0; row < rows; row++ {
		start := row * benchDim
		var norm float64
		for offset := 0; offset < benchDim; offset++ {
			value := float32(source.NormFloat64())
			vectors[start+offset] = value
			norm += float64(value) * float64(value)
		}
		scale := float32(1.0 / math.Sqrt(norm))
		for offset := 0; offset < benchDim; offset++ {
			vectors[start+offset] *= scale
		}
	}
	return &Flat{
		meta:    Meta{Rows: rows, Dim: benchDim, BaseIndex: 1, Version: "bench"},
		vectors: vectors,
	}
}

func BenchmarkSearch(b *testing.B) {
	query := make([]float32, benchDim)
	for offset := range query {
		query[offset] = float32(1.0 / math.Sqrt(benchDim))
	}

	for _, shape := range benchShapes {
		b.Run(shape.name, func(b *testing.B) {
			flat := benchIndex(shape.rows)
			// Reported as a rate so the two shapes are comparable, NOT as
			// evidence about what binds. Every byte of the table feeds exactly
			// one multiply-add, so MB/s and FLOP/s are proportional by
			// construction and a flat MB/s across shapes shows only that cost
			// is linear in table size -- which a compute-bound scan does too.
			// It is compute-bound: ~2.1 G FMA/s is about half of one core's
			// scalar FMA rate, and Go does not auto-vectorise this loop.
			b.SetBytes(int64(shape.rows) * benchDim * 4)
			b.ReportAllocs()

			b.ResetTimer()
			for iteration := 0; iteration < b.N; iteration++ {
				if _, _, err := flat.Search(query, benchK); err != nil {
					b.Fatalf("Search: %v", err)
				}
			}
		})
	}
}
