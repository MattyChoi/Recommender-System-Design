package catalogue

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/matrix"
)

// The byte-count, endianness, unit-norm and base_index checks are properties of
// the READER and are tested once, in internal/matrix. What is left here is what
// belongs to the catalogue: the JSON artifact, the three lines of
// matrix.Expect this caller supplies, and the MMR refusal.

const testDim = 4

func writeCatalogue(t *testing.T, dir string, n int) string {
	t.Helper()
	category := make([]int32, n+1)
	subcategory := make([]int32, n+1)
	for item := 1; item <= n; item++ {
		category[item] = int32(item%3) + 1
		subcategory[item] = int32(item%5) + 1
	}
	blob, err := json.Marshal(payload{
		Version:     "cat-v1",
		Category:    category,
		Subcategory: subcategory,
	})
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "catalogue.json")
	if err := os.WriteFile(path, blob, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// writeVectors drops a content artifact: row 0 zeroed, every other row a unit
// vector, base_index 0.
func writeVectors(t *testing.T, dir string, rows int, baseIndex int32) string {
	t.Helper()
	stem := filepath.Join(dir, "content")

	blob, err := json.Marshal(matrix.Meta{
		Rows: rows, Dim: testDim, BaseIndex: baseIndex,
		Variant: "vec_title_abstract", Version: "vec-v1",
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(stem+".json", blob, 0o600); err != nil {
		t.Fatal(err)
	}

	body := make([]byte, 0, rows*testDim*4)
	for row := range rows {
		for axis := range testDim {
			var value float32
			if row > 0 && axis == row%testDim {
				value = 1
			}
			var word [4]byte
			binary.LittleEndian.PutUint32(word[:], math.Float32bits(value))
			body = append(body, word[:]...)
		}
	}
	if err := os.WriteFile(stem+".bin", body, 0o600); err != nil {
		t.Fatal(err)
	}
	return stem
}

func loadBare(t *testing.T, n int) (*Catalogue, string) {
	t.Helper()
	dir := t.TempDir()
	catalog, err := Load(writeCatalogue(t, dir, n))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	return catalog, dir
}

func TestLoadRefusesABrokenCatalogue(t *testing.T) {
	dir := t.TempDir()
	cases := []struct {
		name string
		body payload
	}{
		{"no rows", payload{Version: "v"}},
		{
			// They index the same items, so a disagreement means one of them
			// came from a different build.
			name: "arrays of different lengths",
			body: payload{Version: "v", Category: []int32{0, 1, 2}, Subcategory: []int32{0, 1}},
		},
		{
			// A real category on row 0 would be attributed to every candidate a
			// short source padded with zeros.
			name: "row 0 not reserved",
			body: payload{Version: "v", Category: []int32{7, 1}, Subcategory: []int32{0, 1}},
		},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			blob, err := json.Marshal(test.body)
			if err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(dir, strings.ReplaceAll(test.name, " ", "-")+".json")
			if err := os.WriteFile(path, blob, 0o600); err != nil {
				t.Fatal(err)
			}
			if _, err := Load(path); err == nil {
				t.Fatal("loaded without complaint")
			}
		})
	}
}

func TestVectorsAreNilUntilLoaded(t *testing.T) {
	catalog, _ := loadBare(t, 4)
	if catalog.HasVectors() {
		t.Fatal("a catalogue with no content artifact reports vectors")
	}
	if catalog.Dim() != 0 || catalog.VectorVersion() != "" {
		t.Fatalf("Dim=%d VectorVersion=%q", catalog.Dim(), catalog.VectorVersion())
	}
	// Nil, not a slice of nils: service.Catalogue documents nil as "MMR off for
	// this request", and rerank branches on the slice itself.
	if got := catalog.Vectors([]int32{1, 2}); got != nil {
		t.Fatalf("Vectors = %v, want nil", got)
	}
}

func TestAnEmptyStemIsNotAnError(t *testing.T) {
	catalog, _ := loadBare(t, 4)
	// MMR is off in the shipped configuration, so a server with no content
	// table is correct rather than degraded.
	if err := catalog.LoadVectors(""); err != nil {
		t.Fatalf(`LoadVectors("") = %v, want nil`, err)
	}
	if catalog.HasVectors() {
		t.Fatal("an empty stem loaded vectors")
	}
}

func TestTheGatherIsPositional(t *testing.T) {
	const n = 6
	catalog, dir := loadBare(t, n)
	if err := catalog.LoadVectors(writeVectors(t, dir, n+1, 0)); err != nil {
		t.Fatalf("LoadVectors: %v", err)
	}
	if catalog.Dim() != testDim || catalog.VectorVersion() != "vec-v1" {
		t.Fatalf("Dim=%d VectorVersion=%q", catalog.Dim(), catalog.VectorVersion())
	}

	// Deliberately not in order and not a prefix: the output is paired with the
	// input by position, and a sorted query would hide a sorted implementation.
	query := []int32{5, 1, 3}
	got := catalog.Vectors(query)
	if len(got) != len(query) {
		t.Fatalf("got %d rows for %d items", len(got), len(query))
	}
	for position, item := range query {
		axis := int(item) % testDim
		if got[position][axis] != 1 {
			t.Fatalf("item %d: got %v, expected a 1 at axis %d", item, got[position], axis)
		}
	}
}

func TestAnOutOfRangeItemYieldsNilRatherThanPanicking(t *testing.T) {
	const n = 3
	catalog, dir := loadBare(t, n)
	if err := catalog.LoadVectors(writeVectors(t, dir, n+1, 0)); err != nil {
		t.Fatal(err)
	}
	got := catalog.Vectors([]int32{2, 99, -1})
	if len(got) != 3 {
		t.Fatalf("got %d rows, want 3", len(got))
	}
	if got[0] == nil || got[1] != nil || got[2] != nil {
		t.Fatalf("expected [row, nil, nil], got %v", got)
	}
}

func TestTheExpectationThisCallerSupplies(t *testing.T) {
	const n = 4
	t.Run("base_index 1 is refused", func(t *testing.T) {
		// An items artifact. Well-formed, wrong table -- this is the check that
		// stops -content-stem pointing at serving/artifacts/items.
		catalog, dir := loadBare(t, n)
		err := catalog.LoadVectors(writeVectors(t, dir, n+1, 1))
		if err == nil {
			t.Fatal("an items artifact loaded as the content table")
		}
		if !strings.Contains(err.Error(), "base_index") {
			t.Fatalf("error does not name the field: %v", err)
		}
		if catalog.HasVectors() {
			t.Fatal("the refused table is attached")
		}
	})

	t.Run("a row count from another build is refused", func(t *testing.T) {
		// One row per item INCLUDING the reserved row, so the matrix is the
		// same length as the category arrays rather than one less.
		catalog, dir := loadBare(t, n)
		if err := catalog.LoadVectors(writeVectors(t, dir, n, 0)); err == nil {
			t.Fatal("a short matrix loaded")
		}
	})

	t.Run("the reserved row is skipped by the norm check", func(t *testing.T) {
		// Row 0 is exactly zero, as policies.normalise leaves it. SkipRows: 1
		// is what makes this load; without it the zero row fails the check.
		catalog, dir := loadBare(t, n)
		if err := catalog.LoadVectors(writeVectors(t, dir, n+1, 0)); err != nil {
			t.Fatalf("a zero row 0 was refused: %v", err)
		}
	})
}

func TestCheckMMR(t *testing.T) {
	const n = 4
	bare, _ := loadBare(t, n)
	loaded, dir := loadBare(t, n)
	if err := loaded.LoadVectors(writeVectors(t, dir, n+1, 0)); err != nil {
		t.Fatal(err)
	}

	// The shipped configuration: Lambda 1.0 makes MMR a no-op, so no table is
	// needed and none is demanded.
	if err := bare.CheckMMR(1.0); err != nil {
		t.Fatalf("Lambda 1.0 with no vectors: %v", err)
	}
	// The silently-broken one. Without this refusal the slate is identical to
	// the pure-relevance slate and nothing reports it.
	if err := bare.CheckMMR(0.7); err == nil {
		t.Fatal("Lambda 0.7 with no vectors was accepted")
	}
	if err := loaded.CheckMMR(0.7); err != nil {
		t.Fatalf("Lambda 0.7 with vectors: %v", err)
	}
}
