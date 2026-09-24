package catalogue

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func write(t *testing.T, payload onDisk) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "catalogue.json")
	raw, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshalling: %v", err)
	}
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatalf("writing: %v", err)
	}
	return path
}

func loaded(t *testing.T) *Table {
	t.Helper()
	// Row 0 reserved, then four articles.
	table, err := Load(write(t, onDisk{
		Version:     "v=test",
		Category:    []int32{0, 3, 3, 4, 5},
		Subcategory: []int32{0, 30, 31, 40, 50},
	}))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	return table
}

func TestColumnsAreGatheredByItemID(t *testing.T) {
	table := loaded(t)

	categories := table.Categories([]int32{3, 1})
	subcategories := table.Subcategories([]int32{3, 1})

	// Positional by item id, not by candidate order: item 3 is category 4.
	if categories[0] != 4 || categories[1] != 3 {
		t.Errorf("categories %v, want [4 3]", categories)
	}
	if subcategories[0] != 40 || subcategories[1] != 30 {
		t.Errorf("subcategories %v, want [40 30]", subcategories)
	}
}

// TestAnUnknownItemYieldsTheReservedIndex pins the degradation direction.
//
// This happens exactly when the index is a rebuild ahead of this artifact --
// during a deploy. Returning the reserved value is wrong in a small way;
// panicking takes the request path down in a large one.
func TestAnUnknownItemYieldsTheReservedIndex(t *testing.T) {
	table := loaded(t)

	got := table.Categories([]int32{99, -1})

	if got[0] != 0 || got[1] != 0 {
		t.Errorf("got %v, want the reserved 0 for both", got)
	}
}

// TestMismatchedColumnsAreRefusedAtLoad: past the shorter column every lookup
// silently returns 0, which is the reserved category -- so a whole tail of the
// catalogue reads as one giant uncategorised bucket and the per-category cap
// binds on all of it at once.
func TestMismatchedColumnsAreRefusedAtLoad(t *testing.T) {
	_, err := Load(write(t, onDisk{
		Category:    []int32{0, 1, 2},
		Subcategory: []int32{0, 10},
	}))

	if err == nil {
		t.Fatal("columns of different lengths must be refused")
	}
}

func TestAnEmptyCatalogueIsRefused(t *testing.T) {
	// Loads fine and makes every item uncategorised, which is a quality
	// regression with no error attached.
	if _, err := Load(write(t, onDisk{Version: "v=empty"})); err == nil {
		t.Fatal("an empty catalogue must be refused at load")
	}
}

func TestAMissingFileIsNamed(t *testing.T) {
	if _, err := Load(filepath.Join(t.TempDir(), "absent.json")); err == nil {
		t.Fatal("a missing catalogue must fail at startup, not at the first request")
	}
}

// TestVectorsAreNilWhichDisablesMMR is a test for a DECISION, not behaviour.
//
// ADR 0012 measured MMR as a null on this corpus and ships Lambda 1.0, so the
// content-vector table is megabytes of cache footprint bought for zero effect.
// If Lambda ever moves off 1.0 this test should fail and be the reminder that
// the vectors have to start being loaded.
func TestVectorsAreNilWhichDisablesMMR(t *testing.T) {
	if got := loaded(t).Vectors([]int32{1, 2}); got != nil {
		t.Errorf("Vectors returned %v; MMR is meant to be off", got)
	}
}

func TestTheVersionAndSizeAreReported(t *testing.T) {
	table := loaded(t)

	// Both feed the health endpoint: a catalogue silently a build behind the
	// index is otherwise invisible.
	if table.Version() != "v=test" {
		t.Errorf("version %q", table.Version())
	}
	if table.Size() != 5 {
		t.Errorf("size %d, want 5 including the reserved row", table.Size())
	}
}
