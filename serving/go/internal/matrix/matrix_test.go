package matrix

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const (
	testDim  = 4
	testRows = 5
)

// write drops a <stem>.bin/.json pair. `rows` is written verbatim, so a test
// can declare one shape in the meta and supply another.
func write(t *testing.T, meta Meta, rows [][]float32) string {
	t.Helper()
	stem := filepath.Join(t.TempDir(), "table")

	blob, err := json.Marshal(meta)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(stem+".json", blob, 0o600); err != nil {
		t.Fatal(err)
	}

	body := make([]byte, 0, len(rows)*meta.Dim*4)
	for _, row := range rows {
		for _, value := range row {
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

// unitRows builds axis-aligned unit vectors, with row 0 zeroed when `reserve`
// is set -- the content table's shape.
func unitRows(n int, reserve bool) [][]float32 {
	rows := make([][]float32, n)
	for index := range rows {
		row := make([]float32, testDim)
		if index > 0 || !reserve {
			row[index%testDim] = 1
		}
		rows[index] = row
	}
	return rows
}

func contentMeta() Meta {
	return Meta{Rows: testRows, Dim: testDim, BaseIndex: 0, Variant: "vec_title_abstract", Version: "v1"}
}

func itemsMeta() Meta {
	return Meta{Rows: testRows, Dim: testDim, BaseIndex: 1, Version: "v1"}
}

func content() Expect {
	return Expect{BaseIndex: 0, SkipRows: 1, Remedy: "make content-vectors"}
}

func items() Expect {
	return Expect{BaseIndex: 1, SkipRows: 0, Remedy: "make index-vectors"}
}

func TestTheRoundTripIsPositional(t *testing.T) {
	rows := unitRows(testRows, true)
	loaded, err := Load(write(t, contentMeta(), rows), content())
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if loaded.Rows() != testRows || loaded.Dim() != testDim || loaded.Version() != "v1" {
		t.Fatalf("meta round-trip: %+v", loaded.Meta())
	}
	for index, want := range rows {
		got := loaded.Row(index)
		for axis := range want {
			if got[axis] != want[axis] {
				t.Fatalf("row %d: got %v, want %v", index, got, want)
			}
		}
	}
}

func TestRowIsBoundedRatherThanPanicking(t *testing.T) {
	loaded, err := Load(write(t, contentMeta(), unitRows(testRows, true)), content())
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Row(-1) != nil || loaded.Row(testRows) != nil {
		t.Fatal("an out-of-range index returned a row")
	}
}

func TestAnAppendCannotReachTheNextRow(t *testing.T) {
	loaded, err := Load(write(t, contentMeta(), unitRows(testRows, true)), content())
	if err != nil {
		t.Fatal(err)
	}
	// The three-index slice is what makes this safe. With a two-index slice the
	// append writes into row 2's first element in place, with no allocation and
	// no error.
	row := loaded.Row(1)
	_ = append(row, 42)
	if loaded.Row(2)[0] == 42 {
		t.Fatal("append through one row wrote into the next")
	}
}

func TestTheBaseIndexIsRefusedRatherThanApplied(t *testing.T) {
	// The whole point of the field: an items artifact handed to a content
	// caller must not load. Both files are otherwise perfectly well-formed.
	stem := write(t, itemsMeta(), unitRows(testRows, false))
	_, err := Load(stem, content())
	if err == nil {
		t.Fatal("an items artifact loaded through the content expectation")
	}
	if !strings.Contains(err.Error(), "base_index") {
		t.Fatalf("error does not name the field: %v", err)
	}

	// And the mirror image, so the test cannot pass by refusing everything.
	if _, err := Load(write(t, itemsMeta(), unitRows(testRows, false)), items()); err != nil {
		t.Fatalf("an items artifact was refused by the items expectation: %v", err)
	}
}

func TestTheRowCountCrossCheckIsOptional(t *testing.T) {
	stem := write(t, contentMeta(), unitRows(testRows, true))

	expect := content()
	expect.Rows = testRows + 1
	if _, err := Load(stem, expect); err == nil {
		t.Fatal("a row count from a different build was accepted")
	}

	// Zero means "any", for a caller with nothing to compare against.
	expect.Rows = 0
	if _, err := Load(stem, expect); err != nil {
		t.Fatalf("Rows=0 should not cross-check: %v", err)
	}
}

func TestLoadRefusesABadShape(t *testing.T) {
	cases := []struct {
		name string
		meta Meta
		rows [][]float32
	}{
		{
			// A truncated file has a plausible size for a smaller matrix, which
			// is why the shape is carried rather than inferred.
			name: "a binary shorter than the declared shape",
			meta: contentMeta(),
			rows: unitRows(testRows, true)[:testRows-1],
		},
		{
			name: "a binary longer than the declared shape",
			meta: contentMeta(),
			rows: append(unitRows(testRows, true), make([]float32, testDim)),
		},
		{
			name: "a zero dimension",
			meta: Meta{Rows: testRows, Dim: 0, BaseIndex: 0, Version: "v"},
			rows: unitRows(testRows, true),
		},
		{
			name: "no rows",
			meta: Meta{Rows: 0, Dim: testDim, BaseIndex: 0, Version: "v"},
			rows: nil,
		},
		{
			// A corrupt meta must not become an allocation.
			name: "a shape that is not an allocation",
			meta: Meta{Rows: 1 << 40, Dim: testDim, BaseIndex: 0, Version: "v"},
			rows: unitRows(testRows, true),
		},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			if _, err := Load(write(t, test.meta, test.rows), content()); err == nil {
				t.Fatal("loaded without complaint")
			}
		})
	}
}

func TestRowsThatAreNotUnitNormAreRefused(t *testing.T) {
	rows := unitRows(testRows, true)
	// Long, not short: a long vector wins every inner product it takes part
	// in, which is the version of this failure that looks most like working
	// code.
	for axis := range rows[3] {
		rows[3][axis] *= 3
	}

	_, err := Load(write(t, contentMeta(), rows), content())
	if err == nil {
		t.Fatal("an un-normalised table loaded")
	}
	// Named, so the message points at the row to look at. base_index 0 here,
	// so row 3 is item 3.
	if !strings.Contains(err.Error(), "row 3, item 3") {
		t.Fatalf("error does not name the worst row: %v", err)
	}
}

func TestSkipRowsIsWhatMakesTheTwoArtifactsBothValid(t *testing.T) {
	// Row 0 zeroed, as policies.normalise leaves the reserved OOV row.
	reserved := unitRows(testRows, true)

	// The content caller skips it and loads.
	if _, err := Load(write(t, contentMeta(), reserved), content()); err != nil {
		t.Fatalf("a zero row 0 was refused by the content expectation: %v", err)
	}

	// The items caller does not skip, because its row 0 is a real item -- so
	// the SAME bytes with an items meta must be refused. If SkipRows were
	// hardcoded either way, one of these two tests would fail.
	meta := itemsMeta()
	if _, err := Load(write(t, meta, reserved), items()); err == nil {
		t.Fatal("a zero row loaded through the items expectation, which skips nothing")
	}
}

func TestTheRemedyIsQuotedIntoTheReadError(t *testing.T) {
	// Nothing written at all: the message has to say what builds it.
	stem := filepath.Join(t.TempDir(), "absent")
	_, err := Load(stem, content())
	if err == nil {
		t.Fatal("a missing artifact loaded")
	}
	if !strings.Contains(err.Error(), "make content-vectors") {
		t.Fatalf("error does not name the remedy: %v", err)
	}
}
