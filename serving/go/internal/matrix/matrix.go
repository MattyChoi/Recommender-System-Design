// Package matrix reads float32 matrix artifacts: a raw .bin of little-endian
// values plus a .json declaring its shape.
package matrix

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
)

// UnitNormTolerance is the deviation from 1.0 a row may carry.
//
// Mirrors assert_unit_norm in indexing/build_index.py
const UnitNormTolerance = 1e-3

// maxElements caps the allocation a corrupt or hostile meta can ask for. 2^31
// float32 is 8 GiB, which no artifact in this project approaches.
const maxElements = 1 << 31

// Meta is the sidecar written beside the values.
//
// One struct for both artifacts even though `variant` appears in only one of
// them: an absent JSON key decodes to the zero value, so the cost of the union
// is an empty string, and the alternative is two structs that differ by one
// field and drift.
type Meta struct {
	// Rows and Dim describe the matrix.
	Rows int `json:"rows"`
	Dim  int `json:"dim"`

	// BaseIndex is the item index held by row 0.
	//
	// 1 for items, which drops the reserved OOV row because it must never be a
	// candidate. 0 for content, which keeps it because that table is looked up
	// BY item id.
	BaseIndex int32 `json:"base_index"`

	// Variant is vec_title_abstract or vec_title, on the artifacts built from
	// the content table. Empty on the others.
	Variant string `json:"variant"`

	Version string `json:"version"`
}

// Expect is what the CALLER knows and the reader cannot.
type Expect struct {
	// BaseIndex is the exact value this caller's artifact must declare.
	BaseIndex int32

	// Rows cross-checks the row count against another artifact the caller has
	// already loaded. Zero means "any", for a caller with nothing to compare
	// against.
	Rows int

	// SkipRows exempts leading rows from the norm check.
	SkipRows int

	// Remedy is the make target that produces this artifact, quoted into the
	// read errors. A loader that says a file is missing without saying what
	// builds it has moved the question rather than answered it.
	Remedy string
}

type Matrix struct {
	meta Meta
	// values is ONE CONTIGUOUS BLOCK, row-major. A [][]float32 would be one
	// allocation and one pointer chase per row, and both callers read it in
	// the request path.
	values []float32
}

// Load reads <stem>.json and <stem>.bin, and refuses anything either caller
// would consume incorrectly.
func Load(stem string, expect Expect) (*Matrix, error) {
	// The meta FIRST, so the binary is checked against a declared shape rather
	// than the shape being guessed from the binary.
	header, err := os.ReadFile(stem + ".json")
	if err != nil {
		return nil, fmt.Errorf("read %s.json: %w (run `%s`)", stem, err, expect.Remedy)
	}
	var meta Meta
	if err := json.Unmarshal(header, &meta); err != nil {
		return nil, fmt.Errorf("parse %s.json: %w", stem, err)
	}
	if meta.Rows <= 0 || meta.Dim <= 0 {
		return nil, fmt.Errorf("%s.json declares a %dx%d matrix", stem, meta.Rows, meta.Dim)
	}
	// Divided rather than multiplied, so the guard cannot itself overflow.
	if meta.Rows > maxElements/meta.Dim {
		return nil, fmt.Errorf(
			"%s.json declares %dx%d, which is not an allocation this reads", stem, meta.Rows, meta.Dim)
	}
	if meta.BaseIndex != expect.BaseIndex {
		return nil, fmt.Errorf(
			"%s.json declares base_index %d, this caller requires %d; "+
				"the two matrix artifacts differ in exactly this field, so a mismatch "+
				"means the stem points at the wrong one",
			stem, meta.BaseIndex, expect.BaseIndex)
	}
	if expect.Rows != 0 && meta.Rows != expect.Rows {
		return nil, fmt.Errorf(
			"%s.json declares %d rows but the caller holds %d items; "+
				"the two artifacts are from different builds",
			stem, meta.Rows, expect.Rows)
	}

	body, err := os.ReadFile(stem + ".bin")
	if err != nil {
		return nil, fmt.Errorf("read %s.bin: %w (run `%s`)", stem, err, expect.Remedy)
	}
	want := meta.Rows * meta.Dim * 4
	if len(body) != want {
		return nil, fmt.Errorf(
			"%s.bin is %d bytes, meta declares %dx%d float32 (%d bytes)",
			stem, len(body), meta.Rows, meta.Dim, want)
	}

	values := make([]float32, meta.Rows*meta.Dim)
	for position := range values {
		// Little-endian, EXPLICITLY. Casting the byte slice is faster and
		// silently wrong on a big-endian host -- and produces numbers rather
		// than an error.
		values[position] = math.Float32frombits(
			binary.LittleEndian.Uint32(body[position*4:]),
		)
	}

	loaded := &Matrix{meta: meta, values: values}
	if err := loaded.checkUnitNorm(expect.SkipRows); err != nil {
		return nil, fmt.Errorf("%s: %w", stem, err)
	}
	return loaded, nil
}

// checkUnitNorm refuses a table an inner product would rank by magnitude.
//
// The WORST row is reported rather than the first one past tolerance. A build
// that de-normalised everything and a build with one bad row produce very
// different messages that way, and how bad it got is the first thing worth
// knowing.
func (m *Matrix) checkUnitNorm(skip int) error {
	worst, worstRow := 0.0, -1
	for row := skip; row < m.meta.Rows; row++ {
		start := row * m.meta.Dim
		var sum float32
		for _, value := range m.values[start : start+m.meta.Dim] {
			sum += value * value
		}
		if deviation := math.Abs(math.Sqrt(float64(sum)) - 1.0); deviation > worst {
			worst, worstRow = deviation, row
		}
	}
	if worst > UnitNormTolerance {
		return fmt.Errorf(
			"version %s is not unit-norm (worst deviation %.4g at row %d, item %d); "+
				"inner product would rank by magnitude rather than cosine",
			m.meta.Version, worst, worstRow, int32(worstRow)+m.meta.BaseIndex)
	}
	return nil
}

// Row returns one row, or nil when the index is outside the matrix.
//
// A SUB-SLICE of the loaded block, not a copy: both callers only read it, and
// copying per request to guard against a write nobody makes would cost more
// than the work the row is for. Capacity-bounded with the three-index form, so
// an append cannot reach into the next row.
func (m *Matrix) Row(index int) []float32 {
	if index < 0 || index >= m.meta.Rows {
		return nil
	}
	start := index * m.meta.Dim
	return m.values[start : start+m.meta.Dim : start+m.meta.Dim]
}

// Values is the whole block, row-major, for a caller that scans it linearly
// rather than by row -- which is what an exact search does.
func (m *Matrix) Values() []float32 { return m.values }

func (m *Matrix) Meta() Meta      { return m.meta }
func (m *Matrix) Rows() int       { return m.meta.Rows }
func (m *Matrix) Dim() int        { return m.meta.Dim }
func (m *Matrix) Version() string { return m.meta.Version }
