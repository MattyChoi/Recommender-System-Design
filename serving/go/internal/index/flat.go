// Package index is the RETRIEVAL FALLBACK, not the retrieval path.
//
// # Where this sits
//
// ADR 0013 puts the FAISS index and the two-tower user encoder in a Python
// sidecar, reached over gRPC with its own deadline, because a FAISS index is a
// C++ artifact the Go process cannot read and a PyTorch forward pass is not
// something it can run either. The healthy path does not come through here.
//
// This is rung 2 of that ADR's degradation ladder. When the sidecar is
// unreachable but §14.4's Redis cache still holds the user's embedding, the
// orchestrator runs exact search in-process rather than dropping its strongest
// source: correct results, more CPU, no ANN. With no cached embedding either,
// the source reports degraded and the popularity fallback serves the slate.
//
// **This is a fallback and not a baseline.** ADR 0002 chose HNSW on a measured
// recall/QPS curve, and a server sitting on this rung is serving correct
// results without the index it was built around -- invisible in every system
// metric, which is exactly why HealthResponse.index_kind reports "flat" and
// why Flat.Kind() exists. BenchmarkSearch measures what this rung costs so the
// degradation has a number attached; it is not an argument for skipping the
// index.
//
// It is also the fourth cross-language parity surface in this project: the
// sidecar and this file must agree on the top-k, or a sidecar outage silently
// serves a different slate than the healthy path. Owed, not paid.
//
// # What the index is not allowed to forget
//
// Inner product is cosine ONLY because the towers L2-normalise, and an
// un-normalised table still loads, still searches and still returns plausible
// neighbours -- it ranks by |u||v|cos instead of cos, so long vectors win and
// nothing in the output says so. indexing/build_index.py refuses such a table
// at build time; Load refuses it here, because the artifact can be replaced
// without the builder running again.
package index

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
)

// Meta is the sidecar written beside the vectors.
type Meta struct {
	// Rows and Dim describe the matrix. Carried rather than inferred from the
	// file size: a truncated file has a plausible size for a smaller matrix,
	// and inferring would silently serve the prefix.
	Rows int `json:"rows"`
	Dim  int `json:"dim"`

	// BaseIndex is the item index held by row 0 of this matrix.
	//
	// One, not zero. Index 0 is the reserved out-of-vocabulary row and is not
	// an article, so it never enters an index -- the same shift
	// indexing/build_index.py applies when it drops the first row and adds one
	// back to every id FAISS returns. Stated in the artifact rather than
	// assumed in the reader, because an off-by-one here returns real ids for
	// the wrong articles.
	BaseIndex int32 `json:"base_index"`

	// Version is the snapshot, echoed into the response's index_version.
	Version string `json:"version"`
}

// Flat is an exact inner-product index held in memory.
type Flat struct {
	meta Meta
	// vectors is one contiguous block, row-major. A [][]float32 would be one
	// allocation and one pointer chase per row; the search loop is the hot
	// path and reads it linearly.
	vectors []float32
}

// UnitNormTolerance matches assert_unit_norm in indexing/build_index.py. The
// same number in both places for the same reason: a table that one accepts and
// the other rejects is a deployment that fails after the build passed.
const UnitNormTolerance = 1e-3

// Load reads the matrix and its sidecar. The stem is the path without an
// extension: stem.json and stem.bin.
func Load(stem string) (*Flat, error) {
	raw, err := os.ReadFile(stem + ".json")
	if err != nil {
		return nil, fmt.Errorf("reading index meta: %w", err)
	}
	var meta Meta
	if err := json.Unmarshal(raw, &meta); err != nil {
		return nil, fmt.Errorf("parsing %s.json: %w", stem, err)
	}
	if meta.Rows <= 0 || meta.Dim <= 0 {
		return nil, fmt.Errorf("index meta declares %d rows of %d", meta.Rows, meta.Dim)
	}

	body, err := os.ReadFile(stem + ".bin")
	if err != nil {
		return nil, fmt.Errorf("reading index vectors: %w", err)
	}
	want := meta.Rows * meta.Dim * 4
	if len(body) != want {
		// Checked rather than trusted. A short read here is a matrix whose rows
		// are offset from the row they claim to be, which searches fine and
		// returns the wrong article for every query.
		return nil, fmt.Errorf(
			"%s.bin is %d bytes, meta declares %dx%d float32 (%d bytes)",
			stem, len(body), meta.Rows, meta.Dim, want,
		)
	}

	vectors := make([]float32, meta.Rows*meta.Dim)
	for position := range vectors {
		// Little-endian, explicitly. The alternative is casting the byte slice,
		// which is faster and silently wrong on a big-endian host -- and
		// produces numbers rather than an error, which is the failure this
		// project has been bitten by before.
		bits := binary.LittleEndian.Uint32(body[position*4:])
		vectors[position] = math.Float32frombits(bits)
	}

	flat := &Flat{meta: meta, vectors: vectors}
	if err := flat.checkUnitNorm(); err != nil {
		return nil, err
	}
	return flat, nil
}

func (f *Flat) checkUnitNorm() error {
	worst := 0.0
	for row := 0; row < f.meta.Rows; row++ {
		var sum float32
		start := row * f.meta.Dim
		for _, value := range f.vectors[start : start+f.meta.Dim] {
			sum += value * value
		}
		if deviation := math.Abs(math.Sqrt(float64(sum)) - 1.0); deviation > worst {
			worst = deviation
		}
	}
	if worst > UnitNormTolerance {
		return fmt.Errorf(
			"index vectors are not unit-norm (worst deviation %.4g); inner product "+
				"would rank by magnitude rather than cosine", worst,
		)
	}
	return nil
}

// Meta reports what was loaded, for the health endpoint.
func (f *Flat) Meta() Meta { return f.meta }

// Kind is "flat". Reported so that a server running exact search is saying so
// rather than being assumed to be running the index ADR 0002 chose.
func (f *Flat) Kind() string { return "flat" }

// Dim is the query width callers must supply.
func (f *Flat) Dim() int { return f.meta.Dim }

// Search returns the k best item indices for one query, best first.
//
// Exact, so there is no recall parameter and no efSearch to drift. The scan is
// a single pass with a k-sized insertion list rather than a sort of every
// score: k is 100-ish against 160K rows, so sorting the full array would spend
// ~20x the work of the scan itself on candidates nobody will look at.
//
// # Ties keep the EARLIER row, and FAISS cannot be matched on this
//
// The rule here is numpy's: the first maximum wins, which is also what
// `rerank.Select` does, so the codebase has one tie-break convention.
//
// ⚠️ **FAISS does not.** Measured against `IndexFlatIP` on two identical rows,
// same index and same query:
//
//	k=1    keeps the EARLIER row   (a first-maximum scan)
//	k=5    keeps the LATER row     (a heap whose replacement admits equals)
//	k=100  keeps the LATER row
//
// So there is no single order to match -- FAISS's depends on k, because the
// k=1 path and the heap path are different code. Reproducing it would mean
// branching on k in the serving hot path to mirror another library's
// internals, and pinning two undocumented behaviours that can move
// independently on a version bump. The parity test therefore compares tied
// groups as SETS: same items, same scores, order within a tie unconstrained.
//
// A tie needs BIT-IDENTICAL embeddings, so on the shipped `both` arm it is
// measure-zero. It is not measure-zero on the content-only ablation arm, where
// two articles with the same title and category features get the same vector
// with no ID embedding to separate them.
func (f *Flat) Search(query []float32, k int) ([]int32, []float32, error) {
	if len(query) != f.meta.Dim {
		return nil, nil, fmt.Errorf("query has %d dimensions, index has %d", len(query), f.meta.Dim)
	}
	if k <= 0 {
		return nil, nil, fmt.Errorf("k must be positive, got %d", k)
	}
	if k > f.meta.Rows {
		k = f.meta.Rows
	}

	topItems := make([]int32, 0, k)
	topScores := make([]float32, 0, k)

	for row := 0; row < f.meta.Rows; row++ {
		start := row * f.meta.Dim
		var score float32
		// Accumulated in float32, matching the Python side: numpy's matmul
		// accumulates in the input dtype. Summing in float64 would give a
		// slightly different score, invisible until two candidates sit close
		// enough for it to flip their order.
		for offset, value := range f.vectors[start : start+f.meta.Dim] {
			score += value * query[offset]
		}

		// `<=`, so a row merely tying the k-th best is rejected and the earlier
		// row keeps the slot.
		if len(topScores) == k && score <= topScores[k-1] {
			continue
		}
		item := int32(row) + f.meta.BaseIndex
		// Linear insertion into a k-sized list. At k~100 this beats a heap:
		// the comparison above rejects almost every row, so the insertion runs
		// rarely and the list stays in cache.
		//
		// STRICTLY less, so the scan stops AT an equal score and the new row is
		// placed after it. Ties keep the earlier row; see the note on Search.
		place := len(topScores)
		for place > 0 && topScores[place-1] < score {
			place--
		}
		if len(topScores) < k {
			topScores = append(topScores, 0)
			topItems = append(topItems, 0)
		}
		copy(topScores[place+1:], topScores[place:])
		copy(topItems[place+1:], topItems[place:])
		topScores[place] = score
		topItems[place] = item
	}
	return topItems, topScores, nil
}

// Stem builds the artifact path a version directory holds.
func Stem(root, version, name string) string {
	return filepath.Join(root, version, name)
}
