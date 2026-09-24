// Package catalogue holds the static per-item attributes the request path
// needs: category and subcategory indices.
//
// In process, not fetched. These move only when the catalogue is rebuilt, and
// a few hundred thousand int32s is under a megabyte -- a network hop per
// request spent on data that stands still is latency for nothing. The feature
// gateway deliberately does not serve them: `item_stats` carries a `category`
// STRING, and the ranker was fitted on the integer index `load_item_tables`
// assigns. Two different things under one word.
package catalogue

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/service"
)

// Table is an in-memory catalogue, indexed by item id with row 0 reserved.
type Table struct {
	category    []int32
	subcategory []int32
	version     string
}

type onDisk struct {
	Version     string  `json:"version"`
	Category    []int32 `json:"category"`
	Subcategory []int32 `json:"subcategory"`
}

// Load reads what scripts/dump_catalogue.py writes.
func Load(path string) (*Table, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading catalogue %s: %w", path, err)
	}
	var decoded onDisk
	if err := json.Unmarshal(raw, &decoded); err != nil {
		return nil, fmt.Errorf("parsing catalogue %s: %w", path, err)
	}
	if len(decoded.Category) == 0 {
		return nil, fmt.Errorf("catalogue %s is empty", path)
	}
	if len(decoded.Category) != len(decoded.Subcategory) {
		// They index the same items. Mismatched, every lookup past the shorter
		// one silently returns 0 -- which is the reserved category, so a whole
		// tail of the catalogue would read as one giant uncategorised bucket
		// and the per-category cap would bind on all of it at once.
		return nil, fmt.Errorf(
			"catalogue %s has %d categories and %d subcategories",
			path, len(decoded.Category), len(decoded.Subcategory),
		)
	}
	return &Table{
		category:    decoded.Category,
		subcategory: decoded.Subcategory,
		version:     decoded.Version,
	}, nil
}

// Vectors returns nil, which disables MMR for every request.
//
// Deliberate, and ADR 0012 is the reason: MMR is a measured NULL on this
// corpus -- the ranker's top 10 already scores 0.590 intra-list diversity, so
// there is nothing to deduplicate -- and the shipped Lambda of 1.0 makes it a
// no-op anyway. Loading a content-vector table per serving process to feed a
// policy that provably does nothing would be megabytes and a cache footprint
// bought for zero measured effect. If Lambda ever moves off 1.0, this is the
// method that has to start returning something.
func (t *Table) Vectors([]int32) [][]float32 { return nil }

// Categories backs both the ranker's category_idx and the per-category cap.
func (t *Table) Categories(items []int32) []int32 { return gather(t.category, items) }

// Subcategories backs the ranker's subcategory_idx.
func (t *Table) Subcategories(items []int32) []int32 { return gather(t.subcategory, items) }

// Version identifies the snapshot, for the health endpoint.
func (t *Table) Version() string { return t.version }

// Size is the number of rows including the reserved row 0.
func (t *Table) Size() int { return len(t.category) }

// gather reads one column for a candidate list.
//
// An out-of-range id yields 0, the reserved index, rather than panicking. The
// alternative is a request path that dies on a candidate the catalogue has not
// heard of -- which happens exactly when the index is a rebuild ahead of this
// artifact, i.e. during a deploy. Returning the reserved value is wrong in a
// small way; taking the process down is wrong in a large one. The count of
// such items is what would make it visible, and that is owed.
func gather(column []int32, items []int32) []int32 {
	out := make([]int32, len(items))
	for index, item := range items {
		if item >= 0 && int(item) < len(column) {
			out[index] = column[item]
		}
	}
	return out
}

// Compile-time proof this satisfies the port the pipeline reads through.
var _ service.Catalogue = (*Table)(nil)
