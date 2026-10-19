package catalogue

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/matrix"
	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/service"
)

// Catalogue holds per-item attributes indexed by INTERNAL index, so every
// lookup is a slice index rather than a map hit. At 400 candidates per request
// that difference is real.
type Catalogue struct {
	categories    []int32
	subcategories []int32
	version       string

	// vectors is nil when no content artifact was loaded
	vectors *matrix.Matrix
}

// payload mirrors what scripts/artifacts/dump_catalogue.py writes. `category`
// and `subcategory` are ARRAYS positional by item_idx rather than objects keyed
// by it: the position IS the key, so an object would cost a parse and a map for
// nothing.
type payload struct {
	Version        string  `json:"version"`
	Category       []int32 `json:"category"`
	Subcategory    []int32 `json:"subcategory"`
	NCategories    int32   `json:"n_categories"`
	NSubcategories int32   `json:"n_subcategories"`
}

// Load reads serving/artifacts/catalogue.json and refuses anything the pipeline
// would consume incorrectly.
//
// The dumper checks the last two of these already. Checking again is the
// two-places argument the id map makes: the dumper guards the BUILD, the loader
// guards against a file edited, truncated or swapped afterwards.
func Load(path string) (*Catalogue, error) {
	blob, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w (run `make catalogue`)", path, err)
	}

	var decoded payload
	if err := json.Unmarshal(blob, &decoded); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	if len(decoded.Category) == 0 {
		return nil, fmt.Errorf("%s holds no rows (run `make catalogue`)", path)
	}
	// They index the same items, so a length disagreement means one of them
	// came from a different build.
	if len(decoded.Subcategory) != len(decoded.Category) {
		return nil, fmt.Errorf("%s: %d categories but %d subcategories",
			path, len(decoded.Category), len(decoded.Subcategory))
	}
	// Row 0 is the reserved OOV row. A real category sitting on it would be
	// attributed to every candidate a short source padded with zeros.
	if decoded.Category[0] != 0 || decoded.Subcategory[0] != 0 {
		return nil, fmt.Errorf("%s: row 0 is not reserved (category %d, subcategory %d)",
			path, decoded.Category[0], decoded.Subcategory[0])
	}

	return &Catalogue{
		categories:    decoded.Category,
		subcategories: decoded.Subcategory,
		version:       decoded.Version,
	}, nil
}

// LoadVectors attaches the content matrix from <stem>.bin and <stem>.json.
func (c *Catalogue) LoadVectors(stem string) error {
	if stem == "" {
		return nil
	}
	loaded, err := matrix.Load(stem, matrix.Expect{
		BaseIndex: 0,
		Rows:      len(c.categories),
		SkipRows:  1,
		Remedy:    "make content-vectors",
	})
	if err != nil {
		return err
	}
	c.vectors = loaded
	return nil
}

// CheckMMR refuses the one configuration that is silently broken.
//
// MMR has TWO independent off-switches and only one of them is visible in the
// config: rerank.Options.Lambda, and whether this catalogue holds vectors at
// all. With Lambda below 1.0 and no table, rerank.objectiveAt short-circuits on
// its nil-vectors branch and returns the raw score -- so the slate is
// byte-identical to the pure-relevance one, with no error, nothing in
// degraded_sources and nothing in stage_counts. The knob reads as on and the
// feature is off, for the life of the deployment.
//
// Called from main, beside the other startup refusals, because that is the only
// place both facts are in scope.
func (c *Catalogue) CheckMMR(lambda float64) error {
	if lambda >= 1.0 || c.HasVectors() {
		return nil
	}
	return fmt.Errorf(
		"-lambda %.2f turns MMR on but the catalogue holds no content vectors, so it "+
			"would have no effect and report nothing: run `make content-vectors` and "+
			"pass -content-stem, or leave -lambda at 1.0", lambda)
}

// HasVectors reports whether MMR can run at all.
func (c *Catalogue) HasVectors() bool { return c.vectors != nil }

// Vectors returns nil when no content table was loaded, which DISABLES MMR for
// the request rather than failing it
func (c *Catalogue) Vectors(items []int32) [][]float32 {
	if c.vectors == nil {
		return nil
	}
	out := make([][]float32, len(items))
	for position, item := range items {
		out[position] = c.vectors.Row(int(item))
	}
	return out
}

// lookup is the shape both accessors share: an output sized up front and filled
// POSITIONALLY, never appended to, because the caller pairs it with `items` by
// index and a short return would shift every later candidate's category.
//
// An id outside the table yields 0, the OOV category, rather than an error.
func lookup(table []int32, items []int32) []int32 {
	out := make([]int32, len(items))
	for position, item := range items {
		if item >= 0 && int(item) < len(table) {
			out[position] = table[item]
		}
	}
	return out
}

// Categories backs the ranker's category_idx column.
func (c *Catalogue) Categories(items []int32) []int32 { return lookup(c.categories, items) }

// Subcategories backs the ranker's subcategory_idx column AND the per-category
// cap.
func (c *Catalogue) Subcategories(items []int32) []int32 { return lookup(c.subcategories, items) }

func (c *Catalogue) Size() int       { return len(c.categories) }
func (c *Catalogue) Version() string { return c.version }

// Dim is 0 when no content table was loaded.
func (c *Catalogue) Dim() int {
	if c.vectors == nil {
		return 0
	}
	return c.vectors.Dim()
}

// VectorVersion is empty when no content table was loaded. NOT required to
// equal Version: the categories come from one dumper and the vectors from
// another, and a build can legitimately refresh one. Carried so the startup log
// prints both and a mismatch is visible.
func (c *Catalogue) VectorVersion() string {
	if c.vectors == nil {
		return ""
	}
	return c.vectors.Version()
}

var _ service.Catalogue = (*Catalogue)(nil)
