// Package features assembles the ranker's input matrix.
//
// **This is the third cross-language parity surface**, after the blend and the
// re-ranker, and the one `service/ports.go` has been calling owed since Part K.
// The offline half is models/ranking/dataset.py:build, which produced every
// ranking number this project has measured. This one runs in the request path.
// They are two programs and they must agree.
//
// The structure deliberately mirrors the Python: build a NAMED column per
// candidate, then emit them in the order the graph was exported with. Python
// does `np.stack([columns[name] for name in chosen_features])`; this does the
// same lookup for the same reason. A port that assembled the row positionally
// would be shorter and would silently produce a different matrix the first
// time anyone reordered FEATURES.
//
// # Where the columns come from, and why it is three places
//
//   - MODEL columns (`retrieval_score`, `content_similarity`) come from the
//     retrieval sidecar, which holds the tower, the item embeddings and the
//     content table. ADR 0013.
//   - BLEND columns (`two_tower_rank`, `trending_rank`, `n_sources`) exist
//     only here: they are a property of what retrieval did this request.
//   - STORE columns (`prior_clicks`, `train_clicks`, and the `is_cold_item`
//     derived from one of them) come from the feature gateway, and
//     `category_idx`/`subcategory_idx` from the in-process catalogue.
//
// Only the orchestrator sees all three, which is why this is its job rather
// than the ranker's or the gateway's.
package features

import (
	"context"
	"fmt"

	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/retrieval"
	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/service"
)

// MaxHistory must equal MAX_HISTORY in data_pipeline/features/user_history.py.
//
// `history_length` is the count of real history entries the model was fitted
// on, and the tower pools the same capped list. A serving path that counted
// more than it pooled would report a user as better-known than the embedding
// it just built for them.
const MaxHistory = 50

// Ranker column names, from models/ranking/dataset.py:FEATURES. Spelled out
// rather than imported from the exported sidecar so that a rename on either
// side is a compile-time constant to update, not a runtime surprise.
const (
	ColRetrievalScore    = "retrieval_score"
	ColTwoTowerRank      = "two_tower_rank"
	ColTrendingRank      = "trending_rank"
	ColSources           = "n_sources"
	ColPriorClicks       = "prior_clicks"
	ColTrainClicks       = "train_clicks"
	ColContentSimilarity = "content_similarity"
	ColHistoryLength     = "history_length"
	ColColdItem          = "is_cold_item"
	ColCategory          = "category_idx"
	ColSubcategory       = "subcategory_idx"
)

// Store names for the two cumulative counters, per
// serving/features/columns.py:RANKER_COLUMN_SOURCE. The store names a column
// by what it measures; the model names it by what it was taught to call it.
const (
	StorePriorClicks = "item_impressions_cum"
	StoreTrainClicks = "item_clicks_cum"
)

// ItemStore fetches per-item features. Satisfied by sources.Gateway.
type ItemStore interface {
	ItemRows(ctx context.Context, items []int32) ([][]float32, []string, []bool, error)
}

// Builder assembles one request's matrix.
type Builder struct {
	Items     ItemStore
	Catalogue service.Catalogue

	// Names is the column order the exported graph was built for, read from
	// the `model.columns.json` sidecar the ONNX export writes. Not a default:
	// the order is the one part of the contract no shape check can catch, and
	// a builder that invented it would be asserting rather than reading.
	Names []string
}

// Build produces the matrix in Names order.
func (b *Builder) Build(ctx context.Context, in service.BuildInput) (service.Features, error) {
	if len(b.Names) == 0 {
		return service.Features{}, fmt.Errorf("no column order configured; read model.columns.json")
	}
	if len(in.Items) == 0 {
		return service.Features{Names: b.Names}, nil
	}

	stored, storeNames, _, err := b.Items.ItemRows(ctx, in.Items)
	if err != nil {
		return service.Features{}, fmt.Errorf("item features: %w", err)
	}
	index := positions(storeNames)

	categories := b.Catalogue.Categories(in.Items)
	subcategories := b.Catalogue.Subcategories(in.Items)

	// history_length counts the SAME entries the tower pooled, by the same
	// rule: truncate to MaxHistory first, then drop the reserved OOV row.
	//
	// Order matters and is copied from `recent()` in
	// serving/retrieval/service.py, which is itself copied from what the
	// loader hands the offline build. Filtering before truncating would let a
	// padded history contribute more real entries than the tower saw, and a
	// plain len() would count the padding itself -- either way the ranker is
	// told the user is better known than the embedding just built for them.
	window := in.User.History
	if len(window) > MaxHistory {
		window = window[:MaxHistory]
	}
	history := 0
	for _, item := range window {
		if item > 0 {
			history++
		}
	}

	rows := make([][]float32, len(in.Items))
	incomplete := 0
	for position, item := range in.Items {
		score, hasScore := in.Model.Score[item]
		similarity, hasSimilarity := in.Model.Similarity[item]
		if !hasScore || !hasSimilarity {
			// Counted, not hidden. Zero is not "unknown" to this model:
			// retrieval_score is a cosine, so a zero-filled row asserts the
			// candidate is orthogonal to the user.
			incomplete++
		}

		trainClicks := at(stored, index, position, StoreTrainClicks)

		column := map[string]float32{
			ColRetrievalScore:    score,
			ColTwoTowerRank:      rankOf(in.Ranks, "two_tower", position),
			ColTrendingRank:      rankOf(in.Ranks, "trending", position),
			ColSources:           sourceCount(in.Ranks, position),
			ColPriorClicks:       at(stored, index, position, StorePriorClicks),
			ColTrainClicks:       trainClicks,
			ColContentSimilarity: similarity,
			ColHistoryLength:     float32(history),
			// Derived here rather than fetched, exactly as Python derives it:
			// `(train_clicks == 0)`. A store column claiming to be this flag
			// could disagree with the count sitting next to it.
			ColColdItem:    boolean(trainClicks == 0),
			ColCategory:    lookup(categories, position),
			ColSubcategory: lookup(subcategories, position),
		}

		row := make([]float32, len(b.Names))
		for slot, name := range b.Names {
			value, known := column[name]
			if !known {
				// Refused rather than zero-filled. An unknown column name means
				// the graph was exported against a FEATURES list this build
				// does not implement, and every row would carry a confident
				// zero in that slot.
				return service.Features{}, fmt.Errorf(
					"column %q at slot %d is not built here; the graph and this "+
						"orchestrator are from different versions", name, slot,
				)
			}
			row[slot] = value
		}
		rows[position] = row
	}

	return service.Features{Names: b.Names, Rows: rows, Incomplete: incomplete}, nil
}

// positions indexes the store's columns by name, so a reordering on the
// gateway side is absorbed here rather than shifting every value.
func positions(names []string) map[string]int {
	out := make(map[string]int, len(names))
	for index, name := range names {
		out[name] = index
	}
	return out
}

func at(rows [][]float32, index map[string]int, position int, name string) float32 {
	slot, known := index[name]
	if !known || position >= len(rows) || slot >= len(rows[position]) {
		return 0
	}
	return rows[position][slot]
}

// rankOf reads one source's rank for a candidate, defaulting to Absent.
//
// Absent rather than zero, and it matters: zero is the BEST rank a source can
// give. A missing source defaulting to 0 would tell the ranker that every
// candidate was every absent source's top pick.
func rankOf(ranks map[string][]int32, source string, position int) float32 {
	values, known := ranks[source]
	if !known || position >= len(values) {
		return float32(retrieval.Absent)
	}
	return float32(values[position])
}

// sourceCount is how many sources proposed this candidate.
//
// Counted over every source present in the map, matching Python's
// `sum((ranks[name] < ABSENT) for name in ranks)` -- over what retrieval
// actually ran, not over a configured list. A build with one source makes this
// column constant, which is correct and is what the ranker will have seen if
// it was trained on the same configuration.
func sourceCount(ranks map[string][]int32, position int) float32 {
	count := 0
	for _, values := range ranks {
		if position < len(values) && values[position] < retrieval.Absent {
			count++
		}
	}
	return float32(count)
}

func lookup(values []int32, position int) float32 {
	if position >= len(values) {
		return 0
	}
	return float32(values[position])
}

func boolean(value bool) float32 {
	if value {
		return 1
	}
	return 0
}

// Compile-time proof this satisfies the port the pipeline builds through.
var _ service.FeatureBuilder = (*Builder)(nil)
