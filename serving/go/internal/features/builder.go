// Package features assembles the ranker's input matrix.

// # Where the columns come from,
//
//   - MODEL columns (`retrieval_score`, `content_similarity`) come from the
//     FAISS index sidecar
//   - BLEND columns (`two_tower_rank`, `trending_rank`, `n_sources`) exist only
//     here: they are a property of what retrieval did THIS request.
//   - STORE columns (`prior_clicks`, `train_clicks`, and the `is_cold_item`
//     derived from one of them) come from the feature gateway, and
//     `category_idx`/`subcategory_idx` from the in-process catalogue.

package features

import (
	"context"
	"fmt"

	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/retrieval"
	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/service"
)

// MaxHistory must equal MAX_HISTORY in data_pipeline/features/user_history.py.
const MaxHistory = 50

// Ranker column names, from models/ranking/dataset.py:FEATURES.
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
	Store     ItemStore
	Catalogue service.Catalogue

	// Names is the column order the exported graph was built for, read from
	// `model.columns.json`
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

	itemFeats, itemFeatNames, _, err := b.Store.ItemRows(ctx, in.Items)
	if err != nil {
		return service.Features{}, fmt.Errorf("item features: %w", err)
	}
	index := positions(itemFeatNames)

	categories := b.Catalogue.Categories(in.Items)
	subcategories := b.Catalogue.Subcategories(in.Items)

	// truncate then filter, exactly like how it was done in training and in the
	// FAISS index service
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
		// fetch retrieval
		retrievalScore, hasRetScore := in.Model.Score[item]
		similarity, hasSimilarity := in.Model.Similarity[item]
		if !hasRetScore || !hasSimilarity {
			incomplete++
		}

		trainClicks := at(itemFeats, index, position, StoreTrainClicks)

		column := map[string]float32{
			ColRetrievalScore:    retrievalScore,
			ColTwoTowerRank:      rankOf(in.Ranks, "two_tower", position),
			ColTrendingRank:      rankOf(in.Ranks, "trending", position),
			ColSources:           sourceCount(in.Ranks, position),
			ColPriorClicks:       at(itemFeats, index, position, StorePriorClicks),
			ColTrainClicks:       trainClicks,
			ColContentSimilarity: similarity,
			ColHistoryLength:     float32(history),
			ColColdItem:          boolean(trainClicks == 0),
			ColCategory:          lookup(categories, position),
			ColSubcategory:       lookup(subcategories, position),
		}

		row := make([]float32, len(b.Names))
		for slot, name := range b.Names {
			value, known := column[name]
			if !known {
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
func rankOf(ranks map[string][]int32, source string, position int) float32 {
	values, known := ranks[source]
	if !known || position >= len(values) {
		return float32(retrieval.Absent)
	}
	return float32(values[position])
}

// sourceCount is how many sources proposed this candidate.
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
