// Package service is the request pipeline: retrieve, filter, rank, re-rank.
//
// It orchestrates I/O and owns no model. Everything it depends on is an
// interface in this file.
package service

import (
	"context"
	"time"
)

// User is everything the pipeline knows about the caller. The retrievers and
// the ranker must see the SAME snapshot of a user.
type User struct {
	ID string

	// Feats and Names travel together to maintain feature order.
	Feats []float32
	Names []string

	// History is recent items as INTERNAL indices, most recent first, already
	// translated by the feature gateway.
	History []int32

	// Found is false when the store had no row and defaults were served. Not an
	// error -- the pipeline degrades rather than failing on a feature miss
	Found bool
}

// Candidates is one source's answer.
type Candidates struct {
	Items []int32

	// Scores is `retrieval_score` per item, or nil from a source with no model
	// -- a precomputed trending list has ids and nothing else.
	Scores []float32

	// Similarity is `content_similarity` per item, or nil for the same reason.
	Similarity []float32
}

// Columns is the per-item model output the feature builder needs, keyed by item id.
type Columns struct {
	Score      map[int32]float32
	Similarity map[int32]float32
}

// BuildInput is everything the feature builder assembles a row from.
type BuildInput struct {
	User  User
	Items []int32
	// Ranks is per-source retrieval rank, keyed by source name, each aligned
	// with Items.
	Ranks map[string][]int32
	// Model carries what the scoring sources computed.
	Model Columns
}

// Features is one request's model input: one row per candidate, columns in the
// order the exported graph expects.
type Features struct {
	Names []string
	Rows  [][]float32

	// Incomplete counts rows where a MODEL-derived column had no value and was
	// zero-filled.
	Incomplete int
}

// UserStore fetches User before the fan-out from the Feast gateway.
type UserStore interface {
	Fetch(ctx context.Context, userID string, maxHistory int) (User, error)
}

// Retriever is one candidate source
type Retriever interface {
	Name() string
	Budget() time.Duration

	// Retrieve returns candidates best-first. An error or a missed deadline is
	// NOT fatal; the caller records the source as degraded and continues.
	Retrieve(ctx context.Context, user User) (Candidates, error)
}

// FeatureBuilder assembles the model input for one request.
type FeatureBuilder interface {
	Build(ctx context.Context, in BuildInput) (Features, error)
}

// Ranker scores a built feature matrix. Behind this in production is Triton's
// onnxruntime backend holding the exported MMoE graph
type Ranker interface {
	Score(ctx context.Context, features Features) ([]float64, error)
}

// SeenList answers "have we already shown this user this item".
type SeenList interface {
	Blocked(ctx context.Context, userID string, itemIDs []int32) ([]bool, error)
	Record(ctx context.Context, userID string, itemIDs []int32) error
}

// Catalogue supplies the per-item attributes the policy layer needs.
type Catalogue interface {
	// Vectors are L2-normalised content vectors for MMR. Returning nil disables
	// MMR for the request rather than failing it.
	Vectors(items []int32) [][]float32
	// Categories back the ranker's category_idx column.
	Categories(items []int32) []int32

	// Subcategories back the ranker's subcategory_idx column AND the
	// per-category cap.
	Subcategories(items []int32) []int32
}

// Fallback produces a slate when the pipeline cannot. We use popularity because
// it needs no user features, no index and no model.
type Fallback interface {
	Popular(ctx context.Context, n int) ([]int32, error)
}
