// Package service is the request pipeline: retrieve, filter, rank, re-rank.
//
// It orchestrates I/O and owns no model. Everything it depends on is an
// interface in this file, for one reason that is worth stating plainly: the
// interesting behaviour of this stage is what it does when a dependency is
// SLOW or BROKEN, and that is exactly what a real dependency will not do on
// demand. A fake retriever that sleeps past its budget is the only way to test
// the deadline that matters in production.
package service

import (
	"context"
	"time"
)

// User is everything the pipeline knows about the caller, fetched ONCE before
// the fan-out.
//
// Once, and before, for a reason §14.2 states and this type enforces: the
// retrievers and the ranker must see the SAME snapshot of a user. Two lookups
// mid-request can straddle a materialisation and produce a slate retrieved for
// who the user was and ranked for who they are, which is not a state any test
// would think to construct.
type User struct {
	// ID is the external id -- Feast's entity key, and what the retrieval
	// sidecar takes. There is no internal user index in the serving path:
	// the tower embeds no user id, so one would exist only to key a cache.
	ID string

	// Feats and Names travel together. Order is the one part of this contract
	// that nothing downstream can check by shape -- a permuted vector is the
	// right length and the right dtype and produces a believable embedding for
	// a user who does not exist.
	Feats []float32
	Names []string

	// History is recent items as INTERNAL indices, most recent first, already
	// translated by the feature gateway.
	History []int32

	// Found is false when the store had no row and defaults were served. Not
	// an error -- §14.2 degrades rather than failing on a feature miss -- but a
	// miss rate that climbs is the signal that materialisation has stopped.
	Found bool
}

// UserStore fetches User before the fan-out.
//
// Behind it is the Python feature gateway: Feast's online store is written
// through Feast's SDK in a Feast-internal protobuf layout, so this is a gRPC
// hop rather than a Redis read. Reimplementing that encoding here would couple
// the request path to a third-party wire format whose changes look like
// corrupt features rather than like a version mismatch.
type UserStore interface {
	Fetch(ctx context.Context, userID string, maxHistory int) (User, error)
}

// Retriever is one candidate source.
type Retriever interface {
	Name() string

	// Budget is this source's own deadline. Per-source rather than shared:
	// Part I measured that a source can contribute features without
	// contributing candidates, so a slow one is worth dropping rather than
	// waiting for -- partial results beat a timeout.
	Budget() time.Duration

	// Retrieve returns candidates best-first. An error or a missed deadline is
	// not fatal; the caller records the source as degraded and continues.
	//
	// Takes the whole User rather than an id: the two-tower source needs the
	// feature vector and the history to encode a query at all, and the cheap
	// sources simply ignore them.
	Retrieve(ctx context.Context, user User) (Candidates, error)
}

// Candidates is one source's answer.
//
// More than a list of ids, because two of the ranker's eleven columns are
// MODEL-derived and only a source holding the model can produce them.
// `retrieval_score` and `content_similarity` are computed by the retrieval
// sidecar, which already has the tower, the item embeddings and the content
// table; recomputing them in the orchestrator would need a second copy of all
// three and would be a skew surface for nothing.
type Candidates struct {
	// Items in the source's own order, best first.
	Items []int32

	// Scores is `retrieval_score` per item, or nil from a source that has no
	// model -- a precomputed trending list has ids and nothing else.
	Scores []float32

	// Similarity is `content_similarity` per item, or nil for the same reason.
	Similarity []float32
}

// Columns is the per-item model output the feature builder needs, keyed by
// item id.
//
// A MAP rather than parallel slices, and this is the one place in the pipeline
// where that is right. Everywhere else the arrays are positional and a map
// would lose the order the blend decided; here the lookup crosses the blend,
// which dedupes and reorders, so an item's score has to find it again by
// identity rather than by position.
type Columns struct {
	Score      map[int32]float32
	Similarity map[int32]float32
}

// BuildInput is everything the feature builder assembles a row from.
//
// A struct rather than five positional arguments, because four of them are
// slices and maps of numbers: a caller that transposed two would compile, run,
// and produce a matrix the ranker scores without complaint.
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
//
// **Names travels with Rows on purpose.** Column ORDER is the one part of the
// serving contract that could not be baked into the ONNX graph -- the
// reciprocal on ranks, the log1p on counts and the fitted standardiser are all
// inside it. So order is checked rather than assumed: the export writes it
// beside the model as `model.columns.json`, and a Ranker compares what it was
// handed against what the graph was built for. A permuted column is a
// correctly-shaped input that scores a different world with no error raised
// anywhere, and it is the one failure here that nothing else can catch.
type Features struct {
	Names []string
	Rows  [][]float32

	// Incomplete counts rows where a MODEL-derived column had no value and was
	// zero-filled. Non-zero whenever the retrieval sidecar degraded to rung 2,
	// which serves candidates but cannot produce `retrieval_score` or
	// `content_similarity`.
	//
	// Carried rather than swallowed because zero is not "unknown" to this
	// model: `retrieval_score` is a cosine, so a zero-filled row claims the
	// candidate is ORTHOGONAL to the user -- a confident statement, made by
	// accident, on every candidate at once. The pipeline reports it; it does
	// not fail on it, because a degraded slate beats no slate.
	Incomplete int
}

// FeatureBuilder assembles the model input for one request.
//
// Its own port, because the columns come from three places -- the blend
// (retrieval score, per-source ranks, source count), the catalogue (click
// counts, category, cold flag) and the feature store (history length) -- and
// only the orchestrator sees all three. Folding it into the Ranker would have a
// model client reaching back for data it has no business knowing about.
//
// **This is the third cross-language parity surface**, after the blend and the
// re-ranker, and it is now paid: internal/features asserts equality with
// models.ranking.dataset.feature_columns against a generated fixture, the same
// way the other two do.
type FeatureBuilder interface {
	Build(ctx context.Context, in BuildInput) (Features, error)
}

// Ranker scores a built feature matrix. Behind this in production is Triton's
// onnxruntime backend holding the exported MMoE graph; the orchestrator never
// links a model.
//
// ADR 0003 records that this model measured WORSE than the LightGBM booster it
// replaces here -- -0.0040 NDCG@10, about four times the measured noise floor
// -- and why it ships regardless.
type Ranker interface {
	Score(ctx context.Context, features Features) ([]float64, error)
}

// SeenList answers "have we already shown this user this item".
//
// The implementation is a Redis-backed Bloom filter, whose error is ONE-SIDED
// by construction: it may hide a fresh item, and it cannot re-show a seen one.
// Part L measured the failure mode at the other end -- past capacity the filter
// saturates and blocks EVERY candidate -- which is why Blocked's answer is
// treated as advice by the selector rather than as a veto.
type SeenList interface {
	Blocked(ctx context.Context, userID string, items []int32) ([]bool, error)
	Record(ctx context.Context, userID string, items []int32) error
}

// Catalogue supplies the per-item attributes the policy layer needs. Held in
// process: it is a few megabytes and changes only when the index is rebuilt,
// so a network hop per request would be latency spent on static data.
type Catalogue interface {
	// Vectors are L2-normalised content vectors for MMR. Returning nil
	// disables MMR for the request rather than failing it.
	Vectors(items []int32) [][]float32
	// Categories back the per-category cap, and the ranker's category_idx.
	Categories(items []int32) []int32
	// Subcategories back the ranker's subcategory_idx. Held here rather than
	// fetched with the other item features because they are STATIC -- an
	// article's category does not change -- and a network hop per request
	// spent on data that only moves when the catalogue is rebuilt is latency
	// for nothing.
	Subcategories(items []int32) []int32
}

// Fallback produces a slate when the pipeline cannot. Popularity needs no user
// features, no index and no model, which is what makes it usable precisely
// when the things that do have failed.
type Fallback interface {
	Popular(ctx context.Context, n int) ([]int32, error)
}
