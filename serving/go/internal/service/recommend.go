package service

import (
	"context"
	"sort"
	"sync"
	"time"

	"go.opentelemetry.io/otel/attribute"
	otelcodes "go.opentelemetry.io/otel/codes"

	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/rerank"
	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/retrieval"
	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/tracing"
)

// Config is everything about the pipeline that is a decision rather than a
// dependency. Each default cites where it came from; a serving constant with
// no provenance is a number someone typed.
type Config struct {
	// Quotas per retriever, in Retrievers order. Part I measured that at a
	// fixed budget no blend beats giving every slot to the strongest source,
	// so the production configuration is [max, 0, 0, ...] -- the other sources
	// still tag candidates with their rank, which costs no slot.
	Quotas []int

	// MaxCandidates entering the ranker.
	MaxCandidates int

	// Deadline for the whole request. docs/design.md budgets 90ms end to end.
	Deadline time.Duration

	// RankDeadline is the ranker's slice of it. Exceeded, the pipeline ships
	// the blended retrieval order rather than failing -- which is a real
	// degradation and is reported as one.
	RankDeadline time.Duration

	// FeatureDeadline bounds the pre-fan-out user fetch. docs/design.md
	// budgets 8ms for it. Its own deadline rather than a share of the
	// request's, because it is the one stage that runs BEFORE anything else
	// and so is the only one that can spend the whole budget alone.
	FeatureDeadline time.Duration

	// MaxHistory is how much of the user's history to fetch. The tower pools
	// at most 50, so more is bytes on the wire nothing reads.
	MaxHistory int

	// Lambda for MMR. ADR 0012 ships 1.0: MMR is a measured null on this
	// corpus because the ranker's top 10 already scores 0.590 intra-list
	// diversity, so there is nothing to deduplicate.
	Lambda float64

	// CapPerCategory, nil to disable. ADR 0012 keeps 3 as a cheap guardrail
	// that binds rarely and binds exactly when one subcategory floods a slate.
	CapPerCategory *int

	// Epsilon and ExploreSlots. ADR 0012 ships 0.1 over the last 2 of 10 slots:
	// 84% more catalogue coverage for -0.0005 NDCG, and the only source of
	// logged propensities an off-policy estimator can consume.
	Epsilon      float64
	ExploreSlots int

	ModelVersion string
	IndexVersion string
}

// Service is the orchestrator.
type Service struct {
	Retrievers []Retriever
	Users      UserStore
	Features   FeatureBuilder
	Ranker     Ranker
	Seen       SeenList
	Catalogue  Catalogue
	Fallback   Fallback
	Config     Config

	// Now is injectable so the latency fields can be asserted in tests without
	// sleeping. Nil means time.Now.
	Now func() time.Time
}

// Request is one call's inputs, in INTERNAL terms: item ids here are the
// integer indices the models were trained on, not the external ids the API
// speaks. Translation happens once, at the transport boundary, so nothing in
// the pipeline has to carry two vocabularies.
type Request struct {
	UserID string

	// Surface is home_rail | search | detail_page. Carried but not yet acted
	// on: per-surface policy is a Config split this pipeline does not have.
	// Recorded here rather than dropped at the transport layer so that when it
	// does start mattering, the plumbing is not the change.
	Surface string

	NumResults int

	// Exclude is what the CALLER has already rendered, on top of the
	// server's own seen-list. Not the same thing: the seen-list is what we
	// showed, this is what the client knows it is showing right now.
	Exclude []int32
}

// Result is the pipeline's output, in the shape the gRPC layer serialises.
// Kept separate from the generated protobuf type so the pipeline can be tested
// without a transport, and so a field is added here deliberately rather than
// because the proto grew one.
type Result struct {
	Items []int32

	// Scores is the RANKER's raw output for each served item.
	Scores []float64
	// Objective is the value each slot was ordered by, after the policy layer.
	// Equal to Scores when no policy fired, which is the shipped configuration
	// (ADR 0012 sets Lambda to 1.0); different the moment a cap or an
	// exploration slot binds. Both are carried because the API promises the
	// ordering value and a debugger wants the ranker's.
	Objective []float64

	Propensity []float64

	// Sources names the retrievers that PROPOSED each served item, aligned
	// with Items. Proposed, not scored: Part I's blend hands one source the
	// whole quota while the others tag every candidate with their own rank at
	// no slot cost, so "which sources scored this" would name all of them and
	// answer nothing.
	Sources [][]string

	UsedFallback    bool
	DegradedSources []string
	StageCounts     map[string]int
	StageLatency    map[string]time.Duration
	TotalLatency    time.Duration
}

// degradedRanker is the name reported when the ranker missed its deadline or
// errored and the blended retrieval order was served instead. A named constant
// because an alert fires on it.
const degradedRanker = "ranker"

// fallbackSource is the source name reported for a popularity slate.
const fallbackSource = "popularity"

// degradedFeatures is reported when the feature gateway could not be reached
// and the pipeline ran against a cold user. Named because it alerts: every
// downstream stage is quietly worse when this fires, and none of them fail.
const degradedFeatures = "features"

// degradedColumns is reported when the ranker scored rows whose model-derived
// columns were zero-filled. Distinct from degradedRanker: the model RAN, on
// worse input. Same slate shape, worse ordering, no error anywhere.
const degradedColumns = "feature_columns"

func (s *Service) now() time.Time {
	if s.Now != nil {
		return s.Now()
	}
	return time.Now()
}

// Recommend runs the pipeline. It returns an error only when it cannot produce
// a slate at all; every recoverable failure degrades and is reported in the
// Result, because **a recommender that silently degrades looks healthy**.
func (s *Service) Recommend(ctx context.Context, userID string, numResults int) (Result, error) {
	return s.Run(ctx, Request{UserID: userID, NumResults: numResults})
}

// Run is the pipeline. Recommend is the two-argument form for the common case;
// this one takes everything the API can send.
func (s *Service) Run(ctx context.Context, request Request) (Result, error) {
	userID, numResults := request.UserID, request.NumResults
	started := s.now()
	result := Result{
		StageCounts:  map[string]int{},
		StageLatency: map[string]time.Duration{},
	}

	ctx, cancel := context.WithTimeout(ctx, s.Config.Deadline)
	defer cancel()

	// Instrumented directly rather than through an injected interface: the
	// OpenTelemetry API is a no-op with no provider registered, so this costs
	// a couple of nil checks when tracing is off and the tests configure
	// nothing.
	ctx, span := tracing.Tracer().Start(ctx, "recommend")
	defer span.End()

	// --- 0. The user, fetched once, before anything fans out -----------------
	//
	// Before, because the two-tower source cannot encode a query without the
	// feature vector and the history. Once, because the retrievers and the
	// ranker must see the same snapshot: two lookups mid-request can straddle
	// a materialisation and produce a slate retrieved for who the user was and
	// ranked for who they are.
	var user User
	s.stage(ctx, "features", &result, func(inner context.Context) {
		user = s.fetchUser(inner, userID, &result)
	})

	// --- 1. Retrieval fan-out, each source on its own deadline ---------------
	var collected []retrieval.Source
	var model Columns
	s.stage(ctx, "retrieval", &result, func(inner context.Context) {
		var degraded []string
		collected, model, degraded = s.fanOut(inner, user)
		result.DegradedSources = append(result.DegradedSources, degraded...)
	})

	blended, err := retrieval.Blend(collected, s.Config.MaxCandidates, s.Config.Quotas)
	if err != nil {
		return result, err
	}
	result.StageCounts["retrieved"] = len(blended.Items)

	// --- 2. Filter -----------------------------------------------------------
	candidates := blended.Items
	var blocked []bool
	s.stage(ctx, "filter", &result, func(inner context.Context) {
		blocked = s.blocked(inner, userID, candidates, request.Exclude, &result)
	})
	result.StageCounts["after_filter"] = countUnblocked(blocked, len(candidates))

	// **Only an empty candidate set triggers the fallback, not an empty
	// FILTERED set.** A saturated Bloom filter blocks every candidate -- Part L
	// measured exactly that past capacity -- and falling back there would throw
	// away a hundred good candidates because a filter misbehaved. The selector
	// already yields an all-blocking mask rather than returning a blank slate,
	// so the filter stays advice. The count is still recorded, because
	// `after_filter` dropping to zero is the signal that the filter is
	// saturated and needs its capacity revisited.
	if len(candidates) == 0 {
		return s.servePopularity(ctx, numResults, started, result)
	}

	// --- 3. Rank, on a hard deadline ----------------------------------------
	var scores []float64
	var ranked bool
	var incomplete int
	s.stage(ctx, "rank", &result, func(inner context.Context) {
		scores, ranked, incomplete = s.score(inner, BuildInput{
			User:  user,
			Items: candidates,
			Ranks: ranksByName(blended),
			Model: model,
		})
	})
	if !ranked {
		result.DegradedSources = append(result.DegradedSources, degradedRanker)
	}
	// Reported separately from a failed ranker, because it is a different
	// failure with the same appearance: the model ran, on rows whose
	// model-derived columns were zero-filled because retrieval degraded to
	// rung 2. The slate is real and the scores are worse, and nothing else in
	// the response would say so.
	if incomplete > 0 {
		result.StageCounts["incomplete_rows"] = incomplete
		result.DegradedSources = append(result.DegradedSources, degradedColumns)
	}
	result.StageCounts["scored"] = len(scores)

	// --- 4. Re-rank ----------------------------------------------------------
	var slate rerank.Slate
	s.stage(ctx, "rerank", &result, func(context.Context) {
		slate, err = rerank.Select(scores, candidates, numResults, rerank.Options{
			Vectors:      s.Catalogue.Vectors(candidates),
			Lambda:       s.Config.Lambda,
			Categories:   s.Catalogue.Categories(candidates),
			Cap:          s.Config.CapPerCategory,
			Blocked:      blocked,
			Epsilon:      s.Config.Epsilon,
			ExploreSlots: s.Config.ExploreSlots,
			Rand:         nil,
		})
	})
	if err != nil {
		return result, err
	}
	result.StageCounts["served"] = len(slate.Items)

	result.Items = slate.Items
	result.Propensity = slate.Propensity
	result.Objective = slate.Objective
	result.Sources = sourcesFor(blended, slate.Rows)
	result.Scores = make([]float64, len(slate.Rows))
	for index, row := range slate.Rows {
		result.Scores[index] = scores[row]
	}

	// Recording what was shown is what makes the seen-list mean anything on the
	// NEXT request. A failure here degrades silently by design: the slate is
	// already correct, and refusing to return it because bookkeeping failed
	// would turn a future duplicate into a present outage.
	if s.Seen != nil {
		_ = s.Seen.Record(ctx, userID, slate.Items)
	}

	// Recorded on the ROOT span, after everything that can degrade has run, so
	// a collector tail-sampling on `recsys.degraded` keeps the whole trace
	// rather than one leaf of it.
	tracing.Degraded(span, result.DegradedSources)
	span.SetAttributes(
		attribute.Bool(tracing.AttrFallback, result.UsedFallback),
		attribute.Int(tracing.AttrIncomplete, result.StageCounts["incomplete_rows"]),
		attribute.Int(tracing.AttrCandidates, len(candidates)),
	)

	result.TotalLatency = s.now().Sub(started)
	return result, nil
}

// stage runs one pipeline step inside its own span and records its duration.
//
// One helper rather than a span per call site, because the span name and the
// StageLatency key must agree: a waterfall whose bars are named differently
// from the metrics is two views of the same request that cannot be lined up.
func (s *Service) stage(ctx context.Context, name string, result *Result, run func(context.Context)) {
	ctx, span := tracing.Tracer().Start(ctx, name)
	defer span.End()

	started := s.now()
	run(ctx)
	result.StageLatency[name] = s.now().Sub(started)
}

// fetchUser gets the caller's features and history, degrading to a cold user.
//
// Degrading rather than failing is §14.2's rule, and the direction is a
// decision: a cold user still gets a slate, from the sources that need no
// features and from the ranker's remaining columns. Failing the request would
// turn one materialisation gap into an outage for everybody it touches.
func (s *Service) fetchUser(ctx context.Context, userID string, result *Result) User {
	cold := User{ID: userID}
	if s.Users == nil {
		return cold
	}

	fetchCtx, cancel := context.WithTimeout(ctx, s.Config.FeatureDeadline)
	defer cancel()

	user, err := s.Users.Fetch(fetchCtx, userID, s.Config.MaxHistory)
	if err != nil {
		result.DegradedSources = append(result.DegradedSources, degradedFeatures)
		return cold
	}
	// A store MISS is not a degradation -- a genuinely new user has no row and
	// that is the correct answer. Only a failed call is. Conflating them would
	// make the degraded-source alert fire on normal cold-start traffic, which
	// is how an alert gets muted.
	user.ID = userID
	return user
}

// fanOut queries every retriever in parallel, each under its own budget.
func (s *Service) fanOut(ctx context.Context, user User) ([]retrieval.Source, Columns, []string) {
	sources := make([]retrieval.Source, len(s.Retrievers))
	answers := make([]Candidates, len(s.Retrievers))
	degradedFlags := make([]bool, len(s.Retrievers))

	var wait sync.WaitGroup
	for index, source := range s.Retrievers {
		wait.Add(1)
		go func(index int, source Retriever) {
			defer wait.Done()
			sourceCtx, cancel := context.WithTimeout(ctx, source.Budget())
			defer cancel()

			// A span per source, inside the retrieval span. This is the
			// picture §16.1 wants: parallel bars, each ending at its own
			// budget. The budget is an attribute so a bar that stops exactly
			// at its deadline reads as a DROPPED source rather than a fast one
			// that happened to return nothing -- from the timings alone those
			// are identical.
			sourceCtx, span := tracing.Tracer().Start(sourceCtx, "retrieve")
			defer span.End()
			tracing.Source(span, source.Name(), source.Budget())

			got, err := source.Retrieve(sourceCtx, user)
			// Results are written to a PRE-SIZED slice at a fixed index rather
			// than appended under a mutex. Appending would make the source
			// order depend on which goroutine finished first, and source order
			// is not cosmetic here: Quotas[i] belongs to Retrievers[i], so a
			// reordered slice silently hands the budget to a different source.
			if err != nil {
				degradedFlags[index] = true
				got = Candidates{}
				// Recorded on the span, not just counted: the trace is what
				// answers "what else was happening when this source dropped",
				// and RecordError is what a tail sampler can key on.
				span.RecordError(err)
				span.SetStatus(otelcodes.Error, "source degraded")
			}
			span.SetAttributes(attribute.Int(tracing.AttrCandidates, len(got.Items)))
			answers[index] = got
			sources[index] = retrieval.Source{Name: source.Name(), Top: got.Items}
		}(index, source)
	}
	wait.Wait()

	// Merged AFTER the wait, on one goroutine. Writing into shared maps from
	// inside the fan-out would be a data race that `go test -race` catches and
	// a release build turns into a corrupted map at load.
	model := Columns{
		Score:      map[int32]float32{},
		Similarity: map[int32]float32{},
	}
	for _, answer := range answers {
		for position, item := range answer.Items {
			// Length-checked per source rather than assumed: a source that
			// returned fewer scores than items would otherwise pair each item
			// with the next item's score from the index onward.
			if position < len(answer.Scores) {
				model.Score[item] = answer.Scores[position]
			}
			if position < len(answer.Similarity) {
				model.Similarity[item] = answer.Similarity[position]
			}
		}
	}

	var degraded []string
	for index, failed := range degradedFlags {
		if failed {
			degraded = append(degraded, s.Retrievers[index].Name())
		}
	}
	sort.Strings(degraded)
	return sources, model, degraded
}

// blocked asks the seen-list, degrading to "nothing blocked" on failure.
//
// The direction of that degradation is a decision. Failing open re-shows items
// the user has seen, which is a visible annoyance; failing closed would drop
// candidates the filter cannot vouch for, which silently shrinks the slate. The
// annoyance is recoverable and the empty page is not.
func (s *Service) blocked(
	ctx context.Context, userID string, items, exclude []int32, result *Result,
) []bool {
	if len(items) == 0 {
		return nil
	}

	var mask []bool
	if s.Seen != nil {
		got, err := s.Seen.Blocked(ctx, userID, items)
		if err != nil || len(got) != len(items) {
			result.DegradedSources = append(result.DegradedSources, "seen")
		} else {
			mask = got
		}
	}
	if len(exclude) == 0 {
		return mask
	}

	// The caller's exclusions merge into the SAME mask rather than pruning the
	// candidate slice. Pruning would be the obvious move and is wrong twice
	// over: it shifts every index, breaking the positional alignment the whole
	// pipeline runs on (blend ranks, feature rows, scores), and it would make a
	// client that excludes everything produce a blank page. Merged here, the
	// selector's existing rule applies -- a mask that blocks everything yields.
	//
	// A fresh slice: the seen-list's is the callee's memory, and a Redis client
	// that pooled its buffers would have this write show up in another request.
	merged := make([]bool, len(items))
	copy(merged, mask)
	drop := make(map[int32]struct{}, len(exclude))
	for _, item := range exclude {
		drop[item] = struct{}{}
	}
	for index, item := range items {
		if _, found := drop[item]; found {
			merged[index] = true
		}
	}
	return merged
}

// score runs the ranker under its own deadline, reporting whether it ran.
//
// On failure the candidates keep the order retrieval produced. That ordering is
// not arbitrary -- it is the retrieval score, which Part K measured at 0.0716
// NDCG@10 against the ranker's 0.1283. So the degraded path serves roughly 56%
// of the quality rather than an error page.
func (s *Service) score(ctx context.Context, in BuildInput) ([]float64, bool, int) {
	items := in.Items
	descending := make([]float64, len(items))
	for index := range items {
		descending[index] = float64(len(items) - index)
	}
	if s.Ranker == nil || s.Features == nil {
		return descending, false, 0
	}

	// Feature construction shares the ranker's deadline rather than having one
	// of its own. They are one step from the caller's point of view -- a slow
	// feature fetch and a slow model are the same symptom and the same
	// mitigation -- and splitting the budget would mean tuning two numbers that
	// only ever move together.
	rankCtx, cancel := context.WithTimeout(ctx, s.Config.RankDeadline)
	defer cancel()

	built, err := s.Features.Build(rankCtx, in)
	if err != nil || len(built.Rows) != len(items) {
		return descending, false, 0
	}

	scores, err := s.Ranker.Score(rankCtx, built)
	if err != nil || len(scores) != len(items) {
		// Zero incomplete rows, not built.Incomplete: the ranker did not run,
		// so no row was scored on zero-filled columns. Reporting the count
		// here would attribute a degradation to a stage that never happened.
		return descending, false, 0
	}
	return scores, true, built.Incomplete
}

// ranksByName flattens the blend's per-source ranks for the feature builder.
//
// A map here and a slice in the blend, deliberately: the blend needs ORDER
// because quotas are positional, while the builder needs LOOKUP because it
// writes named columns -- `two_tower_rank` and `trending_rank` are two of the
// eleven. Converting once at the boundary is cheaper than either side carrying
// the other's shape.
func ranksByName(blended retrieval.Blended) map[string][]int32 {
	out := make(map[string][]int32, len(blended.Ranks))
	for _, source := range blended.Ranks {
		out[source.Name] = source.Ranks
	}
	return out
}

// sourcesFor names the retrievers that proposed each served item.
//
// Read off the RANKS rather than off the source lists: Absent is exactly the
// statement "this source did not propose this candidate", and it is already
// computed. Iterating blended.Ranks (a slice, in input order) rather than a map
// also makes the output order deterministic -- ranging a map here would emit
// the same set in a different order on every call, which is the kind of diff
// that makes a response log impossible to compare against itself.
func sourcesFor(blended retrieval.Blended, rows []int) [][]string {
	out := make([][]string, len(rows))
	for position, row := range rows {
		var names []string
		for _, source := range blended.Ranks {
			if row < len(source.Ranks) && source.Ranks[row] != retrieval.Absent {
				names = append(names, source.Name)
			}
		}
		out[position] = names
	}
	return out
}

// servePopularity is the path that must never itself fail for an avoidable
// reason: no user features, no index, no model.
func (s *Service) servePopularity(
	ctx context.Context, numResults int, started time.Time, result Result,
) (Result, error) {
	result.UsedFallback = true
	if s.Fallback == nil {
		result.TotalLatency = s.now().Sub(started)
		return result, nil
	}

	stage := s.now()
	items, err := s.Fallback.Popular(ctx, numResults)
	result.StageLatency["fallback"] = s.now().Sub(stage)
	if err != nil {
		result.TotalLatency = s.now().Sub(started)
		return result, err
	}

	result.Items = items
	result.Scores = make([]float64, len(items))
	result.Objective = make([]float64, len(items))
	result.Propensity = make([]float64, len(items))
	result.Sources = make([][]string, len(items))
	for index := range items {
		result.Propensity[index] = rerank.Deterministic
		// Named, not left empty. A fallback slate whose items claim no source
		// reads as a bug in the attribution; naming it says what actually
		// happened, and `used_fallback` says it again at the response level.
		result.Sources[index] = []string{fallbackSource}
	}
	result.StageCounts["served"] = len(items)
	result.TotalLatency = s.now().Sub(started)
	return result, nil
}

func countUnblocked(blocked []bool, total int) int {
	if blocked == nil {
		return total
	}
	free := 0
	for _, isBlocked := range blocked {
		if !isBlocked {
			free++
		}
	}
	return free
}
