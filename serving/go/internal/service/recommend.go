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

const (
	degradedFeatures = "features"
	fallbackSource   = "popularity"
	degradedRanker   = "ranker"
	degradedColumns  = "feature_columns"
)

type Config struct {
	Quotas        []int
	MaxCandidates int

	// Deadline for the whole request
	Deadline time.Duration

	// RankDeadline is the ranker's slice. Exceeded, the pipeline ships the
	// blended retrieval order rather than failing
	RankDeadline time.Duration

	// FeatureDeadline bounds the pre-fan-out user fetch. 8ms. Its OWN deadline
	// rather than a share of the request's
	FeatureDeadline time.Duration

	MaxHistory int

	// reranking parameters
	Lambda         float64
	CapPerCategory *int
	Epsilon        float64
	ExploreSlots   int

	ModelVersion string
	IndexVersion string
}

type Service struct {
	Retrievers []Retriever
	Store      UserStore
	Features   FeatureBuilder
	Ranker     Ranker
	Seen       SeenList
	Catalogue  Catalogue
	Fallback   Fallback
	Config     Config
	Now        func() time.Time
}

type Request struct {
	UserID     string
	Surface    string
	NumResults int
	// Exclude is what the CALLER has already rendered, on top of the server's
	// own seen-list.
	Exclude []int32
}

type Result struct {
	Items           []int32
	Scores          []float64
	Objective       []float64
	Propensity      []float64
	Sources         [][]string
	UsedFallback    bool
	DegradedSources []string
	StageCounts     map[string]int
	StageLatency    map[string]time.Duration
	TotalLatency    time.Duration
}

func (s *Service) Recommend(ctx context.Context, userID string, numResults int) (Result, error) {
	return s.Run(ctx, Request{UserID: userID, NumResults: numResults})
}

func (s *Service) Run(ctx context.Context, request Request) (Result, error) {
	userID, numResults := request.UserID, request.NumResults
	started := s.now()

	result := Result{
		StageCounts:  map[string]int{},
		StageLatency: map[string]time.Duration{},
	}

	ctx, cancel := context.WithTimeout(ctx, s.Config.Deadline)
	defer cancel()

	ctx, span := tracing.Tracer().Start(ctx, "recommend")
	defer span.End()

	// --- step 0: fetch the user features from the feature gateway ---
	var user User
	s.stage(ctx, "features", &result, func(inner context.Context) {
		user = s.fetchUser(inner, userID, &result)
	})

	// --- step 1: retrieval fan-out ---
	var sources []retrieval.Source
	var model Columns
	s.stage(ctx, "retrieval", &result, func(inner context.Context) {
		var degraded []string
		sources, model, degraded = s.fanOut(inner, user)
		result.DegradedSources = append(result.DegradedSources, degraded...)
	})

	blended, err := retrieval.Blend(sources, s.Config.MaxCandidates, s.Config.Quotas)
	if err != nil {
		return result, err
	}
	result.StageCounts["retrieved"] = len(blended.Items)

	// --- serve popularity slate if there are no retrieved candidates
	candidates := blended.Items
	if len(candidates) == 0 {
		return s.servePopularity(ctx, numResults, started, result)
	}

	// --- step 2: filter using the seen-list and purposely excluded items
	var blocked []bool
	s.stage(ctx, "filter", &result, func(inner context.Context) {
		blocked = s.blocked(inner, userID, candidates, request.Exclude, &result)
	})
	result.StageCounts["after_filter"] = countUnblocked(blocked, len(candidates))

	// --- step 3: ranking ---
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
	if incomplete > 0 {
		result.StageCounts["incomplete_rows"] = incomplete
		result.DegradedSources = append(result.DegradedSources, degradedColumns)
	}
	result.StageCounts["scored"] = len(scores)

	// --- step 4: Re-rank ---
	var slate rerank.Slate
	s.stage(ctx, "rerank", &result, func(inner context.Context) {
		slate, err = rerank.Select(scores, candidates, numResults, rerank.Options{
			Vectors:      s.Catalogue.Vectors(candidates),
			Lambda:       s.Config.Lambda,
			Categories:   s.Catalogue.Subcategories(candidates),
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
	result.Objective = slate.Objective
	result.Propensity = slate.Propensity
	result.Sources = sourcesFor(blended, slate.Rows)
	result.Scores = make([]float64, len(slate.Rows))
	for index, row := range slate.Rows {
		result.Scores[index] = scores[row]
	}

	// --- step 5: add to the seen-list ---
	if s.Seen != nil {
		_ = s.Seen.Record(ctx, userID, slate.Items)
	}

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
func (s *Service) stage(ctx context.Context, name string, result *Result, run func(context.Context)) {
	ctx, span := tracing.Tracer().Start(ctx, name)
	defer span.End()

	started := s.now()
	run(ctx)
	result.StageLatency[name] = s.now().Sub(started)
}

func (s *Service) now() time.Time {
	if s.Now != nil {
		return s.Now()
	}
	return time.Now()
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

	}
	result.StageCounts["served"] = len(items)
	result.TotalLatency = s.now().Sub(started)
	return result, nil
}

// fetch the user features from the feature store gateway, degrading to a cold user
func (s *Service) fetchUser(ctx context.Context, userID string, result *Result) User {
	cold := User{ID: userID}

	if s.Store == nil {
		return cold
	}

	fetchCtx, cancel := context.WithTimeout(ctx, s.Config.FeatureDeadline)
	defer cancel()

	user, err := s.Store.Fetch(fetchCtx, userID, s.Config.MaxHistory)
	if err != nil {
		result.DegradedSources = append(result.DegradedSources, degradedFeatures)
		return cold
	}

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

			sourceCtx, span := tracing.Tracer().Start(sourceCtx, "retrieve")
			defer span.End()
			tracing.Source(span, source.Name(), source.Budget())

			got, err := source.Retrieve(sourceCtx, user)
			if err != nil {
				degradedFlags[index] = true
				got = Candidates{}
				span.RecordError(err)
				span.SetStatus(otelcodes.Error, "source degraded")
			}
			span.SetAttributes(attribute.Int(tracing.AttrCandidates, len(got.Items)))
			answers[index] = got
			sources[index] = retrieval.Source{Name: source.Name(), Top: got.Items}
		}(index, source)
	}
	wait.Wait()

	model := Columns{
		Score:      map[int32]float32{},
		Similarity: map[int32]float32{},
	}
	for _, answer := range answers {
		for position, item := range answer.Items {
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
func (s *Service) score(ctx context.Context, in BuildInput) ([]float64, bool, int) {
	items := in.Items
	descending := make([]float64, len(items))
	for index := range items {
		descending[index] = float64(len(items) - index)
	}
	if s.Ranker == nil || s.Features == nil {
		return descending, false, 0
	}

	// Feature construction shares the ranker's deadline
	rankCtx, cancel := context.WithTimeout(ctx, s.Config.RankDeadline)
	defer cancel()

	// fetch the item features from the feature gateway
	built, err := s.Features.Build(rankCtx, in)
	if err != nil || len(built.Rows) != len(items) {
		return descending, false, 0
	}

	scores, err := s.Ranker.Score(rankCtx, built)
	if err != nil || len(scores) != len(items) {
		return descending, false, 0
	}
	return scores, true, built.Incomplete
}

// ranksByName flattens the blend's per-source ranks for the feature builder.
func ranksByName(blended retrieval.Blended) map[string][]int32 {
	out := make(map[string][]int32, len(blended.Ranks))
	for _, source := range blended.Ranks {
		out[source.Name] = source.Ranks
	}
	return out
}

// sourcesFor names the retrievers that proposed each served item.
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
