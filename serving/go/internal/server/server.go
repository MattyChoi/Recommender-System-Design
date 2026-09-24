package server

import (
	"context"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/experiments"
	pb "github.com/MattyChoi/Recommender-System-Design/serving/go/internal/pb"
	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/service"
)

// Defaults for a request that does not say. Both are limits rather than
// preferences: the ceiling exists because numResults sizes the re-ranker's
// per-slot pass over every candidate, so an unbounded value is a cheap way for
// one caller to spend the whole box.
const (
	defaultNumResults = 10
	maxNumResults     = 100
)

// Probe reports what the health endpoint can say about the loaded index.
//
// Separate from the retrievers themselves so that readiness does not have to
// fabricate a query, and because two of these fields exist to answer questions
// that no system metric can. See the HealthResponse comments in recsys.proto:
// a server that quietly failed over from HNSW to exact search is CORRECT and
// six times slower, and a config drift from efSearch 512 back to 128 is
// invisible everywhere else.
type Probe interface {
	// Ready returns nil when the server can serve, or the reason it cannot.
	Ready(ctx context.Context) error
	// Kind is "hnsw" or "flat".
	Kind() string
	// EFSearch is the efSearch actually loaded, not the one configured.
	EFSearch() int32
}

// Recommender adapts the pipeline to the wire.
type Recommender struct {
	pb.UnimplementedRecommenderServer

	Service *service.Service
	IDs     IDs
	Probe   Probe

	// Experiment assigns the caller to a variant, reported in every response.
	//
	// Assignment happens HERE rather than in the pipeline because it is a
	// property of the request, not of the recommendation: the variant has to
	// be logged whether or not the pipeline reached the stage the experiment
	// is about, including on the fallback path. A variant recorded only when
	// the happy path ran would bias every estimate toward healthy requests.
	//
	// Zero value means no experiment is running and the field stays empty --
	// which is honest, and different from a fabricated "control".
	Experiment experiments.Experiment

	// Observer records what the pipeline reported. Nil disables it.
	//
	// An interface rather than a concrete recorder, so this package does not
	// depend on a metrics library to be testable -- and so the thing being
	// observed is the Result rather than the protobuf. The response carries
	// stage latencies truncated to whole milliseconds, and a 5ms re-rank
	// budget measured in whole milliseconds has four useful values.
	Observer Observer
}

// Observer receives one call per completed pipeline run.
type Observer interface {
	Observed(result service.Result, variant string)
}

// GetRecommendations validates, translates, runs the pipeline, and translates
// back. It contains no policy: anything that decides what a user sees belongs
// in service, where it can be tested without a transport.
func (r *Recommender) GetRecommendations(
	ctx context.Context, request *pb.RecommendRequest,
) (*pb.RecommendResponse, error) {
	if request.GetUserId() == "" {
		return nil, status.Error(codes.InvalidArgument, "user_id is required")
	}

	numResults := int(request.GetNumResults())
	switch {
	case numResults == 0:
		numResults = defaultNumResults
	case numResults < 0:
		return nil, status.Errorf(
			codes.InvalidArgument, "num_results must not be negative, got %d", numResults,
		)
	case numResults > maxNumResults:
		// Refused rather than clamped. Clamping would return a short slate that
		// looks like the catalogue ran out, and the caller would go looking for
		// the missing items in the wrong system.
		return nil, status.Errorf(
			codes.InvalidArgument, "num_results %d exceeds the limit of %d",
			numResults, maxNumResults,
		)
	}

	// Assigned BEFORE the pipeline runs, so the variant is known even if the
	// request degrades or falls back.
	variant := r.Experiment.Assign(request.GetUserId())

	result, err := r.Service.Run(ctx, service.Request{
		UserID:     request.GetUserId(),
		Surface:    request.GetSurface(),
		NumResults: numResults,
		Exclude:    r.internalIDs(request.GetExcludeItemIds()),
	})
	if err != nil {
		return nil, status.Errorf(codes.Internal, "recommend: %v", err)
	}

	// Recorded BEFORE the response is built, so a translation failure below
	// does not lose the pipeline's own account of what happened. That failure
	// means the id map and the index disagree -- exactly when the degradation
	// counters are worth having.
	if r.Observer != nil {
		r.Observer.Observed(result, variant)
	}

	response, err := r.response(result)
	if err != nil {
		return nil, err
	}
	response.ExperimentVariant = variant
	return response, nil
}

// internalIDs translates the caller's exclusions, dropping ones this build has
// never heard of.
//
// Dropping is right HERE and would be wrong in the response direction. An
// exclusion naming an unknown item is a no-op by construction -- an item that
// is not in the map cannot be in the candidate set either -- so failing the
// request would reject a page over a stale id the caller was holding harmlessly.
func (r *Recommender) internalIDs(external []string) []int32 {
	if len(external) == 0 {
		return nil
	}
	internal := make([]int32, 0, len(external))
	for _, id := range external {
		if index, found := r.IDs.Index(id); found {
			internal = append(internal, index)
		}
	}
	return internal
}

// response translates a Result. Every field the proto documents as load-bearing
// is filled here; the ones that are not are named in the comments, not omitted
// in silence.
func (r *Recommender) response(result service.Result) (*pb.RecommendResponse, error) {
	items := make([]*pb.ScoredItem, len(result.Items))
	for position, item := range result.Items {
		external, found := r.IDs.External(item)
		if !found {
			// The whole request fails rather than the slot being skipped.
			//
			// Skipping is the tempting move and is the worse outage: it returns
			// a short slate that every metric reads as success, while the cause
			// -- an id map and an index built from different snapshots -- keeps
			// mistranslating the items it DOES know. A map off by one snapshot
			// serves real ids pointing at the wrong articles. Loud is better.
			return nil, status.Errorf(
				codes.Internal,
				"served item index %d is absent from item map %q; "+
					"the index and the map are from different builds",
				item, r.IDs.Version(),
			)
		}
		items[position] = &pb.ScoredItem{
			ItemId: external,
			// The ORDERING value, per the proto: the objective the slot was
			// chosen by, not the ranker's raw output. They differ exactly when
			// a policy fires, which is when someone is most likely to be
			// reading this field to work out why.
			Score:      float32(at(result.Objective, position)),
			Propensity: float32(at(result.Propensity, position)),
			Position:   int32(position),
			Sources:    sourcesAt(result.Sources, position),
			DebugScores: map[string]float32{
				"ranker_score": float32(at(result.Scores, position)),
			},
		}
	}

	response := &pb.RecommendResponse{
		Items:           items,
		ModelVersion:    r.Service.Config.ModelVersion,
		IndexVersion:    r.Service.Config.IndexVersion,
		LatencyMs:       int32(result.TotalLatency.Milliseconds()),
		UsedFallback:    result.UsedFallback,
		DegradedSources: result.DegradedSources,
		StageCounts:     make(map[string]int32, len(result.StageCounts)),
		StageLatencyMs:  make(map[string]int32, len(result.StageLatency)),
		// ExperimentVariant is set by the caller, after this returns: it is a
		// property of the REQUEST rather than of the result, and has to be
		// known even when the pipeline degraded or fell back.
	}
	for stage, count := range result.StageCounts {
		response.StageCounts[stage] = int32(count)
	}
	for stage, elapsed := range result.StageLatency {
		// Milliseconds, truncated, because that is what the proto declares. A
		// sub-millisecond stage reporting 0 is honest at this resolution; the
		// per-stage histogram that needs more lives in metrics, not here.
		response.StageLatencyMs[stage] = int32(elapsed.Milliseconds())
	}
	return response, nil
}

// Health answers without touching the pipeline, so that "can this box serve?"
// stays answerable when the answer is no.
func (r *Recommender) Health(
	ctx context.Context, _ *pb.HealthRequest,
) (*pb.HealthResponse, error) {
	response := &pb.HealthResponse{
		ModelVersion: r.Service.Config.ModelVersion,
		IndexVersion: r.Service.Config.IndexVersion,
	}
	if r.Probe == nil {
		response.Detail = "no index probe configured"
		return response, nil
	}

	response.IndexKind = r.Probe.Kind()
	response.IndexEfSearch = r.Probe.EFSearch()
	if err := r.Probe.Ready(ctx); err != nil {
		// Not ready is a successful ANSWER, not a failed call: a health check
		// that returns a gRPC error is indistinguishable from one that could
		// not reach the server, and those need different responses at 3am.
		response.Detail = err.Error()
		return response, nil
	}
	response.Ready = true
	return response, nil
}

// at reads a parallel array that may be shorter than Items.
//
// The arrays are built together and should always match. "Should" is the
// problem: an index panic here takes down the process on a response that was
// otherwise correct, so a missing value degrades to zero and the mismatch is
// caught by the tests rather than by the pager.
func at(values []float64, index int) float64 {
	if index < len(values) {
		return values[index]
	}
	return 0
}

func sourcesAt(values [][]string, index int) []string {
	if index < len(values) {
		return values[index]
	}
	return nil
}

// Compile-time proof that this type is the service the proto declares. Without
// it, a signature drifting out of step with a regenerated stub shows up as a
// confusing error at the registration call site instead of here.
var _ pb.RecommenderServer = (*Recommender)(nil)
