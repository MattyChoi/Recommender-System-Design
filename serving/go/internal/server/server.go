package server

import (
	"context"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/experiments"
	pb "github.com/MattyChoi/Recommender-System-Design/serving/go/internal/pb"
	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/service"
)

const (
	defaultNumResults = 10
	maxNumResults     = 100
)

// Probe reports what the health endpoints say
type Probe interface {
	Ready(ctx context.Context) error
	// Kind is "hnsw" or "flat".
	Kind() string
	EFSearch() int32
}

// Observer receives one call per completed pipeline run.
type Observer interface {
	Observed(result service.Result, variant string)
}

// Recommender adapts the pipeline to the grpc wire.
type Recommender struct {
	pb.UnimplementedRecommenderServer

	Service    *service.Service
	IDs        IDs
	Probe      Probe
	Experiment experiments.Experiment
	Observer   Observer
}

func (r *Recommender) GetRecommendations(ctx context.Context, request *pb.RecommendRequest) (*pb.RecommendResponse, error) {
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

	// assign the experiment variant
	variant := r.Experiment.Assign(request.GetUserId())

	// run the recommend pipeline
	result, err := r.Service.Run(ctx, service.Request{
		UserID:     request.GetUserId(),
		Surface:    request.GetSurface(),
		NumResults: numResults,
		Exclude:    r.itemIndices(request.GetExcludeItemIds()),
	})
	if err != nil {
		return nil, status.Errorf(codes.Internal, "recommend: %v", err)
	}

	// record what the pipeline reported
	if r.Observer != nil {
		r.Observer.Observed(result, variant)
	}

	// translate the response from the pipeline
	response, err := r.response(result)
	if err != nil {
		return nil, err
	}
	response.ExperimentVariant = variant

	return response, nil

}

func (r *Recommender) Health(ctx context.Context, _ *pb.HealthRequest) (*pb.HealthResponse, error) {
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
		response.Detail = err.Error()
		return response, nil
	}
	response.Ready = true

	return response, nil
}

func (r *Recommender) response(result service.Result) (*pb.RecommendResponse, error) {
	items := make([]*pb.ScoredItem, len(result.Items))
	for position, item_idx := range result.Items {
		item_id, found := r.IDs.External(item_idx)
		if !found {
			return nil, status.Errorf(
				codes.Internal,
				"served item index %d is absent from item map %q; "+
					"the index and the map are from different builds",
				item_idx, r.IDs.Version(),
			)
		}
		items[position] = &pb.ScoredItem{
			ItemId: item_id,
			Score:  float32(at(result.Objective, position)),
			DebugScores: map[string]float32{
				"ranker_score": float32(at(result.Scores, position)),
			},
			Sources:    sourcesAt(result.Sources, position),
			Propensity: float32(at(result.Propensity, position)),
			Position:   int32(position),
		}
	}

	response := &pb.RecommendResponse{
		Items:           items,
		ModelVersion:    r.Service.Config.ModelVersion,
		IndexVersion:    r.Service.Config.IndexVersion,
		LatencyMs:       int32(result.TotalLatency.Milliseconds()),
		UsedFallback:    result.UsedFallback,
		DegradedSources: result.DegradedSources,
		StageLatencyMs:  make(map[string]int32, len(result.StageLatency)),
		StageCounts:     make(map[string]int32, len(result.StageCounts)),
	}

	for stage, elapsed := range result.StageLatency {
		response.StageLatencyMs[stage] = int32(elapsed.Milliseconds())
	}
	for stage, count := range result.StageCounts {
		response.StageCounts[stage] = int32(count)
	}

	return response, nil
}

// translate the item id strings to their corresponding item indices
func (r *Recommender) itemIndices(itemIDs []string) []int32 {
	if len(itemIDs) == 0 {
		return nil
	}
	item_idx := make([]int32, len(itemIDs))
	for _, id := range itemIDs {
		if index, found := r.IDs.Index(id); found {
			item_idx = append(item_idx, index)
		}
	}
	return item_idx
}

func sourcesAt(sources [][]string, position int) []string {
	if position < len(sources) {
		return sources[position]
	}
	return nil
}

func at(values []float64, position int) float64 {
	if position < len(values) {
		return values[position]
	}
	return 0
}

var _ pb.RecommenderServer = (*Recommender)(nil)
