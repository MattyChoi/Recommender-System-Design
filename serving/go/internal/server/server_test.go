package server

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/experiments"
	pb "github.com/MattyChoi/Recommender-System-Design/serving/go/internal/pb"
	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/service"
)

// --- A pipeline small enough to reason about ---------------------------------
//
// Real fakes rather than a mocked Service: the interesting bugs at this
// boundary are about what happens to values on the way THROUGH, and a stubbed
// pipeline would let a translation error agree with itself.

type fixedRetriever struct{ items []int32 }

func (f fixedRetriever) Name() string          { return "two_tower" }
func (f fixedRetriever) Budget() time.Duration { return time.Second }

func (f fixedRetriever) Retrieve(context.Context, service.User) (service.Candidates, error) {
	return service.Candidates{Items: f.items}, nil
}

// emptyCatalogue disables MMR and caps by returning nil, which is the shipped
// configuration: ADR 0012 sets Lambda to 1.0 because MMR measured as a null.
type emptyCatalogue struct{}

func (emptyCatalogue) Vectors([]int32) [][]float32   { return nil }
func (emptyCatalogue) Categories([]int32) []int32    { return nil }
func (emptyCatalogue) Subcategories([]int32) []int32 { return nil }

func testServer(t *testing.T, known map[string]int32) *Recommender {
	t.Helper()
	table, err := NewTable(known, "test-snapshot")
	if err != nil {
		t.Fatalf("NewTable: %v", err)
	}
	return &Recommender{
		IDs: table,
		Service: &service.Service{
			Retrievers: []service.Retriever{fixedRetriever{items: []int32{1, 2, 3, 4, 5}}},
			Catalogue:  emptyCatalogue{},
			Config: service.Config{
				Quotas:        []int{5},
				MaxCandidates: 5,
				Deadline:      time.Second,
				RankDeadline:  time.Second,
				Lambda:        1.0,
				ModelVersion:  "mmoe-v3",
				IndexVersion:  "hnsw-2026-09",
			},
		},
	}
}

func fullMap() map[string]int32 {
	return map[string]int32{"N1": 1, "N2": 2, "N3": 3, "N4": 4, "N5": 5}
}

func itemIDs(response *pb.RecommendResponse) []string {
	out := make([]string, len(response.GetItems()))
	for index, item := range response.GetItems() {
		out[index] = item.GetItemId()
	}
	return out
}

func equal(left, right []string) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}

// --- Translation out ---------------------------------------------------------

func TestTheSlateComesBackAsExternalIDsInOrder(t *testing.T) {
	server := testServer(t, fullMap())

	response, err := server.GetRecommendations(context.Background(), &pb.RecommendRequest{
		UserId: "U1", NumResults: 3,
	})
	if err != nil {
		t.Fatalf("GetRecommendations: %v", err)
	}

	if got := itemIDs(response); !equal(got, []string{"N1", "N2", "N3"}) {
		t.Errorf("items %v, want [N1 N2 N3]", got)
	}
	for index, item := range response.GetItems() {
		if item.GetPosition() != int32(index) {
			t.Errorf("slot %d reports position %d", index, item.GetPosition())
		}
		// Every slot here was filled by the greedy rule, so the propensity
		// carries no information for an estimator -- and that is precisely what
		// 1.0 is defined to mean.
		if item.GetPropensity() != 1.0 {
			t.Errorf("slot %d propensity %v, want 1.0", index, item.GetPropensity())
		}
		if len(item.GetSources()) != 1 || item.GetSources()[0] != "two_tower" {
			t.Errorf("slot %d sources %v, want [two_tower]", index, item.GetSources())
		}
	}
	if response.GetModelVersion() != "mmoe-v3" || response.GetIndexVersion() != "hnsw-2026-09" {
		t.Error("a response that cannot name its artifacts cannot be attributed after a rebuild")
	}
}

func TestNoExperimentLeavesTheVariantEmpty(t *testing.T) {
	// Empty is honest. A fabricated "control" that no assignment produced is
	// worse than a missing field, because an analysis keyed on it looks valid.
	server := testServer(t, fullMap())

	response, err := server.GetRecommendations(context.Background(), &pb.RecommendRequest{
		UserId: "U1", NumResults: 2,
	})
	if err != nil {
		t.Fatalf("GetRecommendations: %v", err)
	}

	if response.GetExperimentVariant() != "" {
		t.Errorf("variant %q with no experiment configured", response.GetExperimentVariant())
	}
}

func TestTheVariantIsReportedAndSticky(t *testing.T) {
	server := testServer(t, fullMap())
	server.Experiment = experiments.Experiment{
		ID:       "ranker_v2",
		Variants: []experiments.Variant{{Name: "control", Percent: 100}},
	}

	first, err := server.GetRecommendations(context.Background(), &pb.RecommendRequest{
		UserId: "U1", NumResults: 2,
	})
	if err != nil {
		t.Fatalf("GetRecommendations: %v", err)
	}
	second, _ := server.GetRecommendations(context.Background(), &pb.RecommendRequest{
		UserId: "U1", NumResults: 2,
	})

	if first.GetExperimentVariant() != "control" {
		t.Errorf("variant %q, want control", first.GetExperimentVariant())
	}
	if second.GetExperimentVariant() != first.GetExperimentVariant() {
		t.Error("the same user got two variants; per-user metrics would be meaningless")
	}
}

// TestTheVariantSurvivesTheFallbackPath is why assignment happens before the
// pipeline runs.
//
// A variant recorded only when the happy path completed would bias every
// off-policy estimate toward healthy requests -- the degraded ones, which are
// exactly where an experiment's effect may differ, would carry no arm at all.
func TestTheVariantSurvivesTheFallbackPath(t *testing.T) {
	server := testServer(t, fullMap())
	// No retrievers: the candidate set is empty and the pipeline falls back.
	server.Service.Retrievers = nil
	server.Service.Config.Quotas = nil
	server.Service.Fallback = fixedFallback{items: []int32{1, 2}}
	server.Experiment = experiments.Experiment{
		ID:       "ranker_v2",
		Variants: []experiments.Variant{{Name: "treatment", Percent: 100}},
	}

	response, err := server.GetRecommendations(context.Background(), &pb.RecommendRequest{
		UserId: "U1", NumResults: 2,
	})
	if err != nil {
		t.Fatalf("GetRecommendations: %v", err)
	}

	if !response.GetUsedFallback() {
		t.Fatal("expected the fallback path")
	}
	if response.GetExperimentVariant() != "treatment" {
		t.Errorf("variant %q lost on the fallback path", response.GetExperimentVariant())
	}
}

type fixedFallback struct{ items []int32 }

func (f fixedFallback) Popular(_ context.Context, n int) ([]int32, error) {
	if n > len(f.items) {
		n = len(f.items)
	}
	return f.items[:n], nil
}

// TestAnUnmappableServedItemFailsTheRequest is the one that matters most here.
//
// The tempting alternative is to skip the slot. That returns a short slate
// which every metric reads as success, while the underlying cause -- an id map
// and an index built from different snapshots -- keeps mistranslating the items
// it DOES know, serving real ids that point at the wrong articles.
func TestAnUnmappableServedItemFailsTheRequest(t *testing.T) {
	partial := fullMap()
	delete(partial, "N3")
	server := testServer(t, partial)

	response, err := server.GetRecommendations(context.Background(), &pb.RecommendRequest{
		UserId: "U1", NumResults: 3,
	})

	if err == nil {
		t.Fatalf("expected a refusal, got a slate of %d items", len(response.GetItems()))
	}
	if status.Code(err) != codes.Internal {
		t.Errorf("code %v, want Internal", status.Code(err))
	}
	if !strings.Contains(err.Error(), "test-snapshot") {
		t.Errorf("the error should name the map that came up short, got %q", err)
	}
}

func TestTheScoreReportedIsTheOrderingValue(t *testing.T) {
	server := testServer(t, fullMap())

	response, err := server.GetRecommendations(context.Background(), &pb.RecommendRequest{
		UserId: "U1", NumResults: 2,
	})
	if err != nil {
		t.Fatalf("GetRecommendations: %v", err)
	}

	// With no ranker wired the pipeline degrades to the blended retrieval
	// order, scored descending -- so slot 0 outscores slot 1 and the raw
	// ranker score agrees with the ordering value. They agree here BECAUSE no
	// policy fired; the point of carrying both is the case where one does.
	first, second := response.GetItems()[0], response.GetItems()[1]
	if !(first.GetScore() > second.GetScore()) {
		t.Errorf("scores %v then %v are not descending", first.GetScore(), second.GetScore())
	}
	if got := first.GetDebugScores()["ranker_score"]; got != first.GetScore() {
		t.Errorf("ranker_score %v, ordering score %v; expected agreement with no policy firing",
			got, first.GetScore())
	}
}

func TestTheRankerIsReportedDegradedWhenItIsNotWired(t *testing.T) {
	server := testServer(t, fullMap())

	response, err := server.GetRecommendations(context.Background(), &pb.RecommendRequest{
		UserId: "U1", NumResults: 3,
	})
	if err != nil {
		t.Fatalf("GetRecommendations: %v", err)
	}

	// A recommender that silently degrades looks healthy. Serving the
	// retrieval order is a real quality loss (Part K: 0.0716 against 0.1283)
	// and has to arrive labelled as one.
	found := false
	for _, source := range response.GetDegradedSources() {
		if source == "ranker" {
			found = true
		}
	}
	if !found {
		t.Errorf("degraded_sources %v should name the ranker", response.GetDegradedSources())
	}
	if response.GetStageCounts()["retrieved"] != 5 {
		t.Errorf("stage_counts %v should record 5 retrieved", response.GetStageCounts())
	}
}

// --- Translation in ----------------------------------------------------------

func TestTheCallersExclusionsAreHonoured(t *testing.T) {
	server := testServer(t, fullMap())

	response, err := server.GetRecommendations(context.Background(), &pb.RecommendRequest{
		UserId: "U1", NumResults: 3, ExcludeItemIds: []string{"N2"},
	})
	if err != nil {
		t.Fatalf("GetRecommendations: %v", err)
	}

	// Still three items: the exclusion masks a candidate, it does not shorten
	// the slate. Masking rather than pruning is also what keeps the positional
	// alignment the rest of the pipeline runs on.
	if got := itemIDs(response); !equal(got, []string{"N1", "N3", "N4"}) {
		t.Errorf("items %v, want [N1 N3 N4]", got)
	}
}

// TestAnUnknownExclusionIsIgnored is the counterpart to the unmappable-item
// test above, and goes the other way on purpose: an exclusion naming an item
// this build has never heard of is a no-op by construction, since an unmapped
// item cannot be in the candidate set either. Failing the page over a stale id
// the caller was holding harmlessly would be the wrong trade.
func TestAnUnknownExclusionIsIgnored(t *testing.T) {
	server := testServer(t, fullMap())

	response, err := server.GetRecommendations(context.Background(), &pb.RecommendRequest{
		UserId: "U1", NumResults: 3, ExcludeItemIds: []string{"N999"},
	})
	if err != nil {
		t.Fatalf("an unknown exclusion must not fail the request: %v", err)
	}
	if got := itemIDs(response); !equal(got, []string{"N1", "N2", "N3"}) {
		t.Errorf("items %v, want the unfiltered [N1 N2 N3]", got)
	}
}

// --- Validation --------------------------------------------------------------

func TestRequestValidation(t *testing.T) {
	server := testServer(t, fullMap())

	cases := []struct {
		name    string
		request *pb.RecommendRequest
	}{
		{"no user", &pb.RecommendRequest{NumResults: 3}},
		{"negative num_results", &pb.RecommendRequest{UserId: "U1", NumResults: -1}},
		// Refused rather than clamped: a clamped request returns a short slate
		// that looks like the catalogue ran out, sending the caller to look for
		// the missing items in the wrong system.
		{"num_results past the ceiling", &pb.RecommendRequest{UserId: "U1", NumResults: 1000}},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			_, err := server.GetRecommendations(context.Background(), testCase.request)
			if status.Code(err) != codes.InvalidArgument {
				t.Errorf("code %v, want InvalidArgument (err: %v)", status.Code(err), err)
			}
		})
	}
}

func TestAnUnsetNumResultsTakesTheDefault(t *testing.T) {
	server := testServer(t, fullMap())

	response, err := server.GetRecommendations(context.Background(), &pb.RecommendRequest{
		UserId: "U1",
	})
	if err != nil {
		t.Fatalf("GetRecommendations: %v", err)
	}
	// Five candidates, a default of ten: the slate is what exists, not what was
	// asked for. Asserting 5 rather than 10 pins that a short candidate set is
	// served short rather than padded.
	if len(response.GetItems()) != 5 {
		t.Errorf("got %d items from 5 candidates", len(response.GetItems()))
	}
}

// --- The id map --------------------------------------------------------------

// TestAnUninvertibleMapIsRefusedAtLoad: two external ids on one index means the
// reverse direction has no answer, so some slate would be served under another
// article's id. Caught at startup, where it is one line in a log rather than a
// silent mistranslation.
func TestAnUninvertibleMapIsRefusedAtLoad(t *testing.T) {
	_, err := NewTable(map[string]int32{"N1": 1, "N2": 1}, "broken")
	if err == nil {
		t.Fatal("a map that is not invertible must be refused")
	}
	if !strings.Contains(err.Error(), "invertible") {
		t.Errorf("the error should say what is wrong, got %q", err)
	}
}

// --- Health ------------------------------------------------------------------

type probe struct {
	err      error
	kind     string
	efSearch int32
}

func (p probe) Ready(context.Context) error { return p.err }
func (p probe) Kind() string                { return p.kind }
func (p probe) EFSearch() int32             { return p.efSearch }

func TestHealthAnswersRatherThanErroring(t *testing.T) {
	server := testServer(t, fullMap())
	server.Probe = probe{err: errors.New("index still loading"), kind: "hnsw", efSearch: 512}

	response, err := server.Health(context.Background(), &pb.HealthRequest{})

	// A health check that returns a gRPC error is indistinguishable from one
	// that could not reach the server, and those need different responses.
	if err != nil {
		t.Fatalf("not-ready must be an answer, not an error: %v", err)
	}
	if response.GetReady() {
		t.Error("ready should be false while the index is loading")
	}
	if response.GetDetail() == "" {
		t.Error("a bare false sends someone to the logs")
	}
	// Reported even when not ready: a server that quietly failed over to exact
	// search is correct and six times slower, and this is the one call that says so.
	if response.GetIndexKind() != "hnsw" || response.GetIndexEfSearch() != 512 {
		t.Errorf("index reported as %q/%d", response.GetIndexKind(), response.GetIndexEfSearch())
	}
}

func TestHealthIsReadyWhenTheProbeIs(t *testing.T) {
	server := testServer(t, fullMap())
	server.Probe = probe{kind: "hnsw", efSearch: 512}

	response, err := server.Health(context.Background(), &pb.HealthRequest{})
	if err != nil {
		t.Fatalf("Health: %v", err)
	}
	if !response.GetReady() {
		t.Errorf("ready should be true, detail: %q", response.GetDetail())
	}
}
