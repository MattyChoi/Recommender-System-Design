package service

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"
)

// These tests are about DEGRADATION, because that is the behaviour a serving
// orchestrator is actually judged on. A pipeline that works when every
// dependency answers is the easy half; the half that matters is what it returns
// when a retriever hangs, a ranker times out, or a filter blocks everything --
// and each of those failures is silent unless the response says otherwise.
//
// Everything here runs against fakes. A real dependency cannot be asked to be
// slow on demand, which is the whole reason the ports exist.

type fakeRetriever struct {
	name   string
	budget time.Duration
	items  []int32
	delay  time.Duration
	err    error
}

func (f fakeRetriever) Name() string          { return f.name }
func (f fakeRetriever) Budget() time.Duration { return f.budget }
func (f fakeRetriever) Retrieve(ctx context.Context, _ User) (Candidates, error) {
	if f.err != nil {
		return Candidates{}, f.err
	}
	if f.delay > 0 {
		select {
		case <-time.After(f.delay):
		case <-ctx.Done():
			// The source overran its own budget. Returning the context error
			// rather than partial results is what makes it DROPPED rather than
			// waited on.
			return Candidates{}, ctx.Err()
		}
	}
	// Scores mirror the item order so the fan-out's merge has something to
	// key, and a source with no model leaves Similarity nil -- which is the
	// shape a precomputed trending list really has.
	scores := make([]float32, len(f.items))
	for index := range f.items {
		scores[index] = float32(len(f.items) - index)
	}
	return Candidates{Items: f.items, Scores: scores}, nil
}

// fakeFeatures builds one column per candidate: its position. Enough for the
// orchestrator's contract, which is about SHAPE and deadlines rather than about
// what the columns mean.
type fakeFeatures struct {
	err   error
	delay time.Duration
	// short returns fewer rows than candidates, to exercise the length guard.
	short bool
}

func (f fakeFeatures) Build(ctx context.Context, in BuildInput) (Features, error) {
	if f.err != nil {
		return Features{}, f.err
	}
	if f.delay > 0 {
		select {
		case <-time.After(f.delay):
		case <-ctx.Done():
			return Features{}, ctx.Err()
		}
	}
	count := len(in.Items)
	if f.short && count > 0 {
		count--
	}
	rows := make([][]float32, count)
	for index := range rows {
		rows[index] = []float32{float32(index)}
	}
	return Features{Names: []string{"position"}, Rows: rows}, nil
}

type fakeRanker struct {
	scores []float64
	delay  time.Duration
	err    error
}

func (f fakeRanker) Score(ctx context.Context, features Features) ([]float64, error) {
	if f.err != nil {
		return nil, f.err
	}
	if f.delay > 0 {
		select {
		case <-time.After(f.delay):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	if f.scores != nil {
		return f.scores, nil
	}
	scores := make([]float64, len(features.Rows))
	for index := range scores {
		scores[index] = float64(index) // deliberately ASCENDING, see below
	}
	return scores, nil
}

type fakeSeen struct {
	blocked  []bool
	err      error
	recorded []int32
}

func (f *fakeSeen) Blocked(_ context.Context, _ string, items []int32) ([]bool, error) {
	if f.err != nil {
		return nil, f.err
	}
	if f.blocked != nil {
		return f.blocked, nil
	}
	return make([]bool, len(items)), nil
}

func (f *fakeSeen) Record(_ context.Context, _ string, items []int32) error {
	f.recorded = append(f.recorded, items...)
	return nil
}

// fakeUsers is the feature gateway. `seen` records what each retriever was
// handed, which is how the "one snapshot" property is asserted.
type fakeUsers struct {
	user  User
	err   error
	delay time.Duration
	calls int
}

func (f *fakeUsers) Fetch(ctx context.Context, userID string, _ int) (User, error) {
	f.calls++
	if f.err != nil {
		return User{}, f.err
	}
	if f.delay > 0 {
		select {
		case <-time.After(f.delay):
		case <-ctx.Done():
			return User{}, ctx.Err()
		}
	}
	user := f.user
	user.ID = userID
	return user, nil
}

// recordingRetriever captures the User it was handed, so the pre-fan-out fetch
// can be checked for actually reaching the sources that need it.
type recordingRetriever struct {
	name string
	got  []User
	mu   sync.Mutex
}

func (r *recordingRetriever) Name() string          { return r.name }
func (r *recordingRetriever) Budget() time.Duration { return time.Second }
func (r *recordingRetriever) Retrieve(_ context.Context, user User) (Candidates, error) {
	// The fan-out is concurrent, so this is a real race without the lock --
	// and `go test -race` is the only thing that would say so.
	r.mu.Lock()
	defer r.mu.Unlock()
	r.got = append(r.got, user)
	return Candidates{
		Items:      []int32{1, 2, 3},
		Scores:     []float32{0.9, 0.8, 0.7},
		Similarity: []float32{0.3, 0.2, 0.1},
	}, nil
}

type emptyCatalogue struct{}

func (emptyCatalogue) Vectors(_ []int32) [][]float32   { return nil }
func (emptyCatalogue) Categories(_ []int32) []int32    { return nil }
func (emptyCatalogue) Subcategories(_ []int32) []int32 { return nil }

type fakeFallback struct{ items []int32 }

func (f fakeFallback) Popular(_ context.Context, n int) ([]int32, error) {
	if n > len(f.items) {
		n = len(f.items)
	}
	return f.items[:n], nil
}

func newService(retrievers []Retriever, quotas []int) *Service {
	return &Service{
		Retrievers: retrievers,
		Features:   fakeFeatures{},
		Ranker:     fakeRanker{},
		Seen:       &fakeSeen{},
		Catalogue:  emptyCatalogue{},
		Fallback:   fakeFallback{items: []int32{900, 901, 902}},
		Config: Config{
			Quotas:          quotas,
			MaxCandidates:   10,
			Deadline:        time.Second,
			RankDeadline:    200 * time.Millisecond,
			FeatureDeadline: 200 * time.Millisecond,
			MaxHistory:      50,
			Lambda:          1.0,
		},
	}
}

// --- The pre-fan-out user fetch ----------------------------------------------

func TestEverySourceSeesTheSameUserSnapshot(t *testing.T) {
	// The reason the fetch happens ONCE, before the fan-out. Two lookups
	// mid-request can straddle a materialisation and produce a slate
	// retrieved for who the user was and ranked for who they are -- a state
	// no test would think to construct and no metric would show.
	first := &recordingRetriever{name: "a"}
	second := &recordingRetriever{name: "b"}
	users := &fakeUsers{user: User{Feats: []float32{1, 2}, History: []int32{9}, Found: true}}

	service := newService([]Retriever{first, second}, []int{5, 5})
	service.Users = users

	if _, err := service.Recommend(context.Background(), "U1", 3); err != nil {
		t.Fatalf("Recommend: %v", err)
	}

	if users.calls != 1 {
		t.Errorf("the gateway was called %d times, want exactly 1", users.calls)
	}
	for _, source := range []*recordingRetriever{first, second} {
		if len(source.got) != 1 {
			t.Fatalf("%s saw %d users", source.name, len(source.got))
		}
		got := source.got[0]
		if got.ID != "U1" || len(got.Feats) != 2 || len(got.History) != 1 {
			t.Errorf("%s was handed %+v", source.name, got)
		}
	}
}

func TestADeadFeatureGatewayServesAColdUserAndSaysSo(t *testing.T) {
	// Degrade, do not 500. A cold user still gets a slate from the sources
	// that need no features; failing would turn one materialisation gap into
	// an outage for everyone it touches.
	source := &recordingRetriever{name: "a"}
	service := newService([]Retriever{source}, []int{5})
	service.Users = &fakeUsers{err: errors.New("gateway down")}

	result, err := service.Recommend(context.Background(), "U1", 3)
	if err != nil {
		t.Fatalf("a dead gateway must not fail the request: %v", err)
	}

	if len(result.Items) == 0 {
		t.Error("a cold user should still be served")
	}
	if !contains(result.DegradedSources, "features") {
		t.Errorf("degraded_sources %v should name the gateway", result.DegradedSources)
	}
	// The identity survives even when the features do not: the sidecar keys
	// its embedding cache on it.
	if source.got[0].ID != "U1" {
		t.Errorf("the user id should survive a feature miss, got %q", source.got[0].ID)
	}
}

// TestAUserTheStoreHasNeverSeenIsNotADegradation separates the two events that
// a naive implementation collapses. A genuinely new user has no row, and that
// is the CORRECT answer -- reporting it as degraded would make the alert fire
// on normal cold-start traffic, which is how an alert gets muted.
func TestAUserTheStoreHasNeverSeenIsNotADegradation(t *testing.T) {
	service := newService([]Retriever{&recordingRetriever{name: "a"}}, []int{5})
	service.Users = &fakeUsers{user: User{Found: false}}

	result, err := service.Recommend(context.Background(), "U-new", 3)
	if err != nil {
		t.Fatalf("Recommend: %v", err)
	}

	if contains(result.DegradedSources, "features") {
		t.Errorf("a store miss is not a degradation; got %v", result.DegradedSources)
	}
}

func TestASlowFeatureGatewayIsBoundedByItsOwnDeadline(t *testing.T) {
	service := newService([]Retriever{&recordingRetriever{name: "a"}}, []int{5})
	service.Config.FeatureDeadline = 20 * time.Millisecond
	service.Users = &fakeUsers{delay: time.Second}

	started := time.Now()
	result, err := service.Recommend(context.Background(), "U1", 3)
	elapsed := time.Since(started)

	if err != nil {
		t.Fatalf("Recommend: %v", err)
	}
	// It runs BEFORE everything else, so it is the one stage that can spend
	// the whole request budget on its own.
	if elapsed > 500*time.Millisecond {
		t.Errorf("the fetch ran for %v, past its own 20ms deadline", elapsed)
	}
	if !contains(result.DegradedSources, "features") {
		t.Errorf("a timed-out fetch should be reported, got %v", result.DegradedSources)
	}
}

func TestASlowSourceIsDroppedAndReported(t *testing.T) {
	service := newService([]Retriever{
		fakeRetriever{name: "fast", budget: 50 * time.Millisecond, items: []int32{1, 2, 3}},
		fakeRetriever{
			name: "slow", budget: 10 * time.Millisecond,
			items: []int32{7, 8}, delay: 300 * time.Millisecond,
		},
	}, []int{5, 5})

	got, err := service.Recommend(context.Background(), "u1", 3)
	if err != nil {
		t.Fatalf("Recommend: %v", err)
	}

	if len(got.DegradedSources) != 1 || got.DegradedSources[0] != "slow" {
		t.Errorf("degraded: got %v, want [slow]", got.DegradedSources)
	}
	if len(got.Items) == 0 {
		t.Fatal("partial results beat a timeout; the fast source should still have served")
	}
	if got.UsedFallback {
		t.Error("one dropped source is not a fallback condition")
	}
}

func TestAFailingSourceDoesNotFailTheRequest(t *testing.T) {
	service := newService([]Retriever{
		fakeRetriever{name: "broken", budget: time.Second, err: errors.New("boom")},
		fakeRetriever{name: "ok", budget: time.Second, items: []int32{4, 5, 6}},
	}, []int{5, 5})

	got, err := service.Recommend(context.Background(), "u1", 2)
	if err != nil {
		t.Fatalf("Recommend: %v", err)
	}
	if len(got.Items) != 2 {
		t.Errorf("items: got %v, want 2 served from the healthy source", got.Items)
	}
	if len(got.DegradedSources) != 1 || got.DegradedSources[0] != "broken" {
		t.Errorf("degraded: got %v, want [broken]", got.DegradedSources)
	}
}

// TestSourceOrderSurvivesTheFanOut is the concurrency control. Quotas[i]
// belongs to Retrievers[i], so if the fan-out appended results in completion
// order the budget would silently go to whichever source answered first --
// producing a full, plausible slate built from the wrong source.
//
// **MaxCandidates is pinned to the quota, and that is what makes this test
// mean anything.** The first version left it at 10: `primary` filled its 3
// slots, the top-up then correctly spent the remaining 7 on `secondary`, and
// the ranker put secondary's items on top. Worse than a false failure, the
// assertion was INVERTED -- had the fan-out mispaired quotas, secondary would
// have taken the first slots, primary would have arrived via the top-up at
// higher indices, the ascending fake scores would have ranked primary's items
// first, and the test would have passed on the bug it exists to catch.
func TestSourceOrderSurvivesTheFanOut(t *testing.T) {
	service := newService([]Retriever{
		// The quota holder is deliberately the SLOWER of the two.
		fakeRetriever{
			name: "primary", budget: time.Second,
			items: []int32{11, 12, 13}, delay: 30 * time.Millisecond,
		},
		fakeRetriever{name: "secondary", budget: time.Second, items: []int32{21, 22, 23}},
	}, []int{3, 0})
	// No slack for the top-up, so the pool is exactly what the quota bought.
	service.Config.MaxCandidates = 3

	got, err := service.Recommend(context.Background(), "u1", 3)
	if err != nil {
		t.Fatalf("Recommend: %v", err)
	}
	if len(got.Items) != 3 {
		t.Fatalf("served %v, want 3", got.Items)
	}
	for _, item := range got.Items {
		if item < 11 || item > 13 {
			t.Fatalf("served %v; the quota belongs to `primary`, so every slot must be 11-13",
				got.Items)
		}
	}
}

// TestTheTopUpSpendsSlotsAnUnderFilledQuotaLeft pins the behaviour the test
// above had to exclude, so that excluding it is not mistaken for denying it.
// A source that cannot fill its allocation must not shrink the candidate pool.
func TestTheTopUpSpendsSlotsAnUnderFilledQuotaLeft(t *testing.T) {
	service := newService([]Retriever{
		fakeRetriever{name: "primary", budget: time.Second, items: []int32{11, 12}},
		fakeRetriever{name: "secondary", budget: time.Second, items: []int32{21, 22, 23}},
	}, []int{5, 0})
	service.Config.MaxCandidates = 5

	got, err := service.Recommend(context.Background(), "u1", 5)
	if err != nil {
		t.Fatalf("Recommend: %v", err)
	}
	if got.StageCounts["retrieved"] != 5 {
		t.Errorf("retrieved %d, want 5: primary's 2 plus 3 topped up from secondary",
			got.StageCounts["retrieved"])
	}
}

func TestEverySourceDeadFallsBackToPopularity(t *testing.T) {
	service := newService([]Retriever{
		fakeRetriever{name: "a", budget: time.Second, err: errors.New("down")},
		fakeRetriever{name: "b", budget: time.Second, err: errors.New("down")},
	}, []int{5, 5})

	got, err := service.Recommend(context.Background(), "u1", 2)
	if err != nil {
		t.Fatalf("Recommend: %v", err)
	}
	if !got.UsedFallback {
		t.Error("no candidates at all is exactly the fallback's case")
	}
	if len(got.Items) != 2 {
		t.Errorf("fallback served %v, want 2 popular items", got.Items)
	}
	if got.StageCounts["served"] != 2 {
		t.Errorf("stage_counts: got %v", got.StageCounts)
	}
}

// TestASlowRankerShipsRetrievalOrder pins the degradation the guide calls for
// and names what it costs: Part K measured retrieval order at 0.0716 NDCG@10
// against the ranker's 0.1283, so this path serves ~56% of the quality rather
// than an error.
func TestASlowRankerShipsRetrievalOrder(t *testing.T) {
	service := newService([]Retriever{
		fakeRetriever{name: "only", budget: time.Second, items: []int32{31, 32, 33}},
	}, []int{5})
	service.Ranker = fakeRanker{delay: 300 * time.Millisecond}
	service.Config.RankDeadline = 10 * time.Millisecond

	got, err := service.Recommend(context.Background(), "u1", 3)
	if err != nil {
		t.Fatalf("Recommend: %v", err)
	}

	if len(got.Items) != 3 {
		t.Fatalf("items: got %v, want the blended order served", got.Items)
	}
	// Retrieval order preserved, best first.
	if got.Items[0] != 31 || got.Items[1] != 32 || got.Items[2] != 33 {
		t.Errorf("items: got %v, want [31 32 33] in retrieval order", got.Items)
	}
	if !contains(got.DegradedSources, degradedRanker) {
		t.Errorf("a skipped ranker must be reported; degraded = %v", got.DegradedSources)
	}
	if got.UsedFallback {
		t.Error("a degraded ranker is not the popularity fallback; the two mean different things")
	}
}

// TestTheRankerActuallyReordersWhenItRuns is the control for the test above.
// Without it, a Score that was never called would satisfy that assertion
// exactly as well as one that was.
func TestTheRankerActuallyReordersWhenItRuns(t *testing.T) {
	service := newService([]Retriever{
		fakeRetriever{name: "only", budget: time.Second, items: []int32{31, 32, 33}},
	}, []int{5})
	// The default fake scores ASCENDING, so a ranker that ran must invert
	// retrieval order.
	got, err := service.Recommend(context.Background(), "u1", 3)
	if err != nil {
		t.Fatalf("Recommend: %v", err)
	}
	if got.Items[0] != 33 {
		t.Errorf("items: got %v, want the ranker's order [33 32 31]", got.Items)
	}
	if contains(got.DegradedSources, degradedRanker) {
		t.Error("the ranker ran; it must not be reported as degraded")
	}
}

// TestASaturatedSeenListDoesNotEmptyTheSlate is Part L's measured failure mode
// as a serving test. Past capacity a Bloom filter answers "seen" to every
// candidate. The filter is advice: the selector yields rather than returning a
// blank page, and the fallback is NOT triggered -- discarding a hundred good
// candidates because a filter misbehaved would be a worse outage than the one
// it prevents.
func TestASaturatedSeenListDoesNotEmptyTheSlate(t *testing.T) {
	service := newService([]Retriever{
		fakeRetriever{name: "only", budget: time.Second, items: []int32{41, 42, 43}},
	}, []int{5})
	service.Seen = &fakeSeen{blocked: []bool{true, true, true}}

	got, err := service.Recommend(context.Background(), "u1", 3)
	if err != nil {
		t.Fatalf("Recommend: %v", err)
	}
	if len(got.Items) != 3 {
		t.Fatalf("items: got %v, want a full slate despite the filter", got.Items)
	}
	if got.UsedFallback {
		t.Error("an over-blocking filter is not a retrieval failure")
	}
	if got.StageCounts["after_filter"] != 0 {
		t.Errorf("after_filter should still record 0 as the saturation signal, got %v",
			got.StageCounts)
	}
}

// TestASlowFeatureBuildDegradesLikeASlowRanker: from the caller's side a slow
// feature fetch and a slow model are the same symptom with the same mitigation,
// so they share a deadline and produce the same reported degradation.
func TestASlowFeatureBuildDegradesLikeASlowRanker(t *testing.T) {
	service := newService([]Retriever{
		fakeRetriever{name: "only", budget: time.Second, items: []int32{81, 82, 83}},
	}, []int{5})
	service.Features = fakeFeatures{delay: 300 * time.Millisecond}
	service.Config.RankDeadline = 10 * time.Millisecond

	got, err := service.Recommend(context.Background(), "u1", 3)
	if err != nil {
		t.Fatalf("Recommend: %v", err)
	}
	if got.Items[0] != 81 {
		t.Errorf("items: got %v, want retrieval order [81 82 83]", got.Items)
	}
	if !contains(got.DegradedSources, degradedRanker) {
		t.Errorf("a feature build that missed the deadline must be reported; got %v",
			got.DegradedSources)
	}
}

// TestAShortFeatureMatrixIsRefused guards the positional zip. The orchestrator
// pairs scores with candidates by index, so a matrix with one row missing would
// rank every item by the next item's score -- a full, plausible, wrong slate.
func TestAShortFeatureMatrixIsRefused(t *testing.T) {
	service := newService([]Retriever{
		fakeRetriever{name: "only", budget: time.Second, items: []int32{91, 92, 93}},
	}, []int{5})
	service.Features = fakeFeatures{short: true}

	got, err := service.Recommend(context.Background(), "u1", 3)
	if err != nil {
		t.Fatalf("Recommend: %v", err)
	}
	if !contains(got.DegradedSources, degradedRanker) {
		t.Errorf("a short feature matrix must degrade, not be scored; got %v",
			got.DegradedSources)
	}
	if got.Items[0] != 91 {
		t.Errorf("items: got %v, want the retrieval order preserved", got.Items)
	}
}

func TestABrokenSeenListFailsOpenAndSaysSo(t *testing.T) {
	service := newService([]Retriever{
		fakeRetriever{name: "only", budget: time.Second, items: []int32{51, 52}},
	}, []int{5})
	service.Seen = &fakeSeen{err: errors.New("redis down")}

	got, err := service.Recommend(context.Background(), "u1", 2)
	if err != nil {
		t.Fatalf("Recommend: %v", err)
	}
	if len(got.Items) != 2 {
		t.Errorf("items: got %v; failing open re-shows, failing closed empties the page", got.Items)
	}
	if !contains(got.DegradedSources, "seen") {
		t.Errorf("a filter that could not answer must be reported; got %v", got.DegradedSources)
	}
}

func TestWhatWasServedIsRecorded(t *testing.T) {
	seen := &fakeSeen{}
	service := newService([]Retriever{
		fakeRetriever{name: "only", budget: time.Second, items: []int32{61, 62, 63}},
	}, []int{5})
	service.Seen = seen

	got, err := service.Recommend(context.Background(), "u1", 2)
	if err != nil {
		t.Fatalf("Recommend: %v", err)
	}
	if len(seen.recorded) != len(got.Items) {
		t.Errorf("recorded %v, served %v -- the seen-list is meaningless if the two differ",
			seen.recorded, got.Items)
	}
}

func TestStageCountsAndLatenciesArePopulated(t *testing.T) {
	service := newService([]Retriever{
		fakeRetriever{name: "only", budget: time.Second, items: []int32{71, 72, 73, 74}},
	}, []int{5})

	got, err := service.Recommend(context.Background(), "u1", 2)
	if err != nil {
		t.Fatalf("Recommend: %v", err)
	}
	for _, stage := range []string{"retrieved", "after_filter", "scored", "served"} {
		if _, ok := got.StageCounts[stage]; !ok {
			t.Errorf("stage_counts missing %q: %v", stage, got.StageCounts)
		}
	}
	for _, stage := range []string{"retrieval", "filter", "rank", "rerank"} {
		if _, ok := got.StageLatency[stage]; !ok {
			t.Errorf("stage_latency missing %q: %v", stage, got.StageLatency)
		}
	}
	if got.StageCounts["served"] != 2 {
		t.Errorf("served: got %d, want 2", got.StageCounts["served"])
	}
}

func contains(haystack []string, needle string) bool {
	for _, value := range haystack {
		if value == needle {
			return true
		}
	}
	return false
}
