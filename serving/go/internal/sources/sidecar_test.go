package sources

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc"

	pb "github.com/MattyChoi/Recommender-System-Design/serving/go/internal/pb"
	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/service"
)

// fakeRetrieval stands in for the sidecar. It EMBEDS the generated client
// interface rather than implementing every method: the embedded nil satisfies
// the type and the one method below shadows it, so any other RPC panics --
// which is the behaviour worth having, since a test that silently exercised an
// unimplemented call would be testing nothing.
//
// The mutex is not decoration. `make go-race` is a gate now, and the
// concurrency test below calls this from several goroutines; a fake that
// raced would fail the gate on its own account and obscure whether the code
// under test is clean.
type fakeRetrieval struct {
	pb.RetrievalClient

	mu       sync.Mutex
	captured *pb.RetrieveRequest
	calls    int

	response *pb.RetrieveResponse
	err      error
}

func (f *fakeRetrieval) Retrieve(
	_ context.Context, in *pb.RetrieveRequest, _ ...grpc.CallOption,
) (*pb.RetrieveResponse, error) {
	f.mu.Lock()
	f.captured = in
	f.calls++
	f.mu.Unlock()

	if f.err != nil {
		return nil, f.err
	}
	return f.response, nil
}

func (f *fakeRetrieval) request() *pb.RetrieveRequest {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.captured
}

// fakeFallback is rung 2. `ok` false means the user has no cached embedding,
// which is a MISS rather than a failure.
type fakeFallback struct {
	items []int32
	ok    bool
	err   error
	calls int
	mu    sync.Mutex
}

func (f *fakeFallback) Search(_ context.Context, _ string, _ int) ([]int32, bool, error) {
	f.mu.Lock()
	f.calls++
	f.mu.Unlock()
	return f.items, f.ok, f.err
}

func newSidecar(client pb.RetrievalClient) *Sidecar {
	return &Sidecar{
		client:     client,
		SourceName: "two_tower",
		Deadline:   25 * time.Millisecond,
		K:          100,
	}
}

func testUser() service.User {
	return service.User{
		ID:      "U1",
		Feats:   []float32{1, 2, 3, 4},
		History: []int32{9, 8, 7},
		Found:   true,
	}
}

// answer builds a WELL-FORMED response: three arrays of equal length, which is
// what the sidecar guarantees by filtering its OOV rows before computing the
// model columns. Tests that want the ragged case build it explicitly.
func answer(items ...int32) *pb.RetrieveResponse {
	scores := make([]float32, len(items))
	similarity := make([]float32, len(items))
	for index := range items {
		scores[index] = float32(len(items) - index)
		similarity[index] = float32(index) / 10
	}
	return &pb.RetrieveResponse{
		Items:             items,
		Scores:            scores,
		ContentSimilarity: similarity,
		IndexKind:         "hnsw",
		IndexVersion:      "v=2026-09-22T12:00Z",
	}
}

// --- The request on the wire -------------------------------------------------

func TestTheRequestCarriesTheFetchedUser(t *testing.T) {
	fake := &fakeRetrieval{response: answer(1, 2, 3)}
	sidecar := newSidecar(fake)
	sidecar.EFSearch = 128

	got, err := sidecar.Retrieve(context.Background(), testUser())
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}

	sent := fake.request()
	if sent.GetUserId() != "U1" {
		t.Errorf("user id %q, want U1", sent.GetUserId())
	}
	// Sent rather than fetched by the sidecar: the orchestrator's pre-fan-out
	// call already paid for these, and a second read could see a different
	// snapshot than the one the ranker's features came from.
	if len(sent.GetUserFeats()) != 4 || len(sent.GetHistory()) != 3 {
		t.Errorf("feats %v history %v", sent.GetUserFeats(), sent.GetHistory())
	}
	if sent.GetK() != 100 || sent.GetEfSearch() != 128 {
		t.Errorf("k %d efSearch %d", sent.GetK(), sent.GetEfSearch())
	}
	if len(got.Items) != 3 || len(got.Scores) != 3 || len(got.Similarity) != 3 {
		t.Errorf("got %+v", got)
	}
}

// TestRaggedModelColumnsDegradeRatherThanShift is the guard on three arrays
// that travel separately.
//
// A short scores array pairs each item with the NEXT item's score from that
// point on: a complete, plausible, wrong ranking with nothing reporting it.
// The sidecar filters its OOV rows before computing these precisely so the
// lengths agree, so a mismatch means the two sides are out of step.
func TestRaggedModelColumnsDegradeRatherThanShift(t *testing.T) {
	sidecar := newSidecar(&fakeRetrieval{response: &pb.RetrieveResponse{
		Items:             []int32{1, 2, 3},
		Scores:            []float32{0.9, 0.8},
		ContentSimilarity: []float32{0.3, 0.2, 0.1},
		IndexKind:         "hnsw",
	}})

	_, err := sidecar.Retrieve(context.Background(), testUser())

	if err == nil {
		t.Fatal("3 items with 2 scores must not be zipped")
	}
	if !strings.Contains(err.Error(), "2 scores") {
		t.Errorf("the error should show the arithmetic, got %q", err)
	}
}

func TestAnUnsetEFSearchLeavesTheServersDefault(t *testing.T) {
	// Zero means "whatever the server is configured with", and the response
	// says which. Sending a value here would override ADR 0002's 512 from the
	// client side, invisibly.
	fake := &fakeRetrieval{response: answer(1)}

	if _, err := newSidecar(fake).Retrieve(context.Background(), testUser()); err != nil {
		t.Fatalf("Retrieve: %v", err)
	}

	if fake.request().GetEfSearch() != 0 {
		t.Errorf("efSearch %d, want 0", fake.request().GetEfSearch())
	}
}

// --- What answered -----------------------------------------------------------

func TestASuccessRecordsWhichIndexAnswered(t *testing.T) {
	fake := &fakeRetrieval{response: answer(1)}
	sidecar := newSidecar(fake)

	if _, err := sidecar.Retrieve(context.Background(), testUser()); err != nil {
		t.Fatalf("Retrieve: %v", err)
	}

	// Recorded on SUCCESS, because the interesting case is a success: a
	// sidecar that fell back to exact search internally is correct, much
	// slower, and says so only here.
	kind, version := sidecar.LastIndex()
	if kind != "hnsw" || version != "v=2026-09-22T12:00Z" {
		t.Errorf("LastIndex() = %q, %q", kind, version)
	}
}

func TestASidecarServingFlatIsReportedAsFlat(t *testing.T) {
	// Built from answer() rather than by hand, so the three model arrays stay
	// well-formed: this test is about what the sidecar SAYS answered, and a
	// hand-rolled response missing its columns would fail on the length guard
	// instead and prove nothing about the reporting.
	response := answer(1)
	response.IndexKind = "flat"
	response.IndexVersion = "v=1"
	sidecar := newSidecar(&fakeRetrieval{response: response})

	if _, err := sidecar.Retrieve(context.Background(), testUser()); err != nil {
		t.Fatalf("Retrieve: %v", err)
	}

	if kind, _ := sidecar.LastIndex(); kind != "flat" {
		t.Errorf("kind %q; a quiet failover to exact search must not read as hnsw", kind)
	}
}

// --- The degradation ladder --------------------------------------------------

func TestWithNoFallbackTheSourceSimplyDegrades(t *testing.T) {
	sidecar := newSidecar(&fakeRetrieval{err: errors.New("unavailable")})

	_, err := sidecar.Retrieve(context.Background(), testUser())

	if err == nil {
		t.Fatal("expected an error the fan-out can record as degraded")
	}
	if !strings.Contains(err.Error(), "retrieval sidecar") {
		t.Errorf("the error should name the source, got %q", err)
	}
}

func TestRungTwoAnswersWhenTheSidecarIsUnreachable(t *testing.T) {
	fallback := &fakeFallback{items: []int32{4, 5, 6}, ok: true}
	sidecar := newSidecar(&fakeRetrieval{err: errors.New("connection refused")})
	sidecar.Fallback = fallback

	got, err := sidecar.Retrieve(context.Background(), testUser())
	if err != nil {
		t.Fatalf("rung 2 should answer: %v", err)
	}

	if len(got.Items) != 3 || got.Items[0] != 4 {
		t.Errorf("got %v, want the fallback's candidates", got.Items)
	}
	// Ids only. Rung 2 has the item embeddings but neither the content table
	// nor the tower's view of this user, so it cannot produce the model
	// columns -- and inventing them would be worse than their absence.
	if got.Scores != nil || got.Similarity != nil {
		t.Errorf("rung 2 must not fabricate model columns: %+v", got)
	}
	if fallback.calls != 1 {
		t.Errorf("the fallback ran %d times", fallback.calls)
	}
}

// TestRungTwoReportsFlatAndForgetsTheSidecarsVersion pins a distinction that is
// easy to get wrong in the convenient direction.
func TestRungTwoReportsFlatAndForgetsTheSidecarsVersion(t *testing.T) {
	fake := &fakeRetrieval{response: answer(1)}
	sidecar := newSidecar(fake)
	sidecar.Fallback = &fakeFallback{items: []int32{4}, ok: true}

	// A healthy call first, so there IS a version to wrongly carry over.
	if _, err := sidecar.Retrieve(context.Background(), testUser()); err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	fake.err = errors.New("gone")
	if _, err := sidecar.Retrieve(context.Background(), testUser()); err != nil {
		t.Fatalf("rung 2 should answer: %v", err)
	}

	kind, version := sidecar.LastIndex()
	if kind != "flat" {
		t.Errorf("kind %q, want flat", kind)
	}
	// The version the SIDECAR last reported says nothing about what this
	// process just searched. A health endpoint claiming an index version it
	// did not use is worse than one admitting it does not know.
	if version != "" {
		t.Errorf("version %q should not survive a fallback", version)
	}
}

// TestACacheMissReturnsTheSidecarsErrorNotTheFallbacks is the one that decides
// where whoever is on call starts looking.
func TestACacheMissReturnsTheSidecarsErrorNotTheFallbacks(t *testing.T) {
	sidecar := newSidecar(&fakeRetrieval{err: errors.New("deadline exceeded")})
	// ok=false: the user has no cached embedding, so rung 2 cannot answer.
	sidecar.Fallback = &fakeFallback{ok: false}

	_, err := sidecar.Retrieve(context.Background(), testUser())

	if err == nil {
		t.Fatal("expected the request to degrade")
	}
	// "no cached embedding" is a CONSEQUENCE of the sidecar being down.
	// Reporting it as the cause sends someone to Redis to debug a gRPC outage.
	if !strings.Contains(err.Error(), "deadline exceeded") {
		t.Errorf("the error should name the sidecar failure, got %q", err)
	}
}

func TestAFailingFallbackAlsoReportsTheSidecar(t *testing.T) {
	sidecar := newSidecar(&fakeRetrieval{err: errors.New("unavailable")})
	sidecar.Fallback = &fakeFallback{err: errors.New("redis down"), ok: true}

	_, err := sidecar.Retrieve(context.Background(), testUser())

	if err == nil || !strings.Contains(err.Error(), "unavailable") {
		t.Errorf("the original cause should survive, got %v", err)
	}
}

func TestAHealthySidecarNeverTouchesTheFallback(t *testing.T) {
	// The control. Without it, every test above would pass on an
	// implementation that ran the fallback unconditionally.
	fallback := &fakeFallback{items: []int32{99}, ok: true}
	sidecar := newSidecar(&fakeRetrieval{response: answer(1, 2)})
	sidecar.Fallback = fallback

	got, err := sidecar.Retrieve(context.Background(), testUser())
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}

	if fallback.calls != 0 {
		t.Error("rung 2 ran while rung 1 was answering")
	}
	if len(got.Items) != 2 {
		t.Errorf("got %v, want the sidecar's candidates", got.Items)
	}
}

// --- Concurrency -------------------------------------------------------------

// TestLastIndexIsSafeUnderTheFanOut is why `make go-race` exists.
//
// One Sidecar is held as a singleton and called from a goroutine per request.
// Before the mutex, lastKind and lastVersion were plain fields written on
// every call: a genuine data race that `go vet` and plain `go test` both pass.
// This test is meaningless without -race and decisive with it.
func TestLastIndexIsSafeUnderTheFanOut(t *testing.T) {
	sidecar := newSidecar(&fakeRetrieval{response: answer(1)})

	var wait sync.WaitGroup
	for worker := 0; worker < 16; worker++ {
		wait.Add(2)
		go func() {
			defer wait.Done()
			_, _ = sidecar.Retrieve(context.Background(), testUser())
		}()
		go func() {
			defer wait.Done()
			_, _ = sidecar.LastIndex()
		}()
	}
	wait.Wait()

	if kind, _ := sidecar.LastIndex(); kind != "hnsw" {
		t.Errorf("kind %q after the fan-out", kind)
	}
}

// --- The port ----------------------------------------------------------------

func TestTheSourceReportsItsNameAndBudget(t *testing.T) {
	// Both are read by the orchestrator: the name keys quotas and
	// degraded_sources, and Config.Quotas is POSITIONAL, so a source whose
	// name drifts from its slot hands the budget to a different retriever.
	sidecar := newSidecar(&fakeRetrieval{response: answer()})

	if sidecar.Name() != "two_tower" {
		t.Errorf("name %q", sidecar.Name())
	}
	if sidecar.Budget() != 25*time.Millisecond {
		t.Errorf("budget %v", sidecar.Budget())
	}
}
