package service

import (
	"context"
	"slices"
	"testing"
	"time"
)

// Fakes prefixed `shadow` so they cannot collide with recommend_test.go's.

type shadowSource struct{ items []int32 }

func (s shadowSource) Name() string          { return "shadow_source" }
func (s shadowSource) Budget() time.Duration { return 50 * time.Millisecond }

func (s shadowSource) Retrieve(context.Context, User) (Candidates, error) {
	return Candidates{Items: s.items}, nil
}

type shadowBuilder struct{ rows int }

func (b shadowBuilder) Build(context.Context, BuildInput) (Features, error) {
	return Features{Rows: make([][]float32, b.rows)}, nil
}

// shadowScorer returns a fixed score vector, optionally after a delay. The
// delay respects cancellation, so a shadow abandoned with the request does not
// outlive the test.
type shadowScorer struct {
	scores []float64
	delay  time.Duration
}

func (r shadowScorer) Score(ctx context.Context, _ Features) ([]float64, error) {
	if r.delay > 0 {
		select {
		case <-time.After(r.delay):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	return r.scores, nil
}

type shadowCatalogue struct{}

func (shadowCatalogue) Vectors([]int32) [][]float32         { return nil }
func (shadowCatalogue) Categories(items []int32) []int32    { return make([]int32, len(items)) }
func (shadowCatalogue) Subcategories(items []int32) []int32 { return make([]int32, len(items)) }

type shadowFallback struct{}

func (shadowFallback) Popular(context.Context, int) ([]int32, error) { return []int32{900}, nil }

type shadowSink struct{ records []Record }

func (s *shadowSink) EmitAsync(rec Record) { s.records = append(s.records, rec) }

// shadowService wires four candidates, no store and no seen-list: this file is
// about the shadow, and every other stage is exercised elsewhere.
func shadowService(sink *shadowSink, live, shade Ranker) *Service {
	return &Service{
		Retrievers:  []Retriever{shadowSource{items: []int32{10, 11, 12, 13}}},
		Features:    shadowBuilder{rows: 4},
		Ranker:      live,
		Shadow:      shade,
		ShadowModel: "ranker_v2",
		Catalogue:   shadowCatalogue{},
		Fallback:    shadowFallback{},
		Impressions: sink,
		Config: Config{
			Quotas:          []int{4},
			MaxCandidates:   4,
			Deadline:        time.Second,
			RankDeadline:    50 * time.Millisecond,
			FeatureDeadline: 50 * time.Millisecond,
			Lambda:          1.0,
		},
	}
}

func TestAShadowDoesNotChangeTheSlate(t *testing.T) {
	live := shadowScorer{scores: []float64{1, 2, 3, 4}}
	request := Request{UserID: "u", NumResults: 4}

	plain := &shadowSink{}
	without, err := shadowService(plain, live, nil).Run(context.Background(), request)
	if err != nil {
		t.Fatalf("without a shadow: %v", err)
	}

	shaded := &shadowSink{}
	shadow := shadowScorer{scores: []float64{100, 200, 300, 400}}
	with, err := shadowService(shaded, live, shadow).Run(context.Background(), request)
	if err != nil {
		t.Fatalf("with a shadow: %v", err)
	}

	// The whole promise of shadow mode, and the only one that cannot be
	// compromised: a model nobody is served must not move what is served.
	if !slices.Equal(without.Items, with.Items) {
		t.Errorf("slate changed: %v without the shadow, %v with it", without.Items, with.Items)
	}
	if !slices.Equal(without.Scores, with.Scores) {
		t.Errorf("scores changed: %v against %v", without.Scores, with.Scores)
	}
}

func TestASlowShadowDoesNotExtendTheRequest(t *testing.T) {
	sink := &shadowSink{}
	pipeline := shadowService(
		sink,
		shadowScorer{scores: []float64{1, 2, 3, 4}},
		shadowScorer{scores: []float64{100, 200, 300, 400}, delay: 300 * time.Millisecond},
	)

	started := time.Now()
	if _, err := pipeline.Run(context.Background(), Request{UserID: "u", NumResults: 4}); err != nil {
		t.Fatalf("run: %v", err)
	}

	// Generous bound: the point is that the request did not wait ~300ms, not
	// that it took any particular time on a loaded CI runner.
	if took := time.Since(started); took > 150*time.Millisecond {
		t.Errorf("waited %v on a 300ms shadow; a shadow must never be waited for", took)
	}

	if len(sink.records) != 1 {
		t.Fatalf("logged %d records, want 1", len(sink.records))
	}
	if sink.records[0].ShadowScores != nil {
		t.Errorf("a shadow that missed the slate must be ABSENT, got %v",
			sink.records[0].ShadowScores)
	}
	if got := pipeline.ShadowDropped(); got != 1 {
		t.Errorf("dropped = %d, want 1; an uncounted drop is a silently dead shadow", got)
	}
}

func TestTheShadowIsReindexedOntoTheServedSlots(t *testing.T) {
	// The subtle one. ImpressionContext's item_ids, scores and objective are
	// aligned with each other and with position, so a shadow column left in
	// CANDIDATE order joins to the wrong items -- with matching lengths and
	// plausible values, so nothing downstream could notice.
	pipeline := &Service{ShadowModel: "ranker_v2"}

	answered := make(chan []float64, 1)
	answered <- []float64{100, 200, 300, 400} // by candidate index

	// Ranking reversed the candidates, so the slate served 3, 2, 1, 0.
	got, model := pipeline.shadowAt(answered, []int{3, 2, 1, 0})

	want := []float64{400, 300, 200, 100}
	if !slices.Equal(got, want) {
		t.Errorf("shadow scores %v, want %v", got, want)
	}
	if model != "ranker_v2" {
		t.Errorf("model %q, want ranker_v2", model)
	}
	if n := pipeline.ShadowLogged(); n != 1 {
		t.Errorf("logged = %d, want 1", n)
	}
}

func TestAShadowThatHasNotAnsweredIsDropped(t *testing.T) {
	pipeline := &Service{ShadowModel: "ranker_v2"}

	got, model := pipeline.shadowAt(make(chan []float64), []int{0, 1})
	if got != nil || model != "" {
		t.Errorf("got %v / %q, want nil and empty", got, model)
	}
	if n := pipeline.ShadowDropped(); n != 1 {
		t.Errorf("dropped = %d, want 1", n)
	}
}

func TestAFailedShadowIsDroppedRatherThanPadded(t *testing.T) {
	pipeline := &Service{ShadowModel: "ranker_v2"}

	answered := make(chan []float64, 1)
	answered <- nil // what startShadow sends on an error or a short vector

	got, model := pipeline.shadowAt(answered, []int{0, 1})
	if got != nil {
		t.Errorf("got %v; zeros would read as a model that scored everything zero", got)
	}
	if model != "" {
		t.Errorf("model %q, want empty: a named model with no scores is a lie", model)
	}
	if n := pipeline.ShadowDropped(); n != 1 {
		t.Errorf("dropped = %d, want 1", n)
	}
}

func TestShadowModeOffCountsNothing(t *testing.T) {
	pipeline := &Service{}

	got, model := pipeline.shadowAt(nil, []int{0, 1})
	if got != nil || model != "" {
		t.Errorf("got %v / %q, want nil and empty", got, model)
	}
	// Not a drop. A shadow nobody configured has not failed, and counting it
	// would make the drop counter useless as an alert.
	if pipeline.ShadowDropped() != 0 || pipeline.ShadowLogged() != 0 {
		t.Errorf("counted %d dropped and %d logged with no shadow configured",
			pipeline.ShadowDropped(), pipeline.ShadowLogged())
	}
}
