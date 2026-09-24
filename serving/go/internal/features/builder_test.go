package features

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/retrieval"
	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/service"
)

// featureOrder is models/ranking/dataset.py:FEATURES, verbatim. The real order
// comes from the exported graph's model.columns.json; this is what that file
// contains today and what the tests assert against.
var featureOrder = []string{
	ColRetrievalScore,
	ColTwoTowerRank,
	ColTrendingRank,
	ColSources,
	ColPriorClicks,
	ColTrainClicks,
	ColContentSimilarity,
	ColHistoryLength,
	ColColdItem,
	ColCategory,
	ColSubcategory,
}

type fakeItems struct {
	rows  [][]float32
	names []string
	err   error
	calls int
}

func (f *fakeItems) ItemRows(
	_ context.Context, items []int32,
) ([][]float32, []string, []bool, error) {
	f.calls++
	if f.err != nil {
		return nil, nil, nil, f.err
	}
	return f.rows, f.names, make([]bool, len(items)), nil
}

type fakeCatalogue struct {
	categories    []int32
	subcategories []int32
}

func (f fakeCatalogue) Vectors([]int32) [][]float32   { return nil }
func (f fakeCatalogue) Categories([]int32) []int32    { return f.categories }
func (f fakeCatalogue) Subcategories([]int32) []int32 { return f.subcategories }

// A two-candidate request with every column distinguishable from every other,
// so a transposition cannot pass by coincidence.
func input() service.BuildInput {
	return service.BuildInput{
		User:  service.User{ID: "U1", History: []int32{9, 8, 7}},
		Items: []int32{11, 22},
		Ranks: map[string][]int32{
			"two_tower": {0, 1},
			"trending":  {5, retrieval.Absent},
		},
		Model: service.Columns{
			Score:      map[int32]float32{11: 0.91, 22: 0.82},
			Similarity: map[int32]float32{11: 0.31, 22: 0.22},
		},
	}
}

func newBuilder(names []string) (*Builder, *fakeItems) {
	items := &fakeItems{
		// prior_clicks then train_clicks, per RANKER_COLUMN_SOURCE.
		rows:  [][]float32{{500, 7}, {600, 0}},
		names: []string{StorePriorClicks, StoreTrainClicks},
	}
	return &Builder{
		Items:     items,
		Catalogue: fakeCatalogue{categories: []int32{3, 4}, subcategories: []int32{30, 40}},
		Names:     names,
	}, items
}

func build(t *testing.T, names []string) service.Features {
	t.Helper()
	builder, _ := newBuilder(names)
	got, err := builder.Build(context.Background(), input())
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	return got
}

// --- The full row ------------------------------------------------------------

func TestEveryColumnLandsInItsDeclaredSlot(t *testing.T) {
	got := build(t, featureOrder)

	// Written out longhand rather than computed, so this fails if the assembly
	// changes rather than moving with it.
	want := [][]float32{
		// score, tt_rank, trend_rank, n_sources, prior, train, sim, hist, cold, cat, subcat
		{0.91, 0, 5, 2, 500, 7, 0.31, 3, 0, 3, 30},
		{0.82, 1, float32(retrieval.Absent), 1, 600, 0, 0.22, 3, 1, 4, 40},
	}
	if len(got.Rows) != len(want) {
		t.Fatalf("%d rows, want %d", len(got.Rows), len(want))
	}
	for row := range want {
		for slot := range want[row] {
			if got.Rows[row][slot] != want[row][slot] {
				t.Errorf("row %d slot %d (%s) = %v, want %v",
					row, slot, featureOrder[slot], got.Rows[row][slot], want[row][slot])
			}
		}
	}
}

// TestTheOrderFollowsNamesNotTheAssembly is the point of building a named map
// and emitting from it. The graph's column order is the one part of the serving
// contract no shape check can catch downstream.
func TestTheOrderFollowsNamesNotTheAssembly(t *testing.T) {
	reversed := make([]string, len(featureOrder))
	for index, name := range featureOrder {
		reversed[len(featureOrder)-1-index] = name
	}

	normal := build(t, featureOrder)
	flipped := build(t, reversed)

	width := len(featureOrder)
	for slot := 0; slot < width; slot++ {
		if flipped.Rows[0][slot] != normal.Rows[0][width-1-slot] {
			t.Fatalf("slot %d did not follow Names", slot)
		}
	}
}

func TestASubsetOfColumnsIsHonoured(t *testing.T) {
	// FEATURES is configurable on the Python side too (`features=` on build),
	// so an ablation arm exporting three columns must work here unchanged.
	got := build(t, []string{ColColdItem, ColRetrievalScore})

	if len(got.Rows[0]) != 2 || got.Rows[0][0] != 0 || got.Rows[0][1] != 0.91 {
		t.Errorf("row %v", got.Rows[0])
	}
}

// TestAnUnknownColumnIsRefused: a graph exported against a FEATURES list this
// build does not implement would otherwise put a confident zero in that slot on
// every row, and the model would score it.
func TestAnUnknownColumnIsRefused(t *testing.T) {
	builder, _ := newBuilder([]string{ColRetrievalScore, "dwell_time_p50"})

	_, err := builder.Build(context.Background(), input())

	if err == nil {
		t.Fatal("an unimplemented column must be refused, not zero-filled")
	}
	if !strings.Contains(err.Error(), "dwell_time_p50") {
		t.Errorf("the error should name the column, got %q", err)
	}
}

func TestNoConfiguredOrderIsRefused(t *testing.T) {
	builder := &Builder{Items: &fakeItems{}, Catalogue: fakeCatalogue{}}

	if _, err := builder.Build(context.Background(), input()); err == nil {
		t.Fatal("a builder with no column order must refuse rather than invent one")
	}
}

// --- The blend columns -------------------------------------------------------

// TestAMissingSourceRanksAbsentNotZero is the one that would be silently wrong.
//
// Zero is the BEST rank a source can give. A missing source defaulting to 0
// tells the ranker that every candidate was that source's top pick -- a strong,
// false signal on a column the model was trained to trust.
func TestAMissingSourceRanksAbsentNotZero(t *testing.T) {
	in := input()
	delete(in.Ranks, "trending")
	builder, _ := newBuilder(featureOrder)

	got, err := builder.Build(context.Background(), in)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	slot := indexOf(featureOrder, ColTrendingRank)
	for row := range got.Rows {
		if got.Rows[row][slot] != float32(retrieval.Absent) {
			t.Errorf("row %d trending_rank = %v, want Absent", row, got.Rows[row][slot])
		}
	}
}

func TestSourceCountIgnoresAbsentRanks(t *testing.T) {
	got := build(t, featureOrder)
	slot := indexOf(featureOrder, ColSources)

	// Item 11 was proposed by both sources; item 22 by two_tower only, because
	// trending recorded Absent for it.
	if got.Rows[0][slot] != 2 || got.Rows[1][slot] != 1 {
		t.Errorf("n_sources = %v, %v; want 2, 1", got.Rows[0][slot], got.Rows[1][slot])
	}
}

// --- The store columns -------------------------------------------------------

// TestStoreColumnsAreReadByNameNotPosition: the gateway owns its own order, and
// a reordering there must be absorbed here rather than shifting every value.
func TestStoreColumnsAreReadByNameNotPosition(t *testing.T) {
	builder, items := newBuilder(featureOrder)
	items.names = []string{StoreTrainClicks, StorePriorClicks}
	items.rows = [][]float32{{7, 500}, {0, 600}}

	got, err := builder.Build(context.Background(), input())
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	if got.Rows[0][indexOf(featureOrder, ColPriorClicks)] != 500 {
		t.Error("prior_clicks followed position rather than name")
	}
	if got.Rows[0][indexOf(featureOrder, ColTrainClicks)] != 7 {
		t.Error("train_clicks followed position rather than name")
	}
}

// TestColdItemIsDerivedFromTrainClicks mirrors Python's `train_clicks == 0`.
// Derived rather than fetched, so it cannot disagree with the count beside it.
func TestColdItemIsDerivedFromTrainClicks(t *testing.T) {
	got := build(t, featureOrder)
	cold := indexOf(featureOrder, ColColdItem)
	train := indexOf(featureOrder, ColTrainClicks)

	for row := range got.Rows {
		wantCold := float32(0)
		if got.Rows[row][train] == 0 {
			wantCold = 1
		}
		if got.Rows[row][cold] != wantCold {
			t.Errorf("row %d: train=%v cold=%v", row, got.Rows[row][train], got.Rows[row][cold])
		}
	}
}

func TestAFailingItemStoreFailsTheBuild(t *testing.T) {
	// Not degraded to zeros: five of eleven columns would be a confident zero,
	// and the pipeline already has a path for a build that fails -- it ships
	// the retrieval order and reports the ranker degraded.
	builder, items := newBuilder(featureOrder)
	items.err = errors.New("gateway down")

	if _, err := builder.Build(context.Background(), input()); err == nil {
		t.Fatal("a failed item fetch must not produce a matrix")
	}
}

// --- history_length ----------------------------------------------------------

func TestHistoryLengthIsCappedLikeTheTowerPools(t *testing.T) {
	// The gateway already truncates, but the cap now lives in two languages.
	// A serving path that counted more than it pooled would report a user as
	// better-known than the embedding just built for them.
	in := input()
	in.User.History = make([]int32, MaxHistory+25)
	for index := range in.User.History {
		in.User.History[index] = int32(index + 1)
	}
	builder, _ := newBuilder(featureOrder)

	got, err := builder.Build(context.Background(), in)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	if got.Rows[0][indexOf(featureOrder, ColHistoryLength)] != float32(MaxHistory) {
		t.Errorf("history_length = %v, want %d",
			got.Rows[0][indexOf(featureOrder, ColHistoryLength)], MaxHistory)
	}
}

// TestPaddingDoesNotCountAsHistory pins the other half of the same rule.
//
// Index 0 is the reserved OOV row that short lists pad with. Counted, it makes
// a barely-known user look well-known to the ranker while the tower pooled
// nothing extra -- two columns disagreeing about the same person.
func TestPaddingDoesNotCountAsHistory(t *testing.T) {
	in := input()
	in.User.History = []int32{9, 0, 8, 0, 0}
	builder, _ := newBuilder(featureOrder)

	got, err := builder.Build(context.Background(), in)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	if got.Rows[0][indexOf(featureOrder, ColHistoryLength)] != 2 {
		t.Errorf("history_length = %v, want 2",
			got.Rows[0][indexOf(featureOrder, ColHistoryLength)])
	}
}

// --- The rung-2 case ---------------------------------------------------------

// TestMissingModelColumnsAreCountedNotHidden is the reason Features carries a
// count at all.
//
// Zero is not "unknown" to this model: retrieval_score is a cosine, so a
// zero-filled row asserts the candidate is ORTHOGONAL to the user -- a
// confident statement, made by accident, on every candidate at once.
func TestMissingModelColumnsAreCountedNotHidden(t *testing.T) {
	in := input()
	in.Model = service.Columns{Score: map[int32]float32{}, Similarity: map[int32]float32{}}
	builder, _ := newBuilder(featureOrder)

	got, err := builder.Build(context.Background(), in)
	if err != nil {
		t.Fatalf("rung 2 must still produce a matrix: %v", err)
	}

	if got.Incomplete != 2 {
		t.Errorf("Incomplete = %d, want 2", got.Incomplete)
	}
	if got.Rows[0][indexOf(featureOrder, ColRetrievalScore)] != 0 {
		t.Error("the missing column should be zero-filled")
	}
}

func TestACompleteRequestReportsNothingIncomplete(t *testing.T) {
	// The control: without it, the count could be wired to a constant.
	if got := build(t, featureOrder); got.Incomplete != 0 {
		t.Errorf("Incomplete = %d on a complete request", got.Incomplete)
	}
}

func TestOneMissingColumnStillCountsTheRow(t *testing.T) {
	// Similarity present, score absent. A row is incomplete if EITHER
	// model column is missing -- counting only the both-missing case would
	// under-report exactly the partial degradations hardest to notice.
	in := input()
	in.Model.Score = map[int32]float32{11: 0.91}
	builder, _ := newBuilder(featureOrder)

	got, err := builder.Build(context.Background(), in)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if got.Incomplete != 1 {
		t.Errorf("Incomplete = %d, want 1", got.Incomplete)
	}
}

// --- Empty -------------------------------------------------------------------

func TestAnEmptyCandidateSetSkipsTheStore(t *testing.T) {
	builder, items := newBuilder(featureOrder)

	got, err := builder.Build(context.Background(), service.BuildInput{})
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	if len(got.Rows) != 0 {
		t.Errorf("rows %v", got.Rows)
	}
	// Names still travel, so the ranker's column check has something to
	// compare against even on an empty matrix.
	if len(got.Names) != len(featureOrder) {
		t.Error("the column names should survive an empty request")
	}
	if items.calls != 0 {
		t.Error("an empty request is a round trip nobody needs")
	}
}

func indexOf(names []string, want string) int {
	for index, name := range names {
		if name == want {
			return index
		}
	}
	return -1
}
