package rerank

import (
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

const fixturePath = "../../../testdata/rerank_cases.json"

type rerankCase struct {
	Name               string      `json:"name"`
	Scores             []float64   `json:"scores"`
	Items              []int32     `json:"items"`
	K                  int         `json:"k"`
	Vectors            [][]float32 `json:"vectors"`
	Lambda             float64     `json:"lambda"`
	Categories         []int32     `json:"categories"`
	Cap                *int        `json:"cap"`
	Blocked            []bool      `json:"blocked"`
	ExpectedItems      []int32     `json:"expected_items"`
	ExpectedPropensity []float64   `json:"expected_propensity"`
}

func loadCases(t *testing.T) []rerankCase {
	t.Helper()
	raw, err := os.ReadFile(filepath.Clean(fixturePath))
	if err != nil {
		t.Fatalf("reading fixture: %v (run `make fixtures`)", err)
	}
	var cases []rerankCase
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatalf("parsing fixture: %v", err)
	}
	if len(cases) == 0 {
		t.Fatal("fixture is empty; a parity test over no cases passes and proves nothing")
	}
	return cases
}

// TestSelectMatchesPython compares the DECISION -- which items, in which order
// -- rather than the intermediate floats. That is the right assertion for two
// reasons. The decision is what a user sees and what every offline number was
// computed from; and float noise below the margin between two candidates is
// not a disagreement, while noise that crosses it is exactly the failure worth
// catching, and it shows up as a different item.
func TestSelectMatchesPython(t *testing.T) {
	for _, testCase := range loadCases(t) {
		t.Run(testCase.Name, func(t *testing.T) {
			got, err := Select(testCase.Scores, testCase.Items, testCase.K, Options{
				Vectors:    testCase.Vectors,
				Lambda:     testCase.Lambda,
				Categories: testCase.Categories,
				Cap:        testCase.Cap,
				Blocked:    testCase.Blocked,
			})
			if err != nil {
				t.Fatalf("Select: %v", err)
			}

			if !equalInt32(got.Items, testCase.ExpectedItems) {
				t.Errorf("items: got %v, Python gave %v", got.Items, testCase.ExpectedItems)
			}
			if len(got.Propensity) != len(testCase.ExpectedPropensity) {
				t.Fatalf("propensity length: got %d, Python gave %d",
					len(got.Propensity), len(testCase.ExpectedPropensity))
			}
			for index, expected := range testCase.ExpectedPropensity {
				if math.Abs(got.Propensity[index]-expected) > 1e-12 {
					t.Errorf("propensity[%d]: got %v, Python gave %v",
						index, got.Propensity[index], expected)
				}
			}
		})
	}
}

// TestRowsIndexBackIntoTheInput pins the contract the orchestrator relies on:
// Rows must address the caller's arrays, or a label or feature lookup reads
// another candidate's row.
func TestRowsIndexBackIntoTheInput(t *testing.T) {
	for _, testCase := range loadCases(t) {
		t.Run(testCase.Name, func(t *testing.T) {
			got, err := Select(testCase.Scores, testCase.Items, testCase.K, Options{
				Vectors:    testCase.Vectors,
				Lambda:     testCase.Lambda,
				Categories: testCase.Categories,
				Cap:        testCase.Cap,
				Blocked:    testCase.Blocked,
			})
			if err != nil {
				t.Fatalf("Select: %v", err)
			}
			for slot, row := range got.Rows {
				if testCase.Items[row] != got.Items[slot] {
					t.Fatalf("slot %d: Rows says %d (item %d) but Items says %d",
						slot, row, testCase.Items[row], got.Items[slot])
				}
			}
		})
	}
}

// --- Exploration: the arithmetic is pinned, the draws are not ---------------
//
// No fixture can make Go's generator agree with numpy's, so exploration has no
// cross-language parity test. What a downstream estimator actually consumes is
// the propensity, and that IS checked here -- an off-policy estimate is biased
// by a wrong propensity and indifferent to which item the coin chose.

func TestExplorationLeavesEarlySlotsDeterministic(t *testing.T) {
	got, err := Select([]float64{3, 2, 1, 0.5}, []int32{10, 11, 12, 13}, 3, Options{
		Epsilon:      1.0,
		ExploreSlots: 1,
		Rand:         rand.New(rand.NewSource(1)),
	})
	if err != nil {
		t.Fatalf("Select: %v", err)
	}
	if got.Propensity[0] != Deterministic || got.Propensity[1] != Deterministic {
		t.Errorf("early slots should be deterministic, got %v", got.Propensity)
	}
	if got.Propensity[2] >= Deterministic {
		t.Errorf("the exploration slot should carry a real probability, got %v", got.Propensity[2])
	}
}

// TestAlwaysRandomPropensityUsesWhatRemained is the off-by-2x guard. At slot 2
// two of four candidates are already placed, so the denominator is what is
// LEFT, not the original count. Logging 1/4 instead of 1/2 biases every
// downstream estimate by a factor of two.
func TestAlwaysRandomPropensityUsesWhatRemained(t *testing.T) {
	got, err := Select([]float64{3, 2, 1, 0.5}, []int32{10, 11, 12, 13}, 3, Options{
		Epsilon:      1.0,
		ExploreSlots: 1,
		Rand:         rand.New(rand.NewSource(7)),
	})
	if err != nil {
		t.Fatalf("Select: %v", err)
	}
	if math.Abs(got.Propensity[2]-0.5) > 1e-12 {
		t.Errorf("propensity: got %v, want 0.5 (1 of the 2 candidates left)", got.Propensity[2])
	}
}

// TestGreedyBranchCountsBothPaths: when the coin says "be greedy", that item
// could also have come from the random draw, so the propensity is
// (1-eps) + eps/n and never a bare 1-eps.
func TestGreedyBranchCountsBothPaths(t *testing.T) {
	const epsilon = 0.5
	got, err := Select([]float64{3, 2, 1, 0.5}, []int32{10, 11, 12, 13}, 2, Options{
		Epsilon:      epsilon,
		ExploreSlots: 1,
		Rand:         rand.New(rand.NewSource(3)),
	})
	if err != nil {
		t.Fatalf("Select: %v", err)
	}
	remaining := 3.0
	greedyBranch := (1.0 - epsilon) + epsilon/remaining
	randomBranch := epsilon / remaining
	actual := got.Propensity[1]
	if math.Abs(actual-greedyBranch) > 1e-12 &&
		math.Abs(actual-(randomBranch+1.0-epsilon)) > 1e-12 &&
		math.Abs(actual-randomBranch) > 1e-12 {
		t.Errorf("propensity %v matches neither branch (%v or %v)",
			actual, greedyBranch, randomBranch)
	}
	if actual >= Deterministic {
		t.Errorf("an explored slot cannot be certain, got %v", actual)
	}
}

func TestExplorationWithoutARandSourceIsRefused(t *testing.T) {
	_, err := Select([]float64{1, 0.5}, []int32{10, 11}, 1, Options{
		Epsilon:      0.1,
		ExploreSlots: 1,
	})
	if err == nil {
		t.Fatal("expected an error: a run that explores must be reproducible")
	}
}

func TestMismatchedLengthsAreRefused(t *testing.T) {
	if _, err := Select([]float64{1, 2}, []int32{10}, 1, Options{}); err == nil {
		t.Fatal("expected an error for 1 item over 2 scores")
	}
}

func equalInt32(left, right []int32) bool {
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
