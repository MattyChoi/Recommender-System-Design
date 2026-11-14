package experiments

import (
	"fmt"
	"math"
	"testing"
)

func fiftyFifty() []Variant {
	return []Variant{{Name: "control", Percent: 50}, {Name: "treatment", Percent: 50}}
}

func users(n int) []string {
	out := make([]string, n)
	for index := range out {
		out[index] = fmt.Sprintf("U%d", index)
	}
	return out
}

func counts(experimentID string, variants []Variant, userIDs []string) map[string]int {
	got := map[string]int{}
	for _, userID := range userIDs {
		got[Assign(experimentID, userID, variants)]++
	}
	return got
}

// --- The three properties ----------------------------------------------------

func TestAssignmentIsDeterministic(t *testing.T) {
	// Sticky follows from this: a user who flipped variants mid-experiment
	// belongs to neither arm, and every per-user metric becomes meaningless.
	first := Assign("ranker_v2", "U42", fiftyFifty())
	for attempt := 0; attempt < 100; attempt++ {
		if got := Assign("ranker_v2", "U42", fiftyFifty()); got != first {
			t.Fatalf("attempt %d gave %q, first gave %q", attempt, got, first)
		}
	}
}

// TestTheSaltDecorrelatesExperiments is the reason the experiment id is in the
// hash at all.
//
// Without it, bucketing is a property of the USER alone: the same people land
// in the same bucket in every experiment forever, two concurrent tests get
// correlated arms, and an effect from one leaks into the other's estimate.
// Invisible in either experiment's own numbers.
func TestTheSaltDecorrelatesExperiments(t *testing.T) {
	sample := users(2000)

	agree := 0
	for _, userID := range sample {
		if Assign("ranker_v2", userID, fiftyFifty()) ==
			Assign("rerank_lambda", userID, fiftyFifty()) {
			agree++
		}
	}

	// Two independent 50/50 splits agree on about half the users. Anything
	// near 100% means the salt is not reaching the hash.
	rate := float64(agree) / float64(len(sample))
	if math.Abs(rate-0.5) > 0.05 {
		t.Errorf("two experiments agreed on %.1f%% of users; want ~50%%", rate*100)
	}
}

func TestTheSplitIsRoughlyTheRequestedRatio(t *testing.T) {
	// An SRM check on our own assignment code. §17.2: if the ratio is wrong
	// the experiment is invalid regardless of what the primary metric did, so
	// the assignment must not be the thing producing it.
	got := counts("ranker_v2", fiftyFifty(), users(10000))

	for _, name := range []string{"control", "treatment"} {
		share := float64(got[name]) / 10000
		if math.Abs(share-0.5) > 0.02 {
			t.Errorf("%s got %.1f%% of traffic, want ~50%%", name, share*100)
		}
	}
}

func TestAnUnevenSplitIsHonoured(t *testing.T) {
	variants := []Variant{{Name: "control", Percent: 90}, {Name: "treatment", Percent: 10}}

	got := counts("ranker_v2", variants, users(10000))

	share := float64(got["treatment"]) / 10000
	if math.Abs(share-0.10) > 0.015 {
		t.Errorf("treatment got %.1f%%, want ~10%%", share*100)
	}
}

// --- The holdback ------------------------------------------------------------

// TestAPartialAllocationLeavesAHoldback is where this deviates from the
// guide's sketch, deliberately.
//
// That version falls through to `variants[0]`, which is a fine guard against
// float rounding and a serious bug when the allocation deliberately sums to
// less than 100: the entire remainder silently joins the first arm. That is a
// sample-ratio mismatch manufactured by the assignment code, and an SRM
// invalidates the experiment whatever the primary metric says.
func TestAPartialAllocationLeavesAHoldback(t *testing.T) {
	variants := []Variant{{Name: "control", Percent: 10}, {Name: "treatment", Percent: 10}}

	got := counts("ranker_v2", variants, users(10000))

	holdback := float64(got[Unassigned]) / 10000
	if math.Abs(holdback-0.80) > 0.02 {
		t.Errorf("holdback is %.1f%%, want ~80%%", holdback*100)
	}
	// And the first arm must NOT have absorbed it.
	if share := float64(got["control"]) / 10000; math.Abs(share-0.10) > 0.015 {
		t.Errorf("control got %.1f%%; the remainder leaked into the first arm", share*100)
	}
}

func TestAFullAllocationLeavesNoHoldback(t *testing.T) {
	// The control for the test above: with 100% allocated, nobody is
	// unassigned, so the holdback is a property of the allocation rather than
	// an off-by-one at the last boundary.
	if got := counts("ranker_v2", fiftyFifty(), users(5000)); got[Unassigned] != 0 {
		t.Errorf("%d users unassigned from a fully allocated experiment", got[Unassigned])
	}
}

func TestNoVariantsIsUnassignedNotAPanic(t *testing.T) {
	if got := Assign("ranker_v2", "U1", nil); got != Unassigned {
		t.Errorf("got %q", got)
	}
}

// --- Configuration guards ----------------------------------------------------

// TestOverAllocationTruncatesThenStarves documents what actually happens when
// the percentages sum past 100, which is worse than either "it errors" or "it
// is fine" and is the reason Allocated() is worth checking at startup.
//
// Arms are walked in order against a fixed 10,000 buckets, so:
//
//   - the arm that STRADDLES 100% is silently truncated -- it asked for 60%
//     and gets whatever is left, here 40%;
//   - every arm after it is unreachable and gets nothing.
//
// Both are invisible at runtime. The dashboard shows one arm smaller than
// configured and one with no users at all, and neither the service nor the
// analysis says why.
func TestOverAllocationTruncatesThenStarves(t *testing.T) {
	over := Experiment{ID: "x", Variants: []Variant{
		{Name: "a", Percent: 60}, {Name: "b", Percent: 60}, {Name: "c", Percent: 60},
	}}

	if over.Allocated() != 180 {
		t.Errorf("Allocated() = %v, want 180", over.Allocated())
	}

	got := counts("x", over.Variants, users(10000))

	if share := float64(got["a"]) / 10000; math.Abs(share-0.60) > 0.02 {
		t.Errorf("arm a got %.1f%%, want its full 60%%", share*100)
	}
	// Truncated, not honoured: it asked for 60% and only 40% was left.
	if share := float64(got["b"]) / 10000; math.Abs(share-0.40) > 0.02 {
		t.Errorf("arm b got %.1f%%, want the ~40%% remaining", share*100)
	}
	if got["c"] != 0 {
		t.Errorf("arm c got %d users; everything past 100%% is unreachable", got["c"])
	}
	// And nobody is unassigned: the allocation over-covers the space.
	if got[Unassigned] != 0 {
		t.Errorf("%d unassigned under an over-allocation", got[Unassigned])
	}
}

func TestNamesIncludeTheHoldbackOnlyWhenThereIsOne(t *testing.T) {
	partial := Experiment{ID: "x", Variants: []Variant{{Name: "a", Percent: 25}}}
	full := Experiment{ID: "x", Variants: fiftyFifty()}

	if len(partial.Names()) != 2 {
		t.Errorf("a partial allocation should name the holdback: %v", partial.Names())
	}
	if len(full.Names()) != 2 || full.Names()[0] != "control" {
		t.Errorf("a full allocation should name only its arms: %v", full.Names())
	}
}

// TestTheSeparatorKeepsConcatenationsDistinct guards a collision that would
// look like a coincidence.
//
// Without a separator, ("ab", "c") and ("a", "bc") hash identically -- two
// different (experiment, user) pairs sharing a bucket. Rare, silent, and the
// kind of thing that makes one experiment's assignment mysteriously track
// another's.
func TestTheSeparatorKeepsConcatenationsDistinct(t *testing.T) {
	// Compared over many users so a chance agreement on one pair cannot pass.
	agree := 0
	sample := users(500)
	for _, userID := range sample {
		if Assign("ab", userID, fiftyFifty()) == Assign("a", "b"+userID, fiftyFifty()) {
			agree++
		}
	}
	if agree == len(sample) {
		t.Error("('ab', u) and ('a', 'b'+u) assign identically; the salt has no separator")
	}
}
