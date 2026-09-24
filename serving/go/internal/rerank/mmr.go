// Package rerank selects the slate: one greedy pass in which MMR, category
// caps, a block list and exploration all act together.
//
// This mirrors models/reranking/policies.py:select, which produced Part L's
// measured trade-off table. mmr_test.go asserts the two agree on fixtures
// generated from the Python side.
//
// **They are composed in ONE pass, not chained.** Capping after MMR discards
// MMR's pick and promotes a candidate MMR never compared; capping inside the
// loop makes MMR choose the best FEASIBLE candidate, which is what a constraint
// means. Reproducing the composition is therefore part of reproducing the
// behaviour -- a port that ran three filters in sequence would return
// well-formed, differently-ordered slates.
//
// # What is parity-tested and what cannot be
//
// The deterministic path is tested against Python case by case. **Exploration
// is not, and cannot be**: it draws from a pseudo-random source, and no
// fixture can make Go's generator agree with numpy's. What IS pinned on both
// sides is the propensity arithmetic, because that is the part a downstream
// estimator consumes -- an off-policy estimate is biased by a wrong propensity
// and indifferent to which item the coin happened to choose.
package rerank

import (
	"fmt"
	"math/rand"
)

// Deterministic is the propensity of a slot the greedy rule filled. Named
// rather than written as 1.0 at the call site, where it would look like a
// probability someone computed.
const Deterministic = 1.0

// Slate is one request's final ordering with the propensities that produced it.
type Slate struct {
	// Items in served order.
	Items []int32
	// Rows indexes back into the caller's candidate arrays, so labels or
	// features can be recovered without a second lookup.
	Rows []int
	// Propensity is P(this item in this slot | policy). **Logged at selection
	// time or lost**: once the request is over, the candidate set and the draw
	// are both gone, and every off-policy estimate built later is biased.
	Propensity []float64
	// Objective is the value each slot was actually CHOSEN by -- the
	// MMR-adjusted score when MMR is on, the raw score when it is off.
	//
	// It is not scores[Rows[i]] whenever a policy fires, and the difference is
	// the point: reporting the ranker's raw score beside a re-ranked order is
	// how a debugging session starts from a false premise ("the top item scored
	// 0.9, so why is it in slot 4?"). For an exploration slot this is the
	// objective of the item the DRAW landed on, not the greedy maximum that was
	// passed over -- the slot was filled by that item, at that value.
	Objective []float64
}

// Options switches the individual policies on. A nil slice or pointer disables
// its policy, mirroring Python's None.
type Options struct {
	// Vectors are L2-NORMALISED content vectors, one per candidate. Nil
	// disables MMR. Normalisation is the caller's job: doing it per request
	// would renormalise the same vectors on every call.
	Vectors [][]float32
	// Lambda weights relevance against similarity. 1.0 makes MMR a no-op.
	Lambda float64
	// Categories are one per candidate. Nil disables caps.
	Categories []int32
	// Cap is the most slots one category may take. Nil disables caps.
	Cap *int
	// Blocked marks candidates that must not be served -- the seen-list's
	// answer, or any other business rule.
	Blocked []bool
	// Epsilon is the chance an exploration slot is filled uniformly at random.
	Epsilon float64
	// ExploreSlots is how many of the LAST slots explore. Last rather than
	// first: an explored item in slot 0 costs the most relevance.
	ExploreSlots int
	// Rand is required when Epsilon > 0, so a run that explores is reproducible.
	Rand *rand.Rand
}

// Select greedily fills k slots from one request's candidates.
func Select(scores []float64, items []int32, k int, opts Options) (Slate, error) {
	n := len(scores)
	if len(items) != n {
		return Slate{}, fmt.Errorf("%d items for %d scores", len(items), n)
	}
	if opts.Vectors != nil && len(opts.Vectors) != n {
		return Slate{}, fmt.Errorf("%d vectors for %d scores", len(opts.Vectors), n)
	}
	if opts.Categories != nil && len(opts.Categories) != n {
		return Slate{}, fmt.Errorf("%d categories for %d scores", len(opts.Categories), n)
	}
	if opts.Blocked != nil && len(opts.Blocked) != n {
		return Slate{}, fmt.Errorf("%d block flags for %d scores", len(opts.Blocked), n)
	}
	if opts.Epsilon > 0 && opts.Rand == nil {
		return Slate{}, fmt.Errorf("exploration needs a rand source, or the run cannot be reproduced")
	}
	if opts.Epsilon > 0 && opts.ExploreSlots <= 0 {
		return Slate{}, fmt.Errorf("epsilon > 0 with no exploration slots explores nothing")
	}

	available := make([]bool, n)
	for index := range available {
		available[index] = true
	}
	if opts.Blocked != nil {
		// A mask that blocks EVERYTHING returns an empty slate, which is a
		// blank page rather than a degraded one -- and a saturated Bloom
		// filter does exactly that. The business rule yields; the page does not.
		allBlocked := true
		for _, blocked := range opts.Blocked {
			if !blocked {
				allBlocked = false
				break
			}
		}
		if !allBlocked {
			for index, blocked := range opts.Blocked {
				if blocked {
					available[index] = false
				}
			}
		}
	}

	budget := k
	if n < budget {
		budget = n
	}

	slate := Slate{
		Items:      make([]int32, 0, budget),
		Rows:       make([]int, 0, budget),
		Propensity: make([]float64, 0, budget),
		Objective:  make([]float64, 0, budget),
	}
	used := make(map[int32]int)
	// peak is the running MAXIMUM similarity to the chosen set, so MMR stays
	// O(n*k) instead of recomputing every pair at every step. A running maximum
	// and not the similarity to the last pick: with the latter, a candidate
	// identical to slot 0 becomes eligible again as soon as slot 1 is unlike it.
	peak := make([]float64, n)

	for slot := 0; slot < budget; slot++ {
		feasible := make([]bool, n)
		copy(feasible, available)

		if opts.Categories != nil && opts.Cap != nil {
			any := false
			for index := range feasible {
				if feasible[index] && used[opts.Categories[index]] >= *opts.Cap {
					feasible[index] = false
				}
				if feasible[index] {
					any = true
				}
			}
			// A cap that leaves nothing is a cap the slate cannot honour. Fall
			// back to the uncapped set rather than returning a short slate.
			if !any {
				copy(feasible, available)
			}
		}

		candidates := make([]int, 0, n)
		for index, ok := range feasible {
			if ok {
				candidates = append(candidates, index)
			}
		}
		if len(candidates) == 0 {
			break
		}

		greedy := candidates[0]
		best := objectiveAt(scores, peak, opts, candidates[0], len(slate.Items))
		for _, index := range candidates[1:] {
			value := objectiveAt(scores, peak, opts, index, len(slate.Items))
			// Strictly greater, so ties keep the EARLIER candidate -- numpy's
			// argmax does the same, and a port using >= would silently prefer
			// the later one on every tie.
			if value > best {
				best, greedy = value, index
			}
		}

		pick := greedy
		propensity := float64(Deterministic)
		if slot >= budget-opts.ExploreSlots && opts.Epsilon > 0 {
			if opts.Rand.Float64() < opts.Epsilon {
				pick = candidates[opts.Rand.Intn(len(candidates))]
				// Two disjoint paths can land on the same item: the random draw,
				// and the draw happening to choose what greedy would have.
				// Counting only one understates the propensity of exactly the
				// item most likely to be logged.
				propensity = opts.Epsilon / float64(len(candidates))
				if pick == greedy {
					propensity += 1.0 - opts.Epsilon
				}
			} else {
				propensity = (1.0 - opts.Epsilon) + opts.Epsilon/float64(len(candidates))
			}
		}

		// Read BEFORE the appends: objectiveAt takes the size of the chosen set,
		// and MMR's penalty switches on at the first pick. Computing it after
		// would evaluate slot 0 under the rule for slot 1.
		atPick := objectiveAt(scores, peak, opts, pick, len(slate.Items))

		slate.Items = append(slate.Items, items[pick])
		slate.Rows = append(slate.Rows, pick)
		slate.Propensity = append(slate.Propensity, propensity)
		slate.Objective = append(slate.Objective, atPick)
		available[pick] = false
		if opts.Categories != nil {
			used[opts.Categories[pick]]++
		}
		if opts.Vectors != nil {
			for index := range peak {
				if similarity := dot(opts.Vectors[index], opts.Vectors[pick]); similarity > peak[index] {
					peak[index] = similarity
				}
			}
		}
	}

	return slate, nil
}

// objectiveAt is the value MMR maximises, or the raw score when MMR is off or
// nothing has been chosen yet -- matching Python, which applies the penalty
// only once the chosen set is non-empty.
func objectiveAt(scores, peak []float64, opts Options, index, chosen int) float64 {
	if opts.Vectors == nil || chosen == 0 {
		return scores[index]
	}
	return opts.Lambda*scores[index] - (1.0-opts.Lambda)*peak[index]
}

// dot accumulates in float32 and widens once, because the Python side holds
// these vectors as float32 and numpy's matmul accumulates in the input dtype.
// Summing in float64 here would give a slightly different similarity, which is
// invisible until two candidates sit close enough for it to flip the argmax --
// at which point the two languages return different slates and the test that
// catches it looks like a logic bug.
func dot(left, right []float32) float64 {
	var total float32
	for index := range left {
		total += left[index] * right[index]
	}
	return float64(total)
}
