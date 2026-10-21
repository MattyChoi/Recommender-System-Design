package rerank

import (
	"errors"
	"fmt"
	"math/rand"
)

// Deterministic is the propensity of a slot the greedy rule filled.
const Deterministic = 1.0

type Options struct {
	// Vectors are L2-NORMALISED content vectors, one per candidate, or nil to
	// disable MMR.
	Vectors [][]float32

	// Lambda weights relevance against similarity. 1.0 is pure relevance and
	// makes MMR a no-op; 0.0 ignores relevance entirely.
	Lambda float64

	Categories []int32
	Cap        *int
	Blocked    []bool
	Epsilon    float64

	// ExploreSlots is how many of the LAST slots explore.
	ExploreSlots int

	// Rand is required whenever Epsilon is positive, so a run that explores can
	// be reproduced.
	Rand *rand.Rand
}

type Slate struct {
	// Items are catalogue indices in served order
	Items []int32

	// Rows is each served item's index into the candidate block
	Rows []int

	// Propensity is P(this item in this slot | policy, history)
	Propensity []float64

	// Objective is the value the greedy rule saw for the item it took
	Objective []float64
}

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
		return Slate{}, errors.New("exploration needs a Rand, or the run cannot be reproduced")
	}
	if opts.Epsilon > 0 && opts.ExploreSlots <= 0 {
		return Slate{}, errors.New("Epsilon > 0 with no ExploreSlots explores nothing")
	}

	budget := k
	if n < budget {
		budget = n
	}
	slate := Slate{}

	available := make([]bool, n)
	for index := range available {
		available[index] = true
	}

	if opts.Blocked != nil {
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
	used := make(map[int32]int)
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
			// back to the uncapped set rather than returning a SHORT slate.
			if !any {
				copy(feasible, available)
			}
		}

		// Collected in index order
		candidates := make([]int, 0, n)
		for index, ok := range feasible {
			if ok {
				candidates = append(candidates, index)
			}
		}
		// Nothing left to serve. BREAK rather than continue: every later slot
		// would find the same empty set, and the slate is simply shorter than k.
		if len(candidates) == 0 {
			break
		}

		greedy := candidates[0]
		best := objectiveAt(scores, peak, opts, candidates[0], len(slate.Items))
		for _, index := range candidates[1:] {
			value := objectiveAt(scores, peak, opts, index, len(slate.Items))
			// STRICTLY greater, so ties keep the EARLIER candidate
			if value > best {
				best, greedy = value, index
			}
		}

		pick := greedy
		propensity := float64(Deterministic)
		if slot >= budget-opts.ExploreSlots && opts.Epsilon > 0 {
			if opts.Rand.Float64() < opts.Epsilon {
				pick = candidates[opts.Rand.Intn(len(candidates))]
				propensity = opts.Epsilon / float64(len(candidates))
				if pick == greedy {
					propensity += 1.0 - opts.Epsilon
				}
			} else {
				propensity = (1.0 - opts.Epsilon) + opts.Epsilon/float64(len(candidates))
			}
		}

		atPick := objectiveAt(scores, peak, opts, pick, len(slate.Items))

		slate.Items = append(slate.Items, items[pick])
		slate.Rows = append(slate.Rows, pick)
		slate.Propensity = append(slate.Propensity, propensity)
		slate.Objective = append(slate.Objective, atPick)

		// Off the available set, and onto the category's tally. Both happen
		// whether the pick was greedy or explored: an explored item occupies a
		// slot and fills a category quota exactly like any other.
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

// objectiveAt is the value the greedy rule maximises for one candidate.
func objectiveAt(scores, peak []float64, opts Options, index, chosen int) float64 {
	if opts.Vectors == nil || chosen == 0 {
		return scores[index]
	}
	return opts.Lambda*scores[index] - (1.0-opts.Lambda)*peak[index]
}

func dot(left, right []float32) float64 {
	var total float32
	for index := range left {
		total += left[index] * right[index]
	}
	return float64(total)
}
