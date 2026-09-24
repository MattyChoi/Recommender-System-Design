// Package experiments assigns a user to a variant, deterministically.
//
// Three properties, and all three are requirements rather than conveniences:
//
//   - **Deterministic.** The same user gets the same variant on every request,
//     with no lookup and no state. An assignment service that had to be
//     consulted would be another hop and another dependency on the hot path.
//   - **Sticky.** It follows from determinism, and it is what makes a metric
//     per user meaningful: a user who flipped variants mid-experiment belongs
//     to neither arm.
//   - **Salted by experiment.** See Salt below. This is the three-character
//     detail with the large consequence.
package experiments

import (
	"hash/fnv"
	"sort"
)

// Buckets the hash lands in. Ten thousand, so an allocation can be expressed
// to a hundredth of a percent -- and so the granularity is finer than any
// traffic split anyone will write.
const Buckets = 10000

// Unassigned is returned for a user outside every variant's allocation.
//
// **Not the first variant.** The guide's sketch falls through to
// `variants[0]`, which is a reasonable guard against float rounding and a
// serious bug if the allocations deliberately sum to less than 100%: the whole
// unallocated remainder silently joins the first arm. That is a sample-ratio
// mismatch manufactured by the assignment code itself, and §17.2 says an SRM
// invalidates the experiment regardless of what the primary metric did. An
// explicit empty variant is a holdback that the analysis can see and exclude.
const Unassigned = ""

// Variant is one arm and its share of traffic, in percent.
type Variant struct {
	Name    string
	Percent float64
}

// Experiment is one test's arms.
type Experiment struct {
	// ID is the SALT. Two experiments with the same variants and the same
	// users must not produce the same split.
	ID       string
	Variants []Variant
}

// Assign returns the user's variant, or Unassigned.
//
// # Why the experiment id is in the hash
//
// Without it, bucketing is a property of the USER alone: the same people land
// in the same bucket in every experiment forever. Two tests running at once
// then have correlated arms -- the users who got the aggressive ranker also
// got the aggressive re-ranker -- and an effect from one leaks into the other's
// estimate. It is invisible in any single experiment's numbers and shows up as
// results that do not replicate, which is a very hard thing to chase after the
// fact.
//
// FNV-1a rather than a cryptographic hash: this needs uniformity and speed,
// not preimage resistance, and it runs on every request.
func Assign(experimentID, userID string, variants []Variant) string {
	if len(variants) == 0 {
		return Unassigned
	}

	digest := fnv.New64a()
	// Written as one salted string. A separator so that ("ab", "c") and
	// ("a", "bc") are different experiments rather than the same bucket.
	_, _ = digest.Write([]byte(experimentID))
	_, _ = digest.Write([]byte{':'})
	_, _ = digest.Write([]byte(userID))
	bucket := digest.Sum64() % Buckets

	var cumulative float64
	for _, variant := range variants {
		cumulative += variant.Percent
		// Compared in BUCKETS, not in percent, and rounded once at the
		// boundary. Accumulating percentages and converting each time lets
		// float error move a boundary by a bucket, which reassigns a handful
		// of users on every deploy and shows up as a small, permanent,
		// unexplained SRM.
		if bucket < uint64(cumulative*Buckets/100) {
			return variant.Name
		}
	}
	return Unassigned
}

// Assign is the method form, for a configured experiment.
func (e Experiment) Assign(userID string) string {
	return Assign(e.ID, userID, e.Variants)
}

// Allocated is the total percent across variants.
//
// Worth checking at startup rather than trusting, because an allocation
// summing past 100 fails in two ways at once and neither is visible at
// runtime: the arm that STRADDLES the boundary is silently truncated to
// whatever is left, and every arm after it is unreachable. The dashboard shows
// one arm smaller than configured and one with no users, and nothing says why.
func (e Experiment) Allocated() float64 {
	var total float64
	for _, variant := range e.Variants {
		total += variant.Percent
	}
	return total
}

// Names returns the variant names in allocation order, plus Unassigned when
// the allocation leaves a holdback. What an analysis should expect to see.
func (e Experiment) Names() []string {
	names := make([]string, 0, len(e.Variants)+1)
	for _, variant := range e.Variants {
		names = append(names, variant.Name)
	}
	if e.Allocated() < 100 {
		names = append(names, Unassigned)
	}
	sort.Strings(names)
	return names
}
