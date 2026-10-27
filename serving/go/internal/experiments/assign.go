// Package experiments assigns a user to a variant, deterministically.
package experiments

import (
	"hash/fnv"
	"sort"
)

// Buckets the hash lands in
const Buckets = 10000

// Unassigned is returned for a user outside every variant's allocation.
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

// Assign each user a variant
func Assign(experimentID, userID string, variants []Variant) string {
	if len(variants) == 0 {
		return Unassigned
	}

	digest := fnv.New64a()
	_, _ = digest.Write([]byte(experimentID))
	_, _ = digest.Write([]byte{':'})
	_, _ = digest.Write([]byte(userID))
	bucket := digest.Sum64() % Buckets

	var cumulative float64
	for _, variant := range variants {
		cumulative += variant.Percent

		if bucket < uint64(cumulative*Buckets/100) {
			return variant.Name
		}
	}

	return Unassigned
}

func (e Experiment) Assign(userID string) string {
	return Assign(e.ID, userID, e.Variants)
}

// Allocated is the total percent across variants.
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
