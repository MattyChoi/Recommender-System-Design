// Package retrieval merges the candidate lists the sources return.
package retrieval

import "fmt"

// Rank assigned to a candidate a source did not propose. Equal to the
// constant used in models/ranking/dataset.py
const Absent int32 = 10000

type Source struct {
	Name string
	Top  []int32
}

// SourceRanks is one source's position for each blended candidate. Aligned
// with Blended.Items
type SourceRanks struct {
	Name  string
	Ranks []int32
}

// Blended is the candidate set handed to filtering and ranking.
type Blended struct {
	Items []int32
	Ranks []SourceRanks
}

// Blend fills maxCandidates slots by explicit per-source quota, keeping every
// source's rank for every chosen candidate.
func Blend(sources []Source, maxCandidates int, quotas []int) (Blended, error) {
	if len(sources) != len(quotas) {
		return Blended{}, fmt.Errorf("%d quotas for %d sources", len(quotas), len(sources))
	}
	if maxCandidates < 0 {
		return Blended{}, fmt.Errorf("maxCandidates must not be negative, got %d", maxCandidates)
	}

	chosen := make([]int32, 0, maxCandidates)
	seen := make(map[int32]struct{}, maxCandidates)

	// take reports whether the item was NEWLY added. A duplicate is not an error
	// and does not consume quota -- it simply is not taken, which is what lets a
	// later source still tag it with a rank.
	take := func(item int32) bool {
		if item <= 0 {
			return false
		}
		if _, ok := seen[item]; ok {
			return false
		}
		seen[item] = struct{}{}
		chosen = append(chosen, item)
		return true
	}

	for index, source := range sources {
		taken := 0
		for _, item := range source.Top {
			if taken >= quotas[index] || len(chosen) >= maxCandidates {
				break
			}
			if take(item) {
				taken++
			}
		}
	}

	// Top up in SOURCE ORDER, so an under-filled quota is spent rather than lost.
	// A source that returned fewer items than its allocation does not shrink the
	// slate.
	for _, source := range sources {
		for _, item := range source.Top {
			if len(chosen) >= maxCandidates {
				break
			}
			take(item)
		}
	}

	ranks := make([]SourceRanks, 0, len(sources))
	for _, source := range sources {
		// Built by FORWARD iteration so a repeated item keeps its LAST position,
		// matching the Python dict comprehension this mirrors.
		place := make(map[int32]int32, len(source.Top))
		for position, item := range source.Top {
			if item > 0 {
				place[item] = int32(position)
			}
		}
		positions := make([]int32, len(chosen))
		for index, item := range chosen {
			if found, ok := place[item]; ok {
				positions[index] = found
			} else {
				positions[index] = Absent
			}
		}
		ranks = append(ranks, SourceRanks{Name: source.Name, Ranks: positions})
	}
	return Blended{Items: chosen, Ranks: ranks}, nil
}
