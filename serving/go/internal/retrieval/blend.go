// Package retrieval merges the candidate lists the sources return.
//
// This is the serving half of a function that also exists in Python, as
// models/ranking/dataset.py:blend_one. The offline half produced every
// candidate set this project has measured; this one runs in the request path.
// They are two programs and they must agree, which is what
// blend_test.go asserts against fixtures generated from the Python side.
//
// Two differences between the languages are load-bearing here:
//
//   - **Python dicts preserve insertion order and Go maps do not.** The blend's
//     output order is the order candidates were taken -- quota pass first, then
//     top-up -- and the ranker downstream reads the list positionally. A port
//     that accumulated into a map and ranged over it would emit a different
//     order on every run of the same binary, which is a nondeterministic
//     serving path that no single test run would catch.
//   - **Source order is part of the input**, because quotas[i] belongs to
//     sources[i]. It is carried as a slice for that reason; an object keyed by
//     source name cannot express it.
package retrieval

import "fmt"

// Absent is the rank given to a candidate a source did not propose. One past
// any real cutoff, so "absent" sorts worse than every rank a source could give.
// Must equal ABSENT in models/ranking/dataset.py: the ranker was trained on
// rows carrying this exact value, and a serving path that sent a different
// sentinel would feed the model a number it has never seen.
const Absent int32 = 10000

// Source is one retriever's ranked output for one request, best first.
type Source struct {
	Name string
	Top  []int32
}

// SourceRanks is one source's position for each blended candidate, aligned
// with Blended.Items. A slice of these rather than a map, so the order matches
// the input and can be serialised without losing it.
type SourceRanks struct {
	Name  string
	Ranks []int32
}

// Blended is the candidate set handed to filtering and ranking.
type Blended struct {
	// Items in serving order: the quota pass, then the top-up.
	Items []int32
	// Ranks carries one entry per source, in input order.
	Ranks []SourceRanks
}

// Blend fills maxCandidates slots by explicit per-source quota, keeping every
// source's rank for every chosen candidate.
//
// A source contributes candidates, features, or both. Part I measured that no
// equal-slot blend beats handing the whole budget to the strongest retriever on
// this corpus, so the usual production configuration gives one source every
// slot -- and the others still tag each candidate with their own rank, which
// costs no slot and is a feature the ranker uses.
//
// Item ids at or below zero are skipped everywhere: zero is the reserved
// out-of-vocabulary row that short source lists pad with, and serving it would
// put a placeholder in front of a user.
func Blend(sources []Source, maxCandidates int, quotas []int) (Blended, error) {
	if len(quotas) != len(sources) {
		return Blended{}, fmt.Errorf("%d quotas for %d sources", len(quotas), len(sources))
	}
	if maxCandidates < 0 {
		return Blended{}, fmt.Errorf("maxCandidates must not be negative, got %d", maxCandidates)
	}

	chosen := make([]int32, 0, maxCandidates)
	seen := make(map[int32]struct{}, maxCandidates)

	// take reports whether the item was newly added. A duplicate is not an
	// error and does not consume quota -- it simply is not taken, which is what
	// lets a later source still tag it with a rank.
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

	// Top up in source order, so an under-filled quota is spent rather than
	// lost. A source that returned fewer items than its allocation does not
	// shrink the slate.
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
		// Built by forward iteration so a repeated item keeps its LAST
		// position, matching the Python dict comprehension this mirrors.
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
