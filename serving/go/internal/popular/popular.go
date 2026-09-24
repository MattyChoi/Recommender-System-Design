// Package popular is the slate served when the pipeline cannot produce one.
//
// **It has no runtime dependencies, and that is the whole design.** §14.2
// requires the fallback to need no user features, no index and no model; this
// adds no Redis and no network. A fallback that can fail is a fallback on the
// one path whose entire job is to always return something, and the failure
// would arrive exactly when other things are already failing.
//
// The cost is staleness: the list freezes at build time. ADR 0011 measured the
// decayed-popularity optimum on this corpus at a half-life of roughly 29
// MINUTES, which a static artifact cannot track. That price is only right
// because this path is rare -- if the fallback rate stops being rare, the
// staleness stops being acceptable and the trade needs revisiting rather than
// the artifact regenerating.
package popular

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/service"
)

// List is a ranked slate held in memory.
type List struct {
	items   []int32
	version string
}

type onDisk struct {
	Version string  `json:"version"`
	Items   []int32 `json:"items"`
}

// Load reads what scripts/dump_popular.py writes.
func Load(path string) (*List, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading popularity fallback %s: %w", path, err)
	}
	var decoded onDisk
	if err := json.Unmarshal(raw, &decoded); err != nil {
		return nil, fmt.Errorf("parsing %s: %w", path, err)
	}
	if len(decoded.Items) == 0 {
		// An empty list loads fine and returns an empty slate on the one path
		// that exists to never do that. Refused at startup, where it is a line
		// in a log rather than a blank page.
		return nil, fmt.Errorf("popularity fallback %s is empty", path)
	}
	for position, item := range decoded.Items {
		if item <= 0 {
			// Index 0 is the reserved OOV row. Served, it puts a placeholder
			// in front of a user at the moment the system is already degraded.
			return nil, fmt.Errorf(
				"popularity fallback %s holds a reserved index at position %d", path, position,
			)
		}
	}
	return &List{items: decoded.Items, version: decoded.Version}, nil
}

// Popular returns the top n, or everything if the list is shorter.
//
// Never an error. The signature carries one because the port does, and the
// port does because a future implementation might fetch -- but this one
// reads a slice it already holds, so the only way it could fail is a
// programming error, and returning short beats returning nothing.
func (l *List) Popular(_ context.Context, n int) ([]int32, error) {
	if n <= 0 {
		return nil, nil
	}
	if n > len(l.items) {
		n = len(l.items)
	}
	// A copy, not a sub-slice of the shared backing array. The caller writes
	// this into a Result that the re-ranker and the transport both touch, and
	// a slice aliasing the loaded artifact would let one request's mutation
	// corrupt the fallback for every request after it.
	out := make([]int32, n)
	copy(out, l.items[:n])
	return out, nil
}

// Version identifies the snapshot, so a stale fallback is attributable.
func (l *List) Version() string { return l.version }

// Size is how many items are held.
func (l *List) Size() int { return len(l.items) }

// Compile-time proof this satisfies the port the pipeline falls back through.
var _ service.Fallback = (*List)(nil)
