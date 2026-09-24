// Package server is the gRPC boundary: proto in, proto out, and the pipeline
// in between. It holds no policy. Everything here is translation, validation
// and the honest reporting of what the pipeline said it did.
package server

import (
	"encoding/json"
	"fmt"
	"os"
)

// IDs translates between the two vocabularies this system has.
//
// The API speaks the CATALOGUE's ids -- "N1234", the string MIND ships. The
// models speak a dense integer index, because an embedding table is indexed by
// position. `data_pipeline/transform/id_maps.py` builds the mapping and writes
// it to bronze as `item_map`; everything downstream, including every offline
// number this project has measured, is in terms of `item_idx`.
//
// The translation lives at this boundary and nowhere else. A pipeline that
// carried both vocabularies would invite the mistake where an external id is
// compared against an internal one -- both are "the item id", both type-check
// in the places it matters, and the result is a filter that silently matches
// nothing.
type IDs interface {
	// Index maps an external id to the internal one. False for an id the map
	// does not know.
	Index(external string) (int32, bool)
	// External maps back. False for an index outside the map, which means the
	// map and the index were built from different snapshots.
	External(index int32) (string, bool)
	// Version identifies the snapshot, for the response's index_version.
	Version() string
}

// Table is an in-memory IDs. A few hundred thousand entries is a handful of
// megabytes and it changes only when the index is rebuilt, so a lookup service
// would be a network hop per request spent on static data.
type Table struct {
	forward map[string]int32
	reverse map[int32]string
	version string
}

// itemMap is the on-disk shape: the id map, dumped for serving.
//
// A JSON OBJECT rather than a positional array. The array would be smaller and
// is the obvious encoding, but it silently asserts that the indices are dense
// and zero-based, and index 0 is the reserved out-of-vocabulary row. An
// explicit mapping cannot express that assumption, so it cannot get it wrong.
type itemMap struct {
	// Version is the snapshot the map was dumped from.
	Version string `json:"version"`
	// Items maps external id to internal index.
	Items map[string]int32 `json:"items"`
}

// LoadIDs reads the map that scripts/dump_item_map.py writes.
func LoadIDs(path string) (*Table, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading item map %s: %w", path, err)
	}
	var decoded itemMap
	if err := json.Unmarshal(raw, &decoded); err != nil {
		return nil, fmt.Errorf("parsing item map %s: %w", path, err)
	}
	if len(decoded.Items) == 0 {
		// An empty map parses fine and produces a server that answers every
		// request with "unknown item". Refused at load, where it is one line in
		// a startup log rather than a 100% error rate.
		return nil, fmt.Errorf("item map %s is empty", path)
	}
	return NewTable(decoded.Items, decoded.Version)
}

// NewTable builds the reverse direction and refuses a map that cannot have one.
func NewTable(forward map[string]int32, version string) (*Table, error) {
	reverse := make(map[int32]string, len(forward))
	for external, index := range forward {
		// Two external ids on one index means the reverse direction has no
		// answer: whichever wins, some slate will be served under another
		// article's id. The Spark job's contract tests already assert item_id
		// is unique, so this fires when the dump is stale or hand-edited --
		// which is exactly when nobody is watching for it.
		if clash, found := reverse[index]; found {
			return nil, fmt.Errorf(
				"index %d maps to both %q and %q; the map is not invertible",
				index, clash, external,
			)
		}
		reverse[index] = external
	}
	return &Table{forward: forward, reverse: reverse, version: version}, nil
}

func (t *Table) Index(external string) (int32, bool) {
	index, found := t.forward[external]
	return index, found
}

func (t *Table) External(index int32) (string, bool) {
	external, found := t.reverse[index]
	return external, found
}

func (t *Table) Version() string { return t.version }
