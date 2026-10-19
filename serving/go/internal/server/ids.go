package server

import (
	"encoding/json"
	"fmt"
	"os"
)

type IDs interface {
	Index(external string) (int32, bool)
	External(index int32) (string, bool)
	Size() int
	Version() string
}

type Table struct {
	forward map[string]int32
	reverse []string
	version string
}

func (t *Table) Index(external string) (int32, bool) {
	index, found := t.forward[external]
	return index, found
}

func (t *Table) External(index int32) (string, bool) {
	if index < 0 || int(index) >= len(t.reverse) || t.reverse[index] == "" {
		return "", false
	}
	return t.reverse[index], true
}

func (t *Table) Size() int { return len(t.forward) }

func (t *Table) Version() string { return t.version }

type itemMap struct {
	Version string           `json:"version"`
	Items   map[string]int32 `json:"items"`
}

func LoadIDs(path string) (*Table, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading item map: %w", err)
	}
	var decoded itemMap
	if err := json.Unmarshal(raw, &decoded); err != nil {
		return nil, fmt.Errorf("parsing %s: %w", path, err)
	}
	return NewTable(decoded.Items, decoded.Version)
}

// NewTable inverts the map, REFUSING a map two ids share an index in.
func NewTable(forward map[string]int32, version string) (*Table, error) {
	highest := int32(-1)
	for _, index := range forward {
		if index > highest {
			highest = index
		}
	}
	reverse := make([]string, highest+1)
	for external, index := range forward {
		if index < 0 {
			return nil, fmt.Errorf("item %q has negative index %d", external, index)
		}
		if existing := reverse[index]; existing != "" {
			return nil, fmt.Errorf(
				"index %d is claimed by both %q and %q; the map is not invertible",
				index, existing, external,
			)
		}
		reverse[index] = external
	}
	return &Table{forward: forward, reverse: reverse, version: version}, nil
}
