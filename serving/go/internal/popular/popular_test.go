package popular

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func write(t *testing.T, payload onDisk) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "popular.json")
	raw, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshalling: %v", err)
	}
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatalf("writing: %v", err)
	}
	return path
}

func loaded(t *testing.T) *List {
	t.Helper()
	list, err := Load(write(t, onDisk{Version: "v=test", Items: []int32{5, 9, 2, 7}}))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	return list
}

func TestPopularReturnsTheTopNInOrder(t *testing.T) {
	got, err := loaded(t).Popular(context.Background(), 2)
	if err != nil {
		t.Fatalf("Popular: %v", err)
	}

	if len(got) != 2 || got[0] != 5 || got[1] != 9 {
		t.Errorf("got %v, want [5 9]", got)
	}
}

func TestAskingForMoreThanExistsReturnsWhatExists(t *testing.T) {
	// Short, not padded. The alternative on this path is an error, and an
	// error here is a blank page at the moment the system is already degraded.
	got, err := loaded(t).Popular(context.Background(), 50)
	if err != nil {
		t.Fatalf("Popular: %v", err)
	}

	if len(got) != 4 {
		t.Errorf("got %d items from a 4-item list", len(got))
	}
}

// TestTheReturnedSliceDoesNotAliasTheArtifact is the bug that would only
// appear under load.
//
// The caller writes this into a Result that the re-ranker and the transport
// both touch. A sub-slice of the loaded array would let one request's
// mutation corrupt the fallback for every request after it -- intermittent,
// unreproducible, and worst exactly when the fallback is busiest.
func TestTheReturnedSliceDoesNotAliasTheArtifact(t *testing.T) {
	list := loaded(t)

	first, err := list.Popular(context.Background(), 2)
	if err != nil {
		t.Fatalf("Popular: %v", err)
	}
	first[0] = 999

	second, _ := list.Popular(context.Background(), 2)
	if second[0] != 5 {
		t.Errorf("the artifact was mutated through a returned slice: %v", second)
	}
}

func TestZeroOrNegativeIsEmptyNotAnError(t *testing.T) {
	for _, n := range []int{0, -1} {
		got, err := loaded(t).Popular(context.Background(), n)
		if err != nil || len(got) != 0 {
			t.Errorf("n=%d: got %v, %v", n, got, err)
		}
	}
}

// --- What the loader refuses -------------------------------------------------

func TestAnEmptyListIsRefusedAtLoad(t *testing.T) {
	// It loads fine and returns an empty slate on the one path that exists to
	// never do that. Caught at startup, where it is a log line.
	if _, err := Load(write(t, onDisk{Version: "v=empty"})); err == nil {
		t.Fatal("an empty fallback must be refused at load")
	}
}

func TestAReservedIndexIsRefusedAtLoad(t *testing.T) {
	// Index 0 is the OOV row. Served, it puts a placeholder in front of a user
	// at the moment the system is already degraded.
	_, err := Load(write(t, onDisk{Items: []int32{5, 0, 7}}))

	if err == nil {
		t.Fatal("a reserved index must be refused")
	}
}

func TestAMissingFileIsNamed(t *testing.T) {
	if _, err := Load(filepath.Join(t.TempDir(), "absent.json")); err == nil {
		t.Fatal("a missing fallback must fail at startup, not at the first degraded request")
	}
}

func TestTheVersionIsReported(t *testing.T) {
	// A fallback frozen at build time ages; ADR 0011 puts the decay optimum at
	// ~29 minutes on this corpus. The version is what makes a stale slate
	// attributable rather than mysterious.
	if loaded(t).Version() != "v=test" {
		t.Errorf("version %q", loaded(t).Version())
	}
}
