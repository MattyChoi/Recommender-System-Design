package impressions

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	pb "github.com/MattyChoi/Recommender-System-Design/serving/go/internal/pb"
)

func event(id string) *pb.InteractionEvent {
	// dwell_ms and propensity left at zero ON PURPOSE: they are two of the
	// fields the EmitUnpopulated test below is about.
	return &pb.InteractionEvent{
		EventId: id, UserId: "U7", ItemId: "N11",
		EventType: "impression", EventTsMs: 1, IngestTsMs: 2,
	}
}

func lines(t *testing.T, dir, topic string) []string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(dir, topic+".jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	return strings.Split(strings.TrimSuffix(string(raw), "\n"), "\n")
}

func TestOneFilePerTopic(t *testing.T) {
	dir := t.TempDir()
	p, err := NewFileProducer(dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, topic := range []string{TopicEvents, TopicContext} {
		if err := p.Produce(topic, "U7", time.Now(), event("e1")); err != nil {
			t.Fatal(err)
		}
	}
	if err := p.Close(); err != nil {
		t.Fatal(err)
	}
	for _, topic := range []string{TopicEvents, TopicContext} {
		if got := lines(t, dir, topic); len(got) != 1 {
			t.Errorf("%s has %d lines", topic, len(got))
		}
	}
}

func TestAReopenAppendsRatherThanTruncates(t *testing.T) {
	// O_APPEND against O_TRUNC, and the only test that can tell them apart.
	// A probe that compares behaviour across a restart needs both halves of
	// the run, and a truncating open loses the first one silently.
	dir := t.TempDir()
	for _, id := range []string{"before", "after"} {
		p, err := NewFileProducer(dir)
		if err != nil {
			t.Fatal(err)
		}
		if err := p.Produce(TopicEvents, "U7", time.Now(), event(id)); err != nil {
			t.Fatal(err)
		}
		if err := p.Close(); err != nil {
			t.Fatal(err)
		}
	}
	if got := lines(t, dir, TopicEvents); len(got) != 2 {
		t.Fatalf("got %d lines across two runs, want 2", len(got))
	}
}

func TestTheJsonUsesProtoNamesAndKeepsZeroValues(t *testing.T) {
	// The slip this catches was shipped once: protojson's DEFAULTS emit
	// lowerCamelCase and omit proto3 zero values, so a JSON consumer reads
	// thirteen NULL columns and nothing errors. Asserted on the KEYS, because
	// that is what a reader binds to.
	dir := t.TempDir()
	p, err := NewFileProducer(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := p.Produce(TopicEvents, "U7", time.Now(), event("e1")); err != nil {
		t.Fatal(err)
	}
	if err := p.Close(); err != nil {
		t.Fatal(err)
	}

	var decoded map[string]any
	if err := json.Unmarshal([]byte(lines(t, dir, TopicEvents)[0]), &decoded); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"event_ts_ms", "ingest_ts_ms", "request_id"} {
		if _, ok := decoded[key]; !ok {
			t.Errorf("no %q key -- UseProtoNames is off", key)
		}
	}
	if _, ok := decoded["eventTsMs"]; ok {
		t.Error("lowerCamelCase keys present -- UseProtoNames is off")
	}
	for _, key := range []string{"dwell_ms", "propensity", "session_id"} {
		if _, ok := decoded[key]; !ok {
			t.Errorf("no %q key -- EmitUnpopulated is off", key)
		}
	}
	// The DESCRIPTOR's count, not a literal 13. A hard-coded number turns "a
	// field was added to the proto and the debug sink now prints it" into a
	// failure that reads like a bug, which is how a correct assertion gets
	// deleted instead of understood.
	declared := (&pb.InteractionEvent{}).ProtoReflect().Descriptor().Fields().Len()
	if len(decoded) != declared {
		t.Errorf("%d keys in the JSON, %d fields in the proto -- EmitUnpopulated "+
			"should print every one", len(decoded), declared)
	}
}

func TestCloseTwiceIsSafe(t *testing.T) {
	// A second Close must not close the same fd again: on a reused fd number
	// that is a close of somebody else's file, which is the worst available
	// bug here and exactly one `defer` away in main.
	dir := t.TempDir()
	p, err := NewFileProducer(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := p.Produce(TopicEvents, "U7", time.Now(), event("e1")); err != nil {
		t.Fatal(err)
	}
	if err := p.Close(); err != nil {
		t.Fatal(err)
	}
	if err := p.Close(); err != nil {
		t.Fatalf("the second Close returned %v", err)
	}
}

func TestConcurrentProducesDoNotInterleaveALine(t *testing.T) {
	// What the single Write buys. Two syscalls per record would let a second
	// writer land between the object and its newline, and the symptom is a
	// parse error on a record that was fine.
	//
	// Run this under `make go-race` as well: the unsynchronised version of
	// open() fails there long before the line check below notices.
	dir := t.TempDir()
	p, err := NewFileProducer(dir)
	if err != nil {
		t.Fatal(err)
	}

	var wg sync.WaitGroup
	for i := range 64 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = p.Produce(TopicEvents, "U7", time.Now(), event(strings.Repeat("x", i%40)))
		}()
	}
	wg.Wait()
	if err := p.Close(); err != nil {
		t.Fatal(err)
	}

	got := lines(t, dir, TopicEvents)
	if len(got) != 64 {
		t.Fatalf("got %d lines, want 64", len(got))
	}
	for i, line := range got {
		var parsed map[string]any
		if err := json.Unmarshal([]byte(line), &parsed); err != nil {
			t.Fatalf("line %d does not parse: %v", i, err)
		}
	}
}

func TestAnUnusableDirectoryFailsAtConstruction(t *testing.T) {
	// Not on the first served request, where it would be swallowed as an
	// AsyncLog drop and counted with three other things.
	file := filepath.Join(t.TempDir(), "notadir")
	if err := os.WriteFile(file, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := NewFileProducer(filepath.Join(file, "under")); err == nil {
		t.Fatal("a directory under a regular file must be refused")
	}
}
