package impressions

import (
	"errors"
	"strconv"
	"testing"
	"time"

	"google.golang.org/protobuf/proto"

	pb "github.com/MattyChoi/Recommender-System-Design/serving/go/internal/pb"
	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/service"
)

// An INTERNAL test package, deliberately: these reach l.now to freeze the
// clock and KafkaProducer's unexported fields below. The alternative is
// exporting a clock setter for the tests' benefit, which is a worse trade.

var (
	served = time.Date(2026, 3, 1, 12, 0, 0, 0, time.UTC)
	// 40ms later, so ingest - served is a number a test can recognise rather
	// than a zero that two identical clocks would also produce.
	ingested = served.Add(40 * time.Millisecond)
)

type captured struct {
	topic     string
	key       string
	eventTime time.Time
	msg       proto.Message
}

// recorder is a Producer that remembers. failAt makes the Nth call (1-based)
// fail, for the partial-write case.
type recorder struct {
	calls  []captured
	failAt int
}

func (r *recorder) Produce(topic, key string, eventTime time.Time, msg proto.Message) error {
	r.calls = append(r.calls, captured{topic, key, eventTime, msg})
	if r.failAt != 0 && len(r.calls) == r.failAt {
		return errors.New("boom")
	}
	return nil
}

type mapper map[int32]string

func (m mapper) External(index int32) (string, bool) {
	name, ok := m[index]
	return name, ok
}

func fixture(t *testing.T, known mapper) (*Log, *recorder) {
	t.Helper()
	rec := &recorder{}
	l := New(rec, known)
	l.now = func() time.Time { return ingested }
	return l, rec
}

func record() service.Record {
	return service.Record{
		UserID:     "U7",
		Surface:    "home_rail",
		Variant:    "treatment",
		RequestID:  "req-1",
		ServedAt:   served,
		ItemIDs:    []int32{11, 22, 33},
		Objective:  []float64{3.0, 2.0, 1.0},
		Scores:     []float64{3.5, 2.5, 1.5},
		Propensity: []float64{0.9, 0.1, 0.5},
		UserFeats:  []float32{1, 2, 3, 4},
		Degraded:   []string{"ranker"},
	}
}

func events(t *testing.T, rec *recorder) []*pb.InteractionEvent {
	t.Helper()
	var out []*pb.InteractionEvent
	for _, call := range rec.calls {
		if event, ok := call.msg.(*pb.InteractionEvent); ok {
			out = append(out, event)
		}
	}
	return out
}

func TestTheContextRowGoesFirst(t *testing.T) {
	// The ordering argument is in a comment on Emit; this is what makes it a
	// property. Events before context means a partial failure leaves fact rows
	// that look complete and silently carry no variant.
	l, rec := fixture(t, mapper{11: "N11", 22: "N22", 33: "N33"})
	if err := l.Emit(record()); err != nil {
		t.Fatal(err)
	}
	if rec.calls[0].topic != TopicContext {
		t.Fatalf("first write went to %q, want %q", rec.calls[0].topic, TopicContext)
	}
	if got := len(rec.calls); got != 4 {
		t.Fatalf("got %d writes, want 1 context + 3 events", got)
	}
}

func TestOnePerSlotWithDensePositions(t *testing.T) {
	l, rec := fixture(t, mapper{11: "N11", 22: "N22", 33: "N33"})
	if err := l.Emit(record()); err != nil {
		t.Fatal(err)
	}
	for slot, event := range events(t, rec) {
		if int(event.Position) != slot {
			t.Errorf("slot %d reported position %d", slot, event.Position)
		}
		if event.EventId != "req-1:"+strconv.Itoa(slot) {
			t.Errorf("slot %d event_id %q", slot, event.EventId)
		}
	}
}

func TestEachSlotKeepsItsOwnPropensity(t *testing.T) {
	// THE test in this file. A short or shifted propensity slice pairs each
	// slot with its neighbour's probability, and §17.4's estimator divides by
	// it without complaining -- a complete, plausible, wrong dataset.
	l, rec := fixture(t, mapper{11: "N11", 22: "N22", 33: "N33"})
	if err := l.Emit(record()); err != nil {
		t.Fatal(err)
	}
	want := []float32{0.9, 0.1, 0.5}
	for slot, event := range events(t, rec) {
		if event.Propensity != want[slot] {
			t.Errorf("slot %d propensity %v, want %v", slot, event.Propensity, want[slot])
		}
	}
}

func TestAMismatchedPropensityLengthWritesNOTHING(t *testing.T) {
	// Asserting on the COUNT, not just the error: the check has to run before
	// the context write, or a refused record still leaves a context row with
	// no facts and a consumer counting requests sees one that never happened.
	l, rec := fixture(t, mapper{11: "N11"})
	broken := record()
	broken.Propensity = []float64{0.9}

	if err := l.Emit(broken); err == nil {
		t.Fatal("a 3-item slate with 1 propensity must be refused")
	}
	if len(rec.calls) != 0 {
		t.Fatalf("%d writes happened before the refusal", len(rec.calls))
	}
}

func TestAnUnmappedItemIsSkippedCountedAndLeavesItsSlotEmpty(t *testing.T) {
	// The surviving slots keep their ORIGINAL positions: 0 and 2, not 0 and 1.
	// That is correct -- the item really was served in slot 1 -- and it means a
	// consumer counting rows per request_id gets k-1 for this request. Which is
	// what impression_unmapped_total is for.
	l, rec := fixture(t, mapper{11: "N11", 33: "N33"})
	if err := l.Emit(record()); err != nil {
		t.Fatal(err)
	}
	got := events(t, rec)
	if len(got) != 2 {
		t.Fatalf("got %d events, want 2", len(got))
	}
	if got[0].Position != 0 || got[1].Position != 2 {
		t.Errorf("positions %d and %d, want 0 and 2", got[0].Position, got[1].Position)
	}
	if l.Unmapped() != 1 {
		t.Errorf("Unmapped() = %d, want 1", l.Unmapped())
	}
}

func TestEventTimeIsServedAndIngestIsTheSinksOwnClock(t *testing.T) {
	// Three things at once, and the third is the one with no error attached
	// anywhere: the eventTime handed to Produce becomes the KAFKA RECORD
	// timestamp, and §5.4 is why it must be served and not ingest. Pass the
	// wrong clock and every downstream window silently becomes processing
	// time.
	l, rec := fixture(t, mapper{11: "N11", 22: "N22", 33: "N33"})
	if err := l.Emit(record()); err != nil {
		t.Fatal(err)
	}
	for _, call := range rec.calls {
		if !call.eventTime.Equal(served) {
			t.Errorf("%s got record timestamp %v, want %v", call.topic, call.eventTime, served)
		}
		if call.key != "U7" {
			t.Errorf("%s keyed on %q, want the user id", call.topic, call.key)
		}
	}
	for _, event := range events(t, rec) {
		if event.EventTsMs != served.UnixMilli() {
			t.Errorf("event_ts_ms %d, want %d", event.EventTsMs, served.UnixMilli())
		}
		if event.IngestTsMs != ingested.UnixMilli() {
			t.Errorf("ingest_ts_ms %d, want %d", event.IngestTsMs, ingested.UnixMilli())
		}
	}
}

func TestOneIngestTimestampForTheWholeSlate(t *testing.T) {
	// A clock read per slot would make ingest - served per-row jitter instead
	// of the log's lag, which is the only reading that number has.
	ticks := 0
	rec := &recorder{}
	l := New(rec, mapper{11: "N11", 22: "N22", 33: "N33"})
	l.now = func() time.Time {
		ticks++
		return ingested.Add(time.Duration(ticks) * time.Millisecond)
	}
	if err := l.Emit(record()); err != nil {
		t.Fatal(err)
	}
	if ticks != 1 {
		t.Fatalf("the clock was read %d times for one slate", ticks)
	}
}

func TestTheContextCarriesWhatNoFactRowCan(t *testing.T) {
	// §6.5's requirement, asserted: "log the actual feature vectors served in
	// production to Kafka". InteractionEvent has no field for any of these four.
	l, rec := fixture(t, mapper{11: "N11", 22: "N22", 33: "N33"})
	if err := l.Emit(record()); err != nil {
		t.Fatal(err)
	}
	ctx, ok := rec.calls[0].msg.(*pb.ImpressionContext)
	if !ok {
		t.Fatalf("first write carried %T", rec.calls[0].msg)
	}
	if ctx.Variant != "treatment" {
		t.Errorf("variant %q", ctx.Variant)
	}
	if len(ctx.UserFeats) != 4 {
		t.Errorf("user_feats has %d entries, want 4", len(ctx.UserFeats))
	}
	if len(ctx.Degraded) != 1 || ctx.Degraded[0] != "ranker" {
		t.Errorf("degraded %v", ctx.Degraded)
	}
	if len(ctx.Scores) != 3 || len(ctx.Objective) != 3 {
		t.Errorf("scores %v objective %v", ctx.Scores, ctx.Objective)
	}
}

func TestTwoEmitsOfOneRecordProduceIdenticalEventIds(t *testing.T) {
	// What makes Kafka's at-least-once redelivery and a whole-record retry
	// after a partial failure dedupable rather than duplicative.
	l, rec := fixture(t, mapper{11: "N11", 22: "N22", 33: "N33"})
	for range 2 {
		if err := l.Emit(record()); err != nil {
			t.Fatal(err)
		}
	}
	got := events(t, rec)
	for slot := range 3 {
		if got[slot].EventId != got[slot+3].EventId {
			t.Errorf("slot %d: %q then %q", slot, got[slot].EventId, got[slot+3].EventId)
		}
	}
}

func TestAFailedSlotStopsAndReportsWhatCameBefore(t *testing.T) {
	// The partial write is DOCUMENTED behaviour, not an accident: slots before
	// the failure are already out, and the deterministic event_id above is
	// what makes retrying the whole record safe.
	l, rec := fixture(t, mapper{11: "N11", 22: "N22", 33: "N33"})
	rec.failAt = 3 // context, slot 0, then slot 1 fails

	if err := l.Emit(record()); err == nil {
		t.Fatal("a producer error must reach Emit")
	}
	if len(rec.calls) != 3 {
		t.Fatalf("got %d writes, want it to stop at the failure", len(rec.calls))
	}
}
