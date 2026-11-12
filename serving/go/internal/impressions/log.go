// Package impressions writes the served slate to the event log: one
// InteractionEvent per slot on `impressions`, plus one ImpressionContext per
// request on `impression_context`.
package impressions

import (
	"fmt"
	"sync/atomic"
	"time"

	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"

	pb "github.com/MattyChoi/Recommender-System-Design/serving/go/internal/pb"
	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/service"
)

const (
	// The EXISTING topic, shared with data_pipeline/replay.
	TopicEvents  = "impressions"
	TopicContext = "impression_context"
)

// eventType is what these rows are. The stream is shared with the raw
// interaction log, where click, watch and purchase rows arrive from the client;
// "impression" rows come from here and from the replay harness.
const eventType = "impression"

// marshal is the DEBUG encoder, and only that.
//
// The TOPIC carries Confluent-framed protobuf, built by the registry serializer
// in kafka.go. This is what FileProducer writes for a run with no broker, so it
// is explicitly NOT a byte-for-byte mirror of the stream -- `make consume` is
// what makes the stream readable instead.
//
// UseProtoNames so the debug output still reads like the .proto, and
// EmitUnpopulated so a zero-valued field is visible rather than missing, which
// is the property records.py argued for: "a consumer that has to branch on
// whether a key exists is a consumer that will break the day the field starts
// arriving."
var marshal = protojson.MarshalOptions{UseProtoNames: true, EmitUnpopulated: true}

// Item map, internal index -> external id.
//
// Declared here as ONE method rather than imported from package server: the
// sink needs the reverse direction only, and *server.Table satisfies this
type Mapper interface {
	External(index int32) (string, bool)
}

// Producer is all this package needs of a transport.
//
// The MESSAGE, not bytes: encoding is a transport concern, which is what lets
// the file producer write readable JSON while the Kafka producer writes the
// same JSON with a record timestamp, from one unchanged sink.
//
// eventTime is a SEPARATE argument and not read off the message, because the
// two messages name it differently (`event_ts_ms` and `served_ts_ms`) and a
// producer that had to switch on the concrete type to find it would be a
// producer that silently stamps the wrong one on the next message added.
type Producer interface {
	Produce(topic, key string, eventTime time.Time, msg proto.Message) error
}

type Log struct {
	producer Producer
	mapper   Mapper
	now      func() time.Time

	// unmapped is items the vocabulary could not resolve.
	unmapped atomic.Uint64
}

func New(producer Producer, mapper Mapper) *Log {
	return &Log{producer: producer, mapper: mapper, now: time.Now}
}

func (l *Log) Unmapped() uint64 { return l.unmapped.Load() }

func (l *Log) Emit(rec service.Record) error {
	if len(rec.Propensity) != len(rec.ItemIDs) {
		return fmt.Errorf("impression %s: %d items, %d propensities",
			rec.RequestID, len(rec.ItemIDs), len(rec.Propensity))
	}

	served := rec.ServedAt.UnixMilli()

	// produce the impression context first, so the Flink job can join on it and know the
	// request's objective and user features before it reads the per-slot events.
	if err := l.producer.Produce(
		TopicContext, rec.UserID, rec.ServedAt, &pb.ImpressionContext{
			RequestId:  rec.RequestID,
			UserId:     rec.UserID,
			Variant:    rec.Variant,
			ServedTsMs: served,
			ItemIds:    rec.ItemIDs,
			Scores:     rec.Scores,
			Objective:  rec.Objective,
			UserFeats:  rec.UserFeats,
			Degraded:   rec.Degraded,
			UserFound:  rec.UserFound,
			// Nil stays nil: proto3 writes an absent repeated field, which is
			// what distinguishes "no shadow ran" from "the shadow scored zero".
			ShadowScores: rec.ShadowScores,
			ShadowModel:  rec.ShadowModel,
		}); err != nil {
		return fmt.Errorf("producing context %s: %w", rec.RequestID, err)
	}

	// ONE clock read for the whole slate, AFTER the context write, so every
	// row of one request carries the same ingest timestamp and the difference
	// from served is the log's lag rather than per-row jitter.
	ingest := l.now().UnixMilli()

	for position, index := range rec.ItemIDs {
		itemID, ok := l.mapper.External(index)
		if !ok {
			// Dropped and COUNTED, never written with an empty item_id
			l.unmapped.Add(1)
			continue
		}

		event := &pb.InteractionEvent{
			EventId:    fmt.Sprintf("%s:%d", rec.RequestID, position),
			UserId:     rec.UserID,
			ItemId:     itemID,
			EventType:  eventType,
			EventTsMs:  served,
			IngestTsMs: ingest,
			Position:   int32(position),
			RequestId:  rec.RequestID,
			Surface:    rec.Surface,
			Propensity: float32(rec.Propensity[position]),
		}

		if err := l.producer.Produce(
			TopicEvents, rec.UserID, rec.ServedAt, event); err != nil {
			return fmt.Errorf("producing %s: %w", event.EventId, err)
		}
	}
	return nil
}

var _ service.Log = (*Log)(nil)
