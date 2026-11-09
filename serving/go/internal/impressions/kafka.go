package impressions

import (
	"fmt"
	"log"
	"sync/atomic"
	"time"

	"github.com/confluentinc/confluent-kafka-go/v2/kafka"
	"github.com/confluentinc/confluent-kafka-go/v2/schemaregistry"
	"github.com/confluentinc/confluent-kafka-go/v2/schemaregistry/serde"
	"github.com/confluentinc/confluent-kafka-go/v2/schemaregistry/serde/protobuf"
	pbuf "google.golang.org/protobuf/proto"
)

// flushTimeout bounds Close. Long enough for a full local queue to reach a
// healthy broker, short enough that a dead one does not hold a rollout open --
// and AsyncLog.Close has already spent up to closeTimeout draining into here.
const (
	flushTimeout  = 5 * time.Second
	lingerTimeout = 20 // in milliseconds
	maxQueueSize  = 200000
)

type KafkaProducer struct {
	producer   *kafka.Producer
	serializer *protobuf.Serializer

	// Topic names held as FIELDS because kafka.TopicPartition wants a
	// *string, not a string -- one of the API's rough edges. One stable
	// address per topic rather than one per message.
	events  string
	context string

	// failed is delivery failures, reported asynchronously
	failed atomic.Uint64
	done   chan struct{}
}

func NewKafkaProducer(
	bootstrapServers, registryURL, eventsTopic, contextTopic string,
) (*KafkaProducer, error) {
	registry, err := schemaregistry.NewClient(schemaregistry.NewConfig(registryURL))
	if err != nil {
		return nil, fmt.Errorf("schema registry at %s: %w", registryURL, err)
	}

	config := protobuf.NewSerializerConfig()
	config.AutoRegisterSchemas = false
	config.UseLatestVersion = true

	serializer, err := protobuf.NewSerializer(registry, serde.ValueSerde, config)
	if err != nil {
		return nil, fmt.Errorf("protobuf serializer: %w", err)
	}

	producer, err := kafka.NewProducer(&kafka.ConfigMap{
		"bootstrap.servers": bootstrapServers,
		// IDEMPOTENCE is not the same guarantee as the deterministic
		// event_id, and both earn their place. This dedupes librdkafka's OWN
		// retries inside one producer session, so a timeout-then-resend
		// cannot double-write. It does not survive a process restart or a
		// consumer reprocessing the topic -- that window is event_id's.
		"enable.idempotence": true,

		// Set explicitly even though idempotence implies it
		"acks": "all",

		// matching replay/producer.py.
		"linger.ms":                    lingerTimeout,
		"compression.type":             "lz4",
		"queue.buffering.max.messages": maxQueueSize,
	})
	if err != nil {
		return nil, fmt.Errorf("kafka producer at %s: %w", bootstrapServers, err)
	}

	k := &KafkaProducer{
		producer:   producer,
		serializer: serializer,
		events:     eventsTopic,
		context:    contextTopic,
		done:       make(chan struct{}),
	}
	go k.reportDeliveries()
	return k, nil
}

// reportDeliveries drains the producer's event channel.
func (k *KafkaProducer) reportDeliveries() {
	defer close(k.done)
	for event := range k.producer.Events() {
		switch value := event.(type) {
		case *kafka.Message:
			if value.TopicPartition.Error != nil {
				k.failed.Add(1)
			}
		case kafka.Error:
			if value.IsFatal() {
				log.Printf("kafka producer FATAL: %v", value)
			}
		}
	}
}

// Failed is delivery failures reported after the fact.
func (k *KafkaProducer) Failed() uint64 { return k.failed.Load() }

func (k *KafkaProducer) Produce(
	topic, key string, eventTime time.Time, msg pbuf.Message,
) error {
	name, err := k.topic(topic)
	if err != nil {
		return err
	}

	payload, err := k.serializer.Serialize(topic, msg)
	if err != nil {
		return fmt.Errorf("serializing for %s: %w", topic, err)
	}

	// ASYNCHRONOUS. This enqueues and returns; the only errors it reports are
	// a full local queue and a malformed message. Delivery lands on Events()
	// milliseconds to seconds later, which is what Failed() is for.
	if err := k.producer.Produce(&kafka.Message{
		TopicPartition: kafka.TopicPartition{Topic: name, Partition: kafka.PartitionAny},
		// Keyed on the user, per §5.4. Kafka hashes the key itself;
		// PartitionAny means "use the partitioner", not "pick at random".
		// The key stays PLAIN BYTES -- only the value is framed, because the
		// serializer was built with serde.ValueSerde.
		Key:       []byte(key),
		Value:     payload,
		Timestamp: eventTime,
	}, nil); err != nil {
		return fmt.Errorf("enqueueing for %s: %w", topic, err)
	}
	return nil
}

// topic maps a logical name onto its stable address, and refuses anything else.
func (k *KafkaProducer) topic(name string) (*string, error) {
	switch name {
	case k.events:
		return &k.events, nil
	case k.context:
		return &k.context, nil
	default:
		return nil, fmt.Errorf("unknown impression topic %q", name)
	}
}

// Close flushes, then shuts the client down.
func (k *KafkaProducer) Close() error {
	remaining := k.producer.Flush(int(flushTimeout / time.Millisecond))

	// Close also closes Events(), which is what lets reportDeliveries return.
	k.producer.Close()
	<-k.done

	if remaining > 0 {
		return fmt.Errorf("kafka flush abandoned %d undelivered records", remaining)
	}
	return nil
}

var _ Producer = (*KafkaProducer)(nil)
