# Kafka, for the replay harness.
#
# These targets shell INTO the broker container, so they use the PLAINTEXT
# listener it advertises as localhost:9092. A client on the host reaches the
# same address through the published port; Flink, inside the compose network,
# uses the INTERNAL listener at kafka:29092 instead.

.PHONY: topic delete_topic replay consume offsets

KAFKA_CONTAINER ?= recsys-kafka
KAFKA_BROKER    ?= localhost:9092
TOPIC           ?= impressions
PARTITIONS      ?= 3
REPLAY_SPLIT    ?= train
REPLAY_ARGS     ?=
CONSUME_ARGS    ?=
KAFKA_EXEC       = docker exec $(KAFKA_CONTAINER) /opt/kafka/bin

topic:  ## Create the replay topic -- run ONCE before the first replay
	$(KAFKA_EXEC)/kafka-topics.sh --bootstrap-server $(KAFKA_BROKER) \
	    --create --if-not-exists --topic $(TOPIC) \
	    --partitions $(PARTITIONS) --replication-factor 1
	$(KAFKA_EXEC)/kafka-configs.sh --bootstrap-server $(KAFKA_BROKER) \
	    --alter --entity-type topics --entity-name $(TOPIC) \
	    --add-config retention.ms=-1
	$(KAFKA_EXEC)/kafka-topics.sh --bootstrap-server $(KAFKA_BROKER) \
	    --describe --topic $(TOPIC)

delete_topic:  ## Delete the replay topic -- run ONCE to reset the replay harness
	$(KAFKA_EXEC)/kafka-topics.sh --bootstrap-server $(KAFKA_BROKER) \
	    --delete --topic $(TOPIC)

# Run `make topic` first. Producing to a topic that does not exist auto-creates
# it with ONE partition, and on one partition the producer's user_id keying is
# untestable -- ordering is trivially global. Partition count cannot be lowered
# later, so the only fix is deleting the topic.
replay:  ## Replay bronze onto Kafka (make replay REPLAY_SPLIT=dev REPLAY_ARGS="--speed 0")
	uv run python -m data_pipeline.replay.producer --split $(REPLAY_SPLIT) $(REPLAY_ARGS)

consume:  ## Tail the replay topic (make consume CONSUME_ARGS=--from-beginning)
	docker exec -it $(KAFKA_CONTAINER) /opt/kafka/bin/kafka-console-consumer.sh \
	    --bootstrap-server $(KAFKA_BROKER) --topic $(TOPIC) \
	    --property print.key=true \
	    --property print.partition=true \
	    --property print.timestamp=true $(CONSUME_ARGS)

# End offsets, which equal the message count on a topic that has never been
# compacted and whose retention has not yet deleted a segment -- true for a
# demo topic, not in general. Counts ACCUMULATE across replays; `make topic`
# does not reset them.
offsets:  ## Message count per partition of the replay topic
	@$(KAFKA_EXEC)/kafka-get-offsets.sh --bootstrap-server $(KAFKA_BROKER) --topic $(TOPIC) \
	  | awk -F: '{printf "  partition %s: %8d\n", $$2, $$3; t += $$3} END {printf "  %-11s %8d\n", "total:", t}'
