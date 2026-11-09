# Kafka, for the replay harness.
#
# These targets shell INTO the broker container, so they use the PLAINTEXT
# listener it advertises as localhost:9092. A client on the host reaches the
# same address through the published port; Flink, inside the compose network,
# uses the INTERNAL listener at kafka:29092 instead.

.PHONY: topic delete_topic replay consume offsets schemas schemas-check

KAFKA_CONTAINER ?= recsys-kafka
KAFKA_BROKER    ?= localhost:9092
SR_URL          ?= http://localhost:8081
TOPIC           ?= impressions
PARTITIONS      ?= 3
REPLAY_SPLIT    ?= train
REPLAY_ARGS     ?=
CONSUME_ARGS    ?=
KAFKA_EXEC       = docker exec $(KAFKA_CONTAINER) /opt/kafka/bin
# The two subjects, under the default TopicNameStrategy: <topic>-value.
SCHEMA_SUBJECTS = impressions:events.proto impression_context:impressions.proto
# Pinned on each SUBJECT rather than inherited from the registry-wide default in
# docker-compose.yml, so changing that default cannot quietly loosen the rule
# for topics that consumers replay from offset 0.
SCHEMA_COMPATIBILITY ?= BACKWARD_TRANSITIVE

# Every topic with a registered schema, so a new subject gets its topic for
# free. `make topic TOPIC=x` narrows it to just x.
ifeq ($(origin TOPIC),command line)
TOPICS ?= $(TOPIC)
else
TOPICS ?= $(foreach p,$(SCHEMA_SUBJECTS),$(firstword $(subst :, ,$(p))))
endif

topic:  ## Create the event topics (impressions, impression_context) -- run ONCE before the first replay
	@for t in $(TOPICS); do \
	  $(KAFKA_EXEC)/kafka-topics.sh --bootstrap-server $(KAFKA_BROKER) \
	      --create --if-not-exists --topic $$t \
	      --partitions $(PARTITIONS) --replication-factor 1 || exit 1; \
	  $(KAFKA_EXEC)/kafka-configs.sh --bootstrap-server $(KAFKA_BROKER) \
	      --alter --entity-type topics --entity-name $$t \
	      --add-config retention.ms=-1 || exit 1; \
	  $(KAFKA_EXEC)/kafka-topics.sh --bootstrap-server $(KAFKA_BROKER) \
	      --describe --topic $$t || exit 1; \
	done

delete_topic:  ## Delete the replay topic -- run ONCE to reset the replay harness
	$(KAFKA_EXEC)/kafka-topics.sh --bootstrap-server $(KAFKA_BROKER) \
	    --delete --topic $(TOPIC)

# Run `make topic` first. Producing to a topic that does not exist auto-creates
# it with ONE partition, and on one partition the producer's user_id keying is
# untestable -- ordering is trivially global. Partition count cannot be lowered
# later, so the only fix is deleting the topic.
replay:  ## Replay bronze onto Kafka (make replay REPLAY_SPLIT=dev REPLAY_ARGS="--speed 0")
	uv run python -m data_pipeline.replay.producer --split $(REPLAY_SPLIT) $(REPLAY_ARGS)

consume:  ## Decode a protobuf topic. make consume TOPIC=impressions N=5
	uv run python scripts/writeups/decode_events.py --topic $(TOPIC) -n $(or $(N),5)

# End offsets, which equal the message count on a topic that has never been
# compacted and whose retention has not yet deleted a segment -- true for a
# demo topic, not in general. Counts ACCUMULATE across replays; `make topic`
# does not reset them.
offsets:  ## Message count per partition of the replay topic
	@$(KAFKA_EXEC)/kafka-get-offsets.sh --bootstrap-server $(KAFKA_BROKER) --topic $(TOPIC) \
	  | awk -F: '{printf "  partition %s: %8d\n", $$2, $$3; t += $$3} END {printf "  %-11s %8d\n", "total:", t}'

schemas:  ## Register the event schemas with the Schema Registry
	@test -n "$(SCHEMA_SUBJECTS)" || { echo "SCHEMA_SUBJECTS is empty"; exit 1; }
	@for pair in $(SCHEMA_SUBJECTS); do \
	  topic=$${pair%%:*}; file=$${pair##*:}; \
	  echo "  $$topic-value  <-  serving/proto/$$file  ($(SCHEMA_COMPATIBILITY))"; \
	  resp=$$(curl -sS -X PUT "$(SR_URL)/config/$$topic-value" \
	        -H 'Content-Type: application/vnd.schemaregistry.v1+json' \
	        -d '{"compatibility":"$(SCHEMA_COMPATIBILITY)"}'); \
	  printf "%s" "$$resp" | jq -e '.compatibility' > /dev/null 2>&1 \
	    || { echo "  COMPATIBILITY REFUSED: $$resp"; exit 1; }; \
	  resp=$$(jq -Rs '{schemaType:"PROTOBUF", schema:.}' < serving/proto/$$file \
	    | curl -sS -X POST "$(SR_URL)/subjects/$$topic-value/versions" \
	        -H 'Content-Type: application/vnd.schemaregistry.v1+json' -d @-); \
	  printf "%s" "$$resp" | jq -e '.id' > /dev/null 2>&1 \
	    || { echo "  REFUSED: $$resp"; exit 1; }; \
	done

# The inverse, and the gate worth having in `make check`: ask whether a schema
# WOULD be accepted, without registering it.
schemas-check:
	@test -n "$(SCHEMA_SUBJECTS)" || { echo "SCHEMA_SUBJECTS is empty"; exit 1; }
	@for pair in $(SCHEMA_SUBJECTS); do \
	  topic=$${pair%%:*}; file=$${pair##*:}; \
	  resp=$$(jq -Rs '{schemaType:"PROTOBUF", schema:.}' < serving/proto/$$file \
	    | curl -sS -X POST \
	        "$(SR_URL)/compatibility/subjects/$$topic-value/versions/latest" \
	        -H 'Content-Type: application/vnd.schemaregistry.v1+json' -d @-); \
	  printf "%s" "$$resp" | jq -e '.is_compatible == true' > /dev/null 2>&1 \
	    || { echo "  $$topic-value: NOT compatible: $$resp"; exit 1; }; \
	  echo "  $$topic-value  compatible"; \
	done
