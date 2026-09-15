.PHONY: proto help up down clean raw bronze silver gold feast parity eval sweep results gap coverage compare baselines torch-env data \
        topic delete_topic replay consume offsets \
        train index serve bench demo lint fmt types test check

.DEFAULT_GOAL := help


-include .env
export UV_ENV_FILE=.env

MIND_SIZE ?= small
SPLITS ?= train dev
FORCE ?=
FORCE_ID_MAPS ?=
REPLAY_SPLIT ?= train
REPLAY_ARGS ?=

# --- Kafka, for the replay harness ---------------------------------------
# These targets shell INTO the broker container, so they use the PLAINTEXT
# listener it advertises as localhost:9092. A client on the Mac reaches the
# same address through the published port; Flink, inside the compose network,
# uses the INTERNAL listener at kafka:29092 instead.
KAFKA_CONTAINER ?= recsys-kafka
KAFKA_BROKER    ?= localhost:9092
TOPIC           ?= impressions
PARTITIONS      ?= 3
CONSUME_ARGS    ?=
KAFKA_EXEC       = docker exec $(KAFKA_CONTAINER) /opt/kafka/bin

# Feast variables
FEAST_START ?= 2019-11-09T00:00:00		# MIND dataset date range
FEAST_END   ?= 2019-11-16T00:00:00
FEAST_REPO ?= data_pipeline/features/recsys_store/feature_repo

# Parity sample for make parity
PARITY_SAMPLE ?= 200


# Options for evaluating model metrics
MODELS ?= random popularity recency content covisit als als_item
MODEL ?= random
EVAL_SPLIT ?= dev
HALF_LIVES ?= 0.02 0.05 0.1 0.25 1 3
# 24h, chosen from `make coverage`, not from the manual. 1h left 91% of dev
# slates with every candidate tied at 0.0; 24h is where slate reach plateaus.
COVISIT_MAX_GAP ?= 86400

BASELINE ?= recency
CANDIDATE ?= decayed_popularity@0.02
# Seconds: 1h, 6h, 24h, 72h. The 1h default came from the manual, not this corpus.
WINDOWS ?= 3600 21600 86400 259200


help:  ## Show this help
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

up:
	docker compose up -d --wait
	docker exec recsys-minio sh -c 'mc alias set local http://localhost:9000 "$$MINIO_ROOT_USER" "$$MINIO_ROOT_PASSWORD" >/dev/null && mc mb --ignore-existing local/mlflow local/recsys'

down:  ## Stop and remove containers, keeping data
	docker compose down

clean:  ## Stop everything and DESTROY all volumes
	docker compose down -v

proto:  ## Regenerate protobuf stubs from serving/proto/*.proto
	mkdir -p common/pb serving/go/internal/pb && touch common/pb/__init__.py
	uv run python -m grpc_tools.protoc -I serving/proto \
	    --python_out=common/pb --pyi_out=common/pb --grpc_python_out=common/pb \
	    serving/proto/*.proto
	uv run python -c "import pathlib,re;[f.write_text(re.sub(r'^import (\w+_pb2)', r'from common.pb import \1', f.read_text(), flags=re.M)) for f in pathlib.Path('common/pb').glob('*_pb2*.py')]"
	protoc -I serving/proto \
	    --go_out=serving/go/internal/pb --go_opt=paths=source_relative \
	    --go-grpc_out=serving/go/internal/pb --go-grpc_opt=paths=source_relative \
	    serving/proto/*.proto

raw:  ## Fetch MIND into paths.raw (override: make download MIND_SIZE=large)
	uv run python -m data_pipeline.ingest.download_mind --size $(MIND_SIZE) --splits $(SPLITS) $(if $(FORCE),--force)

bronze:  ## raw -> bronze (rebuild an existing layer: make bronze FORCE=1)
	uv run python -m data_pipeline.ingest.bronze --splits $(SPLITS) $(if $(FORCE),--force) $(if $(FORCE_ID_MAPS),--force-id-maps)

silver:  ## bronze -> silver (rebuild an existing layer: make silver FORCE=1)
	uv run python -m data_pipeline.transform.silver --splits $(SPLITS) $(if $(FORCE),--force)

# Writes the three feature series to MinIO, so `make up` first. The label table
# stays on disk. `RECSYS_STORAGE__BACKEND=local make gold` puts everything back
# on disk and needs nothing running.
gold:  ## silver -> gold feature tables (rebuild an existing layer: make gold FORCE=1)
	uv run python -m data_pipeline.features.gold --splits $(SPLITS) $(if $(FORCE),--force)

# Reads the series from MinIO, so `make up` first. Note that apply RESOLVES the
# source paths and bakes them into the registry: running this under a different
# backend than the one you built with leaves a registry pointing at data that is
# not there. `RECSYS_STORAGE__BACKEND=local make feast` for the on-disk build.
feast:  ## Register feature definitions and materialise them into Redis
	uv run feast -c $(FEAST_REPO) apply
	uv run feast -c $(FEAST_REPO) materialize $(FEAST_START) $(FEAST_END) \
	    --views item_stats --views user_stats --views user_category_stats

# Reads bronze, not silver: silver's session_id was cut with the very threshold
# this measures, so measuring there would let the old answer pick the new one.
gap:  ## Measure the inter-impression gap, to set session.gap_minutes
	uv run python -m data_pipeline.transform.session_gap --split train

eval:  ## Score a model into evaluation/results/ (make eval MODEL=random)
	uv run python -m evaluation.offline.run_eval --model $(MODEL) --split $(EVAL_SPLIT) \
	    --max-gap-seconds $(COVISIT_MAX_GAP)

# One card per half-life, so the results directory holds the ablation. The curve is
# the deliverable: a lone tuned half-life reads as a number someone picked.
sweep:  ## Half-life curve for decayed popularity (make sweep HALF_LIVES="0.5 1 3 7")
	uv run python -m evaluation.offline.run_eval_sweep \
	    --split $(EVAL_SPLIT) --half-lives $(HALF_LIVES)

# Reach, not quality, and no cards written. A model that scores nothing and a
# model that ranks badly produce near-identical cards; this tells them apart
# cheaply, before spending eval runs on windows that cannot move the metric.
coverage:
	uv run python -m evaluation.offline.covisit_coverage \
	    --split $(EVAL_SPLIT) --windows $(WINDOWS)

compare:  ## Paired bootstrap between two models (make compare BASELINE=recency CANDIDATE=decayed_popularity@0.02)
	uv run python -m evaluation.offline.compare \
	    --baseline $(BASELINE) --candidate $(CANDIDATE) --split $(EVAL_SPLIT)

baselines:  ## Re-score every baseline and rebuild docs/results.md from one commit
	@for model in $(MODELS); do \
	    $(MAKE) --no-print-directory eval MODEL=$$model || exit 1; \
	done
	$(MAKE) --no-print-directory sweep HALF_LIVES="0.02 0.05 0.1 0.25"
	$(MAKE) --no-print-directory sweep HALF_LIVES="0.5 1 2 3"
	$(MAKE) --no-print-directory results

# Rebuilt from the cards, never hand-edited: a table that disagrees with the
# JSON it quotes is worse than no table. Pure stdlib, so no cluster is needed.
results:  ## Rebuild docs/results.md from evaluation/results/*.json
	uv run python -m evaluation.offline.results_table

# Compares what Feast materialised against what the gold series says it should
# hold. NOT the skew report -- both sides are offline reads; see docs/ for the
# distinction. Needs Redis and MinIO, so `make up && make feast` first.
parity:  ## Materialisation parity: Redis vs the gold series
	uv run python -m data_pipeline.features.parity --sample $(PARITY_SAMPLE)

data: raw bronze silver gold  ## Build every layer under data/

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

# Prints torch version, selected device and whether autocast is on, so a metric
# in MLflow can always be traced back to the hardware that produced it.
torch-env:  ## Report the torch device this machine will train on
	uv run python -c "from common.torch_env import describe, select_device; \
	    print(describe(select_device()))"

train:  ## Train retrieval + ranking models
	@echo "TODO: training"; exit 1

index:  ## Build the FAISS index
	@echo "TODO: index build"; exit 1

serve:  ## Run the gRPC orchestrator
	@echo "TODO: serving"; exit 1

bench:  ## Latency + recall/QPS benchmarks
	@echo "TODO: benchmarks"; exit 1

# Runs against the LOCAL gold layer on purpose.
demo: export RECSYS_STORAGE__BACKEND = local
demo:  ## End-to-end demo; must work from a clean clone, with nothing running
	@echo "TODO: demo"; exit 1

lint:  ## ruff check + format check
	uv run ruff check .
	uv run ruff format --check .

fmt:  ## Apply ruff formatting and autofixes
	uv run ruff check --fix --exit-zero .
	uv run ruff format .

types:  ## mypy
	uv run mypy .

test:  ## pytest
	uv run pytest

check:  ## lint + types + test (what CI runs)
	$(MAKE) lint
	$(MAKE) types
	$(MAKE) test
