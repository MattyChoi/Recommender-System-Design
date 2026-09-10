.PHONY: proto help up down clean raw bronze silver gold feast data replay train index serve bench demo lint fmt types test check

.DEFAULT_GOAL := help

MIND_SIZE ?= small
SPLITS ?= train dev
FORCE ?=
FEAST_START ?= 2019-11-09T00:00:00		# MIND dataset date range
FEAST_END   ?= 2019-11-16T00:00:00
FEAST_REPO ?= data_pipeline/features/recsys_store/feature_repo

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
	uv run python -m data_pipeline.ingest.bronze --splits $(SPLITS) $(if $(FORCE),--force)

silver:  ## bronze -> silver (rebuild an existing layer: make silver FORCE=1)
	uv run python -m data_pipeline.transform.silver --splits $(SPLITS) $(if $(FORCE),--force)

gold:  ## silver -> gold feature tables (rebuild an existing layer: make gold FORCE=1)
	uv run python -m data_pipeline.features.gold --splits $(SPLITS) $(if $(FORCE),--force)

feast:  ## Register feature definitions and materialise them into Redis
	uv run feast -c $(FEAST_REPO) apply
	uv run feast -c $(FEAST_REPO) materialize $(FEAST_START) $(FEAST_END)
# 	uv run feast -c $(FEAST_REPO) materialize 2019-11-09T00:00:00 2019-11-16T00:00:00

data: raw bronze silver gold  ## Build every layer under data/

replay:  ## Drive the pipeline from the Kafka replay harness
	@echo "TODO: replay harness"; exit 1

train:  ## Train retrieval + ranking models
	@echo "TODO: training"; exit 1

index:  ## Build the FAISS index
	@echo "TODO: index build"; exit 1

serve:  ## Run the gRPC orchestrator
	@echo "TODO: serving"; exit 1

bench:  ## Latency + recall/QPS benchmarks
	@echo "TODO: benchmarks"; exit 1

demo:  ## End-to-end demo; must work from a clean clone
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
