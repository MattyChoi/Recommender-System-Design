.PHONY: help up down clean download bronze data replay train index serve bench demo lint fmt types test check

.DEFAULT_GOAL := help

MIND_SIZE ?= small
SPLITS ?= train dev
FORCE ?=

help:  ## Show this help
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

up:  ## Bring up local infra (kafka, redis, minio, mlflow, prometheus, grafana)
	docker compose up -d

down:  ## Stop and remove containers, keeping data
	docker compose down

clean:  ## Stop everything and DESTROY all volumes
	docker compose down -v

download:  ## Fetch MIND into paths.raw (override: make download MIND_SIZE=large)
	uv run python -m data_pipeline.ingest.download --size $(MIND_SIZE) --splits $(SPLITS) $(if $(FORCE),--force)

bronze:  ## raw -> bronze (rebuild an existing layer: make bronze FORCE=1)
	uv run python -m data_pipeline.ingest.mind --splits $(SPLITS) $(if $(FORCE),--force)

data: download bronze  ## Build raw + bronze under data/ (silver, gold: TODO)

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
