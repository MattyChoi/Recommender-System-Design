# Local stack: Kafka, Redis, MinIO, MLflow, Triton.
#
# `up` waits for every healthcheck before returning, then creates the two
# buckets MLflow and the artifact dumpers expect. Creating them here rather
# than in a container entrypoint keeps the bucket names in one place and lets
# `make up` be idempotent.

.PHONY: up down clean image image-size image-run sidecar-image sidecars-up sidecars-down \
        minio-image minio-image-kind


export HOST_IP := $(shell hostname -I 2>/dev/null | awk '{print $$1}')

# MinIO is built from source (no more published images -- see
# infra/docker/minio.Dockerfile), so `up` builds it the first time it is missing.
up:  ## Start the local stack and create the MinIO buckets
	@docker image inspect $(MINIO_IMAGE) >/dev/null 2>&1 || $(MAKE) minio-image
	@test -n "$(HOST_IP)" || echo "  warning: HOST_IP is empty; Prometheus will fall back to host-gateway"
	docker compose up -d --wait
	docker exec recsys-minio sh -c 'mc alias set local http://localhost:9000 "$$MINIO_ROOT_USER" "$$MINIO_ROOT_PASSWORD" >/dev/null && mc mb --ignore-existing local/mlflow local/recsys'
	@echo "  scrape targets:"
	@curl -s localhost:9090/api/v1/targets \
	  | python3 -c "import json,sys; [print(f\"    {t['labels'].get('job'):22} {t['health']}\") for t in json.load(sys.stdin)['data']['activeTargets']]" \
	  || echo "    (could not read Prometheus targets)"

down:  ## Stop and remove containers, keeping data
	docker compose down

# Destroys the MinIO and MLflow volumes, which is where every logged run and
# every published artifact lives. There is no undo.
clean:  ## Stop everything and DESTROY all volumes
	docker compose down -v


# ----------------------------------------------------------------- the images

IMAGE         ?= recsys/serving:dev
SIDECAR_IMAGE ?= recsys/sidecars:dev

COMPOSE_PROJECT ?= $(notdir $(CURDIR))
COMPOSE_NETWORK ?= $(COMPOSE_PROJECT)_default

serving-image:  ## Build the orchestrator image
	docker build -f infra/docker/serving.Dockerfile -t $(IMAGE) .

# ONE image for both Python sidecars; the compose service picks the module.
sidecar-image:  ## Build the Python sidecar image (features + retrieval)
	docker build -f infra/docker/sidecars.Dockerfile -t $(SIDECAR_IMAGE) .

image:
	$(MAKE) serving-image
	$(MAKE) sidecar-image
image-size:  ## Report both image sizes
	@docker images --format '  {{.Repository}}:{{.Tag}}  {{.Size}}' $(IMAGE)
	@docker images --format '  {{.Repository}}:{{.Tag}}  {{.Size}}' $(SIDECAR_IMAGE)

# The two sidecars sit behind the `sidecars` profile so a plain `make up` does
# not start them: the host-run workflow via `make serve-all` stays the default
# and nothing silently takes their ports.
sidecars-up: up  ## Start the whole stack plus the containerised sidecars
	docker compose --profile sidecars up -d --wait

sidecars-down:  ## Stop the containerised sidecars, leaving the stack up
	docker compose --profile sidecars stop features retrieval

# Run the orchestrator image against the compose stack. The artifacts are
# mounted at the paths the binary's own defaults already name -- see WORKDIR in
# serving.Dockerfile -- so item-map, catalogue, popular and columns need no
# flags. The two OPTIONAL stems do, because their defaults are empty.
IMAGE_FEATURES_HOST  ?= features
IMAGE_RETRIEVAL_HOST ?= retrieval
IMAGE_STEMS = \
  $(if $(wildcard $(INDEX_STEM).json),-index-stem $(INDEX_STEM),) \
  $(if $(wildcard $(CONTENT_STEM).json),-content-stem $(CONTENT_STEM),)

image-run:  ## Run the orchestrator image against the compose stack
	@test -f serving/artifacts/item_map.json \
	  || { echo "  serving/artifacts is empty -- run 'make artifacts' first"; exit 1; }
	@test -f serving/triton/ranker/1/model.columns.json \
	  || { echo "  model.columns.json missing -- run 'make onnx' first"; exit 1; }
	@test -n "$(wildcard $(INDEX_STEM).json)" \
	  || echo "  note: no $(INDEX_STEM).json -- exact search fallback disabled (run: make index-vectors)"
	@test -n "$(wildcard $(CONTENT_STEM).json)" \
	  || echo "  note: no $(CONTENT_STEM).json -- MMR disabled (run: make content-vectors)"
	@for c in recsys-kafka recsys-triton recsys-schema-registry recsys-redis \
	          recsys-features recsys-retrieval; do \
	  docker inspect -f '{{.State.Running}}' $$c 2>/dev/null | grep -q true \
	    || { echo "  $$c is not running -- run 'make sidecars-up' first"; exit 1; }; \
	done
	docker run --rm -it \
	    --name recsys-orchestrator \
	    --network $(COMPOSE_NETWORK) \
	    --add-host recsys-host:$(or $(HOST_IP),host-gateway) \
	    -p $(RECSYS_PORT):$(RECSYS_PORT) -p 9102:9102 \
	    -v "$(CURDIR)/serving/artifacts:/repo/serving/artifacts:ro" \
	    -v "$(CURDIR)/serving/triton/ranker/1:/repo/serving/triton/ranker/1:ro" \
	    $(IMAGE) \
	      -port $(RECSYS_PORT) \
	      -features $(call addresses,$(IMAGE_FEATURES_HOST),$(FEATURES_PORT),$(FEATURES_PROCESSES)) \
	      -retrieval $(call addresses,$(IMAGE_RETRIEVAL_HOST),$(RETRIEVAL_PORT),$(RETRIEVAL_PROCESSES)) \
	      $(IMAGE_STEMS) \
	      -triton triton:8001 \
	      -redis redis://redis:6379 \
	      -kafka kafka:29092 \
	      -schema-registry http://schema-registry:8081 \
	      -warm-wait 60s

# The server tag names the image; the mc tag rides along inside it. Bump both
# here, and the image name in docker-compose.yml and infra/k8s/*.yaml with them.
MINIO_TAG   ?= RELEASE.2025-09-07T16-13-09Z
MC_TAG      ?= RELEASE.2025-04-16T18-13-26Z
MINIO_IMAGE ?= recsys/minio:$(MINIO_TAG)
KIND_CLUSTER ?= recsys

# `- <` sends the Dockerfile with NO build context: it clones what it needs, so
# there is nothing in the repo worth shipping to the daemon.
minio-image:  ## Build MinIO + mc from source (~1 min first time)
	docker build -t $(MINIO_IMAGE) \
	  --build-arg MINIO_TAG=$(MINIO_TAG) --build-arg MC_TAG=$(MC_TAG) \
	  - < infra/docker/minio.Dockerfile

# kind nodes have no registry behind them, so the image is side-loaded.
minio-image-kind: minio-image  ## Build MinIO and load it into the kind cluster
	kind load docker-image $(MINIO_IMAGE) --name $(KIND_CLUSTER)
