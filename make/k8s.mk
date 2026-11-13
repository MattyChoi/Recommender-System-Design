# The orchestrator on a local Kubernetes cluster (kind).
#
# Credentials come from .env, which the root Makefile -includes. They are
# exported and referenced as $$VAR rather than $(VAR), so make prints the
# variable name and never the value.

.PHONY: kind-all kind-preflight kind-up kind-load kind-push kind-status kind-down \
        kind-guard backend-bundle kind-feast feast-gold-check kind-smoke

KIND_CONTEXT = kind-$(KIND_CLUSTER)
MINIO_FORWARD_PORT ?= 9900
MANIFESTS = infra/k8s
REDIS_FORWARD_PORT ?= 6390

TRITON_IMAGE ?= nvcr.io/nvidia/tritonserver:25.12-py3
BACKEND_BUNDLE = build/backend-bundle
BACKEND_PREFIX = backend-artifacts
FEAST_HOST_REGISTRY = $(FEAST_REPO)/registry.db
BACKENDS = recsys-retrieval recsys-features recsys-triton

# Everything backend-bundle links in. One list, read by the bundle AND by the
# preflight, so the two can never disagree about what is required.
BACKEND_INPUTS = $(INDEX_CHECKPOINT) data/gold/item_content/part-00000.parquet \
                 $(FEAST_HOST_REGISTRY) $(ITEM_MAP) $(CATEGORY_MAP) \
                 serving/triton/ranker/1/model.onnx serving/triton/ranker/1/model.onnx.data

# What `feast materialize` reads, in the COMPOSE stack's object store. The one
# input kind-all cannot produce itself, so it is checked before anything slow.
# <bucket>/gold, matching gold_location() for storage.bucket = recsys.
FEAST_GOLD = recsys/gold
FEAST_GOLD_TABLES = item_hourly_features user_hourly_features user_category_cross_features

KIND_TOOLS = kind kubectl docker uv grpcurl
# Not 15051: that is the host-run orchestrator's port under `make serve-all`.
SMOKE_FORWARD_PORT ?= 15951
SMOKE_REQUESTS ?= 30

export MINIO_ROOT_USER
export MINIO_ROOT_PASSWORD

# From nothing (or from any partial state) to a cluster serving real slates.
# Sequential $(MAKE) calls, NOT prerequisites: under `make -j` prerequisites
# run in parallel, and every step here needs the one before it. Each step is
# idempotent, so re-running after a failure resumes rather than starts over.
kind-all:  ## Preflight, then up -> load -> push -> feast -> smoke: a fully serving cluster
	@$(MAKE) --no-print-directory kind-preflight
	$(MAKE) --no-print-directory kind-up
	$(MAKE) --no-print-directory kind-load
	$(MAKE) --no-print-directory kind-push
	$(MAKE) --no-print-directory kind-feast
	$(MAKE) --no-print-directory kind-smoke
	@echo
	@echo "  cluster is serving real slates. Try:"
	@echo "    kubectl port-forward svc/recsys-api 15051:15051"

# Seconds, against the ten minutes of image loading after it. Everything here
# is an input kind-all READS and cannot build: tools, credentials, the dumped
# artifacts and the gold tables. No kind-guard -- the cluster may not exist yet.
kind-preflight:  ## Check every input kind-all needs, before anything slow
	@for t in $(KIND_TOOLS); do \
	  command -v $$t >/dev/null || { echo "  missing tool: $$t"; exit 1; }; \
	done
	@test -n "$$MINIO_ROOT_USER" \
	  || { echo "  MINIO_ROOT_USER is unset; .env supplies it"; exit 1; }
	@for f in $(ITEM_MAP) $(RANKER_COLUMNS) $(BACKEND_INPUTS); do \
	  test -f "$$f" || { echo "  missing $$f -- see make help for the target that writes it"; exit 1; }; \
	done
	@version=$$(cat data/index/CURRENT 2>/dev/null) \
	  && test -f "data/index/$$version/index.faiss" \
	  || { echo "  data/index/CURRENT names no index; run: make index"; exit 1; }
	@$(MAKE) --no-print-directory feast-gold-check
	@echo "  preflight ok"

# Starts ONLY the compose MinIO -- not `make up`, which would also start Triton
# on the GPU, Kafka and the rest, none of which materialisation needs. The
# credentials are expanded INSIDE the container, from its own env.
feast-gold-check:
	@docker compose up -d --wait minio >/dev/null 2>&1 \
	  || { echo "  could not start the compose MinIO (docker compose up minio)"; exit 1; }
	@docker exec recsys-minio sh -c '\
	  mc alias set local http://localhost:9000 "$$MINIO_ROOT_USER" "$$MINIO_ROOT_PASSWORD" >/dev/null; \
	  missing=; \
	  for t in $(FEAST_GOLD_TABLES); do \
	    test -n "$$(mc ls "local/$(FEAST_GOLD)/$$t/" 2>/dev/null)" \
	      || { echo "  missing s3://$(FEAST_GOLD)/$$t in the compose MinIO"; missing=1; }; \
	  done; \
	  test -z "$$missing"' \
	  || { echo "  Feast materialises from these: build them with the data pipeline,"; \
	       echo "  or copy them in from wherever they were built."; exit 1; }

# Every target that CHANGES the cluster depends on this. Without it a stray
# kubeconfig context turns `make kind-push` into a rolling restart of whatever
# cluster happens to be current, which may not be a toy.
kind-guard:
	@current=$$(kubectl config current-context 2>/dev/null || echo none); \
	  test "$$current" = "$(KIND_CONTEXT)" \
	  || { echo "  context is $$current, not $(KIND_CONTEXT); refusing"; exit 1; }

kind-up:  ## Create the kind cluster, load MinIO, write the secret, apply everything
	@kind get clusters 2>/dev/null | grep -qx $(KIND_CLUSTER) \
	  || kind create cluster --name $(KIND_CLUSTER)
	@$(MAKE) --no-print-directory kind-guard
	@# AFTER the cluster exists, not as a prerequisite: minio-image-kind ends in
	@# a `kind load`, which needs something to load into. Here rather than in
	@# kind-load because MinIO changes once a cluster, not once a code change.
	$(MAKE) --no-print-directory minio-image-kind
	@test -n "$$MINIO_ROOT_USER" \
	  || { echo "  MINIO_ROOT_USER is unset; .env supplies it"; exit 1; }
	@# `create --dry-run=client | apply` is the idempotent form: plain create
	@# fails on an existing secret, so re-running kind-up would break. Still
	@# @-prefixed, so no literal reaches the terminal.
	@kubectl create secret generic recsys-minio \
	    --from-literal=accesskey="$$MINIO_ROOT_USER" \
	    --from-literal=secretkey="$$MINIO_ROOT_PASSWORD" \
	    --dry-run=client -o yaml | kubectl apply -f -
	kubectl apply -f $(MANIFESTS)/
	@echo
	@echo "  recsys-api sits in Init:Error until: make kind-load kind-push"

kind-load: kind-guard  ## Build the orchestrator + sidecar images and side-load them (and Triton) into kind
	$(MAKE) --no-print-directory serving-image sidecar-image
	@# kind nodes are containers with no registry behind them, so the images are
	@# pushed into them directly. Re-run after every rebuild.
	kind load docker-image $(IMAGE) $(SIDECAR_IMAGE) --name $(KIND_CLUSTER)
	@# Triton is ~20GB and never rebuilt here, so NOT `kind load`: under the
	@# containerd image store the host ID is the multi-arch INDEX digest while
	@# the node files it by the amd64 config ID, so kind never sees a match and
	@# re-imports every time -- with --all-platforms, which then fails on the
	@# arm64 layers that were never pulled ("content digest ... not found").
	@# Skip by tag instead, and import a single-platform save.
	@set -e; for node in $$(kind get nodes --name $(KIND_CLUSTER)); do \
	    if docker exec $$node crictl inspecti $(TRITON_IMAGE) >/dev/null 2>&1; then \
	        echo "$(TRITON_IMAGE): already on $$node"; continue; fi; \
	    docker image inspect $(TRITON_IMAGE) >/dev/null 2>&1 || docker pull --platform linux/amd64 $(TRITON_IMAGE); \
	    echo "$(TRITON_IMAGE): importing into $$node (several minutes)"; \
	    docker save --platform linux/amd64 $(TRITON_IMAGE) \
	        | docker exec -i $$node ctr --namespace=k8s.io images import --digests --snapshotter=overlayfs -; \
	done

kind-push: kind-guard  ## Publish artifacts into the cluster's MinIO and roll the pods
	@kubectl rollout status deploy/recsys-minio --timeout=120s
	@# The bucket is a Job, and publishing into a bucket that does not exist yet
	@# fails as a bare 404 from the S3 client.
	@kubectl wait --for=condition=complete job/recsys-minio-bucket --timeout=120s
	@# ONE shell for the whole block: each recipe line is its own process, so a
	@# port-forward started on its own line would outlive the recipe. The trap
	@# is what makes it exit-safe, including on a failed publish.
	@kubectl port-forward svc/recsys-minio $(MINIO_FORWARD_PORT):9000 >/dev/null & \
	  forward=$$!; trap 'kill $$forward 2>/dev/null' EXIT; \
	  ready=""; \
	  for _ in $$(seq 60); do \
	    curl -sf http://localhost:$(MINIO_FORWARD_PORT)/minio/health/ready >/dev/null \
	      && { ready=1; break; }; \
	    sleep 0.5; \
	  done; \
	  test -n "$$ready" \
	    || { echo "  the forward never became ready on $(MINIO_FORWARD_PORT)"; exit 1; }; \
	  RECSYS_STORAGE__ENDPOINT_URL=http://localhost:$(MINIO_FORWARD_PORT) \
	    $(MAKE) --no-print-directory artifacts-push && \
	  $(MAKE) --no-print-directory backend-bundle && \
	  RECSYS_STORAGE__ENDPOINT_URL=http://localhost:$(MINIO_FORWARD_PORT) \
	  RECSYS_STORAGE__ARTIFACTS_PREFIX=$(BACKEND_PREFIX) \
	    uv run python -m $(DUMPERS).publish --source $(BACKEND_BUNDLE)
	@# Every init container reads CURRENT once, at pod start, so a push alone
	@# changes nothing that is running. A ROLLING restart honours the readiness
	@# flip, so new pods pull the new version and the old ones drain. Backends
	@# first: the orchestrator's warm-up probes them.
	kubectl rollout restart $(addprefix deploy/,$(BACKENDS))
	@for d in $(BACKENDS); do kubectl rollout status deploy/$$d --timeout=360s || exit 1; done
	kubectl rollout restart deploy/recsys-api
	kubectl rollout status deploy/recsys-api --timeout=180s
	@echo
	@echo "  features serve zeros until Redis is materialised: make kind-feast"

# What the backends' init containers pull: one directory per backend, so each
# mirrors only its own slice. SYMLINKS, not copies -- the publisher reads
# through them, and there is ~700MB here. Files, never directories: the
# publisher's rglob does not descend into a symlinked directory.
backend-bundle:  ## Stage the retrieval / Feast / Triton files kind-push publishes
	@version=$$(cat data/index/CURRENT) \
	  && test -f "data/index/$$version/index.faiss" \
	  || { echo "  data/index/CURRENT names no index; run: make index"; exit 1; }
	@for f in $(BACKEND_INPUTS); do \
	  test -f "$$f" || { echo "  missing $$f"; exit 1; }; \
	done
	rm -rf $(BACKEND_BUNDLE)
	@version=$$(cat data/index/CURRENT); b=$(BACKEND_BUNDLE); \
	  mkdir -p $$b/retrieval/index/$$version $$b/retrieval/gold/item_content \
	           $$b/features $$b/triton/ranker/1 && \
	  ln -s $(CURDIR)/data/index/CURRENT                $$b/retrieval/index/CURRENT && \
	  ln -s $(CURDIR)/data/index/$$version/index.faiss  $$b/retrieval/index/$$version/index.faiss && \
	  ln -s $(CURDIR)/$(INDEX_CHECKPOINT)               $$b/retrieval/tower.pt && \
	  ln -s $(CURDIR)/data/gold/item_content/part-00000.parquet \
	                                                    $$b/retrieval/gold/item_content/part-00000.parquet && \
	  ln -s $(CURDIR)/$(FEAST_HOST_REGISTRY)            $$b/features/registry.db && \
	  ln -s $(CURDIR)/$(ITEM_MAP)                       $$b/features/item_map.json && \
	  ln -s $(CURDIR)/$(CATEGORY_MAP)                   $$b/features/category_map.json && \
	  ln -s $(CURDIR)/serving/triton/ranker/1/model.onnx      $$b/triton/ranker/1/model.onnx && \
	  ln -s $(CURDIR)/serving/triton/ranker/1/model.onnx.data $$b/triton/ranker/1/model.onnx.data
	@# The ONE difference from compose's model config. Rewritten, not kept as a
	@# second file, and refused if the rewrite matched nothing -- a config that
	@# still says KIND_GPU would fail to load on a GPU-less node.
	@grep -q 'KIND_GPU' serving/triton/ranker/config.pbtxt \
	  || { echo "  config.pbtxt has no KIND_GPU to rewrite; check triton.yaml's assumption"; exit 1; }
	sed 's/KIND_GPU/KIND_CPU/' serving/triton/ranker/config.pbtxt > $(BACKEND_BUNDLE)/triton/ranker/config.pbtxt
	@find -L $(BACKEND_BUNDLE) -type f -printf '  %8k KB  %P\n'

# Feast's online rows, written from the HOST into the cluster's Redis through a
# forward. From the host because materialisation reads the gold feature tables
# from the compose stack's object store, which the cluster deliberately cannot
# reach. Against a COPY of the registry: materialize records its intervals in
# the registry, and the host's own one belongs to the compose stack.
kind-feast: kind-guard  ## Materialise Feast into the cluster's Redis (needs the gold tables)
	@$(MAKE) --no-print-directory feast-gold-check
	@kubectl rollout status deploy/recsys-redis --timeout=120s
	@mkdir -p build && cp $(FEAST_HOST_REGISTRY) build/kind-registry.db
	@kubectl port-forward svc/recsys-redis $(REDIS_FORWARD_PORT):6379 >/dev/null & \
	  forward=$$!; trap 'kill $$forward 2>/dev/null' EXIT; \
	  for _ in $$(seq 40); do \
	    bash -c 'echo > /dev/tcp/127.0.0.1/$(REDIS_FORWARD_PORT)' 2>/dev/null && break; sleep 0.5; \
	  done; \
	  REDIS_URL=localhost:$(REDIS_FORWARD_PORT) FEAST_REGISTRY=$(CURDIR)/build/kind-registry.db \
	    uv run feast -c $(FEAST_REPO) materialize $(FEAST_START) $(FEAST_END) \
	      --views item_stats --views user_stats --views user_category_stats && \
	  keys=$$(uv run python -c "import redis; print(redis.Redis(port=$(REDIS_FORWARD_PORT), db=0).dbsize())") && \
	  echo "  Feast rows in the cluster's Redis: $$keys" && \
	  test "$$keys" -gt 0 \
	    || { echo "  materialize wrote nothing; check FEAST_START/FEAST_END against the data"; exit 1; }
	@# The gateway caches item MISSES for its TTL, so a restart makes the new
	@# rows visible at once rather than within a minute.
	kubectl rollout restart deploy/recsys-features
	kubectl rollout status deploy/recsys-features --timeout=240s

# The end-to-end proof: real slates, not just Ready pods. Ready means "can
# serve a slate" and holds on the popularity fallback too, so it proves nothing
# about the backends. Nor does "not the fallback": trending, co-visitation and
# content still fill a slate with the two-tower retriever DOWN, so a real slate
# here also means NO degraded source. Passes on 3 CONSECUTIVE ones, as the first
# request on each fresh connection pays setup and can legitimately fall back.
kind-smoke: kind-guard  ## Prove the cluster serves real (non-fallback) slates
	@kubectl rollout status deploy/recsys-api --timeout=180s >/dev/null
	@kubectl port-forward svc/recsys-api $(SMOKE_FORWARD_PORT):15051 >/dev/null & \
	  forward=$$!; trap 'kill $$forward 2>/dev/null' EXIT; \
	  ok=0; last=""; \
	  for i in $$(seq $(SMOKE_REQUESTS)); do \
	    last=$$(grpcurl -plaintext -import-path serving/proto -proto recsys.proto \
	      -d "{\"user_id\": \"U$$i\", \"surface\": \"home_rail\", \"num_results\": 10}" \
	      localhost:$(SMOKE_FORWARD_PORT) recsys.v1.Recommender/GetRecommendations 2>&1); \
	    if printf '%s' "$$last" | python3 -c 'import json,sys; r=json.load(sys.stdin); \
	        sys.exit(0 if not r.get("usedFallback") and not r.get("degradedSources") \
	                 and r.get("stageCounts",{}).get("retrieved",0) > 0 else 1)' \
	        2>/dev/null; then ok=$$((ok+1)); else ok=0; fi; \
	    test $$ok -ge 3 && break; \
	    sleep 0.5; \
	  done; \
	  if test $$ok -ge 3; then \
	    printf '%s' "$$last" | python3 -c 'import json,sys; r=json.load(sys.stdin); \
	      print("  smoke ok: %d items, stages %s" % (len(r["items"]), r.get("stageLatencyMs")))'; \
	  else \
	    echo "  no run of 3 real slates in $(SMOKE_REQUESTS) requests; the last response:"; \
	    printf '%s\n' "$$last" | head -40; \
	    echo "  degradedSources names what failed: two_tower -> recsys-retrieval,"; \
	    echo "  features -> recsys-features, ranker -> recsys-triton."; \
	    exit 1; \
	  fi

kind-status:  ## Pods, and whether readiness put them behind the Service
	@kubectl get pods -o wide
	@echo
	@kubectl get endpointslice
	@echo
	@echo "  no addresses means readiness is failing; addresses mean traffic flows"

# Destroys the PVC with it, so the published artifacts go too. They are
# re-publishable from disk, which is why this is not treated as precious --
# unlike the MinIO and MLflow volumes under docker compose.
kind-down:  ## Delete the kind cluster
	kind delete cluster --name $(KIND_CLUSTER)
