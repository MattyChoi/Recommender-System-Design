# Serving: the ONNX export Triton loads, the orchestrator, and the two things
# that have to work from a clean clone.

.PHONY: serve bench demo

RANKER_COLUMNS ?= $(ONNX_OUT:.onnx=.columns.json)

FEATURES_ARGS ?=
RETRIEVAL_ARGS ?= --artifacts data/index --pointer data/index/CURRENT
SERVE_ARGS ?= -content-stem ../../$(CONTENT_STEM)

features-serve:  ## Run the Feast gateway sidecar on :50053
	uv run --extra serving python -m serving.features \
	    --port $(FEATURES_PORT) $(FEATURES_ARGS)

retrieval-serve:  ## Run the FAISS + user-tower sidecar on :50052
	uv run --extra serving python -m serving.retrieval \
	    --port $(RETRIEVAL_PORT) $(RETRIEVAL_ARGS)

serve:  ## Run the gRPC orchestrator on :50051
	@test -f "$(ITEM_MAP)"       || { echo 'run: make item-map';  exit 1; }
	@test -f "$(CATALOGUE)"      || { echo 'run: make catalogue'; exit 1; }
	@test -f "$(POPULAR)"        || { echo 'run: make popular';   exit 1; }
	@test -f "$(RANKER_COLUMNS)" || { echo 'run: make onnx';      exit 1; }
	@test -f "$(INDEX_STEM).json" || \
	  echo '  note: no $(INDEX_STEM).json -- exact search fallback disabled (run: make index-vectors)'
	cd $(GO_DIR) && $(GO) run ./cmd/server \
	    -item-map ../../$(ITEM_MAP) -catalogue ../../$(CATALOGUE) \
	    -popular ../../$(POPULAR) -columns ../../$(RANKER_COLUMNS) \
	    -index-stem ../../$(INDEX_STEM) $(SERVE_ARGS)

bench:  ## Latency + recall/QPS benchmarks
	@echo "TODO: benchmarks"; exit 1

# Runs against the LOCAL gold layer on purpose.
demo: export RECSYS_STORAGE__BACKEND = local
demo:  ## End-to-end demo; must work from a clean clone, with nothing running
	@echo "TODO: demo"; exit 1
