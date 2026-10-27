# Serving: the ONNX export Triton loads, the orchestrator, and the two things
# that have to work from a clean clone.

.PHONY: serve bench demo

RANKER_COLUMNS ?= $(ONNX_OUT:.onnx=.columns.json)

FEATURES_ARGS ?=
RETRIEVAL_ARGS ?= --artifacts data/index --pointer data/index/CURRENT
SERVE_ARGS ?= -content-stem ../../$(CONTENT_STEM)

# Processes per Python sidecar, each on its own port: base, base+10, ...
# (PORT_STRIDE in serving/processes.py). `serve` is handed the same list, so
# the orchestrator and the sidecars cannot disagree about where the copies are.
FEATURES_PROCESSES ?= 4
RETRIEVAL_PROCESSES ?= 4
PORT_STRIDE := 10
# $(call addresses,<base port>,<processes>) -> localhost:<base>,localhost:<base+10>,...
addresses = $(shell seq -s, -f 'localhost:%.0f' $(1) $(PORT_STRIDE) $$(( $(1) + $(PORT_STRIDE) * ($(2) - 1) )))

features-serve:  ## Run the Feast gateway sidecar on :50053 (+10 per extra process)
	uv run --extra serving python -m serving.features \
	    --port $(FEATURES_PORT) --processes $(FEATURES_PROCESSES) $(FEATURES_ARGS)

retrieval-serve:  ## Run the FAISS + user-tower sidecar on :50052 (+10 per extra process)
	uv run --extra serving python -m serving.retrieval \
	    --port $(RETRIEVAL_PORT) --processes $(RETRIEVAL_PROCESSES) $(RETRIEVAL_ARGS)

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
	    -index-stem ../../$(INDEX_STEM) \
	    -features $(call addresses,$(FEATURES_PORT),$(FEATURES_PROCESSES)) \
	    -retrieval $(call addresses,$(RETRIEVAL_PORT),$(RETRIEVAL_PROCESSES)) \
	    $(SERVE_ARGS)

BENCH_TARGET ?= localhost:50051
BENCH_ARGS ?=
bench:  ## Sweep the orchestrator to saturation with ghz
	uv run python -m scripts.writeups.bench_serving --target $(BENCH_TARGET) $(BENCH_ARGS)

# Runs against the LOCAL gold layer on purpose.
demo: export RECSYS_STORAGE__BACKEND = local
demo:  ## End-to-end demo; must work from a clean clone, with nothing running
	@echo "TODO: demo"; exit 1
