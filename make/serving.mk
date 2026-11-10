# Serving: the ONNX export Triton loads, the orchestrator, and the two things
# that have to work from a clean clone.

.PHONY: serve serve-all features-serve retrieval-serve server-build bench demo

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

# Where the orchestrator looks for each sidecar. localhost when all three run
# on the host; the compose SERVICE NAME when the sidecars are containerised and
# the orchestrator is on the same network (see make/docker.mk's image-run).
FEATURES_HOST ?= localhost
RETRIEVAL_HOST ?= localhost

# $(call addresses,<host>,<base port>,<processes>) -> <host>:<base>,<host>:<base+10>,...
addresses = $(shell seq -s, -f '$(1):%.0f' $(2) $(PORT_STRIDE) $$(( $(2) + $(PORT_STRIDE) * ($(3) - 1) )))

SERVER_BIN ?= bin/recsys-server

features-serve:  ## Run the Feast gateway sidecar on :15053 (+10 per extra process)
	uv run --extra serving python -m serving.features \
	    --port $(FEATURES_PORT) --processes $(FEATURES_PROCESSES) $(FEATURES_ARGS)

retrieval-serve:  ## Run the FAISS + user-tower sidecar on :15052 (+10 per extra process)
	uv run --extra serving python -m serving.retrieval \
	    --port $(RETRIEVAL_PORT) --processes $(RETRIEVAL_PROCESSES) $(RETRIEVAL_ARGS)

server-build:  ## Compile the orchestrator binary to serving/go/$(SERVER_BIN)
	cd $(GO_DIR) && $(GO) build -trimpath -o $(SERVER_BIN) ./cmd/server

# Run `make server-build` first
serve:  ## Run the gRPC orchestrator on :15051
	@test -f "$(ITEM_MAP)"       || { echo 'run: make item-map';  exit 1; }
	@test -f "$(CATALOGUE)"      || { echo 'run: make catalogue'; exit 1; }
	@test -f "$(POPULAR)"        || { echo 'run: make popular';   exit 1; }
	@test -f "$(RANKER_COLUMNS)" || { echo 'run: make onnx';      exit 1; }
	@test -f "$(INDEX_STEM).json" || \
	  echo '  note: no $(INDEX_STEM).json -- exact search fallback disabled (run: make index-vectors)'
	cd $(GO_DIR) && ./$(SERVER_BIN) \
	    -item-map ../../$(ITEM_MAP) -catalogue ../../$(CATALOGUE) \
	    -popular ../../$(POPULAR) -columns ../../$(RANKER_COLUMNS) \
	    -index-stem ../../$(INDEX_STEM) \
	    -features $(call addresses,$(FEATURES_HOST),$(FEATURES_PORT),$(FEATURES_PROCESSES)) \
	    -retrieval $(call addresses,$(RETRIEVAL_HOST),$(RETRIEVAL_PORT),$(RETRIEVAL_PROCESSES)) \
	    $(SERVE_ARGS)

serve-all: export PYTHONUNBUFFERED = 1
serve-all: server-build  ## Run both sidecars and the orchestrator together; Ctrl-C stops all three
	@trap 'trap - INT TERM EXIT; kill 0' INT TERM EXIT; \
	( $(MAKE) --no-print-directory features-serve  2>&1 | sed -u 's/^/[features]  /'; kill 0 ) & \
	( $(MAKE) --no-print-directory retrieval-serve 2>&1 | sed -u 's/^/[retrieval] /'; kill 0 ) & \
	( $(MAKE) --no-print-directory serve SERVE_ARGS='$(SERVE_ARGS) -warm-wait 60s' 2>&1 | sed -u 's/^/[serve]     /'; kill 0 ) & \
	wait

BENCH_TARGET ?= localhost:15051
BENCH_ARGS ?=
bench:  ## Sweep the orchestrator to saturation with ghz
	uv run python -m scripts.writeups.bench_serving --target $(BENCH_TARGET) $(BENCH_ARGS)

# Runs against the LOCAL gold layer on purpose.
demo: export RECSYS_STORAGE__BACKEND = local
demo:  ## End-to-end demo; must work from a clean clone, with nothing running
	@echo "TODO: demo"; exit 1
