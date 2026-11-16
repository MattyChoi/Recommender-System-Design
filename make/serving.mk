# Serving: the ONNX export Triton loads, the orchestrator, and the three things
# that run against it -- bench, demo and the ranked smoke test. All three need a
# stack that is already up; none of them starts one.

.PHONY: serve serve-all features-serve retrieval-serve server-build bench demo smoke-ranked

RANKER_COLUMNS ?= $(ONNX_OUT:.onnx=.columns.json)

FEATURES_ARGS ?=
RETRIEVAL_ARGS ?= --artifacts data/index --pointer data/index/CURRENT
SERVE_ARGS ?= -content-stem ../../$(CONTENT_STEM)

# The collector, not Tempo: traces must pass the tail sampler. Defaulted ON
# because -otlp's own default is EMPTY, which disables tracing silently -- a
# healthy Tempo with a provisioned datasource and no spans in it looks like a
# broken backend and is actually an unset flag.
#
# Safe when nothing is listening: otlptracegrpc does not dial at construction,
# so a missing collector costs dropped spans and retry logs, never a failed
# start. `make serve OTLP=` turns tracing off again.
#
# Head ratio stays at the binary's 1.0 default on purpose. Cutting here would
# throw traces away before infra/docker/otel-collector.yaml's policies could
# ask whether they were the degraded ones worth keeping.
OTLP ?= localhost:4317

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

MODEL_VERSION ?= $(shell cat $(ONNX_OUT) $(ONNX_OUT).data 2>/dev/null | sha256sum | cut -c1-12)
INDEX_VERSION ?= $(shell cat data/index/CURRENT 2>/dev/null)

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
	@test -n "$(INDEX_VERSION)"  || \
	  echo '  note: no data/index/CURRENT -- index_version will be empty in responses'
	@test -f "$(INDEX_STEM).json" || \
	  echo '  note: no $(INDEX_STEM).json -- exact search fallback disabled (run: make index-vectors)'
	@test -z "$(OTLP)" || docker inspect -f '{{.State.Running}}' recsys-otel 2>/dev/null | grep -q true || \
	  echo '  note: recsys-otel is not running -- spans will be dropped (run: make up)'
	cd $(GO_DIR) && ./$(SERVER_BIN) \
	    -otlp '$(OTLP)' \
	    -item-map ../../$(ITEM_MAP) -catalogue ../../$(CATALOGUE) \
	    -popular ../../$(POPULAR) -columns ../../$(RANKER_COLUMNS) \
	    -index-stem ../../$(INDEX_STEM) \
	    -model-version '$(MODEL_VERSION)' -index-version '$(INDEX_VERSION)' \
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

# Needs `make serve-all` running, and a GPU for Triton -- which is why this is a
# make target and not a job in .github/workflows/ci.yml. Its mirror,
# tests/test_serving_smoke.py, runs hosted because it needs no backends at all.
smoke-ranked:  ## Assert the full pipeline against a running stack (make serve-all first)
	RECSYS_SMOKE_TARGET=$(BENCH_TARGET) \
	  uv run pytest -m integration tests/test_serving_ranked.py -q

BENCH_ARGS ?=
bench:  ## Sweep the orchestrator to saturation with ghz
	uv run python -m scripts.writeups.bench_serving --target $(BENCH_TARGET) $(BENCH_ARGS)

DEMO_USER ?= U13740
DEMO_ARGS ?=

# RECSYS_STORAGE__BACKEND=local on purpose: the gold layer is read from disk, so
# this needs no MinIO. It does need the ONLINE stack -- Redis behind the feature
# gateway, Triton behind the ranker -- which is what the container guard is for.
# "Needs no container" was never true of this target; it is true of the test
# fixtures, which build their own artifacts and expect every backend to be absent.
#
# Guards rather than prerequisites, deliberately. `artifacts` re-dumps all seven
# outputs including the ONNX export, and `data` would reach for the gated
# download -- so a demo that depended on them would rebuild the world, or fail
# deep inside a target nobody asked for. Each guard names the one command.
demo: export RECSYS_STORAGE__BACKEND = local
demo: server-build  ## One request through the full path, printed (make up && make serve-all first)
	@test -f "$(ITEM_MAP)"       || { echo 'run: make artifacts'; exit 1; }
	@test -f "$(CATALOGUE)"      || { echo 'run: make artifacts'; exit 1; }
	@test -f "$(POPULAR)"        || { echo 'run: make artifacts'; exit 1; }
	@test -f "$(RANKER_COLUMNS)" || { echo 'run: make onnx';      exit 1; }
	@docker inspect -f '{{.State.Running}}' recsys-triton 2>/dev/null | grep -q true \
	  || { echo '  recsys-triton is not running -- run: make up'; exit 1; }
	uv run python -m scripts.demo \
	    --target $(BENCH_TARGET) --user $(DEMO_USER) $(DEMO_ARGS)
