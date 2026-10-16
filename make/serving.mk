# Serving: the ONNX export Triton loads, the orchestrator, and the two things
# that have to work from a clean clone.

.PHONY: serve bench demo

FEATURES_PORT ?= $(FEATURES_REDIS_PORT)
FEATURES_ARGS ?=

features-serve:  ## Run the Feast gateway sidecar on :50053
	uv run --extra serving python -m serving.features \
	    --port $(FEATURES_PORT) $(FEATURES_ARGS)

serve:  ## Run the gRPC orchestrator
	@echo "TODO: serving"; exit 1

bench:  ## Latency + recall/QPS benchmarks
	@echo "TODO: benchmarks"; exit 1

# Runs against the LOCAL gold layer on purpose.
demo: export RECSYS_STORAGE__BACKEND = local
demo:  ## End-to-end demo; must work from a clean clone, with nothing running
	@echo "TODO: demo"; exit 1
