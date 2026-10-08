# Serving: the ONNX export Triton loads, the orchestrator, and the two things
# that have to work from a clean clone.

.PHONY: serve bench demo

serve:  ## Run the gRPC orchestrator
	@echo "TODO: serving"; exit 1

bench:  ## Latency + recall/QPS benchmarks
	@echo "TODO: benchmarks"; exit 1

# Runs against the LOCAL gold layer on purpose.
demo: export RECSYS_STORAGE__BACKEND = local
demo:  ## End-to-end demo; must work from a clean clone, with nothing running
	@echo "TODO: demo"; exit 1
