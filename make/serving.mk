# Serving: the ONNX export Triton loads, the orchestrator, and the two things
# that have to work from a clean clone.

.PHONY: onnx serve bench demo

# Must be a rank-neural checkpoint
RANKER_CHECKPOINT ?= data/checkpoints/ranking/mmoe-two_tower+trending+covisit+content-c100-4c80d49d59df4f4a-7fb9c9cfebba.pt
ONNX_OUT ?= serving/triton/ranker/1/model.onnx

# `--extra serving`: onnx, onnxscript and onnxruntime live there. Listing the
# other extras too, because `uv run --extra X` resolves only X
onnx:  ## Export the serving ranker to ONNX, preprocessing baked into the graph
	@test -f "$(RANKER_CHECKPOINT)" || \
	  { echo "set RANKER_CHECKPOINT=data/checkpoints/ranking/<mmoe run>.pt"; exit 1; }
	uv run --extra serving --extra sharded --extra orchestration \
	    python -m models.export.ranker_onnx $(RANKER_CHECKPOINT) --out $(ONNX_OUT)

serve:  ## Run the gRPC orchestrator
	@echo "TODO: serving"; exit 1

bench:  ## Latency + recall/QPS benchmarks
	@echo "TODO: benchmarks"; exit 1

# Runs against the LOCAL gold layer on purpose.
demo: export RECSYS_STORAGE__BACKEND = local
demo:  ## End-to-end demo; must work from a clean clone, with nothing running
	@echo "TODO: demo"; exit 1
