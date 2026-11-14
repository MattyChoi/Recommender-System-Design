# FAISS index for ANN

.PHONY: benchmark-index index-promote

INDEX_CHECKPOINT ?= data/checkpoints/both-logq-n4u0-b8192e10lr0.001-ab7e1d500b9bf792.pt
INDEX_ROOT ?= data/index
INDEX_KIND ?= hnsw

BENCHMARK_INDEX_ARGS ?=
PROMOTE_ARGS ?=

benchmark-index:  ## Recall/QPS curve: HNSW and IVF-PQ against exact search
	@test -f "$(INDEX_CHECKPOINT)" || \
	  { echo "set INDEX_CHECKPOINT=data/checkpoints/<run>.pt"; exit 1; }
	uv run python -m indexing.benchmark_index $(INDEX_CHECKPOINT) $(INDEX_ARGS)

index-promote:  ## Build, gate and promote an index into INDEX_ROOT
	@test -f "$(INDEX_CHECKPOINT)" || \
	  { echo "set INDEX_CHECKPOINT=data/checkpoints/<run>.pt"; exit 1; }
	uv run python -m indexing.pipeline \
	    --checkpoint $(INDEX_CHECKPOINT) --artifacts $(INDEX_ROOT) \
	    --kind $(INDEX_KIND) $(PROMOTE_ARGS)
