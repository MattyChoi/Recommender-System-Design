# ANN indexing: the recall/QPS curve that decides which index serving ships.

.PHONY: index

INDEX_CHECKPOINT ?= data/checkpoints/both-logq-n4u0-b8192e10lr0.001-ab7e1d500b9bf792.pt
INDEX_ARGS ?=
index:  ## Recall/QPS curve: HNSW and IVF-PQ against exact search
	@test -f "$(INDEX_CHECKPOINT)" || \
	  { echo "set INDEX_CHECKPOINT=data/checkpoints/<run>.pt"; exit 1; }
	uv run python -m indexing.benchmark_index $(INDEX_CHECKPOINT) $(INDEX_ARGS)
