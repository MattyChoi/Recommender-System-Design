# All artifacts that are dumped to disk for the Go server to load at startup.

.PHONY: catalogue item-map popular index-vectors artifacts onnx

ITEM_MAP ?= serving/artifacts/item_map.json

# sidecar is unreachable. MUST come from the same checkpoint the promoted index
# was built from, or it searches a space the cached queries are not in.
INDEX_CHECKPOINT ?= data/checkpoints/both-logq-n4u0-b8192e10lr0.001-ab7e1d500b9bf792.pt

# Must be a rank-neural checkpoint
RANKER_CHECKPOINT ?= data/checkpoints/ranking/mmoe-two_tower+trending+covisit+content-c100-4c80d49d59df4f4a-7fb9c9cfebba.pt
ONNX_OUT ?= serving/triton/ranker/1/model.onnx

catalogue:  ## Dump per-item category indices for the Go server
	uv run python -m scripts.artifacts.dump_catalogue --out serving/artifacts/catalogue.json

item-map:  ## Dump bronze item_map to JSON for the Go server
	uv run python -m scripts.artifacts.dump_item_map --out $(ITEM_MAP)
	
popular:  ## Dump the popularity fallback slate
	@test -f "$(ITEM_MAP)" || { echo 'run: make item-map'; exit 1; }
	uv run python -m scripts.artifacts.dump_popular --item-map $(ITEM_MAP) --out serving/artifacts/popular.json

index-vectors:  ## Dump item embeddings for in-process exact search
	@test -f "$(INDEX_CHECKPOINT)" || \
	  { echo 'set INDEX_CHECKPOINT=data/checkpoints/<run>.pt'; exit 1; }
	uv run python -m scripts.artifacts.dump_index_vectors $(INDEX_CHECKPOINT) --stem serving/artifacts/items

# `--extra serving`: onnx, onnxscript and onnxruntime live there. Listing the
# other extras too, because `uv run --extra X` resolves only X
onnx:  ## Export the serving ranker to ONNX, preprocessing baked into the graph
	@test -f "$(RANKER_CHECKPOINT)" || \
	  { echo "set RANKER_CHECKPOINT=data/checkpoints/ranking/<mmoe run>.pt"; exit 1; }
	uv run --extra serving --extra sharded --extra orchestration \
	    python -m models.export.ranker_onnx $(RANKER_CHECKPOINT) --out $(ONNX_OUT)

artifacts: catalogue item-map popular index-vectors onnx  ## Dump all artifacts for the Go server