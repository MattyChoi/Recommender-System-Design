# Retrieval: the two-tower, its ablations, the embedding-table sizing probes and
# the five candidate sources that feed ranking.

.PHONY: retrieval retrieval-dist ablation bands shard-plan hash-bench sources blend

### Two-tower training
EPOCHS ?= 10
BATCH ?= 8192
LR ?= 1e-3
TRAIN_ARGS ?=

# MLflow must be up (`make up`) unless RECSYS_MLFLOW__ENABLED=false. The arms:
#   make retrieval TRAIN_ARGS="--geometry-every 1"     # the first-epoch collapse
#   make retrieval TRAIN_ARGS="--lr 1e-4 --geometry-every 1"
#   make retrieval TRAIN_ARGS="--no-logq"              # the logQ gate
#   make retrieval TRAIN_ARGS="--no-use-content"       # ID only
#   make retrieval TRAIN_ARGS="--no-use-id"            # content only
#   make retrieval TRAIN_ARGS="--max-negs 0 --uniform-negs 0 --no-logq"   # in-batch only
#   make retrieval TRAIN_ARGS="--max-negs 0 --uniform-negs 0"             # + logQ
#   make retrieval TRAIN_ARGS="--max-negs 0 --uniform-negs 4"             # + uniform
#   make retrieval                                                        # slate negatives
# ex: `make retrieval TRAIN_ARGS="--batch-size 2048 --epochs 40 --patience 5"`
retrieval:  ## gold + item_content -> a trained two-tower, logged to MLflow
	$(DOTENV) $(PYTHON) -m models.retrieval.train --epochs $(EPOCHS) \
	    --batch-size $(BATCH) --lr $(LR) $(TRAIN_ARGS)

# Distributed training under Ray. Ray re-resolves the environment inside each
# worker and has dropped the sharded extras when launched through `uv run`, so
# the driver runs from .venv directly and sources .env itself.
#   make retrieval-dist TRAIN_ARGS="--workers 2 --limit-rows 20000 --epochs 1"
retrieval-dist: PYTHON = .venv/bin/python
retrieval-dist: DOTENV = $(SOURCE_DOTENV)
retrieval-dist: retrieval

# HOLDOUT must match the run that wrote CHECKPOINT
CHECKPOINT ?=
HOLDOUT ?= 12
BANDS_ARGS ?=
BASELINE_NPZ ?=
CANDIDATE_NPZ ?=
ablation:  ## two scored arms -> the paired per-band table and the plot
	@test -n "$(BASELINE_NPZ)" -a -n "$(CANDIDATE_NPZ)" || \
	  { echo "set BASELINE_NPZ= and CANDIDATE_NPZ= to two evaluation/results/retrieval/*.npz"; exit 1; }
	uv run python -m models.retrieval.ablation $(BASELINE_NPZ) $(CANDIDATE_NPZ)

# `make bands CHECKPOINT=data/checkpoints/both-logq-n4u0-0f6e03fc2c6e519e.pt`
bands:  ## a checkpoint -> Recall@k by item popularity band, the cold-start gate
	@test -n "$(CHECKPOINT)" || { echo "set CHECKPOINT=data/checkpoints/<run>.pt"; exit 1; }
	uv run python -m models.retrieval.evaluate $(CHECKPOINT) \
	    --holdout-hours $(HOLDOUT) $(BANDS_ARGS)

SHARD_DEVICE ?= cuda
SHARD_ARGS ?=
shard-plan:  ## Where an embedding table stops fitting on one device
	uv run python -m models.layers.sharding_report \
	    --compute-device $(SHARD_DEVICE) $(SHARD_ARGS)

# Needs NO torchrec: `hashing` imports `sizing`, not `sharded_embeddings`.
# COUNTS_NPZ may be ANY scored arm -- `train_counts` is a property of the split,
# not of the model
COUNTS_NPZ ?= evaluation/results/retrieval/both-logq-n4u0-b8192e10lr0.001-ab7e1d500b9bf792.npz
HASH_ARGS ?=
hash-bench:  ## Hashing-trick collisions, by three denominators
	@test -f "$(COUNTS_NPZ)" || \
	  { echo "set COUNTS_NPZ= to an evaluation/results/retrieval/*.npz"; exit 1; }
	uv run python -m models.layers.hashing --counts-npz $(COUNTS_NPZ) $(HASH_ARGS)

SOURCE_ARGS ?=
sources:  ## Five retrieval sources -> their top-100 per request
	@test -f "$(SOURCE_CHECKPOINT)" || \
	  { echo "set SOURCE_CHECKPOINT=data/checkpoints/<run>.pt"; exit 1; }
	uv run python -m models.retrieval.sources $(SOURCE_CHECKPOINT) $(SOURCE_ARGS)

BLEND_ARGS ?=
blend:  ## Union the sources, then leave-one-source-out
	uv run python -m models.retrieval.blend $(BLEND_ARGS)
