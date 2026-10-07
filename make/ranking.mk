# Ranking: the GBDT bar, the neural contenders that have to clear it, and the
# distributed driver.

.PHONY: rank rank-neural rank-dist

# Trained on the candidates retrieval actually returns, never on random
# negatives: a ranker fitted on easy negatives and served real ones degrades
# while every offline metric still looks healthy.
RANK_ARGS ?=
rank:  ## Fit the GBDT ranker over retrieved candidates and score the funnel
	@test -f "$(SOURCE_CHECKPOINT)" || \
	  { echo "set SOURCE_CHECKPOINT=data/checkpoints/<run>.pt"; exit 1; }
	uv run python -m models.ranking.baselines.train $(SOURCE_CHECKPOINT) $(RANK_ARGS)

# The contender. `rank` above is the bar it has to clear.
#   make rank-neural RANK_ARGS="--model mmoe --epochs 5"
rank-neural:  ## Fit DCN v2 or MMoE over the same candidates
	@test -f "$(SOURCE_CHECKPOINT)" || \
	  { echo "set SOURCE_CHECKPOINT=data/checkpoints/<run>.pt"; exit 1; }
	$(DOTENV) $(PYTHON) -m models.ranking.train $(SOURCE_CHECKPOINT) $(RANK_ARGS)

# Ray under `uv run` re-resolves the environment inside each worker and has
# dropped the sharded extras before; the driver runs from .venv directly for
# the same reason retrieval-dist does.
#   make rank-dist RANK_ARGS="--model dcn --workers 2 --epochs 2"
rank-dist: PYTHON = .venv/bin/python
rank-dist: DOTENV = $(SOURCE_DOTENV)
rank-dist: rank-neural
