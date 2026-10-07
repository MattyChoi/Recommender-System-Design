# Re-ranking: the policy layer over the ranker's output, plus the two probes
# that keep its numbers honest -- the diversity ceiling and the seen-list cost.

.PHONY: headroom seen-bench rerank

# RANKER must be a booster saved by `make rank` -- every arm of the trade-off
# table shares ONE fitted ranker, or the table measures model variance instead
# of policy.
RANKER ?= data/checkpoints/ranking/lgbm-two_tower+trending+covisit+content-c100-a8c6848432dd3e70-a7e02e5b3ab0.txt
RERANK_ARGS ?= --names two_tower trending covisit content --max-candidates 100
SEEN_ARGS ?=

# Run this BEFORE believing any diversity number: it reports the ceiling a
# re-ranker could reach, which is the candidate pool and not the catalogue.
headroom:
	@test -f "$(SOURCE_CHECKPOINT)" || \
	  { echo "set SOURCE_CHECKPOINT=data/checkpoints/<run>.pt"; exit 1; }
	uv run python -m models.reranking.headroom $(SOURCE_CHECKPOINT) $(RERANK_ARGS)

seen-bench:  ## Bloom-filter seen-list: memory against exact storage, and the error rate
	uv run python -m models.reranking.seen_bench $(SEEN_ARGS)

# The seen-filter arm needs Redis, so it is opt-in rather than part of `rerank`:
#   make rerank RERANK_ARGS="... --seen"
rerank:  ## MMR / caps / freshness / exploration -> the relevance-diversity table
	@test -f "$(SOURCE_CHECKPOINT)" || \
	  { echo "set SOURCE_CHECKPOINT=data/checkpoints/<run>.pt"; exit 1; }
	@test -f "$(RANKER)" || { echo "set RANKER= to a booster from 'make rank'"; exit 1; }
	uv run python -m models.reranking.evaluate $(SOURCE_CHECKPOINT) \
	    --ranker $(RANKER) $(RERANK_ARGS)
