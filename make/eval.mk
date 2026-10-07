# Offline evaluation: the baselines, the half-life sweep, the paired bootstrap
# and the table they all feed.
#
# Every run writes a JSON result card under evaluation/results/. docs/baselines.md
# is regenerated from those cards and never hand-edited.

.PHONY: eval sweep coverage compare baselines results

MODELS ?= random popularity recency content covisit als als_item
MODEL ?= random
EVAL_SPLIT ?= dev
HALF_LIVES ?= 0.02 0.05 0.1 0.25 1 3

# 24h, chosen from `make coverage`. 1h left 91% of dev slates with every
# candidate tied at 0.0; 24h is where slate reach plateaus.
COVISIT_MAX_GAP ?= 86400

BASELINE ?= recency
CANDIDATE ?= decayed_popularity@0.02
# Seconds: 1h, 6h, 24h, 72h. None of these is a default -- `make coverage`
# picks the one this corpus supports.
WINDOWS ?= 3600 21600 86400 259200

eval:  ## Score a model into evaluation/results/ (make eval MODEL=random)
	uv run python -m evaluation.offline.run_eval --model $(MODEL) --split $(EVAL_SPLIT) \
	    --max-gap-seconds $(COVISIT_MAX_GAP)

# One card per half-life, so the results directory holds the ablation. The curve is
# the deliverable: a lone tuned half-life reads as a number someone picked.
sweep:  ## Half-life curve for decayed popularity (make sweep HALF_LIVES="0.5 1 3 7")
	uv run python -m evaluation.offline.run_eval_sweep \
	    --split $(EVAL_SPLIT) --half-lives $(HALF_LIVES)

# Reach, not quality, and no cards written. A model that scores nothing and a
# model that ranks badly produce near-identical cards; this tells them apart
# cheaply, before spending eval runs on windows that cannot move the metric.
coverage:
	uv run python -m evaluation.offline.covisit_coverage \
	    --split $(EVAL_SPLIT) --windows $(WINDOWS)

compare:  ## Paired bootstrap between two models (make compare BASELINE=recency CANDIDATE=decayed_popularity@0.02)
	uv run python -m evaluation.offline.compare \
	    --baseline $(BASELINE) --candidate $(CANDIDATE) --split $(EVAL_SPLIT)

baselines:  ## Re-score every baseline and rebuild docs/baselines.md from one commit
	@for model in $(MODELS); do \
	    $(MAKE) --no-print-directory eval MODEL=$$model || exit 1; \
	done
	$(MAKE) --no-print-directory sweep HALF_LIVES="0.02 0.05 0.1 0.25"
	$(MAKE) --no-print-directory sweep HALF_LIVES="0.5 1 2 3"
	$(MAKE) --no-print-directory results

# Rebuilt from the cards, never hand-edited: a table that disagrees with the
# JSON it quotes is worse than no table. Pure stdlib, so no cluster is needed.
results:  ## Rebuild docs/baselines.md from evaluation/results/*.json
	uv run python -m evaluation.offline.results_table
