# Offline evaluation: the baselines, the half-life sweep, the paired bootstrap
# and the table they all feed.
#
# Every run writes a JSON result card under evaluation/results/. docs/baselines.md
# is regenerated from those cards and never hand-edited.

.PHONY: eval sweep coverage compare baselines results quality-gate exposure simulate

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


# The promotion gate. Pure arithmetic over committed cards -- no Spark, no GPU,
# no corpus -- so it runs anywhere, CI included. Producing the CANDIDATE cards is
# the half that needs the corpus, and that is `make eval` and the retrieval and
# ranking evaluate targets. Exits non-zero on a refusal.
CANDIDATE_POINTER ?= evaluation/results/CANDIDATE.json

quality-gate:  ## Refuse a promotion that regresses quality (make quality-gate CANDIDATE_POINTER=...)
	@test -f $(CANDIDATE_POINTER) \
	  || { echo "  no $(CANDIDATE_POINTER): copy PRODUCTION.json and point it at the new cards"; exit 1; }
	uv run python -m evaluation.offline.gate --candidate $(CANDIDATE_POINTER)

# Exposure concentration over time
#   make consume TOPIC=impressions        N=200000 > /tmp/logged.jsonl
#   make consume TOPIC=impression_context N=20000  > /tmp/served.jsonl
#
# Either may be omitted; one curve alone is a weaker claim, not an error.
EXPOSURE_LOGGED ?= /tmp/logged.jsonl
EXPOSURE_SERVED ?= /tmp/served.jsonl
EXPOSURE_WINDOW ?= 5000

exposure:  ## Gini of impressions over items, two policies (make exposure EXPOSURE_LOGGED=... EXPOSURE_SERVED=...)
	$(MAKE) --no-print-directory consume TOPIC=impressions        N=200000 > $(EXPOSURE_LOGGED)
	$(MAKE) --no-print-directory consume TOPIC=impression_context N=20000 > $(EXPOSURE_SERVED)
	uv run python -m scripts.writeups.exposure --window $(EXPOSURE_WINDOW) \
	    $(if $(EXPOSURE_LOGGED),--logged $(EXPOSURE_LOGGED)) \
	    $(if $(EXPOSURE_SERVED),--served $(EXPOSURE_SERVED))


# Determining launch decision. Simulated responses, but a REAL decision rule: SRM
# first, guardrails second, the primary metric's interval third, and the MDE
# when it comes back flat. Calibrated on the served log, so the control arm
# draws from the items this policy actually reaches.
#
#   make consume TOPIC=impression_context N=20000 > /tmp/served.jsonl
#
# SESSIONS=0 asks the power calculation how long the experiment needs to be,
# which is the direction that question is usually asked from.
SIM_SERVED ?= /tmp/served.jsonl
SIM_SESSIONS ?= 0
SIM_LIFT ?= 0.10
SIM_TARGET_LIFT ?= 0.02
SIM_SEED ?= 0

simulate:  ## Simulated A/B test and launch recommendation -> docs/experiment.md
	@test -f "$(SIM_SERVED)" \
	  || { echo "  no $(SIM_SERVED) -- run: make consume TOPIC=impression_context N=20000 > $(SIM_SERVED)"; exit 1; }
	uv run python -m scripts.writeups.simulate_ab \
	    --served $(SIM_SERVED) --sessions $(SIM_SESSIONS) \
	    --relevance-lift $(SIM_LIFT) --target-lift $(SIM_TARGET_LIFT) --seed $(SIM_SEED)
