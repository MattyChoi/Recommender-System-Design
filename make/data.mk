# The medallion layers: raw -> bronze -> silver -> gold, plus the feature store
# and the two measurements that justify a threshold in common/config.py.
#
# Every layer target takes FORCE=1 to rebuild a layer that already exists.
# Without it they are no-ops, so `make data` on a warm tree is cheap.

.PHONY: raw bronze silver gold content feast gap parity data

MIND_SIZE ?= small
SPLITS ?= train dev
MAX_HISTORY ?= 50
MAX_NEGATIVES ?= 20
FORCE ?=
FORCE_ID_MAPS ?=
MIN_COUNT ?= 25

# The encoder ablation knob. Swap it and rebuild with FORCE=1;
ENCODER ?= BAAI/bge-base-en-v1.5
ENCODE_BATCH ?= 256

# Feast variables
# MIND dataset date range
FEAST_START ?= 2019-11-09T00:00:00
FEAST_END   ?= 2019-11-16T00:00:00
FEAST_REPO ?= data_pipeline/features/recsys_store/feature_repo

# Parity sample for make parity
PARITY_SAMPLE ?= 200

raw:  ## Fetch MIND into paths.raw (override: make download MIND_SIZE=large)
	uv run python -m data_pipeline.ingest.download_mind --size $(MIND_SIZE) --splits $(SPLITS) $(if $(FORCE),--force)

bronze:  ## raw -> bronze (rebuild an existing layer: make bronze FORCE=1)
	uv run python -m data_pipeline.ingest.bronze --splits $(SPLITS) $(if $(FORCE),--force) $(if $(FORCE_ID_MAPS),--force-id-maps)

silver:  ## bronze -> silver (rebuild an existing layer: make silver FORCE=1)
	uv run python -m data_pipeline.transform.silver --splits $(SPLITS) $(if $(FORCE),--force)

# Writes the three feature series to MinIO, so `make up` first. The label table
# stays on disk. `RECSYS_STORAGE__BACKEND=local make gold` puts everything back
# on disk and needs nothing running.
gold:  ## silver -> gold feature tables (rebuild an existing layer: make gold FORCE=1)
	uv run python -m data_pipeline.features.gold --splits $(SPLITS) \
	    --max-history $(MAX_HISTORY) --max-negatives $(MAX_NEGATIVES) $(if $(FORCE),--force)

# Writes to gold, but NOT part of `make gold`: this needs torch, a downloaded
# encoder and ideally a GPU, and `make gold` must stay runnable on a machine
# with none of them. Two variants are cached -- title only, matching what the
# content baseline tokenises, and title+abstract, which the model should use.
content:  ## catalogue -> gold/item_content, the frozen sentence vectors for the content tower
	uv run python -m models.retrieval.content_cache --splits $(SPLITS) \
	    --model $(ENCODER) --batch-size $(ENCODE_BATCH) \
	    --min-count $(MIN_COUNT) $(if $(FORCE),--force)

# Reads the series from MinIO, so `make up` first. Note that apply RESOLVES the
# source paths and bakes them into the registry: running this under a different
# backend than the one you built with leaves a registry pointing at data that is
# not there. `RECSYS_STORAGE__BACKEND=local make feast` for the on-disk build.
feast:  ## Register feature definitions and materialise them into Redis
	uv run feast -c $(FEAST_REPO) apply
	uv run feast -c $(FEAST_REPO) materialize $(FEAST_START) $(FEAST_END) \
	    --views item_stats --views user_stats --views user_category_stats

# Reads bronze, not silver: silver's session_id was cut with the very threshold
# this measures, so measuring there would let the old answer pick the new one.
gap:  ## Measure the inter-impression gap, to set session.gap_minutes
	uv run python -m data_pipeline.transform.session_gap --split train

# Compares what Feast materialised against what the gold series says it should
# hold. NOT the skew report -- both sides are offline reads; see docs/ for the
# distinction. Needs Redis and MinIO, so `make up && make feast` first.
parity:  ## Materialisation parity: Redis vs the gold series
	uv run python -m data_pipeline.features.parity --sample $(PARITY_SAMPLE)

data: raw bronze silver gold content ## Build every layer under data/
