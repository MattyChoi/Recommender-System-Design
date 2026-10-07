# Top-level entry point. Everything here is either global (settings every
# fragment needs) or about the build itself; the actual targets live in make/*.mk,
# one fragment per stage of the system:
#
#   make/docker.mk     the local stack
#   make/proto.mk      protobuf stubs, both languages from one .proto
#   make/data.mk       raw -> bronze -> silver -> gold, Feast, parity
#   make/kafka.mk      the replay harness
#   make/eval.mk       offline evaluation and the baselines table
#   make/retrieval.mk  two-tower training, ablations, candidate sources
#   make/ranking.mk    GBDT and neural rankers
#   make/rerank.mk     MMR / caps / freshness / exploration, seen lists
#   make/index.mk      ANN index recall/QPS curve
#   make/serving.mk    ONNX export, orchestrator, demo
#   make/quality.mk    lint, types, tests
#
# A new stage gets a new fragment and one line in the include block at the
# bottom. Each fragment declares its OWN .PHONY and owns the variables only it
# reads; a variable shared by two fragments belongs here instead.

.PHONY: help torch-env

# Set explicitly: without it the default goal would be whichever target the
# first included fragment happens to define.
.DEFAULT_GOAL := help


-include .env
export UV_ENV_FILE=.env

# `uv run` for everything EXCEPT the Ray drivers -- see retrieval-dist.
PYTHON ?= uv run python
DOTENV ?=
SOURCE_DOTENV = set -a; [ -f .env ] && . ./.env; set +a;

# Read by sources, rank, rank-neural, headroom and rerank -- four fragments, so
# it lives here rather than in any one of them.
SOURCE_CHECKPOINT ?= data/checkpoints/both-logq-n4u0-b8192e10lr0.001-ab7e1d500b9bf792.pt

# -h, because the file list is now plural and grep would otherwise prefix every
# line with the fragment it came from. The list is spelled out instead of using
# $(MAKEFILE_LIST) so that .env, which `-include` adds to it, is never read here.
help:  ## Show this help
	@grep -hE '^[a-z-]+:.*?## ' Makefile $(wildcard make/*.mk) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# Prints torch version, selected device and whether autocast is on, so a metric
# in MLflow can always be traced back to the hardware that produced it. Not in
# a fragment: it reports on the machine, not on a stage of the pipeline.
torch-env:  ## Report the torch device this machine will train on
	uv run python -c "from common.torch_env import describe, select_device; \
	    print(describe(select_device()))"


include make/docker.mk
include make/proto.mk
include make/data.mk
include make/kafka.mk
include make/eval.mk
include make/retrieval.mk
include make/ranking.mk
include make/rerank.mk
include make/index.mk
include make/serving.mk
include make/quality.mk
