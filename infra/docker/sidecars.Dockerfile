FROM python:3.13-slim AS build

# uv from its own distroless image: no curl, no install script, no network
# fetch of a shell script at build time.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /repo

# Manifest and lockfile FIRST, so this layer invalidates when the dependency
# set changes and not on every source edit.
# EXPORT, FILTER, then PIP
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv export --frozen --no-dev --extra serving \
        --no-emit-project --no-hashes -o /tmp/requirements.txt \
 && echo "  dropping $(grep -cE '^(nvidia-|triton)' /tmp/requirements.txt) CUDA-only requirement lines" \
 && grep -vE '^(nvidia-|triton)' /tmp/requirements.txt > /tmp/requirements.cpu.txt \
 && uv venv /repo/.venv \
 && UV_TORCH_BACKEND=cpu uv pip install \
        --python /repo/.venv/bin/python -r /tmp/requirements.cpu.txt

COPY common/ ./common/
COPY indexing/ ./indexing/
COPY models/ ./models/
COPY data_pipeline/ ./data_pipeline/
COPY conf/ ./conf/
COPY serving/__init__.py serving/processes.py serving/health.py ./serving/
COPY serving/features/ ./serving/features/
COPY serving/retrieval/ ./serving/retrieval/

ENV PATH="/repo/.venv/bin:$PATH" \
    PYTHONPATH=/repo \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# WORKDIR /repo, matching the orchestrator image, because every default path in
# both sidecars is REPO-RELATIVE: data/index, data/index/CURRENT,
# data/checkpoints/<ckpt>.pt, serving/artifacts/item_map.json. A mount at
# /repo/data/... therefore satisfies them with no flag.
WORKDIR /repo

# Not root. The slim base has no USER line, so without this both sidecars run
# as uid 0 with Redis and MinIO credentials in the environment.
RUN useradd --uid 65532 --no-create-home --shell /usr/sbin/nologin sidecar
USER 65532:65532

# retrieval 15052 + stride 10 per process; features 15053 + stride 10. Four
# processes each by default -- see serving/processes.py. Documentation only.
EXPOSE 15052 15062 15072 15082 15053 15063 15073 15083

# No ENTRYPOINT: compose supplies `python -m serving.features` or
# `python -m serving.retrieval`. One image, two services.
