"""The FAISS + user-tower sidecar, on :50052."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from serving.processes import supervise
from serving.retrieval.cache import DEFAULT_TTL, REDIS_URL
from serving.retrieval.server import serve

load_dotenv()
DEFAULT_PORT = int(os.getenv("RETRIEVAL_PORT", 50052))
DEFAULT_PROCESSES = int(os.getenv("RETRIEVAL_PROCESSES", 4))
DEFAULT_WORKERS = 8

DEFAULT_EF_SEARCH = 512

# MUST be the checkpoint the promoted index was built from -- see the print
# below, and the same warning on INDEX_CHECKPOINT in make/serving_artifacts.mk.
DEFAULT_CHECKPOINT = Path("data/checkpoints/both-logq-n4u0-b8192e10lr0.001-ab7e1d500b9bf792.pt")


def main() -> int:
    parser = argparse.ArgumentParser(description="The FAISS + user-tower sidecar.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="threads per process")
    parser.add_argument(
        "--processes",
        type=int,
        default=DEFAULT_PROCESSES,
        help="server processes, on --port, --port+10, ... Give the orchestrator every address",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--artifacts", type=Path, default=Path("data/index"))
    parser.add_argument("--pointer", type=Path, default=Path("data/index/CURRENT"))
    parser.add_argument("--ef-search", type=int, default=DEFAULT_EF_SEARCH)
    parser.add_argument(
        "--redis-url",
        default=REDIS_URL,
        help="User-embedding cache. Empty string disables it.",
    )
    parser.add_argument("--cache-ttl", type=int, default=DEFAULT_TTL)
    args = parser.parse_args()

    def serve_args(port: int) -> tuple[int, int, Path, Path, Path, int, str, int]:
        return (
            port,
            args.workers,
            args.checkpoint,
            args.artifacts,
            args.pointer,
            args.ef_search,
            args.redis_url,
            args.cache_ttl,
        )

    if args.processes <= 1:
        serve(args)
        return 0
    return supervise(
        "retrieval",
        serve,
        lambda port: (argparse.Namespace(**{**vars(args), "port": port}),),
        args.port,
        args.processes,
    )


if __name__ == "__main__":
    raise SystemExit(main())
