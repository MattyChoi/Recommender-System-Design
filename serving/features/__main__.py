from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

from serving.features.cache import DEFAULT_CAPACITY, DEFAULT_TTL
from serving.features.server import serve
from serving.processes import supervise

load_dotenv()

DEFAULT_PORT = int(os.getenv("FEATURES_PORT", 50053))
DEFAULT_PROCESSES = int(os.getenv("FEATURES_PROCESSES", 4))
DEFAULT_WORKERS = 8


def main() -> int:
    parser = argparse.ArgumentParser(description="The Feast gateway sidecar.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="threads per process")
    parser.add_argument(
        "--processes",
        type=int,
        default=DEFAULT_PROCESSES,
        help="server processes, on --port, --port+10, ... Give the orchestrator every address",
    )
    parser.add_argument("--no-warm", action="store_true", help="skip the startup priming reads")
    parser.add_argument("--item-cache-ttl", type=float, default=DEFAULT_TTL)
    parser.add_argument("--item-cache-size", type=int, default=DEFAULT_CAPACITY)
    parser.add_argument(
        "--no-item-cache",
        action="store_true",
        help="serve every GetItems row from Feast. The A/B arm for the cache's own benchmark",
    )
    args = parser.parse_args()
    if args.processes <= 1:
        serve(args)
        return 0
    return supervise(
        "features",
        serve,
        lambda port: (argparse.Namespace(**{**vars(args), "port": port}),),
        args.port,
        args.processes,
    )


if __name__ == "__main__":
    raise SystemExit(main())
