"""``make features-serve`` -- run the feature gateway.

    uv run python -m serving.features

Everything that can be wrong lives in service.py, where it is testable without
a socket or a Redis. This file opens a port.
"""

from __future__ import annotations

import argparse
import signal
from concurrent import futures
from pathlib import Path

from common.pb import features_pb2_grpc
from serving.features import cache as cache_module
from serving.features.cache import ItemFeatureCache
from serving.features.service import FEATURE_REPO, FeaturesServicer, load, warm

DEFAULT_PORT = 50053

#: Feast's online read is I/O to Redis, so handlers spend their time waiting
#: rather than holding the GIL. Higher than the retrieval sidecar's pool, which
#: is bounded by CPU-bound searches.
DEFAULT_WORKERS = 32


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="The feature gateway (Feast behind gRPC).")
    parser.add_argument("--repo", type=Path, default=FEATURE_REPO)
    parser.add_argument("--item-map", type=Path, default=Path("serving/artifacts/item_map.json"))
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    # Warming costs one round trip per view at startup and saves 60-120 ms on
    # the first real request. Off is available for a fast restart during
    # debugging; it is not a performance choice.
    parser.add_argument("--no-warm", action="store_true", help="skip the startup priming reads")
    # Items only -- see service.py. The TTL trades staleness for GIL time, and
    # the store is already an hour behind, so a minute is small against what
    # is already accepted.
    parser.add_argument("--item-cache-ttl", type=float, default=cache_module.DEFAULT_TTL)
    parser.add_argument("--item-cache-size", type=int, default=cache_module.DEFAULT_CAPACITY)
    parser.add_argument(
        "--no-item-cache",
        action="store_true",
        help="serve every GetItems row from Feast. The A/B arm for the cache's own benchmark",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    import grpc

    args = build_parser().parse_args(argv)
    loaded = load(repo=args.repo, item_map=args.item_map)

    # BEFORE the port opens, so the process is not reachable until it can
    # answer at its warm cost. Opening first and warming after would leave a
    # window where the orchestrator's 8 ms feature budget is unmeetable, and it
    # would report that as a degraded user rather than as a starting gateway.
    warmed = 0.0 if args.no_warm else warm(loaded)

    cache = (
        None
        if args.no_item_cache
        else ItemFeatureCache(ttl=args.item_cache_ttl, capacity=args.item_cache_size)
    )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.workers))
    features_pb2_grpc.add_FeaturesServicer_to_server(  # type: ignore[no-untyped-call]  # generated, untyped
        FeaturesServicer(loaded, cache), server
    )
    server.add_insecure_port(f"[::]:{args.port}")
    server.start()

    print(
        f"feature gateway on :{args.port} · item map {loaded.items.version or '(unversioned)'} "
        f"· {len(loaded.items.mapping):,} items · "
        + ("not warmed" if args.no_warm else f"warmed in {warmed * 1000:.0f}ms")
        + (
            " · item cache OFF"
            if cache is None
            else f" · item cache {cache.ttl:.0f}s ttl, {cache.capacity:,} rows"
        )
    )

    def stop(*_: object) -> None:
        server.stop(grace=5).wait()

    signal.signal(signal.SIGTERM, stop)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
