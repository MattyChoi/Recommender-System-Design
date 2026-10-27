from __future__ import annotations

import argparse
import os
import signal
from concurrent import futures
from types import FrameType

import grpc
from dotenv import load_dotenv

from common.pb import features_pb2_grpc
from serving.features.cache import ItemFeatureCache, DEFAULT_CAPACITY, DEFAULT_TTL
from serving.features.service import FeaturesServicer
from serving.features.store import Store

load_dotenv()

DEFAULT_PORT = int(os.getenv("FEATURES_PORT", 50053))
DEFAULT_WORKERS = 8
GRACE_SECONDS = 10.0


def serve(args: argparse.Namespace) -> None:
    cache = (
        None
        if args.no_item_cache
        else ItemFeatureCache(ttl=args.item_cache_ttl, capacity=args.item_cache_size)
    )
    store = Store(cache=cache)
    warmed = 0.0 if args.no_warm else store.warm()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.workers))
    features_pb2_grpc.add_FeaturesServicer_to_server(FeaturesServicer(store), server)
    server.add_insecure_port(f"[::]:{args.port}")
    
    print(
        f"feature gateway on :{args.port} · item map {store.item_map_version or '(unversioned)'} "
        f"· {len(store._to_id):,} items · "
        + ("not warmed" if args.no_warm else f"warmed in {warmed * 1000:.0f}ms")
        + (
            " · item cache OFF"
            if cache is None
            else f" · item cache {cache.ttl:.0f}s ttl, {cache.capacity:,} rows"
        )
    )

    def drain(signum: int, frame: FrameType | None) -> None:
        print("features gateway draining")
        server.stop(GRACE_SECONDS)

    signal.signal(signal.SIGTERM, drain)
    signal.signal(signal.SIGINT, drain)

    server.start()
    server.wait_for_termination()


def main() -> int:
    parser = argparse.ArgumentParser(description="The Feast gateway sidecar.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--no-warm", action="store_true", help="skip the startup priming reads")
    parser.add_argument("--item-cache-ttl", type=float, default=DEFAULT_TTL)
    parser.add_argument("--item-cache-size", type=int, default=DEFAULT_CAPACITY)
    parser.add_argument(
        "--no-item-cache",
        action="store_true",
        help="serve every GetItems row from Feast. The A/B arm for the cache's own benchmark",
    )
    args = parser.parse_args()
    serve(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
