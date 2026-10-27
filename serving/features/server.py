"""One gateway process: build the Store, bind the port, serve until signalled.

Its own module, not `__main__`, so `--processes` can spawn it: see
serving/processes.py.
"""

from __future__ import annotations

import argparse
import os
import signal
from concurrent import futures
from types import FrameType

import grpc

from common.pb import features_pb2_grpc
from serving.features.cache import ItemFeatureCache
from serving.features.service import FeaturesServicer
from serving.features.store import Store
from serving.processes import EXCLUSIVE_PORT

GRACE_SECONDS = 10.0


def serve(args: argparse.Namespace) -> None:
    cache = (
        None
        if args.no_item_cache
        else ItemFeatureCache(ttl=args.item_cache_ttl, capacity=args.item_cache_size)
    )
    store = Store(cache=cache)
    warmed = 0.0 if args.no_warm else store.warm()

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=args.workers), options=EXCLUSIVE_PORT
    )
    features_pb2_grpc.add_FeaturesServicer_to_server(FeaturesServicer(store), server)
    server.add_insecure_port(f"[::]:{args.port}")

    print(
        f"feature gateway on :{args.port} (pid {os.getpid()}) "
        f"· item map {store.item_map_version or '(unversioned)'} "
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
