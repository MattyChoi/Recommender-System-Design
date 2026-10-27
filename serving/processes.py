"""Several copies of one gRPC sidecar, each on its OWN port.

A Python sidecar is GIL-bound long before the machine is, so more threads do
not help and more processes do. Copy ``i`` binds ``port + PORT_STRIDE * i``
and the orchestrator is given every address (``-retrieval a,b,c``), opening
the same number of connections to each and round-robining over them -- an
exact split.
"""

from __future__ import annotations

import multiprocessing
import os
import signal
from collections.abc import Callable
from types import FrameType
from typing import Any

# OFF explicitly: gRPC Python turns SO_REUSEPORT ON by default on Linux, which
# is exactly what lets a forgotten copy share a port without an error.
EXCLUSIVE_PORT = [("grpc.so_reuseport", 0)]

# 10, not 1: the sidecars' base ports are adjacent (retrieval 50052, features
# 50053), so consecutive ports would collide
PORT_STRIDE = 10


def ports(base: int, processes: int) -> list[int]:
    """The ports ``processes`` copies bind, in order. Copy 0 keeps ``base``."""
    return [base + PORT_STRIDE * index for index in range(max(processes, 1))]


def supervise(
    name: str,
    target: Callable[..., None],
    args_for: Callable[[int], tuple[Any, ...]],
    base: int,
    processes: int,
) -> int:
    """Run ``target(*args_for(port))`` once per port in `ports` and wait.

    The copies are SPAWNED, not forked: nothing gRPC, Feast, torch or FAISS
    survives a fork. So each one loads its own state
    """
    bound = ports(base, processes)
    context = multiprocessing.get_context("spawn")
    children = [
        context.Process(target=target, args=args_for(port), name=f"{name}-{port}") for port in bound
    ]

    def forward(signum: int, frame: FrameType | None) -> None:
        for child in children:
            if child.is_alive() and child.pid is not None:
                os.kill(child.pid, signal.SIGTERM)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    for child in children:
        child.start()
    addresses = ", ".join(f"localhost:{port}" for port in bound)
    print(f"{name} supervisor (pid {os.getpid()}) · {len(bound)} processes · {addresses}")

    for child in children:
        child.join()
    failed = [child.name for child in children if child.exitcode not in (0, -signal.SIGTERM)]
    if failed:
        print(f"{name} processes exited abnormally: {', '.join(failed)}")
        return 1
    return 0
