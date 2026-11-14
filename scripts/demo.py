"""``make demo`` -- one request through the full serving path, printed."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

import grpc

from common.pb import recsys_pb2 as pb
from common.pb import recsys_pb2_grpc as pb_grpc

# PIPELINE order, not the map's. A protobuf map iterates in whatever order the
# wire handed over, and a funnel printed out of order is worse than no funnel.
STAGES = ("features", "retrieval", "filter", "rank", "rerank", "fallback")
COUNTS = ("retrieved", "unblocked", "scored", "served")

# Keyed by the name the orchestrator reports in degraded_sources. Three of these
# are its own constants (degradedFeatures, degradedRanker, degradedColumns);
# "seen" is a bare literal inline in blocked(). Any other name is a retriever, so
# the lookup falls back rather than raising.
FIXES = {
    "features": "feature gateway unreachable -- make features-serve (Redis: make up)",
    "ranker": "no scores from Triton, or the feature matrix failed to build -- see make up",
    "feature_columns": "model.columns.json disagrees with the built rows -- make onnx",
    "seen": "seen-list unreachable -- make up (Redis)",
}


def wait_for_ready(stub: pb_grpc.RecommenderStub, timeout: float) -> pb.HealthResponse:
    """Poll ``Health`` until the server reports ready, or give up.

    Readiness, not a bare connection. The port opens before the artifacts
    finish loading, and a request issued inside that window is answered from the
    fallback path -- so a demo that raced the warm-up would print a degraded
    slate and blame the wrong thing.

    Args:
        stub: A connected stub.
        timeout: Seconds to wait.

    Returns:
        The health response that reported ready.

    Raises:
        SystemExit: If the deadline passes, carrying the last detail reported.
    """
    deadline = time.monotonic() + timeout
    detail = "no answer yet"
    while time.monotonic() < deadline:
        try:
            # Annotated, not inferred: recsys_pb2_grpc is generated without type
            # information, so every stub call is Any and mypy --strict refuses to
            # return one from a typed function.
            health: pb.HealthResponse = stub.Health(pb.HealthRequest(), timeout=2)
        except grpc.RpcError:
            detail = "cannot reach the orchestrator"
        else:
            if health.ready:
                return health
            detail = health.detail or "not ready"
        time.sleep(1.0)

    raise SystemExit(f"  not ready after {timeout:g}s: {detail}")


def _versions(health: pb.HealthResponse, response: pb.RecommendResponse) -> str:
    """What served the slate, or ``unset`` where nothing said.

    Empty is reported rather than hidden: ``-model-version`` and
    ``-index-version`` default to the empty string, so an unset pair makes every
    logged impression unattributable to the build that produced it, and that is
    worth seeing on the way past.
    """
    model = response.model_version or "unset"
    index = response.index_version or "unset"
    shape = f"{health.index_kind or 'unknown'}, efSearch={health.index_ef_search}"
    return f"model={model}  index={index}  ({shape})"


def show(response: pb.RecommendResponse, health: pb.HealthResponse) -> None:
    """Print the response: provenance, timings, funnel, slate, then caveats."""
    print(f"\n  served by     {_versions(health, response)}")
    if response.experiment_variant:
        print(f"  variant       {response.experiment_variant}")
    print(f"  total         {response.latency_ms} ms")

    stages = [
        f"{name} {response.stage_latency_ms[name]}ms"
        for name in STAGES
        if name in response.stage_latency_ms
    ]
    print(f"  stages        {' · '.join(stages)}")

    funnel = [
        f"{name} {response.stage_counts[name]:,}"
        for name in COUNTS
        if name in response.stage_counts
    ]
    print(f"  funnel        {' -> '.join(funnel)}")

    print(f"\n  {'#':>2}  {'item':<10} {'score':>9}  {'prop':>5}  sources")
    for item in response.items:
        sources = ", ".join(item.sources) or "-"
        print(
            f"  {item.position:>2}  {item.item_id:<10} {item.score:>9.4f}"
            f"  {item.propensity:>5.2f}  {sources}"
        )

    # Last, and spelled out, because this is the part a screenshot omits.
    if response.used_fallback:
        print(
            "\n  INCOMPLETE: this slate came from popular.json, so retrieval produced"
            "\n  nothing and the ranker never ran. Nothing end-to-end was demonstrated."
        )
    for name in response.degraded_sources:
        fix = FIXES.get(name, "retriever down -- make retrieval-serve")
        print(f"\n  degraded: {name} -- {fix}")
    if not response.used_fallback and not response.degraded_sources:
        print("\n  every stage ran; no source degraded.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="localhost:15051")
    parser.add_argument("--user", default="U13740", help="a user id present in the feature store")
    parser.add_argument("--surface", default="home_rail")
    parser.add_argument("--num-results", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=60.0, help="seconds to wait for readiness")
    args = parser.parse_args(argv)

    with grpc.insecure_channel(args.target) as channel:
        stub = pb_grpc.RecommenderStub(channel)
        print(f"  waiting for    {args.target}")
        health = wait_for_ready(stub, args.timeout)

        response: pb.RecommendResponse = stub.GetRecommendations(
            pb.RecommendRequest(
                user_id=args.user,
                surface=args.surface,
                num_results=args.num_results,
            ),
            timeout=10,
        )

    show(response, health)
    # Non-zero on the fallback path only. A degraded source is a real local
    # stack with a piece missing, which the output names; a fallback slate means
    # the demo demonstrated nothing and should not pass quietly.
    return 1 if response.used_fallback else 0


if __name__ == "__main__":
    raise SystemExit(main())
