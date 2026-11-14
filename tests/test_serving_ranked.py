"""The RANKED path, against a fully wired stack. N3 stage 6b.

The mirror of tests/test_serving_smoke.py. That one boots the image with every
backend deliberately absent and asserts the service still answers; this one
assumes retrieval, features and Triton are all up and asserts the opposite
property -- that nothing fell back and nothing degraded.

Both are needed, and neither substitutes for the other: a pipeline that always
degrades gracefully and never actually ranks passes the first file completely.

Needs a running stack, so it is skipped unless RECSYS_SMOKE_TARGET is set:

    make serve-all                 # sidecars + orchestrator, in one terminal
    make smoke-ranked              # this file, in another

It cannot run on a hosted CI runner -- Triton reserves a GPU -- which is why it
is a make target rather than a job in .github/workflows/ci.yml.
"""

from __future__ import annotations

import os
import statistics
import time
from collections.abc import Iterator

import grpc
import pytest
from grpc_health.v1 import health_pb2

from common.pb import recsys_pb2 as pb
from common.pb import recsys_pb2_grpc as pb_grpc
from tests._health import check

TARGET = os.environ.get("RECSYS_SMOKE_TARGET", "")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not TARGET, reason="set RECSYS_SMOKE_TARGET=host:port to a running stack"),
]

# Must match readinessService / upstreamsService in serving/go/cmd/server/main.go.
READINESS_SERVICE = "recsys.v1.Recommender"
UPSTREAMS_SERVICE = "recsys.v1.Recommender.upstreams"

SLATE = 10
READY_TIMEOUT = 30.0
# Four spawned copies per sidecar each load their own model state; on this box
# that took ~1 minute. Matches serve-all's -warm-wait 60s, plus slack.
UPSTREAMS_TIMEOUT = 90.0

# Generous against the 90ms end-to-end budget. This is a SMOKE bound, not a
# latency gate: `make bench` sweeps to saturation and is what actually measures
# p99. A single-request assertion at 90ms would fail on a cold cache and tell
# nobody anything.
BUDGET_MS = 250


def _wait_for_upstreams(channel: grpc.Channel, deadline: float) -> None:
    """Block until the upstreams name says SERVING, or fail.

    A connectable :15051 is not a ready stack: the orchestrator listens at
    once, while each sidecar copy is still loading torch/FAISS/Feast and has
    not bound its port. Requests in that window degrade (a feature lookup
    routed to an unbound copy reports as `ranker`), so every assertion below
    would be measuring the warm-up. The upstreams poller requires EVERY
    address of every backend, which is exactly "fully up".
    """
    status = health_pb2.HealthCheckResponse.UNKNOWN
    while time.monotonic() < deadline:
        try:
            status = check(channel, UPSTREAMS_SERVICE, timeout=2)
        except grpc.RpcError:
            pass  # health service not registered yet
        else:
            if status == health_pb2.HealthCheckResponse.SERVING:
                return
        time.sleep(0.5)
    raise AssertionError(
        f"{TARGET} answers but its upstreams never reached SERVING (last status {status}) "
        f"within {UPSTREAMS_TIMEOUT:.0f}s; is every sidecar copy and Triton up?"
    )


@pytest.fixture(scope="module")
def stub() -> Iterator[pb_grpc.RecommenderStub]:
    with grpc.insecure_channel(TARGET) as channel:
        try:
            grpc.channel_ready_future(channel).result(timeout=READY_TIMEOUT)
        except grpc.FutureTimeoutError:
            raise AssertionError(
                f"nothing answering at {TARGET}; is `make serve-all` up?"
            ) from None
        _wait_for_upstreams(channel, time.monotonic() + UPSTREAMS_TIMEOUT)
        yield pb_grpc.RecommenderStub(channel)


@pytest.fixture(scope="module")
def slate(stub: pb_grpc.RecommenderStub) -> pb.RecommendResponse:
    """One ranked response, reused: these assertions are about one slate's
    properties, and re-requesting per test would let them disagree."""
    return stub.GetRecommendations(
        pb.RecommendRequest(user_id="U1000", surface="home_rail", num_results=SLATE), timeout=10
    )


def test_the_ranker_actually_ran(slate: pb.RecommendResponse) -> None:
    """**The whole point of 6b.**

    `used_fallback` false means retrieval produced candidates and the pipeline
    reached the ranker. An empty `degraded_sources` means none of features,
    ranker, seen-list or feature_columns quietly dropped out on the way. The
    fallback smoke cannot distinguish a working pipeline from a broken one --
    both return a valid slate -- and this is the assertion that can.
    """
    assert not slate.used_fallback, "served popular.json; retrieval produced nothing"
    assert list(slate.degraded_sources) == [], "the slate was ranked, but something degraded"


def test_the_slate_is_well_formed(slate: pb.RecommendResponse) -> None:
    assert len(slate.items) == SLATE
    assert [item.position for item in slate.items] == list(range(SLATE))
    assert len({item.item_id for item in slate.items}) == SLATE, "an item was served twice"
    # Needed by off-policy evaluation: a zero propensity takes the maximum IPS
    # weight under §17.4's clipping rather than erroring.
    assert all(item.propensity > 0 for item in slate.items)
    assert all(item.sources for item in slate.items), "no source named for a served item"


def test_exclusions_are_honoured_on_the_ranked_path(stub: pb_grpc.RecommenderStub) -> None:
    """The contrast that makes the xfail in test_serving_smoke.py meaningful.

    `request.Exclude` is merged in `blocked()`, which the ranked path reaches
    and the fallback path returns before. So this passes while its counterpart
    fails -- and that asymmetry is the defect, stated as two tests rather than
    as a comment.
    """
    first = stub.GetRecommendations(
        pb.RecommendRequest(user_id="U1001", num_results=SLATE), timeout=10
    )
    excluded = [item.item_id for item in first.items[:3]]

    second = stub.GetRecommendations(
        pb.RecommendRequest(user_id="U1001", num_results=SLATE, exclude_item_ids=excluded),
        timeout=10,
    )
    assert not {item.item_id for item in second.items} & set(excluded)


def test_the_response_names_what_served_it(slate: pb.RecommendResponse) -> None:
    """`model_version` and `index_version` are in the response contract so a
    slate can be attributed after the fact. Empty means -model-version and
    -index-version were not passed, which makes every logged impression
    unattributable to the build that produced it."""
    assert slate.model_version, "no model_version; pass -model-version"
    assert slate.index_version, "no index_version; pass -index-version"


def test_requests_stay_inside_a_smoke_bound(stub: pb_grpc.RecommenderStub) -> None:
    """Median over a handful, not a single request: the first request pays
    connection setup and any cold cache, and failing on that measures the
    warm-up rather than the pipeline."""
    latencies = [
        stub.GetRecommendations(
            pb.RecommendRequest(user_id=f"U{2000 + n}", num_results=SLATE), timeout=10
        ).latency_ms
        for n in range(10)
    ]
    assert statistics.median(latencies) < BUDGET_MS, f"median {statistics.median(latencies)}ms"


def test_health_reports_ready_with_everything_up(stub: pb_grpc.RecommenderStub) -> None:
    health = stub.Health(pb.HealthRequest(), timeout=10)
    assert health.ready, f"not ready: {health.detail}"
    assert health.index_kind, "no index kind reported"


@pytest.mark.usefixtures("stub")  # only for its wait: the server must be up
def test_the_upstreams_health_name_agrees() -> None:
    """With all three backends up, the name that nothing routes on should say
    SERVING -- the positive counterpart to the smoke file, which asserts it says
    NOT_SERVING when they are absent. Together they prove the poller moves."""
    with grpc.insecure_channel(TARGET) as channel:
        upstreams = check(channel, UPSTREAMS_SERVICE, timeout=10)
        readiness = check(channel, READINESS_SERVICE, timeout=10)

    assert upstreams == health_pb2.HealthCheckResponse.SERVING
    assert readiness == health_pb2.HealthCheckResponse.SERVING
