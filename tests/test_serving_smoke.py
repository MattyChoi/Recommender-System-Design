"""The orchestrator's degradation contract, against the real image.

Nothing here needs a GPU, a trained model or a sidecar. ``run()`` dials every
backend lazily -- ``warmConnections`` logs a failed probe and carries on -- so
the service starts with Triton and both sidecars absent and answers from
popular.json. "The slate is never empty" is the part of the contract only the
real binary can prove: the Go tests cover the same branch with fakes, while
this covers flag parsing, artifact loading, the gRPC server and distroless.

Exactly four files are fatal at startup (item map, catalogue, popular, the
column sidecar), which is why a fixture of four small JSON documents is enough
to boot the whole service.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import grpc
import pytest
from grpc_health.v1 import health_pb2

from common.pb import recsys_pb2 as pb
from common.pb import recsys_pb2_grpc as pb_grpc
from models.ranking.dataset import FEATURES
from tests._health import check

# Defaulted to "" rather than None so the image name is a str for mypy: the
# skipif below guarantees it is non-empty by the time anything uses it, but a
# marker is not something a type checker can reason about.
IMAGE = os.environ.get("RECSYS_SMOKE_IMAGE", "")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not IMAGE, reason="set RECSYS_SMOKE_IMAGE to a built image"),
]

# Item indices are 1-BASED. Index 0 is the reserved OOV row: popular.go refuses
# it outright, catalogue.go requires it to hold category 0, and ids.go treats an
# unclaimed index as not-found. A fixture numbered from 0 is rejected at startup.
ITEMS = 32
ITEM_INDICES = range(1, ITEMS + 1)
KNOWN_IDS = {f"N{i}" for i in ITEM_INDICES}
PORT = 15099  # not 15051: a dev box usually has the real thing on it
METRICS = ":9199"  # ditto 9102
READY_TIMEOUT = 60.0

# Must match readinessService in serving/go/cmd/server/main.go. Two languages
# with nothing to share a constant through, so it is written down once here and
# asserted against a running server rather than assumed.
READINESS_SERVICE = "recsys.v1.Recommender"
UPSTREAMS_SERVICE = "recsys.v1.Recommender.upstreams"


def _docker(*args: str) -> str:
    """Both streams, because `inspect` on a vanished container answers on stderr."""
    done = subprocess.run(["docker", *args], capture_output=True, text=True, check=False)
    return (done.stdout + done.stderr).strip()


def _running(container: str) -> bool:
    return _docker("inspect", "-f", "{{.State.Running}}", container) == "true"


def _await_ready(container: str, channel: grpc.Channel) -> None:
    """Wait for the port, failing fast if the container has already exited.

    Without the liveness check this waits out READY_TIMEOUT and then reports a
    timeout -- the wrong diagnosis for a process that died at startup, and a
    minute of CI spent arriving at it.
    """
    deadline = time.monotonic() + READY_TIMEOUT
    while True:
        try:
            grpc.channel_ready_future(channel).result(timeout=2)
            return
        except grpc.FutureTimeoutError:
            if not _running(container):
                raise AssertionError("the orchestrator exited at startup; log above") from None
            if time.monotonic() > deadline:
                raise AssertionError("the orchestrator never accepted a connection") from None


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The four files run() refuses to start without.

    Consistent by construction: ITEMS entries at indices 1..ITEMS, so
    popular.json indexes the same catalogue the item map describes. FEATURES is
    IMPORTED, not retyped -- a drifted copy would pass here while serving built
    a different feature vector.
    """
    root = tmp_path_factory.mktemp("artifacts")
    (root / "item_map.json").write_text(
        json.dumps({"version": "smoke", "items": {f"N{i}": i for i in ITEM_INDICES}})
    )
    # Positional by item index, so both arrays are ITEMS+1 long and row 0 is the
    # reserved OOV row the loader refuses to find a real category on. Real values
    # start at 1 for the same reason: 0 is what a short source pads with, and
    # both stay under n_categories / n_subcategories.
    (root / "catalogue.json").write_text(
        json.dumps(
            {
                "version": "smoke",
                "category": [0, *(1 + (i - 1) % 3 for i in ITEM_INDICES)],
                "subcategory": [0, *(1 + (i - 1) % 7 for i in ITEM_INDICES)],
                "n_categories": 4,
                "n_subcategories": 8,
            }
        )
    )
    # Index order IS popularity order here, which is what the ordering assertion
    # below checks the service preserves.
    (root / "popular.json").write_text(
        json.dumps({"version": "smoke", "items": list(ITEM_INDICES)})
    )
    (root / "model.columns.json").write_text(json.dumps({"features": list(FEATURES)}))
    return root


@pytest.fixture(scope="module")
def channel(artifacts: Path) -> Iterator[grpc.Channel]:
    """The image, with every backend deliberately absent.

    Yields the CHANNEL rather than one stub: the health contract is served by
    grpc.health.v1.Health, not by Recommender, and both are wanted against the
    same container.
    """
    flags = {
        "-port": str(PORT),
        "-metrics": METRICS,
        "-columns": "/repo/serving/artifacts/model.columns.json",
        # Load-bearing. The default is localhost:9092, and a set -kafka with no
        # broker makes NewKafkaProducer fail, which IS fatal in run(). The empty
        # string is the documented "disable logging" value.
        "-kafka": "",
        "-warm-wait": "0s",
    }
    # -p, not --network host: on a Docker Desktop / WSL daemon the host namespace
    # is the daemon's VM rather than the shell this process runs in, so a
    # host-networked container binds a port nothing here can reach. Publishing
    # works on either kind of daemon. Redis needs no plumbing at all -- the
    # fallback path never consults the seen-list. Built as a dict and flattened
    # rather than written as one long list, which ruff format would explode.
    command = ["docker", "run", "-d", "-p", f"{PORT}:{PORT}"]
    # The image runs as uid 65532, which cannot even traverse pytest's tmp dir
    # (0700, owned by the invoking user), so the artifact reads fail with
    # EACCES. The binary is static and writes nothing, so running it as the
    # caller is the whole fix -- and it holds on a runner with a different uid,
    # which chmod-ing the tmp chain would not.
    command += ["--user", f"{os.getuid()}:{os.getgid()}"]
    command += ["-v", f"{artifacts}:/repo/serving/artifacts:ro", IMAGE]
    for flag, value in flags.items():
        command += [flag, value]

    # No --rm: a container that exits at startup has to survive long enough for
    # the teardown below to read its log, which is the only account of why.
    container = subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
    try:
        with grpc.insecure_channel(f"localhost:{PORT}") as connection:
            _await_ready(container, connection)
            yield connection
    finally:
        # Printed unconditionally; pytest shows captured output only on failure,
        # and a container smoke that fails without its logs is unfixable in CI.
        print(_docker("logs", container))
        _docker("rm", "-f", container)


@pytest.fixture(scope="module")
def stub(channel: grpc.Channel) -> pb_grpc.RecommenderStub:
    return pb_grpc.RecommenderStub(channel)


def test_the_slate_is_never_empty(stub: pb_grpc.RecommenderStub) -> None:
    response = stub.GetRecommendations(
        pb.RecommendRequest(user_id="U1", surface="home_rail", num_results=10), timeout=5
    )
    assert len(response.items) == 10
    assert response.used_fallback, "no retriever and no ranker must mean the fallback ran"
    assert [item.position for item in response.items] == list(range(10))
    assert {item.item_id for item in response.items} <= KNOWN_IDS


@pytest.mark.xfail(strict=True, reason="servePopularity returns before the filter stage")
def test_exclusions_are_honoured_on_the_fallback_path(stub: pb_grpc.RecommenderStub) -> None:
    """A KNOWN GAP, recorded rather than asserted away.

    `request.Exclude` is merged in `blocked()`, which the fallback path never
    reaches: recommend.go returns at the empty-candidates check one stage
    earlier, and `servePopularity` hands back `Popular(ctx, numResults)`
    untouched. So a caller that says "I already rendered these" is served them
    again precisely when retrieval degrades.

    Ignoring the SEEN-LIST on this path is defensible -- it is a Redis
    round-trip on the one path that must not fail for an avoidable reason.
    Ignoring the caller's own list is not: applying it is free.

    strict=True rather than a deleted test: if the fallback ever learns to
    honour Exclude, this fails as XPASS and the marker comes off.
    """
    excluded = ["N1", "N2", "N3"]
    response = stub.GetRecommendations(
        pb.RecommendRequest(user_id="U2", num_results=5, exclude_item_ids=excluded), timeout=5
    )
    assert not {item.item_id for item in response.items} & set(excluded)


def test_health_admits_it_is_not_ready(stub: pb_grpc.RecommenderStub) -> None:
    """Serving and ready are different states. This is the property N2's k8s
    readiness probe depends on, so it is worth a gate now."""
    health = stub.Health(pb.HealthRequest(), timeout=5)
    assert not health.ready
    assert health.detail, "an unready service must say which probe failed"


def test_readiness_does_not_route_on_upstream_health(channel: grpc.Channel) -> None:
    """**What Kubernetes actually probes**, as opposed to the RPC above.

    `readinessProbe.grpc` speaks grpc.health.v1 and nothing else, so the
    Recommender.Health verdict above is invisible to it. Three names, two
    meanings, asserted against a pod with every backend absent:

    * "" is LIVENESS. SERVING, because the process is fine.
    * READINESS_SERVICE is SERVING too -- this container answers from
      popular.json, as test_the_slate_is_never_empty proves on the same fixture,
      and a pod that can answer belongs in the Service's endpoints.
    * UPSTREAMS_SERVICE carries the all-three verdict. NOT_SERVING here, and
      nothing routes on it.

    Routing on upstream health would mean a Triton outage flipped every pod at
    once, kubelet emptied the endpoints, and a service that could still answer
    returned nothing instead -- a degraded dependency promoted to a total
    outage. This container is the proof that the fallback works, so it is the
    right place to assert the probe does not throw it away.
    """
    assert check(channel) == health_pb2.HealthCheckResponse.SERVING
    assert check(channel, READINESS_SERVICE) == health_pb2.HealthCheckResponse.SERVING
    assert check(channel, UPSTREAMS_SERVICE) == health_pb2.HealthCheckResponse.NOT_SERVING
