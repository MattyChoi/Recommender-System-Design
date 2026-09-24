"""The user-embedding cache.

Worth testing beyond "it stores things" for one reason: ADR 0013 makes this
cache a **cross-process, cross-language contract**. The Go orchestrator reads
these bytes when this service is unreachable, so the encoding and the failure
behaviour are the interesting parts, not the hit path.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest

from serving.retrieval.cache import KEY_PREFIX, UserEmbeddingCache, key_for


class FakeRedis:
    """Just enough Redis. `fail` makes every call raise, which is the state the
    cache has to survive rather than propagate."""

    def __init__(self, fail: bool = False) -> None:
        self.store: dict[str, bytes] = {}
        self.ttls: dict[str, int] = {}
        self.fail = fail

    def get(self, key: str) -> bytes | None:
        if self.fail:
            raise ConnectionError("down")
        return self.store.get(key)

    def setex(self, key: str, ttl: int, value: bytes) -> None:
        if self.fail:
            raise ConnectionError("down")
        self.store[key] = value
        self.ttls[key] = ttl


@pytest.fixture
def client() -> FakeRedis:
    return FakeRedis()


def test_a_stored_vector_round_trips(client: FakeRedis) -> None:
    cache = UserEmbeddingCache(client, dim=4)
    vector = np.array([0.5, -0.25, 0.125, 1.0], dtype=np.float32)

    cache.put("U7", vector)

    got = cache.get("U7")
    assert got is not None
    assert np.array_equal(got, vector)


def test_the_encoding_is_raw_little_endian_float32(client: FakeRedis) -> None:
    """The Go reader has no header to parse, so the layout is the contract.

    Pinned explicitly rather than only round-tripped: a round trip passes
    happily on native-endian bytes, and would keep passing right up until
    something reads them on a machine that disagrees.
    """
    cache = UserEmbeddingCache(client, dim=2)

    cache.put("U1", np.array([1.0, 2.0], dtype=np.float32))

    raw = client.store[key_for("U1")]
    assert raw == b"\x00\x00\x80\x3f\x00\x00\x00\x40"
    assert len(raw) == 2 * 4


def test_the_key_is_namespaced() -> None:
    # An unprefixed integer key is what two subsystems sharing a Redis collide
    # on, and a second process reads these.
    assert key_for("U12") == f"{KEY_PREFIX}:U12"


def test_a_vector_of_the_wrong_width_is_not_returned(client: FakeRedis) -> None:
    """A stale-width entry is a DIFFERENT MODEL's embedding left by a deploy.

    Returning it would either crash the dot product or, if a previous build
    happened to match, retrieve neighbours in a space the index is not in --
    which returns entirely reasonable-looking articles.
    """
    cache = UserEmbeddingCache(client, dim=4)
    client.store[key_for("U3")] = np.array([1.0, 2.0], dtype=np.float32).tobytes()

    assert cache.get("U3") is None
    assert cache.errors == 1


def test_a_dead_cache_degrades_rather_than_raising() -> None:
    cache = UserEmbeddingCache(FakeRedis(fail=True), dim=4)

    assert cache.get("U1") is None
    cache.put("U1", np.zeros(4, dtype=np.float32))  # must not raise

    assert cache.errors == 2


def test_a_wrong_shaped_write_is_dropped(client: FakeRedis) -> None:
    cache = UserEmbeddingCache(client, dim=4)

    cache.put("U1", np.zeros(3, dtype=np.float32))

    assert client.store == {}


def test_the_ttl_is_applied(client: FakeRedis) -> None:
    # Short on purpose: a user embedding is a function of a history that moves,
    # so a long TTL serves a vector describing who they were.
    cache = UserEmbeddingCache(client, dim=2, ttl=42)

    cache.put("U5", np.zeros(2, dtype=np.float32))

    assert client.ttls[key_for("U5")] == 42


def test_the_hit_rate_counts_lookups_not_writes(client: FakeRedis) -> None:
    cache = UserEmbeddingCache(client, dim=2)
    cache.put("U1", np.zeros(2, dtype=np.float32))

    cache.get("U1")
    cache.get("U2")

    assert cache.hit_rate == pytest.approx(0.5)


def test_an_empty_cache_reports_a_zero_hit_rate(client: FakeRedis) -> None:
    # Not a ZeroDivisionError on the first scrape of a fresh process.
    assert UserEmbeddingCache(client, dim=2).hit_rate == 0.0


def test_the_returned_array_does_not_alias_the_buffer(client: FakeRedis) -> None:
    """np.frombuffer gives a READ-ONLY view over the bytes.

    Handed back as-is, the first caller to normalise in place gets
    ValueError: assignment destination is read-only -- on the cache-hit path
    only, so it would pass every test that exercises a miss.
    """
    cache = UserEmbeddingCache(client, dim=2)
    cache.put("U1", np.array([1.0, 2.0], dtype=np.float32))

    got = cache.get("U1")
    assert got is not None
    got[0] = 9.0  # must not raise
    assert cache.get("U1")[0] == pytest.approx(1.0)  # type: ignore[index]


def test_a_missing_key_is_a_miss_not_an_error(client: FakeRedis) -> None:
    cache = UserEmbeddingCache(client, dim=2)

    assert cache.get("U99") is None
    assert (cache.misses, cache.errors) == (1, 0)


def test_connect_names_what_to_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """The error has to say `make up`, matching seen_redis.connect.

    Two subsystems that fail differently on the same dead Redis send whoever is
    debugging in two directions.
    """
    import serving.retrieval.cache as module

    class Dead:
        @staticmethod
        def from_url(url: str) -> Any:
            class Client:
                def ping(self) -> None:
                    raise ConnectionError("refused")

            return Client()

    monkeypatch.setitem(sys.modules, "redis", type("redis", (), {"Redis": Dead}))

    with pytest.raises(RuntimeError, match="make up"):
        module.connect("redis://localhost:6379/2")
