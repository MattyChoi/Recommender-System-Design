"""The user-embedding cache key, a cross-language parity surface.

``serving/retrieval/cache.py`` writes these keys and
``serving/go/internal/sources/fallback.go`` reads them on rung 2. The instant and
bucket below are asserted verbatim in ``fallback_test.go``; a disagreement is not
a failing request but rung 2 silently never finding a vector.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import numpy as np
import pytest

from serving.retrieval.cache import UserEmbeddingCache, hour_bucket, key_for

PINNED = datetime(2026, 11, 9, 13, 30, tzinfo=UTC)
PINNED_BUCKET = 498397


class FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}

    def get(self, key: str) -> bytes | None:
        return self.store.get(key)

    def setex(self, key: str, _ttl: int, value: bytes) -> None:
        self.store[key] = value


def test_the_key_matches_the_go_reader() -> None:
    assert hour_bucket(PINNED) == PINNED_BUCKET
    assert key_for("U1", PINNED) == "uemb:U1:498397"


def test_the_bucket_does_not_depend_on_the_zone() -> None:
    eastern = PINNED.astimezone(timezone(timedelta(hours=-5)))
    assert hour_bucket(eastern) == PINNED_BUCKET


def test_a_naive_datetime_is_refused() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        hour_bucket(datetime(2026, 11, 9, 13, 30))


def test_a_vector_is_not_served_into_the_next_hour() -> None:
    """The bug this key exists to fix: hour_of_day is a tower input."""
    cache = UserEmbeddingCache(FakeRedis(), dim=4)
    vector: Any = np.ones((1, 4), dtype=np.float32)

    cache.put("U1", PINNED, vector)

    assert cache.get("U1", PINNED + timedelta(minutes=29)) is not None
    assert cache.get("U1", PINNED + timedelta(minutes=30)) is None
