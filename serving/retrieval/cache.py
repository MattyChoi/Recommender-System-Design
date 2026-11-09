"""The user-embedding cache in Redis with a short TTL

Achieves two things
    - Keeps a tower forward pass off the hot path for user encoding
    - Go orchestrator hits this cache when the service is unreachable
    for the user embeddings
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import numpy as np
import numpy.typing as npt
from dotenv import load_dotenv

load_dotenv()

# Redis logical database. Feast owns 0 and the seen-list owns 1
REDIS_URL = "redis://" + os.getenv("REDIS_URL", "localhost:6379") + "/2"

# expiry in seconds
DEFAULT_TTL = 300

# Key prefix
KEY_PREFIX = "uemb"


def hour_bucket(moment: datetime) -> int:
    """Whole hours since the Unix epoch, in UTC.

    The user tower takes ``hour_of_day`` and ``day_of_week`` as inputs, so a user's
    embedding is a function of the hour as well as of the user.

    Mirrored by ``hourBucket`` in serving/go/internal/sources/fallback.go, which
    reads these keys.
    """
    if moment.tzinfo is None:
        # A naive datetime would be read in the host's local zone, shifting the
        # bucket by the UTC offset -- the timezone bug this project already hit.
        raise ValueError("hour_bucket needs a timezone-aware datetime")
    return int(moment.timestamp()) // 3600


def key_for(user_id: str, moment: datetime) -> str:
    """The cache key: the EXTERNAL user id plus the hour the vector was encoded for."""
    return f"{KEY_PREFIX}:{user_id}:{hour_bucket(moment)}"


class UserEmbeddingCache:
    """Read-through cache of unit-norm user vectors."""

    def __init__(self, client: Any, dim: int, ttl: int = DEFAULT_TTL) -> None:
        self.client = client
        self.dim = dim
        self.ttl = ttl
        self.hits = 0
        self.misses = 0
        self.errors = 0

    def get(self, user_id: str, moment: datetime) -> npt.NDArray[np.float32] | None:
        try:
            raw = self.client.get(key_for(user_id, moment))
        except Exception:
            self.errors += 1
            return None
        if raw is None:
            self.misses += 1
            return None

        if len(raw) != self.dim * 4:
            self.errors += 1
            return None

        self.hits += 1
        # [1, dim], the shape encode_user returns and FAISS search takes.
        return np.frombuffer(raw, dtype="<f4").astype(np.float32).reshape(1, self.dim)

    def put(self, user_id: str, moment: datetime, vector: npt.NDArray[np.float32]) -> None:
        """Store with the TTL. A failed write is not a failed request."""
        if vector.size != self.dim:
            return
        try:
            # Pin the wire format (little-endian float32, dim*4 bytes) that get() and the
            # Go reader expect; .tobytes() alone would dump whatever dtype/order came in.
            self.client.setex(key_for(user_id, moment), self.ttl, vector.astype("<f4").tobytes())
        except Exception:
            self.errors += 1

    @property
    def hit_rate(self) -> float:
        attempts = self.hits + self.misses
        return self.hits / attempts if attempts else 0.0


def connect(url: str = REDIS_URL) -> Any:
    """A Redis client, or an error naming what to start."""
    try:
        import redis
    except ImportError as exc:
        raise RuntimeError("the `redis` package is not installed") from exc

    client = redis.Redis.from_url(url)
    try:
        client.ping()
    except Exception as exc:
        raise RuntimeError(f"Redis is not answering at {url} -- is it up? (`make up`)") from exc
    return client
