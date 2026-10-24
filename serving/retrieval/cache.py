"""The user-embedding cache in Redis with a short TTL

Achieves two things
    - Keeps a tower forward pass off the hot path for user encoding
    - Go orchestrator hits this cache when the service is unreachable
    for the user embeddings
"""

from __future__ import annotations

import os
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


def key_for(user_id: str) -> str:
    """The cache key, from the EXTERNAL user id."""
    return f"{KEY_PREFIX}:{user_id}"


class UserEmbeddingCache:
    """Read-through cache of unit-norm user vectors."""

    def __init__(self, client: Any, dim: int, ttl: int = DEFAULT_TTL) -> None:
        self.client = client
        self.dim = dim
        self.ttl = ttl
        self.hits = 0
        self.misses = 0
        self.errors = 0

    def get(self, user_id: str) -> npt.NDArray[np.float32] | None:
        try:
            raw = self.client.get(key_for(user_id))
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

    def put(self, user_id: str, vector: npt.NDArray[np.float32]) -> None:
        """Store with the TTL. A failed write is not a failed request."""
        if vector.size != self.dim:
            return
        try:
            # Pin the wire format (little-endian float32, dim*4 bytes) that get() and the
            # Go reader expect; .tobytes() alone would dump whatever dtype/order came in.
            self.client.setex(key_for(user_id), self.ttl, vector.astype("<f4").tobytes())
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
