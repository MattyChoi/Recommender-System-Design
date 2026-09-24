"""The user-embedding cache, and the two jobs it does.

§14.4 item 3 prescribes caching user embeddings in Redis with a short TTL, to
keep a tower forward pass off the hot path. That is the latency job.

The second job is structural and is the reason ADR 0013's degradation ladder
has three rungs rather than two: **this cache is what the Go orchestrator reads
when this service is unreachable.** A cached embedding lets Go run exact search
in-process (rung 2); with no cached embedding there is no query vector and the
request falls to popularity (rung 3). So the format written here is a serving
contract with a reader in another language, not an implementation detail.

The format is deliberately the dullest possible: **raw little-endian float32,
no header, no framing.** A pickle would be unreadable from Go, and JSON would
cost a text round trip on every miss and lose the last bit of every float --
this project has already had float width flip an argmax once. Length is the
only thing that needs checking, and the reader knows the expected dimension.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

#: Redis logical database. Feast owns 0 and the seen-list owns 1; see
#: models/reranking/seen_redis.py for why sharing a keyspace with a feature
#: store is how one FLUSHDB during a materialisation takes everything with it.
DEFAULT_URL = "redis://localhost:6379/2"

#: Seconds. Short on purpose. A user embedding is a function of their history,
#: and history moves: a long TTL serves a vector describing who they were. The
#: cost of expiry is one tower forward pass, which is the thing the cache exists
#: to avoid but is not expensive enough to justify serving a stale user.
DEFAULT_TTL = 300

#: Key prefix. Namespaced because rung 2 means a SECOND process reads these,
#: and an unprefixed integer key is the kind of thing two subsystems collide on.
KEY_PREFIX = "uemb"


def key_for(user_id: str) -> str:
    """The cache key, from the EXTERNAL user id.

    The same string Feast keys its entities on, and the same string the Go
    orchestrator has in hand on rung 2 -- so neither side needs a user_map to
    find this. The tower embeds no user id, so an internal index would exist
    only to key this cache.
    """
    return f"{KEY_PREFIX}:{user_id}"


class UserEmbeddingCache:
    """Read-through cache of unit-norm user vectors.

    Every method degrades rather than raises. A cache is an optimisation, and a
    service that fails a request because its optimisation is down has converted
    a latency problem into an outage.
    """

    def __init__(self, client: Any, dim: int, ttl: int = DEFAULT_TTL) -> None:
        self.client = client
        self.dim = dim
        self.ttl = ttl
        self.hits = 0
        self.misses = 0
        self.errors = 0

    def get(self, user_id: str) -> npt.NDArray[np.float32] | None:
        """The cached vector, or None. Never raises."""
        try:
            raw = self.client.get(key_for(user_id))
        except Exception:
            self.errors += 1
            return None
        if raw is None:
            self.misses += 1
            return None

        # A stored vector of the wrong width is not a cache miss, it is a
        # DIFFERENT MODEL's embedding left behind by a deploy. Using it would
        # either crash on the dot product or, if the dimensions happened to
        # match a previous build, silently retrieve neighbours in a space this
        # index was not built in.
        if len(raw) != self.dim * 4:
            self.errors += 1
            return None

        self.hits += 1
        return np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)

    def put(self, user_id: str, vector: npt.NDArray[np.float32]) -> None:
        """Store with the TTL. A failed write is not a failed request."""
        if vector.shape != (self.dim,):
            return
        try:
            # astype("<f4") rather than .tobytes() on whatever came in: the
            # tower returns native-endian float32, and "native" is an
            # assumption the Go reader would be forced to share silently.
            self.client.setex(key_for(user_id), self.ttl, vector.astype("<f4").tobytes())
        except Exception:
            self.errors += 1

    @property
    def hit_rate(self) -> float:
        """Hits over lookups. A cache that has quietly stopped being hit is a
        latency regression with no error attached, which is why this is
        reported in the response rather than only counted here."""
        looked = self.hits + self.misses
        return self.hits / looked if looked else 0.0


def connect(url: str = DEFAULT_URL) -> Any:
    """A Redis client, or an error naming what to start.

    Mirrors models/reranking/seen_redis.py:connect so the two subsystems fail
    the same way and the message says `make up` in both.
    """
    try:
        import redis
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RuntimeError("the `redis` package is not installed") from exc

    client = redis.Redis.from_url(url)
    try:
        client.ping()
    except Exception as exc:
        raise RuntimeError(f"Redis is not answering at {url} -- is it up? (`make up`)") from exc
    return client
