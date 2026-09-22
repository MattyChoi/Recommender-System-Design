"""The seen-list as it would actually be served: bits in Redis, per user.

``models.reranking.seen`` models the structure to measure its error rate, which
is arithmetic and needs no server. This is the other half -- the parts that only
exist once the bits live somewhere shared, and that the in-memory model cannot
have an opinion about:

**One GET, not k x n GETBITs.** A request tests ~100 candidates against a filter
with 4 hash positions each: 400 round trips at sub-millisecond each blows the
re-rank stage's entire budget. The bitmap is 60-1,200 bytes, so it is cheaper to
fetch the WHOLE thing once and test locally. The structure's cost model at
serving time is dominated by round trips, not by bit arithmetic, and that
inverts the obvious implementation.

**Writes are one transaction.** Setting k bits for each shown item must not
interleave with another request for the same user, or two concurrent writes can
leave a partially-written item -- which, uniquely, CAN produce a false negative
and re-show something. A pipeline with ``transaction=True`` makes the k SETBITs
and the EXPIRE one atomic unit.

⚠️ **The TTL contradicts the one-sided guarantee, and both are load-bearing.**
"Recently shown" needs a window or the filter grows without bound and saturates
(see ``seen_bench``: at 10x capacity it hides everything). The window is an
EXPIRE. But an EXPIRE is a scheduled reset, and the never-a-false-negative
promise holds only while the filter is never cleared. **So the mechanism that
bounds memory is the mechanism that reintroduces the failure users notice.**
The trade is real and unavoidable; what a design can choose is where to put it:
a long TTL re-shows rarely and costs memory, a short one is cheap and re-shows
sooner. It is a product decision about how long "recently" means, and it should
be written down as one rather than inherited from a default.

**Database 1, not 0.** Feast's online store is the same Redis instance. Sharing
a keyspace with a feature store means one ``FLUSHDB`` during a materialisation
takes the seen-lists with it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from models.reranking.seen import sizing

# Feast owns db 0 on this instance.
DEFAULT_URL = "redis://localhost:6379/1"
NAMESPACE = "seen"


def connect(url: str = DEFAULT_URL) -> Any:
    """A Redis client, or an error that says what to start.

    Raises:
        RuntimeError: If the package is missing or the server is unreachable.
            Never a silent fall-back to the in-memory filter: a seen-list that
            quietly stops persisting re-shows items forever and looks fine.
    """
    try:
        import redis
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError("the `redis` package is not installed") from exc

    client = redis.Redis.from_url(url)
    try:
        client.ping()
    except Exception as exc:
        raise RuntimeError(f"Redis is not answering at {url} -- is it up? (`make up`)") from exc
    return client


@dataclass
class RedisSeenList:
    """Per-user Bloom filters held as Redis bitmaps.

    Attributes:
        client: A connected client.
        bits: Bitmap width per user.
        hashes: Positions set per item.
        ttl_seconds: How long "recently shown" lasts. See the module note: this
            is the window that bounds memory AND the reset that breaks the
            one-sided guarantee.
        namespace: Key prefix, so the keyspace is legible in ``SCAN`` output and
            separable from anything else sharing the database.
    """

    client: Any
    bits: int
    hashes: int
    ttl_seconds: int = 7 * 24 * 3600
    namespace: str = NAMESPACE

    @classmethod
    def sized_for(
        cls, client: Any, capacity: int, false_positive_rate: float, **kwargs: Any
    ) -> RedisSeenList:
        """Build one from the capacity and rate, using the shared sizing rules."""
        bits, hashes = sizing(capacity, false_positive_rate)
        return cls(client=client, bits=bits, hashes=hashes, **kwargs)

    def key(self, user: int) -> str:
        return f"{self.namespace}:{user}"

    def _positions(self, item: int) -> npt.NDArray[np.int64]:
        """Bit positions for one item. Identical arithmetic to the in-memory
        model, imported rather than reimplemented so the two cannot drift --
        a filter written by one and read by the other must agree."""
        from models.reranking.seen import BloomFilter

        return BloomFilter(bits=self.bits, hashes=self.hashes)._positions(item)

    def add(self, user: int, items: npt.NDArray[np.int64]) -> None:
        """Record items as shown, atomically, and refresh the window."""
        if not len(items):
            return
        pipe = self.client.pipeline(transaction=True)
        key = self.key(user)
        for item in items.tolist():
            for position in self._positions(int(item)).tolist():
                pipe.setbit(key, position, 1)
        pipe.expire(key, self.ttl_seconds)
        pipe.execute()

    def bitmap(self, user: int) -> npt.NDArray[np.bool_]:
        """The user's whole filter in one round trip.

        Redis numbers bits big-endian within each byte -- bit 0 is the HIGH bit
        of byte 0 -- which is what ``unpackbits`` produces by default. Getting
        this backwards yields a filter that answers plausibly and wrongly.
        """
        raw = self.client.get(self.key(user))
        if raw is None:
            return np.zeros(self.bits, dtype=bool)
        unpacked = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="big")
        # Redis grows the string only as far as the highest bit ever set, so a
        # sparse filter comes back short and must be padded, not reshaped.
        if len(unpacked) < self.bits:
            unpacked = np.pad(unpacked, (0, self.bits - len(unpacked)))
        return unpacked[: self.bits].astype(bool)

    def seen_mask(self, user: int, items: npt.NDArray[np.int64]) -> npt.NDArray[np.bool_]:
        """True where the filter says the item was already shown. One round trip."""
        bitmap = self.bitmap(user)
        return np.fromiter(
            (bool(bitmap[self._positions(int(item))].all()) for item in items.tolist()),
            dtype=bool,
            count=len(items),
        )

    def memory_bytes(self, user: int) -> int:
        """What this key ACTUALLY costs, per Redis, overhead included.

        The bit count understates it: a Redis string carries a key name, an
        object header, an SDS header and a dict entry. At small capacities that
        overhead can rival the payload, which is exactly where a
        bits-only estimate would claim the largest saving.
        """
        used = self.client.memory_usage(self.key(user))
        return int(used) if used is not None else 0

    def load(self, user: int) -> float:
        """Share of bits set. **The alarm to monitor**, not the error rate: it
        moves long before the rate does, and a saturated filter hides every
        candidate rather than a few."""
        return float(self.bitmap(user).mean())


def time_calls(
    filter_: RedisSeenList, user: int, items: npt.NDArray[np.int64], repeats: int = 200
) -> dict[str, float]:
    """Per-call latency for the two operations a request performs.

    Returns milliseconds, because the re-rank stage's budget is quoted in them.
    """
    started = time.perf_counter()
    for _ in range(repeats):
        filter_.seen_mask(user, items)
    read_ms = (time.perf_counter() - started) / repeats * 1000.0

    started = time.perf_counter()
    for _ in range(repeats):
        filter_.add(user, items[:10])
    write_ms = (time.perf_counter() - started) / repeats * 1000.0

    return {"read_ms": read_ms, "write_ms": write_ms, "candidates": float(len(items))}
