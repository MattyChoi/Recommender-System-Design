"""A TTL cache for item feature rows, in front of Feast.

**Why items and not users.** A user row is read once per request by the one
user it belongs to, so a cache of them is a cache of one-hit entries. An ITEM
row is read by every request whose candidate set contains it, and on a news
corpus the candidate sets overlap heavily -- 400 candidates drawn from a 65k
catalogue, skewed toward whatever is currently popular. The same rows are
fetched over and over, within the same second, by different users.

**What it is actually saving.** Measured on BigBox: `GetItems` for 400
candidates costs 9.7 ms at p50 and 126 ms at p99 with nothing else touching the
gateway. The Redis round trip is the smaller half. The larger half is Python
building 400 x 6 = 2,400 feature values into objects on every call -- a big,
short-lived object graph that holds the GIL while it is assembled and then
feeds a generational GC pause, which is what the bimodal tail is (p75 10 ms,
p95 84 ms). That same GIL occupancy is what starves `GetUser`, whose own
service time is 1 ms, into missing its 8 ms budget and collapsing the slate to
popularity.

So this caches the ASSEMBLED ROW, not the Redis response. Skipping the object
construction is the point; skipping the network is a bonus.

**The staleness this buys is real and is bounded on purpose.** `item_hourly_features`
is rebuilt hourly, so a row is at most an hour fresh even with no cache at all
-- but `item_age_hours` and the `_cum` counters move continuously between
materialisations, and ADR 0011 measured a ~29 minute decay optimum on this
corpus. A minute of extra staleness is small against the hour the store is
already behind; an hour of it would not be.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

#: Seconds a row may be reused. Short against the store's hourly cadence, long
#: against any realistic request rate: at 50 rps a popular article is read
#: hundreds of times inside one TTL.
DEFAULT_TTL = 60.0

#: Rows held. The catalogue is ~65k and a row is six floats, so the whole
#: thing fits in a few megabytes; the cap exists so an item map that grew
#: unexpectedly cannot turn a cache into a leak.
DEFAULT_CAPACITY = 100_000


@dataclass(frozen=True)
class Entry:
    """One cached row.

    Attributes:
        values: The item's columns, in `ITEM_COLUMNS` order.
        found: Whether the store had a row. **Cached alongside the values**,
            because a miss is a real answer -- a cold article is normal on a
            news corpus -- and re-asking Feast for a row it does not have costs
            exactly as much as asking for one it does.
        expires_at: Monotonic deadline.
    """

    values: tuple[float, ...]
    found: bool
    expires_at: float


class ItemFeatureCache:
    """Thread-safe, batch-oriented, TTL only.

    **Batch in and batch out, one lock acquisition each.** The caller has 400
    keys, and taking the lock per key would trade the GIL contention this
    exists to remove for lock contention that does the same thing.

    No LRU. Eviction is by expiry, with a capacity backstop that drops the
    soonest-to-expire first. An LRU would need a touch on every read -- a write
    on the hot path, under the lock -- to buy a better eviction order for a
    working set that already fits.
    """

    def __init__(self, ttl: float = DEFAULT_TTL, capacity: int = DEFAULT_CAPACITY) -> None:
        self.ttl = ttl
        self.capacity = capacity
        self._entries: dict[int, Entry] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def take(self, keys: list[int]) -> tuple[dict[int, Entry], list[int]]:
        """Split ``keys`` into what is cached and what has to be fetched.

        Returns:
            ``(hits, misses)``. ``misses`` is DEDUPLICATED and in first-seen
            order: a candidate list may legally repeat an index, and fetching
            it twice would pay the expensive half of this call twice for one
            answer.
        """
        now = time.monotonic()
        hits: dict[int, Entry] = {}
        misses: list[int] = []
        seen: set[int] = set()

        with self._lock:
            for key in keys:
                entry = self._entries.get(key)
                if entry is not None and entry.expires_at > now:
                    hits[key] = entry
                elif key not in seen:
                    seen.add(key)
                    misses.append(key)
            self.hits += len(hits)
            self.misses += len(misses)
        return hits, misses

    def fill(self, rows: dict[int, tuple[tuple[float, ...], bool]]) -> None:
        """Store freshly fetched rows."""
        expires_at = time.monotonic() + self.ttl
        with self._lock:
            for key, (values, found) in rows.items():
                self._entries[key] = Entry(values=values, found=found, expires_at=expires_at)
            if len(self._entries) > self.capacity:
                self._evict()

    def _evict(self) -> None:
        """Drop expired rows, then the soonest-to-expire until under capacity.

        Called with the lock held. Sweeping only on overflow rather than on a
        timer keeps expiry off the read path entirely -- an expired entry is
        simply not a hit, and it costs one dict lookup to find that out.
        """
        now = time.monotonic()
        self._entries = {
            key: entry for key, entry in self._entries.items() if entry.expires_at > now
        }
        if len(self._entries) <= self.capacity:
            return
        ordered = sorted(self._entries.items(), key=lambda pair: pair[1].expires_at)
        for key, _ in ordered[: len(self._entries) - self.capacity]:
            del self._entries[key]

    @property
    def hit_rate(self) -> float:
        """Share of lookups served from cache, over the process's lifetime.

        Reported rather than inferred. §14.4's own lesson: a cache that has
        quietly stopped being hit is a latency regression with no error
        attached, and the retrieval sidecar carries `embedding_cached` on every
        response for exactly this reason.
        """
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def __len__(self) -> int:
        return len(self._entries)
