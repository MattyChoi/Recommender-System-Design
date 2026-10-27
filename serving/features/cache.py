"""A TTL cache for item feature rows, in front of Feast.

**Why items and not users.** A user row is read once per request by the one
user it belongs to, so a cache of them is a cache of one-hit entries. An ITEM
row is read by every request whose candidate set contains it, and on a news
corpus the candidate sets overlap heavily -- 400 candidates drawn from a 65k
catalogue, skewed toward whatever is currently popular. The same rows are
fetched over and over, within the same second, by different users.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

DEFAULT_TTL = 60.0
DEFAULT_CAPACITY = 1_000_000


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

    Batch in and batch out, one lock acquisition each. No LRU. Eviction is by expiry.
    """

    def __init__(self, ttl: float = DEFAULT_TTL, capacity: int = DEFAULT_CAPACITY) -> None:
        self.ttl = ttl
        self.capacity = capacity
        self._entries: OrderedDict[int, Entry] = OrderedDict()
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
                if key in hits or key in seen:
                    continue
                entry = self._entries.get(key)
                if entry is not None and entry.expires_at > now:
                    hits[key] = entry
                    continue
                if entry is not None:
                    del self._entries[key]
                seen.add(key)
                misses.append(key)
            self.hits += len(hits)
            self.misses += len(misses)

        return hits, misses

    def put(self, rows: dict[int, tuple[tuple[float, ...], bool]]) -> None:
        """Store freshly fetched rows."""
        expires_at = time.monotonic() + self.ttl
        with self._lock:
            for key, (values, found) in rows.items():
                self._entries[key] = Entry(values=values, found=found, expires_at=expires_at)
                self._entries.move_to_end(key)
            if len(self._entries) > self.capacity:
                self._evict()

    def _evict(self) -> None:
        """Drop expired rows, then the soonest-to-expire until under capacity."""
        now = time.monotonic()
        while self._entries:
            oldest = next(iter(self._entries.values()))
            if oldest.expires_at > now and len(self._entries) <= self.capacity:
                return
            self._entries.popitem(last=False)

    @property
    def hit_rate(self) -> float:
        """Share of lookups served from cache, over the process's lifetime."""
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def __len__(self) -> int:
        return len(self._entries)
