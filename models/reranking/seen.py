"""A seen-list that is allowed to be wrong, in one direction, on purpose.

Filtering already-shown items needs a membership test per (user, candidate).
Exact sets are the obvious answer and do not survive scale: 10M users x a few
hundred recent items x an 8-byte id is tens of gigabytes of Redis, and it grows
with engagement. A Bloom filter holds the same question in a fixed number of
bits per user and answers it wrongly at a known rate.

**The data structure is not the interesting part; the error DIRECTION is.**
A Bloom filter never says "not seen" about an item it holds, and sometimes says
"seen" about one it does not. So the two failure modes are:

- a **false positive** drops a fresh item the user has never been shown. One of
  a hundred candidates disappears and the slot goes to the next-best item. The
  user sees a marginally worse recommendation and nobody can tell.
- a **false negative** would re-show an item the user just saw. This is the
  failure users actually notice and complain about -- and it is the one a Bloom
  filter **cannot** produce.

The structure's error lands entirely on the side the product can absorb. That
is why it is the right choice here and why "we used a Bloom filter" is a weaker
answer than "we accepted a 1% chance of hiding a good item to make re-showing a
seen item impossible".

⚠️ **The guarantee is one-sided only while the filter is never cleared and
never shared.** A filter reset on a schedule can re-show an item after the
reset; a filter keyed by anything coarser than the user (a cohort, say) will
hide items for people who never saw them, at a rate nobody measured.

⚠️ **And it stops being useful before it stops being correct.** Measured in
``seen_bench``: a filter sized for 100 items and given 1,000 reaches 100% bit
load and answers "seen" to EVERYTHING. It has still never produced a false
negative, and it now hides every candidate -- an empty slate rather than a
degraded one. Capacity is therefore a number to monitor rather than to set
once, and bit **load** is the alarm: it is observable per user and it moves
long before the false-positive rate does.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class BloomFilter:
    """One user's seen-list.

    Not thread-safe and not distributed: this is the offline model of the
    structure, used to measure the false-positive rate that a Redis-backed
    implementation would inherit. The serving version holds the same bits in a
    Redis bitfield and runs the same hash.
    """

    bits: int
    hashes: int

    def __post_init__(self) -> None:
        self._bitmap = np.zeros(self.bits, dtype=bool)

    def _positions(self, item: int) -> npt.NDArray[np.int64]:
        """``k`` bit positions for one item.

        A named digest, never ``hash()``: Python salts ``hash`` per process for
        strings and returns the value itself for small ints, so a filter built
        in one process would not answer the same way in another -- and the
        serving side is a different process by definition.
        """
        digest = hashlib.blake2b(item.to_bytes(8, "big"), digest_size=16).digest()
        base = int.from_bytes(digest[:8], "big")
        step = int.from_bytes(digest[8:], "big") | 1  # odd, so it generates the ring
        return np.array(
            [(base + index * step) % self.bits for index in range(self.hashes)], dtype=np.int64
        )

    def add(self, item: int) -> None:
        self._bitmap[self._positions(item)] = True

    def __contains__(self, item: int) -> bool:
        return bool(self._bitmap[self._positions(item)].all())

    @property
    def load(self) -> float:
        """Share of bits set. Past ~50% the false-positive rate runs away."""
        return float(self._bitmap.mean())


def sizing(capacity: int, false_positive_rate: float) -> tuple[int, int]:
    """Bits and hash count for a target rate, from the standard formulas.

    Args:
        capacity: Items expected per user.
        false_positive_rate: Target, e.g. 0.01.

    Returns:
        ``(bits, hashes)``.

    Raises:
        ValueError: On a non-positive capacity or a rate outside ``(0, 1)``.
    """
    if capacity <= 0:
        raise ValueError(f"capacity must be positive; got {capacity}")
    if not 0.0 < false_positive_rate < 1.0:
        raise ValueError(f"false_positive_rate must be in (0, 1); got {false_positive_rate}")

    bits = math.ceil(-capacity * math.log(false_positive_rate) / (math.log(2) ** 2))
    hashes = max(1, round(bits / capacity * math.log(2)))
    return bits, hashes


def measure_false_positives(
    seen: list[int], probes: list[int], capacity: int, target_rate: float
) -> dict[str, float]:
    """Build a filter from ``seen`` and probe it with items it does not hold.

    Returns the MEASURED rate beside the target, because the formula assumes
    ideal independent hashes and a specific load, and a filter given more items
    than its capacity degrades quietly rather than failing.
    """
    bits, hashes = sizing(capacity, target_rate)
    filter_ = BloomFilter(bits=bits, hashes=hashes)
    for item in seen:
        filter_.add(item)

    held = set(seen)
    clean = [item for item in probes if item not in held]
    positives = sum(1 for item in clean if item in filter_)

    return {
        "bits": float(bits),
        "bytes_per_user": bits / 8.0,
        "hashes": float(hashes),
        "target_rate": target_rate,
        "measured_rate": positives / len(clean) if clean else float("nan"),
        "load": filter_.load,
        "items_held": float(len(held)),
        "probes": float(len(clean)),
    }
