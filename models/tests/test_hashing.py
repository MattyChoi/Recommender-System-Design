"""The hashing trick's three rates, and the two ways the benchmark could lie.

**The first is the hash itself.** A hash with structure in it produces a
collision rate that looks like a property of the technique and is really a
property of the hash. So the measured rate is checked against the closed form it
should match, and the identity map -- which is what the manual's
``hash(item_idx) % b`` actually computes, since ``hash()`` is the identity on
small ints -- is checked to be different from what this module does.

**The second is the denominator, and it is the whole point of the module.**
``catalogue``, ``trained`` and ``traffic`` are three different questions. If they
could not differ, reporting all three would be padding. So there is a
constructed case where all three take different values, built before any of them
is quoted -- the standing rule about checking A and B *can* differ, applied to
three things at once.

No ``importorskip`` here: ``hashing`` deliberately imports ``sizing`` rather than
``sharded_embeddings``, so this half of G4 runs on a machine that has never
installed the ``sharded`` extra.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from models.layers.hashing import (
    bucket_of,
    collisions,
    expected_collision_rate,
)


def _ids(count: int) -> list[str]:
    """MIND-shaped string ids: ``N1`` .. ``N<count>``."""
    return [f"N{n}" for n in range(1, count + 1)]


class TestTheHash:
    def test_it_is_reproducible(self) -> None:
        """blake2b, not ``hash()``. Python salts ``hash()`` on strings per
        process unless PYTHONHASHSEED is pinned, so the manual's version puts an
        item in a different bucket on every run -- the same defect class as
        ``F.shuffle`` in G3's negatives and ``f.rand`` in the random baseline."""
        ids = _ids(500)

        assert np.array_equal(bucket_of(ids, 64), bucket_of(ids, 64))

    def test_it_is_not_the_identity_on_the_index(self) -> None:
        """``hash()`` on an int below 2**61 IS that int, so the manual's
        ``hash(item_idx) % b`` is ``item_idx % b`` -- a perfectly uniform
        round-robin that measures the index, not a hash. Anything learned from
        it would not transfer to a system hashing real string ids."""
        ids = _ids(500)
        identity = np.arange(1, 501) % 64

        assert not np.array_equal(bucket_of(ids, 64), identity)

    def test_every_bucket_is_in_range(self) -> None:
        buckets = bucket_of(_ids(1000), 37)

        assert buckets.min() >= 0
        assert buckets.max() < 37

    @pytest.mark.parametrize("n_items,n_buckets", [(1000, 1000), (5000, 2500), (5000, 500)])
    def test_the_measured_rate_matches_the_closed_form(self, n_items: int, n_buckets: int) -> None:
        """The hash-quality check. A measured rate far above ``1 - (1-1/b)^(n-1)``
        means the hash is clumping, and the collision numbers would then be
        about blake2b rather than about the technique."""
        counts = np.ones(n_items, dtype=np.int64)
        measured = collisions(_ids(n_items), counts, n_buckets).catalogue

        assert measured == pytest.approx(expected_collision_rate(n_items, n_buckets), abs=0.03)


class TestTheClosedForm:
    def test_one_bucket_collides_everything(self) -> None:
        assert expected_collision_rate(100, 1) == pytest.approx(1.0)

    def test_a_single_item_never_collides(self) -> None:
        """``n=1`` has nothing to collide with, whatever the table width. The
        exponent is ``n-1``, and an off-by-one there would read as a plausible
        small rate rather than as a bug."""
        assert expected_collision_rate(1, 1000) == pytest.approx(0.0)

    def test_a_vast_table_barely_collides(self) -> None:
        assert expected_collision_rate(1000, 10_000_000) < 0.001

    def test_zero_buckets_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            expected_collision_rate(100, 0)


class TestTheThreeRates:
    def test_one_bucket_saturates_all_three(self) -> None:
        """The degenerate end, pinned because it is the only point where all
        three rates must agree and it proves they are measuring the same event."""
        ids = _ids(50)
        counts = np.zeros(50, dtype=np.int64)
        counts[:10] = 1

        got = collisions(ids, counts, 1)

        assert got.catalogue == pytest.approx(1.0)
        assert got.trained == pytest.approx(1.0)
        assert got.traffic == pytest.approx(1.0)

    def test_untrained_items_inflate_the_catalogue_rate(self) -> None:
        """**The finding the module exists to report.** 89% of this catalogue
        never receives a gradient, and a collision between two such rows costs
        nothing. So the catalogue rate is not an upper bound on damage that is
        merely loose -- it is answering a different question."""
        ids = _ids(1000)
        counts = np.zeros(1000, dtype=np.int64)
        counts[:50] = 1

        got = collisions(ids, counts, 1000)

        assert got.catalogue > 0.5
        assert got.trained < 0.1

    def _skewed(self, colliding: bool) -> tuple[list[str], npt.NDArray[np.int64]]:
        """50 trained items, all clicked, with one of them clicked 10,000 times.

        ``counts`` does DOUBLE DUTY -- it decides which rows are trained AND how
        much traffic each carries -- and that coupling is correct rather than a
        wart: an item trains *because* it was clicked, so "trained with zero
        clicks" is not a state that exists. An earlier version of these tests
        zeroed the other 49 to isolate the weighting and left a single trained
        item, which cannot collide with another trained item by definition. The
        rates came back 0.0 and the fixture, not the code, was wrong.
        """
        ids = _ids(1000)
        buckets = bucket_of(ids, 1000)
        trained_buckets = buckets[:50]
        occupancy = np.bincount(trained_buckets, minlength=1000)
        shares = occupancy[trained_buckets] > 1

        assert shares.any(), "fixture cannot exercise the difference: no trained collisions"
        assert not shares.all(), "fixture cannot exercise the difference: all trained collide"

        counts = np.zeros(1000, dtype=np.int64)
        counts[:50] = 1
        counts[np.flatnonzero(shares if colliding else ~shares)[0]] = 10_000
        return ids, counts

    def test_traffic_falls_below_trained_when_the_head_item_is_safe(self) -> None:
        """Constructed so the two must differ, because a report quoting two
        numbers that are always equal is quoting one number twice.

        Nearly all the traffic sits on a trained item that does NOT share its
        bucket, so ``traffic`` collapses while ``trained`` -- which counts items,
        not clicks -- does not move.
        """
        ids, counts = self._skewed(colliding=False)

        got = collisions(ids, counts, 1000)

        assert got.trained > 0.05
        assert got.traffic < got.trained / 10

    def test_traffic_exceeds_trained_when_the_head_item_collides(self) -> None:
        """The other direction. Together these bracket the weighting: it moves
        the number both ways, so it is doing work rather than tracking
        ``trained`` by construction."""
        ids, counts = self._skewed(colliding=True)

        got = collisions(ids, counts, 1000)

        assert got.traffic > 0.9
        assert got.trained < 0.5


class TestMemory:
    def test_halving_the_buckets_halves_the_table(self) -> None:
        ids = _ids(1000)
        counts = np.ones(1000, dtype=np.int64)

        got = collisions(ids, counts, 500, dim=64)

        assert got.bytes_saved == 500 * 64 * 4

    def test_a_full_width_table_saves_nothing(self) -> None:
        """The control for the memory column: hashing into as many buckets as
        there are items saves zero bytes and still collides, which is the honest
        summary of what the trick costs before it pays."""
        ids = _ids(1000)
        counts = np.ones(1000, dtype=np.int64)

        got = collisions(ids, counts, 1000, dim=64)

        assert got.bytes_saved == 0
        assert got.catalogue > 0.5
