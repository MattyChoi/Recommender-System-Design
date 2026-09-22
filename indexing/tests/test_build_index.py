"""The index's two silent failure modes.

Neither raises, neither shows up in a recall curve, and both produce a table
that looks fine:

**Un-normalised vectors.** An inner-product index still builds and still returns
neighbours -- it just ranks by ``|u||v|cos`` instead of ``cos``, so long vectors
win. Every number downstream is then about a metric nobody chose.

**The reserved row.** Item indices are 1-based with 0 held for OOV, but FAISS
positions are 0-based over the vectors it was handed. Forget the shift and every
returned id names the article next to the right one -- which is wrong in a way
that still recalls plausible news articles.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from indexing.build_index import (
    POINTS_PER_CENTROID,
    assert_unit_norm,
    build_flat,
    build_hnsw,
    recommended_nlist,
    search,
    to_item_ids,
)

DIM = 16


def _unit(rows: int, seed: int = 0) -> npt.NDArray[np.float32]:
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(rows, DIM))
    scaled: npt.NDArray[np.float32] = np.ascontiguousarray(
        vectors / np.linalg.norm(vectors, axis=1, keepdims=True), dtype=np.float32
    )
    return scaled


class TestTheMetric:
    def test_unit_vectors_are_accepted(self) -> None:
        assert_unit_norm(_unit(8))

    def test_unscaled_vectors_are_refused(self) -> None:
        with pytest.raises(ValueError, match="unit-norm"):
            assert_unit_norm(_unit(8) * 2.0)

    def test_the_two_fixtures_can_actually_differ(self) -> None:
        """Precondition: the check keys on the norm and on nothing else.

        The refused fixture is the accepted one times a scalar, so it has the
        same directions, the same dtype and the same shape. Anything the guard
        might have keyed on instead is held fixed.
        """
        vectors = _unit(8)
        assert np.allclose(
            vectors / np.linalg.norm(vectors, axis=1, keepdims=True),
            (vectors * 2.0) / np.linalg.norm(vectors * 2.0, axis=1, keepdims=True),
        )

    def test_a_single_long_vector_is_enough_to_refuse_the_table(self) -> None:
        """One is enough: it wins every query it is remotely near."""
        vectors = _unit(8)
        vectors[3] *= 1.5

        with pytest.raises(ValueError):
            assert_unit_norm(vectors)


class TestTheReservedRow:
    def test_position_zero_is_item_one(self) -> None:
        assert to_item_ids(np.array([[0, 1, 2]])).tolist() == [[1, 2, 3]]

    def test_a_missing_neighbour_becomes_the_reserved_index(self) -> None:
        """FAISS says -1; every other module in this project says 0."""
        assert to_item_ids(np.array([[5, -1]])).tolist() == [[6, 0]]

    def test_an_exact_search_returns_the_item_it_was_given(self) -> None:
        """End to end: query with row p, get item p + 1 back first."""
        vectors = _unit(32)
        index = build_flat(vectors)

        got = search(index, vectors[7:8], k=3)

        assert got[0, 0] == 8
        assert (got > 0).all()


class TestNlist:
    def test_it_never_asks_for_more_cells_than_it_can_train(self) -> None:
        assert recommended_nlist(65_238) * POINTS_PER_CENTROID <= 65_238

    def test_a_copied_in_4096_would_have_been_undertrained_here(self) -> None:
        """The number this replaces, and why it is not a style preference."""
        assert 4096 * POINTS_PER_CENTROID > 65_238
        assert recommended_nlist(65_238) < 4096

    def test_a_large_catalogue_reaches_the_ceiling(self) -> None:
        assert recommended_nlist(10_000_000) == 4096

    def test_an_empty_catalogue_is_refused(self) -> None:
        with pytest.raises(ValueError):
            recommended_nlist(0)


class TestApproximation:
    def test_hnsw_finds_its_own_vectors(self) -> None:
        """A sanity check on the ruler, not on the model.

        Querying an index with a vector it contains must return that vector
        first. If this fails the metric, the offset or the build is wrong, and
        no recall curve taken afterwards means anything.
        """
        vectors = _unit(256)
        index = build_hnsw(vectors, m=16)
        index.hnsw.efSearch = 64

        got = search(index, vectors[:20], k=5)

        assert (got[:, 0] == np.arange(1, 21)).mean() > 0.95
