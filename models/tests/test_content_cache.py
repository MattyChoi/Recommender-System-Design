"""The index maps, and why a rare value gets no index of its own.

``_index_map`` is four lines and looks like it cannot be wrong, but two of its
properties are load-bearing elsewhere and nothing else checks either.

Index 0 must stay empty. The item tower declares ``padding_idx=0`` on both the
category and subcategory embeddings, so index 0 is the row torch holds at zero
and never trains. Handing a real value that index would give every item in it a
guaranteed-zero embedding, silently.

And the floor is an ablation-integrity control, not a size optimisation. On this
catalogue 91 of 270 subcategories have fewer than five items and at least 25 are
singletons. A singleton subcategory embedding is trained by one item and read by
that item alone -- a per-item free parameter, which is exactly what
``use_id=False`` is supposed to remove in G1's content-only arm. The floor folds
those to 0, where ``padding_idx`` zeroes them.
"""

from __future__ import annotations

from models.retrieval.content_cache import _index_map


class TestTheMapItself:
    def test_indices_are_contiguous_from_one(self) -> None:
        """0 is reserved for padding, so the first real value is 1."""
        got = _index_map(["b", "a", "c"])
        assert sorted(got.values()) == [1, 2, 3]

    def test_the_order_is_sorted_not_encounter(self) -> None:
        """Two builds of the same catalogue must agree on what index 7 means.

        Encounter order would make the map a function of Spark's row order,
        which the reproducibility fix in E-series taught us not to trust.
        """
        assert _index_map(["c", "a", "b"]) == _index_map(["a", "b", "c"])

    def test_nulls_and_empties_get_no_index(self) -> None:
        got = _index_map(["a", None, "", "b"])
        assert got == {"a": 1, "b": 2}


class TestTheFloor:
    def test_the_default_keeps_everything(self) -> None:
        """min_count=1 is the pre-floor behaviour, so the argument alone is a no-op."""
        values = ["a", "a", "b", "c", "c", "c"]
        assert _index_map(values) == _index_map(values, 1)

    def test_a_value_below_the_floor_is_absent(self) -> None:
        """Absent, not mapped to 0 -- the call site's ``.get(v, 0)`` does the fold.

        Keeping the map free of folded keys is what lets the written
        ``*_map.parquet`` describe exactly the indices that exist.
        """
        got = _index_map(["a"] * 5 + ["rare"], min_count=5)
        assert got == {"a": 1}

    def test_the_floor_is_inclusive(self) -> None:
        """A value with exactly min_count items has earned its index."""
        assert "b" in _index_map(["a"] * 9 + ["b"] * 3, min_count=3)

    def test_folding_renumbers_rather_than_leaving_holes(self) -> None:
        """The embedding table is sized from len(map), so a hole would waste a row
        and, worse, let an index exceed the table if anyone sized it from max()."""
        got = _index_map(["a", "a", "rare", "z", "z"], min_count=2)
        assert got == {"a": 1, "z": 2}

    def test_a_floor_above_everything_empties_the_map(self) -> None:
        """Degenerate but reachable by a bad MIN_COUNT, and it must not raise.

        Every item then falls to 0 and the feature stops contributing, which is
        a legible outcome; an exception mid-encode after a GPU pass is not.
        """
        assert _index_map(["a", "b"], min_count=99) == {}
