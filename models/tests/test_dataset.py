"""Assembling the model's inputs, and the two mistakes that would be silent.

The first is alignment. ``content[k]`` must be item ``k``'s vector, and B2 is
explicit about the cost of getting it wrong: one direction is an ``IndexError``
on the highest-numbered item, which is the last one anybody tests by hand; the
other is no error at all, just every item reading its neighbour's vector
forever. The loader defends against it structurally -- index assignment rather
than row order -- and the test below shuffles the input rows to prove that.

The second is feature leakage into the user tower. ``user_cat_affinity`` and its
siblings are keyed on the CANDIDATE's category, so putting one in the user tower
makes the user embedding a function of the item and quietly destroys the
precomputation the whole architecture exists for. Nothing would error. The
guard at the bottom of this file is what stops it being re-added.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from common.config import Settings, load_settings
from models.retrieval.dataset import (
    LOG1P_FEATURES,
    USER_FEATURES,
    USER_FLAGS,
    ItemTables,
    _cyclic,
    _pad_ragged,
    load_item_tables,
)
from tests.test_config import REPO_ROOT

N_ITEMS = 4
DIM = 3


def _write_cache(root: Path, order: list[int]) -> None:
    """Write an item_content cache with rows in the given item_idx order.

    Item ``k`` always gets the vector ``[k, k, k]``, category ``k`` and
    subcategory ``10 - k``, whatever order the rows land in.
    """
    root.mkdir(parents=True, exist_ok=True)
    flat = pa.array(
        np.array([[k] * DIM for k in order], dtype="float32").reshape(-1),
        type=pa.float32(),
    )
    pq.write_table(
        pa.table(
            {
                "item_idx": pa.array(order, type=pa.int32()),
                "category_idx": pa.array(order, type=pa.int32()),
                "subcategory_idx": pa.array([10 - k for k in order], type=pa.int32()),
                "vec_title_abstract": pa.FixedSizeListArray.from_arrays(flat, DIM),
                "vec_title": pa.FixedSizeListArray.from_arrays(flat, DIM),
            }
        ),
        root / "part-00000.parquet",
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    local = load_settings(REPO_ROOT).model_copy(deep=True)
    local.paths.gold = tmp_path
    return local


class TestAlignment:
    @pytest.mark.parametrize(
        "order",
        [[1, 2, 3, 4], [4, 3, 2, 1], [3, 1, 4, 2]],
        ids=["sorted", "reversed", "shuffled"],
    )
    def test_row_order_does_not_change_the_matrix(
        self, settings: Settings, order: list[int]
    ) -> None:
        """The test that matters. Parquet row order is not a contract, and a
        loader that sorted and trusted would pass on the first ordering."""
        _write_cache(Path(settings.paths.gold) / "item_content", order)

        tables = load_item_tables(settings)

        for k in range(1, N_ITEMS + 1):
            assert torch.equal(tables.content[k], torch.full((DIM,), float(k)))
            assert int(tables.category[k]) == k
            assert int(tables.subcategory[k]) == 10 - k

    def test_the_reserved_row_is_zero(self, settings: Settings) -> None:
        """TwoTower zeroes row 0 on the way in, so this is not the last line of
        defence -- but a non-zero row 0 HERE means the item indices are
        misaligned, and the model's zeroing would hide that rather than fix it.
        """
        _write_cache(Path(settings.paths.gold) / "item_content", [3, 1, 4, 2])

        tables = load_item_tables(settings)

        assert torch.equal(tables.content[0], torch.zeros(DIM))
        assert int(tables.category[0]) == 0 and int(tables.subcategory[0]) == 0

    def test_the_matrix_is_n_items_plus_one(self, settings: Settings) -> None:
        _write_cache(Path(settings.paths.gold) / "item_content", [1, 2, 3, 4])

        tables = load_item_tables(settings)

        assert tables.content.shape == (N_ITEMS + 1, DIM)
        assert tables.category.shape == (N_ITEMS + 1,)

    def test_a_gap_in_the_indices_is_refused(self, settings: Settings) -> None:
        """A missing index leaves a zero row that looks exactly like the
        reserved one, so an item with no content would be indistinguishable
        from OOV -- and the model would treat it as padding."""
        _write_cache(Path(settings.paths.gold) / "item_content", [1, 2, 4])

        with pytest.raises(ValueError, match="not dense"):
            load_item_tables(settings)

    def test_an_unknown_variant_is_refused(self, settings: Settings) -> None:
        _write_cache(Path(settings.paths.gold) / "item_content", [1, 2, 3, 4])

        with pytest.raises(ValueError, match="variant must be"):
            load_item_tables(settings, variant="vec_nonsense")

    def test_both_variants_are_readable(self, settings: Settings) -> None:
        """vec_title exists so the neural content tower can be compared against
        F3's baseline with the input held fixed."""
        _write_cache(Path(settings.paths.gold) / "item_content", [1, 2, 3, 4])

        for variant in ("vec_title", "vec_title_abstract"):
            assert isinstance(load_item_tables(settings, variant), ItemTables)


class TestRaggedPadding:
    """Shared by the click history and G3's slate negatives -- both are ragged
    per-request item indices needing the same two guarantees."""

    def test_shorter_histories_are_padded_in_ids_and_mask(self) -> None:
        ids, mask = _pad_ragged([np.array([7, 8])], max_len=4)

        assert ids[0].tolist() == [7, 8, 0, 0]
        assert mask[0].tolist() == [1, 1, 0, 0]

    def test_truncation_drops_the_oldest(self) -> None:
        """The table is most-recent-first, so a prefix keeps the freshest
        clicks. Taking a suffix would silently keep the stalest."""
        ids, mask = _pad_ragged([np.array([9, 8, 7, 6, 5])], max_len=3)

        assert ids[0].tolist() == [9, 8, 7]
        assert mask[0].tolist() == [1, 1, 1]

    @pytest.mark.parametrize("empty", [None, np.array([], dtype="int64")])
    def test_an_absent_history_is_all_padding(self, empty: npt.NDArray[np.int64] | None) -> None:
        """88% of dev's users are new; a left join with no match gives None and
        a user with a snapshot but no clicks gives an empty array. Both are
        ordinary states."""
        ids, mask = _pad_ragged([empty], max_len=3)

        assert ids[0].tolist() == [0, 0, 0]
        assert mask[0].tolist() == [0, 0, 0]

    def test_rows_stay_independent(self) -> None:
        ids, mask = _pad_ragged([np.array([1]), None, np.array([2, 3])], max_len=2)

        assert ids.tolist() == [[1, 0], [0, 0], [2, 3]]
        assert mask.tolist() == [[1, 0], [0, 0], [1, 1]]

    def test_a_request_with_no_negatives_is_all_padding(self) -> None:
        """A slate whose every item was clicked has no row in the negatives
        table, so the left join gives None -- the same shape as a user with no
        history, and it must train on in-batch negatives rather than error."""
        ids, mask = _pad_ragged([None, np.array([5, 6])], max_len=4)

        assert ids[0].tolist() == [0, 0, 0, 0]
        assert mask[0].tolist() == [0, 0, 0, 0]
        assert mask[1].tolist() == [1, 1, 0, 0]


class TestCyclicEncoding:
    def test_adjacent_hours_are_close_across_the_wrap(self) -> None:
        """The whole reason for sin/cos. As raw integers 23 and 0 are the two
        most distant hours in the day, which is exactly backwards."""
        sin, cos = _cyclic(np.array([23.0, 0.0, 12.0]), 24)
        points = np.stack([sin, cos], axis=1)

        wrap = np.linalg.norm(points[0] - points[1])
        opposite = np.linalg.norm(points[1] - points[2])

        assert wrap < opposite

    def test_the_encoding_is_on_the_unit_circle(self) -> None:
        sin, cos = _cyclic(np.arange(24, dtype="float32"), 24)

        assert np.allclose(sin**2 + cos**2, 1.0, atol=1e-6)

    def test_a_full_period_returns_to_the_start(self) -> None:
        sin, cos = _cyclic(np.array([0.0, 7.0]), 7)

        assert math.isclose(sin[0], sin[1], abs_tol=1e-6)
        assert math.isclose(cos[0], cos[1], abs_tol=1e-6)


class TestTheUserTowerSeesNoCrossFeatures:
    def test_no_candidate_dependent_column_is_a_user_feature(self) -> None:
        """The guard for the mistake nothing would report.

        ``user_cat_*`` is keyed on (user, CANDIDATE category), and ``item_*``
        and ``cat_*`` are candidate columns outright. Any of them in the user
        tower makes the user embedding a function of the item, so one user
        vector can no longer be ANN-searched against a precomputed index -- the
        precomputation the two-tower exists for. They belong to Part K's
        ranker, which sees both sides anyway.
        """
        forbidden = ("user_cat_", "item_", "cat_")
        offenders = [name for name in (*USER_FEATURES, *USER_FLAGS) if name.startswith(forbidden)]

        assert not offenders, (
            f"{offenders} depend on the candidate; the user tower must not see "
            "them. Move them to the ranker."
        )

    def test_every_log1p_feature_is_actually_a_user_feature(self) -> None:
        """A typo here would silently skip the transform, leaving an unbounded
        count next to a rate in [0, 1]."""
        assert set(USER_FEATURES) >= LOG1P_FEATURES

    def test_the_feature_order_is_a_tuple(self) -> None:
        """Order is the column order of the tensor the model sees, so it is
        part of the checkpoint contract. A set would reorder between runs."""
        assert isinstance(USER_FEATURES, tuple)
        assert isinstance(USER_FLAGS, tuple)
