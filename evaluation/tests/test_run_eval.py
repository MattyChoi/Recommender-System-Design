"""The eval harness's row ordering.

These tests exist because of a bug that produced no error and no warning: the
report cards were a function of ``spark.master``. ``local[*]`` means "this
machine's core count", ``toPandas()`` returns partition order, and every
ranking metric breaks ties by input order -- so two runs of identical code over
identical data returned non-overlapping cold-item confidence intervals on two
machines, and the only symptom was a results table that would not reproduce.

The property worth testing is not "the sort works". It is that the report is
INVARIANT to the order rows arrive in, which is the thing that was false.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.offline.run_eval import _canonical_order


def _rows(seed: int) -> pd.DataFrame:
    """Four slates of five rows, permuted by ``seed``.

    Scores are deliberately coarse so that ties are the common case rather than
    the exception -- which is the regime the real corpus is in, where ALS scores
    every cold user's whole slate at exactly 0.0.
    """
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "impression_id": np.repeat(["i1", "i2", "i3", "i4"], 5),
            "item_idx": np.tile(np.arange(1, 6), 4),
            "score": np.tile([0.0, 0.0, 0.0, 0.5, 0.0], 4),
            "clicked": np.tile([0, 1, 0, 0, 0], 4),
        }
    )
    return frame.iloc[rng.permutation(len(frame))].reset_index(drop=True)


class TestCanonicalOrder:
    def test_it_is_invariant_to_the_input_permutation(self) -> None:
        """The property the bug violated: arrival order must not survive."""
        first = _canonical_order(_rows(0))
        for seed in range(1, 8):
            pd.testing.assert_frame_equal(first, _canonical_order(_rows(seed)))

    def test_it_is_idempotent(self) -> None:
        once = _canonical_order(_rows(3))
        pd.testing.assert_frame_equal(once, _canonical_order(once))

    def test_it_preserves_every_row(self) -> None:
        """Ordering, not filtering. A sort that quietly drops rows would move
        every metric while looking like it worked."""
        source = _rows(1)
        ordered = _canonical_order(source)

        assert len(ordered) == len(source)
        assert sorted(zip(ordered["impression_id"], ordered["item_idx"], strict=True)) == sorted(
            zip(source["impression_id"], source["item_idx"], strict=True)
        )

    def test_it_keeps_each_slate_contiguous(self) -> None:
        """Not required by the metrics, which group before they sort, but it
        makes the collected frame readable when something has to be debugged by
        eye."""
        ids = _canonical_order(_rows(2))["impression_id"].tolist()
        assert ids == sorted(ids)
        assert len(set(ids)) == 4

    def test_it_does_not_order_ties_by_item_idx(self) -> None:
        """The rejected alternative, asserted so nobody 'simplifies' it back.

        Sorting ties by ``item_idx`` would also be reproducible -- and would
        hand every tie to the lowest-numbered item, which biases ``coverage@k``
        for precisely the flat-slate models that metric exists to catch. A hash
        is reproducible AND unbiased, so at least one slate must come back in an
        order that ascending ``item_idx`` would not produce.
        """
        ordered = _canonical_order(_rows(0))
        per_slate = [
            group["item_idx"].tolist() for _, group in ordered.groupby("impression_id", sort=True)
        ]

        assert any(idx != sorted(idx) for idx in per_slate)

    def test_it_leaves_no_working_column_behind(self) -> None:
        """The tiebreak is scaffolding. Leaking it into the frame would put it
        on the collected rows and, eventually, into something that iterates
        columns."""
        assert "_tiebreak" not in _canonical_order(_rows(0)).columns

    @pytest.mark.parametrize("dtype", ["int32", "int64"])
    def test_the_key_does_not_depend_on_the_index_dtype(self, dtype: str) -> None:
        """``item_idx`` arrives from Parquet and its width is not guaranteed
        across a schema change. The hash is taken over a normalised copy, so a
        narrower column must not renumber every tie and silently move the
        table."""
        wide = _rows(0)
        narrow = wide.assign(item_idx=np.asarray(wide["item_idx"], dtype=dtype))

        pd.testing.assert_frame_equal(
            _canonical_order(wide)[["impression_id", "score", "clicked"]],
            _canonical_order(narrow)[["impression_id", "score", "clicked"]],
        )
