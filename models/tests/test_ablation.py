"""Pairing two arms, and the one failure that would look like a result.

A paired bootstrap earns its tight interval by cancelling per-request variance.
That only works if row *i* is the same request in both files. Two arms scored
against different holdout windows, or across a gold rebuild, still pair
row-for-row and still return a narrow interval -- around a difference between
unrelated requests. Nothing downstream reports it, the plot looks fine, and the
number is fiction. `check_aligned` is the guard, and most of this file exercises
it.

The second theme is that a band too thin to test must not read as a null result.
`BandComparison.result` is `None` there rather than a zero difference.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from models.classes.train import Hits
from models.retrieval.ablation import (
    Arm,
    BandComparison,
    check_aligned,
    compare,
    default_plot,
    load_arm,
    per_user,
    render,
)
from models.retrieval.evaluate import LONG_TAIL_BELOW, band_labels, save, summarise

ROWS = 12


def _arm(name: str, hit: list[int], users: list[int] | None = None) -> Arm:
    """Twelve rows over four users and three bands, by default."""
    return Arm(
        name=name,
        hit=np.array(hit, dtype=bool),
        item_ids=np.arange(1, ROWS + 1, dtype=np.int64),
        user_ids=np.array(users if users is not None else [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]),
        band=np.array([0, 0, 0, 0, 1, 1, 1, 1, 8, 8, 8, 8], dtype=np.int64),
        popularity=np.zeros(ROWS, dtype=bool),
    )


class TestTheAlignmentGuard:
    def test_identical_rows_are_accepted(self) -> None:
        check_aligned(_arm("a", [1] * ROWS), _arm("b", [0] * ROWS))

    def test_different_items_are_refused(self) -> None:
        """Two files from different holdout windows. They would pair
        row-for-row and return an interval around nothing."""
        left = _arm("a", [1] * ROWS)
        right = _arm("b", [0] * ROWS)
        shifted = replace(right, item_ids=right.item_ids + 1)

        with pytest.raises(ValueError, match="item_ids"):
            check_aligned(left, shifted)

    def test_different_users_are_refused(self) -> None:
        """The pairing key itself. Wrong here and every user is compared
        against a stranger."""
        left = _arm("a", [1] * ROWS)
        right = _arm("b", [0] * ROWS, users=[3, 3, 3, 2, 2, 2, 1, 1, 1, 0, 0, 0])

        with pytest.raises(ValueError, match="user_ids"):
            check_aligned(left, right)

    def test_different_banding_is_refused(self) -> None:
        """Same rows, different popularity axis -- which happens if the gold
        tables were rebuilt between the two scorings."""
        left = _arm("a", [1] * ROWS)
        right = _arm("b", [0] * ROWS)
        rebanded = replace(right, band=np.zeros(ROWS, dtype=np.int64))

        with pytest.raises(ValueError, match="band"):
            check_aligned(left, rebanded)

    def test_a_different_row_count_is_refused_before_comparing(self) -> None:
        """np.array_equal on mismatched shapes returns False rather than
        raising, so the shape check is not redundant -- it is what makes the
        message name the real problem."""
        left = _arm("a", [1] * ROWS)
        right = replace(_arm("b", [0] * ROWS), item_ids=np.arange(3, dtype=np.int64))

        with pytest.raises(ValueError, match="item_ids"):
            check_aligned(left, right)

    def test_compare_refuses_rather_than_reporting(self) -> None:
        """The guard has to be inside compare, not a step a caller remembers."""
        left = _arm("a", [1] * ROWS)
        right = replace(_arm("b", [0] * ROWS), band=np.zeros(ROWS, dtype=np.int64))

        with pytest.raises(ValueError):
            compare(left, right, resamples=50)


class TestPerUser:
    def test_it_averages_within_a_user_before_averaging_across(self) -> None:
        """Four rows for one user and one for another must not let the loud
        user count four times -- that is the whole reason the unit is the user."""
        arm = _arm("a", [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        scores = per_user(arm, np.ones(ROWS, dtype=bool))

        assert scores == {"0": 1.0, "1": 0.0, "2": 0.0, "3": 0.0}

    def test_a_mask_restricts_which_rows_count(self) -> None:
        arm = _arm("a", [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        scores = per_user(arm, arm.band == 0)

        # Band 0 is rows 0-3: three of user 0's rows, one of user 1's.
        assert scores == {"0": 1.0, "1": 0.0}


class TestTheComparison:
    def test_every_band_gets_a_row_plus_two_pooled_ones(self) -> None:
        rows = compare(_arm("a", [0] * ROWS), _arm("b", [1] * ROWS), resamples=50)

        assert [row.label for row in rows] == [*band_labels(), f"<{LONG_TAIL_BELOW}", "overall"]
        assert [row.summary for row in rows[-2:]] == [True, True]

    def test_the_long_tail_row_matches_the_per_arm_table(self) -> None:
        """`evaluate.summarise` and `ablation.compare` both report a `<26`
        column. They take it from one `long_tail_mask`, so the G3 table's second
        column cannot mean one thing per arm and another in the comparison."""
        arm = _arm("a", [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        hits = Hits(
            hit=torch.from_numpy(arm.hit),
            item_ids=torch.from_numpy(arm.item_ids),
            user_ids=torch.from_numpy(arm.user_ids),
        )
        label = f"<{LONG_TAIL_BELOW}"

        paired = next(r for r in compare(arm, arm, resamples=50) if r.label == label)
        banded = next(r for r in summarise(hits, arm.band, arm.popularity) if r.label == label)

        assert paired.rows == banded.rows
        assert paired.baseline == pytest.approx(banded.recall)

    def test_the_difference_is_candidate_minus_baseline(self) -> None:
        """Sign errors here invert the entire finding and nothing looks wrong."""
        rows = compare(_arm("worse", [0] * ROWS), _arm("better", [1] * ROWS), resamples=200)
        overall = next(row for row in rows if row.label == "overall")

        assert overall.difference == pytest.approx(1.0)
        assert overall.result is not None
        assert overall.result.difference == pytest.approx(1.0)

    def test_an_untestable_band_has_no_result_rather_than_a_zero(self) -> None:
        """A band with too few users must not read as 'measured, no effect'.
        Bands 2-7 are empty in this fixture."""
        rows = compare(_arm("a", [0] * ROWS), _arm("b", [1] * ROWS), resamples=50)
        empty = next(row for row in rows if row.label == "3-5")

        assert empty.rows == 0
        assert empty.result is None

    def test_an_identical_pair_is_not_significant(self) -> None:
        """The control. Two arms that agree everywhere must produce an interval
        containing zero, or the instrument manufactures effects."""
        rows = compare(_arm("a", [1, 0] * 6), _arm("b", [1, 0] * 6), resamples=500)
        overall = next(row for row in rows if row.label == "overall")

        assert overall.result is not None
        assert not overall.result.significant

    def test_the_overall_row_pools_rows_not_bands(self) -> None:
        """Not the mean of the band means, which would weight a 500-row band
        the same as a 5-row one."""
        arm = _arm("a", [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        rows = compare(arm, arm, resamples=50)
        overall = next(row for row in rows if row.label == "overall")

        assert overall.baseline == pytest.approx(4 / 12)


class TestTheRoundTrip:
    def test_what_evaluate_saves_is_what_ablation_loads(self, tmp_path: Path) -> None:
        """The two modules share a file format and nothing else. A key renamed
        on one side is a KeyError here rather than in the middle of a gate."""
        hits = Hits(
            hit=torch.tensor([True, False, True]),
            item_ids=torch.tensor([1, 2, 3]),
            user_ids=torch.tensor([7, 7, 8]),
        )
        band = np.array([0, 1, 8], dtype=np.int64)
        popularity = np.array([False, True, True])
        destination = tmp_path / "arm.npz"

        save(destination, hits, band, popularity, np.zeros(4, dtype=np.int64))
        loaded = load_arm(destination)

        assert np.array_equal(loaded.hit, hits.hit.numpy())
        assert np.array_equal(loaded.user_ids, hits.user_ids.numpy())
        assert np.array_equal(loaded.band, band)
        assert np.array_equal(loaded.popularity, popularity)
        assert loaded.name == "arm"


class TestThePlotPath:
    def test_two_comparisons_do_not_collide(self) -> None:
        """A fixed default overwrote G2's plot the first time two comparisons ran
        back to back. The second reported success and the first deliverable was
        gone, with nothing saying so."""
        both = _arm("both-logq-n4u0-b8192e10lr0.001-ab7e1d50", [1] * ROWS)
        ident = _arm("id-logq-n4u0-b8192e10lr0.001-e077530a", [0] * ROWS)
        content = _arm("content-logq-n4u0-b8192e10lr0.001-57101852", [0] * ROWS)

        assert default_plot(ident, both) != default_plot(content, both)

    def test_the_path_names_both_arms_without_the_budget(self) -> None:
        both = _arm("both-logq-n4u0-b8192e10lr0.001-ab7e1d50", [1] * ROWS)
        ident = _arm("id-logq-n4u0-b8192e10lr0.001-e077530a", [0] * ROWS)

        assert default_plot(ident, both) == Path("docs/img/both-logq-n4u0-vs-id-logq-n4u0.png")


class TestTheTable:
    def test_an_untestable_band_says_so_rather_than_printing_a_number(self) -> None:
        rows = [
            BandComparison("0", 0, 0, float("nan"), float("nan"), float("nan"), None),
            BandComparison("overall", 4, 2, 0.5, 0.75, 0.1, None),
        ]

        table = render(rows, 100, "arm-a", "arm-b")

        assert "too few users" in table
        assert "arm-a" in table and "arm-b" in table
