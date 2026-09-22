"""The blend, and the two conditions that decide what its table means.

Both are about fairness of comparison rather than about code:

**A union of five 100-item lists is a ~400-item pool.** Quoting its recall beside
a single source's Recall@100 compares a model against a bigger budget and calls
the budget a result. Hence the quota blend.

**Dropping a source does not shrink the budget in production.** The slots go to
whoever is left. A leave-one-out that holds the survivors at their old quota
charges the removed source for a budget cut it did not cause, and every source
then looks load-bearing.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from models.retrieval.blend import (
    attribute,
    check_aligned,
    contributions,
    hit_matrix,
    leave_one_out,
)
from models.retrieval.sources import Retrieved


def _source(name: str, top: list[list[int]], clicked: list[int]) -> Retrieved:
    return Retrieved(
        source=name,
        top=torch.tensor(top, dtype=torch.long),
        item_ids=torch.tensor(clicked, dtype=torch.long),
        user_ids=torch.arange(len(clicked)),
        k=len(top[0]),
    )


class TestAlignment:
    def test_sources_scoring_different_clicks_are_refused(self) -> None:
        a = _source("a", [[1, 2]], [1])
        b = _source("b", [[1, 2]], [2])

        with pytest.raises(ValueError, match="different clicked items"):
            check_aligned([a, b])

    def test_sources_of_different_lengths_are_refused(self) -> None:
        a = _source("a", [[1, 2], [3, 4]], [1, 3])
        b = _source("b", [[1, 2]], [1])

        with pytest.raises(ValueError, match="requests"):
            check_aligned([a, b])


class TestAttribution:
    def test_an_item_both_sources_offer_is_not_unique_to_either(self) -> None:
        a = _source("a", [[1, 2, 0]], [1])
        b = _source("b", [[2, 3, 0]], [1])

        shares = attribute([a, b])

        # union pairs: (0,1), (0,2), (0,3) -- three of them
        assert shares["a"] == (2 / 3, 1 / 3)
        assert shares["b"] == (2 / 3, 1 / 3)

    def test_padding_is_not_a_candidate(self) -> None:
        """Slot 0 is the reserved OOV index and must never enter the pool."""
        a = _source("a", [[1, 0, 0]], [1])
        b = _source("b", [[1, 0, 0]], [1])

        shares = attribute([a, b])

        assert shares["a"] == (1.0, 0.0), "one pair in the union, shared"


class TestQuotaAndRedistribution:
    def test_the_quota_blend_uses_a_fraction_of_each_list(self) -> None:
        a = _source("a", [[1, 2, 3, 4]], [3])
        b = _source("b", [[5, 6, 7, 8]], [3])

        assert hit_matrix([a, b])[0][0], "item 3 is in a's full top-4"
        assert not hit_matrix([a, b], quota=2)[0][0], "and outside its 2-slot quota"

    def test_dropping_a_source_hands_its_slots_to_the_survivors(self) -> None:
        """The design decision, asserted rather than assumed.

        k=4 over two sources is 2 slots each, and the click sits at rank 3 of
        source b -- outside b's quota while a is present. Drop a, and b gets all
        four slots and finds it. So removing a source can IMPROVE the blend, and
        a leave-one-out that froze b at two slots would report the opposite.
        """
        rows = 4
        a = _source("a", [[1, 2, 3, 4]] * rows, [7] * rows)
        b = _source("b", [[5, 6, 7, 8]] * rows, [7] * rows)
        users = np.arange(rows)

        assert not hit_matrix([a, b], quota=2).any(), "with both present the click is unreachable"

        losses = leave_one_out([a, b], users, k=4)

        # difference is (without - with); dropping `a` gains a full hit
        assert losses["a"].difference == pytest.approx(1.0)
        assert losses["b"].difference == pytest.approx(0.0)


class TestContributions:
    def test_only_here_counts_answers_not_candidates(self) -> None:
        """A source can supply a third of the pool and none of the answers."""
        rows = 4
        finder = _source("finder", [[9, 1, 2, 3]] * rows, [9] * rows)
        noise = _source("noise", [[4, 5, 6, 7]] * rows, [9] * rows)
        users = np.arange(rows)

        table = {row.source: row for row in contributions([finder, noise], users, k=4)}

        assert table["finder"].found_only_here == 1.0
        assert table["noise"].found_only_here == 0.0
        assert table["noise"].share == 0.5, "half the pool"
        assert table["noise"].alone == 0.0, "and none of the recall"
