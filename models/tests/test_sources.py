"""The five retrieval sources, and the one property they must all share.

A source's job is to name candidates. The thing that makes the blend ablation
meaningful is that a source which has **nothing to say returns nothing** rather
than filler -- pad a short list with popular items and every source silently
becomes a hybrid with ``trending``, so the leave-one-out table would be
measuring the padding rather than the source.

So the tests that matter here are about the shape of an answer, not its quality.
"""

from __future__ import annotations

import pytest
import torch

from models.classes.dataset import SplitTensors
from models.retrieval.dataloader.batching import make_loader
from models.retrieval.sources import (
    Retrieved,
    covisit_edges,
    covisit_source,
    recent_source,
    top_k_from_scores,
    trending_source,
    two_tower_source,
)
from models.retrieval.train import retrieval_hits
from models.retrieval.two_tower import TwoTower

N_ITEMS = 40
DIM = 16


def _split(rows: int = 12, history: int = 5) -> SplitTensors:
    """A small split whose histories vary in length, including empty."""
    generator = torch.Generator().manual_seed(0)
    ids = torch.randint(1, N_ITEMS + 1, (rows, history), generator=generator)
    mask = torch.zeros(rows, history, dtype=torch.long)
    for row in range(rows):
        mask[row, : row % (history + 1)] = 1
    return SplitTensors(
        user_feats=torch.randn(rows, 9, generator=generator),
        history_ids=ids * mask,
        history_mask=mask,
        item_ids=torch.randint(1, N_ITEMS + 1, (rows,), generator=generator),
        impression_ids=torch.arange(rows),
        user_ids=torch.arange(rows) % 4,
        neg_ids=torch.zeros(rows, 4, dtype=torch.long),
        neg_mask=torch.zeros(rows, 4, dtype=torch.long),
    )


def _tower() -> TwoTower:
    torch.manual_seed(0)
    tower = TwoTower(
        content=torch.randn(N_ITEMS + 1, 32),
        item_category=torch.randint(0, 3, (N_ITEMS + 1,)),
        item_subcategory=torch.randint(0, 5, (N_ITEMS + 1,)),
        n_user_feats=9,
        n_categories=3,
        n_subcategories=5,
    )
    tower.eval()
    return tower


class TestTopK:
    def test_a_short_catalogue_is_padded_with_the_reserved_index(self) -> None:
        """Zero is OOV, so it can never be mistaken for a candidate."""
        top = top_k_from_scores(torch.rand(3, 5), k=10)

        assert top.shape == (3, 10)
        assert (top[:, 5:] == 0).all()
        assert (top[:, :5] > 0).all()

    def test_positive_only_drops_unscored_items(self) -> None:
        """A counting source's zero means no evidence, not weak evidence."""
        scores = torch.zeros(1, 6)
        scores[0, 2] = 1.0

        top = top_k_from_scores(scores, k=4, positive_only=True)

        assert top[0, 0].item() == 3  # 1-based
        assert (top[0, 1:] == 0).all()

    def test_an_unanswerable_request_returns_nothing(self) -> None:
        valid = torch.tensor([True, False])

        top = top_k_from_scores(torch.rand(2, 6), k=3, valid_rows=valid)

        assert (top[0] > 0).all()
        assert (top[1] == 0).all()

    def test_the_two_branches_can_actually_differ(self) -> None:
        """The precondition, before anything is concluded from the pair.

        Both calls see the same scores; only the flag moves. If a fixture made
        them agree, every assertion above would pass while testing nothing.
        """
        scores = torch.zeros(1, 6)
        scores[0, 2] = 1.0

        assert not torch.equal(
            top_k_from_scores(scores, k=4, positive_only=True),
            top_k_from_scores(scores, k=4, positive_only=False),
        )


class TestSources:
    def test_recent_returns_the_history_and_nothing_else(self) -> None:
        split = _split()

        top = recent_source(split.history_ids, split.history_mask, k=8)

        kept = split.history_ids * split.history_mask
        for row in range(len(top)):
            got = {int(value) for value in top[row] if value}
            assert got == {int(value) for value in kept[row] if value}

    def test_a_user_with_no_history_gets_no_candidates(self) -> None:
        """The honest answer, and the reason `reach` is reported."""
        split = _split()
        empty = split.history_mask.sum(dim=1) == 0
        assert empty.any(), "fixture must contain an empty history or this proves nothing"

        top = recent_source(split.history_ids, split.history_mask, k=8)

        assert (top[empty] == 0).all()

    def test_trending_gives_every_request_the_same_list(self) -> None:
        prior = torch.zeros(N_ITEMS + 1, dtype=torch.long)
        prior[3], prior[7] = 5, 9

        top = trending_source(prior, rows=4, k=6)

        assert (top == top[0]).all()
        assert top[0, 0].item() == 7 and top[0, 1].item() == 3
        assert (top[0, 2:] == 0).all(), "only two items were ever clicked"

    def test_covisit_sums_every_edge_from_every_history_item(self) -> None:
        """Two seeds pointing at one neighbour must add, not overwrite."""
        neighbour = torch.tensor([9, 9], dtype=torch.long)
        weight = torch.tensor([0.25, 0.5])
        # CSR over n_rows = N_ITEMS + 1 item slots, so ptr is one longer again.
        # Items 1 and 2 own one edge each, both landing on item 9; every other
        # item owns none, which is what the flat tail encodes.
        ptr = torch.full((N_ITEMS + 2,), 2, dtype=torch.long)
        ptr[0], ptr[1], ptr[2] = 0, 0, 1

        history = torch.tensor([[1, 2], [1, 0]])
        mask = torch.tensor([[1, 1], [1, 0]])

        top = covisit_source(ptr, neighbour, weight, history, mask, N_ITEMS + 1, k=3)

        assert top[0, 0].item() == 9
        assert top[1, 0].item() == 9
        assert (top[:, 1:] == 0).all(), "only one neighbour exists"

    def test_covisit_edges_round_trip_through_csr(self) -> None:
        """Rows arrive in no particular order; the CSR must sort them itself."""

        class _Frame:
            def select(self, *_: str) -> _Frame:
                return self

            def collect(self) -> list[dict[str, float]]:
                return [
                    {"item_id": 2, "related_item_id": 5, "weight": 1.0},
                    {"item_id": 1, "related_item_id": 4, "weight": 2.0},
                    {"item_id": 2, "related_item_id": 6, "weight": 3.0},
                ]

        ptr, neighbour, weight = covisit_edges(_Frame(), n_rows=8)  # type: ignore[arg-type]

        assert int(ptr[1]) == 0 and int(ptr[2]) == 1 and int(ptr[3]) == 3
        assert neighbour[int(ptr[1]) : int(ptr[2])].tolist() == [4]
        assert sorted(neighbour[int(ptr[2]) : int(ptr[3])].tolist()) == [5, 6]
        assert weight[0].item() == 2.0


class TestParityWithTheTrainingLoop:
    def test_the_two_tower_source_agrees_with_retrieval_hits(self) -> None:
        """One scoring expression, two callers, and they must not drift.

        ``retrieval_hits`` returns a hit and throws the candidates away; the
        blend needs the candidates, so the top-k is computed again in
        ``sources``. That duplication is deliberate -- inverting it would make
        ``train.py`` import the blending module -- and this is what keeps it
        honest. A failure here means the scoring expression drifted, not the
        loader: ``drop_last`` is False for validation, so both see every row.
        """
        split, tower, device = _split(), _tower(), torch.device("cpu")

        loader = make_loader(split, N_ITEMS, 4, device, training=False, history_dropout=0.0)
        expected = retrieval_hits(tower, loader, device, k=5).hit
        got = Retrieved(
            "two_tower",
            two_tower_source(tower, split, device, k=5),
            split.item_ids,
            split.user_ids,
            5,
        ).hit

        assert len(got) == len(expected)
        assert torch.equal(got, expected)


class TestRetrieved:
    def test_pool_and_reach_read_the_padding(self) -> None:
        top = torch.tensor([[1, 2, 0], [0, 0, 0], [4, 0, 0]])
        got = Retrieved("t", top, torch.tensor([2, 1, 9]), torch.arange(3), 3)

        assert got.pool.tolist() == [2, 0, 1]
        assert got.reach == pytest.approx(2 / 3)
        assert got.hit.tolist() == [True, False, False]

    def test_the_reserved_index_never_counts_as_a_hit(self) -> None:
        """Every real item id is >= 1, so an all-padding row cannot score."""
        empty = torch.zeros(1, 4, dtype=torch.long)
        got = Retrieved("t", empty, torch.tensor([7]), torch.tensor([0]), 4)

        assert not got.hit.any()
        assert got.recall == 0.0
