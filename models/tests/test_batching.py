"""Batch assembly, and the two things that would poison training silently.

The first is index 0 reaching the candidate pool. It is the reserved OOV row
(B2) with a permanently zero embedding, so as a column it is a candidate every
user is scored against and no user can ever be near. Padding the slate negatives
would put it there in every batch that had a short slate.

The second is history dropout running at validation. It would shorten exactly
the histories the metric is meant to measure, and it would improve nothing
visibly -- the model would simply be scored on inputs it will never see.

Everything runs on CPU by construction; ``tests/test_torch_env.py`` fails the
build if anything under ``models/`` names a device.
"""

from __future__ import annotations

import pytest
import torch

from common.torch_env import select_device
from models.classes.batching import Batch, RowIndices
from models.classes.dataset import SplitTensors
from models.retrieval.dataloader.batching import assemble_pool, make_collate, truncate_history

ROWS = 6
FEATS = 3
HISTORY = 4
NEGS = 2
N_ITEMS = 50


@pytest.fixture
def split() -> SplitTensors:
    """Six requests with histories of length 4, 3, 2, 1, 0 and 4."""
    lengths = [4, 3, 2, 1, 0, 4]
    history_ids = torch.zeros(ROWS, HISTORY, dtype=torch.long)
    history_mask = torch.zeros(ROWS, HISTORY, dtype=torch.long)
    for row, length in enumerate(lengths):
        history_ids[row, :length] = torch.arange(1, length + 1) + row
        history_mask[row, :length] = 1

    # Rows 4 and 5 have a short slate: one real negative and one pad.
    neg_ids = torch.tensor([[11, 12], [13, 14], [15, 16], [17, 18], [19, 0], [21, 0]])
    neg_mask = torch.tensor([[1, 1], [1, 1], [1, 1], [1, 1], [1, 0], [1, 0]])

    return SplitTensors(
        user_feats=torch.arange(ROWS * FEATS, dtype=torch.float32).reshape(ROWS, FEATS),
        history_ids=history_ids,
        history_mask=history_mask,
        item_ids=torch.arange(1, ROWS + 1),
        impression_ids=torch.arange(100, 100 + ROWS),
        # Deliberately fewer users than rows: a per-user aggregation that is
        # secretly per-row would still look right on a 1:1 fixture.
        user_ids=torch.tensor([7, 7, 8, 8, 9, 9]),
        neg_ids=neg_ids,
        neg_mask=neg_mask,
    )


def _generator() -> torch.Generator:
    return torch.Generator(device=select_device("cpu")).manual_seed(0)


class TestTheCandidatePool:
    def test_no_negative_is_the_reserved_index(self, split: SplitTensors) -> None:
        """The one that would be invisible. A padded 0 flattened into the pool
        is a column with an all-zero embedding, competing in every softmax."""
        batch = make_collate(split, N_ITEMS, generator=_generator())(list(range(ROWS)))

        assert int(batch.neg_ids.min()) >= 1

    def test_a_short_slate_is_topped_up_rather_than_padded(self, split: SplitTensors) -> None:
        """Rows 4 and 5 have one real negative each; the second slot must hold a
        drawn item, and must be labelled as drawn."""
        batch = make_collate(split, N_ITEMS, generator=_generator())(list(range(ROWS)))

        assert batch.neg_ids[4, 0] == 19 and batch.neg_is_slate[4, 0]
        assert batch.neg_ids[4, 1] != 0 and not batch.neg_is_slate[4, 1]

    def test_a_full_slate_keeps_every_real_negative(self, split: SplitTensors) -> None:
        batch = make_collate(split, N_ITEMS, generator=_generator())(list(range(ROWS)))

        assert torch.equal(batch.neg_ids[:4], split.neg_ids[:4])
        assert bool(batch.neg_is_slate[:4].all())

    def test_the_uniform_arm_appends_and_labels_its_columns(self, split: SplitTensors) -> None:
        """G3's '+ mixed uniform negatives' row. Extra columns, never marked as
        slate, so they take the closed-form log_q rather than the counter's."""
        batch = make_collate(split, N_ITEMS, uniform_negs=3, generator=_generator())(
            list(range(ROWS))
        )

        assert batch.neg_ids.shape == (ROWS, NEGS + 3)
        assert not bool(batch.neg_is_slate[:, NEGS:].any())
        assert int(batch.neg_ids[:, NEGS:].min()) >= 1


class TestHistoryDropout:
    def test_zero_dropout_changes_nothing(self, split: SplitTensors) -> None:
        """What the validation loader passes. Identity, not 'usually identity'."""
        ids, mask = truncate_history(split.history_ids, split.history_mask, 0.0)

        assert ids is split.history_ids and mask is split.history_mask

    def test_it_never_lengthens_a_history(self, split: SplitTensors) -> None:
        _, mask = truncate_history(split.history_ids, split.history_mask, 1.0, _generator())

        assert bool((mask.sum(1) <= split.history_mask.sum(1)).all())

    def test_what_survives_is_a_prefix(self, split: SplitTensors) -> None:
        """Most-recent-first, so a prefix keeps the freshest clicks. Dropping
        scattered entries would make a gappy sequence no real user has."""
        ids, mask = truncate_history(split.history_ids, split.history_mask, 1.0, _generator())

        for row in range(ROWS):
            kept = int(mask[row].sum())
            assert torch.equal(mask[row][:kept], torch.ones(kept, dtype=torch.long))
            assert torch.equal(ids[row][:kept], split.history_ids[row][:kept])

    def test_the_ids_and_the_mask_agree(self, split: SplitTensors) -> None:
        """Either alone would be enough for the pool, but a caller relying on one
        breaks silently the moment the other changes."""
        ids, mask = truncate_history(split.history_ids, split.history_mask, 1.0, _generator())

        assert bool((ids[mask == 0] == 0).all())

    def test_an_empty_history_stays_empty(self, split: SplitTensors) -> None:
        """Row 4 has none. Sampling in [0, 0] must not produce a phantom entry."""
        _, mask = truncate_history(split.history_ids, split.history_mask, 1.0, _generator())

        assert int(mask[4].sum()) == 0

    def test_a_seed_reproduces_the_batch(self, split: SplitTensors) -> None:
        """An ablation whose arms differ by which histories were truncated is
        not an ablation."""
        first = make_collate(split, N_ITEMS, generator=_generator())(list(range(ROWS)))
        second = make_collate(split, N_ITEMS, generator=_generator())(list(range(ROWS)))

        assert torch.equal(first.history_mask, second.history_mask)
        assert torch.equal(first.neg_ids, second.neg_ids)


class TestTheBatchItself:
    def test_rows_are_selected_not_reordered(self, split: SplitTensors) -> None:
        batch = make_collate(split, N_ITEMS, history_dropout=0.0)([3, 1])

        assert torch.equal(batch.item_ids, torch.tensor([4, 2]))
        assert torch.equal(batch.impression_ids, torch.tensor([103, 101]))
        assert torch.equal(batch.user_feats, split.user_feats[torch.tensor([3, 1])])

    def test_the_index_dataset_is_the_row_count(self) -> None:
        rows = RowIndices(ROWS)

        assert len(rows) == ROWS and rows[2] == 2

    def test_moving_a_batch_keeps_every_field(self, split: SplitTensors) -> None:
        batch = make_collate(split, N_ITEMS, history_dropout=0.0)(list(range(ROWS)))
        moved = batch.to(select_device("cpu"))

        assert isinstance(moved, Batch)
        assert torch.equal(moved.neg_is_slate, batch.neg_is_slate)


class TestThePoolLayout:
    """Single-process, where every gather is an identity -- so what this checks
    is the ORDER, which is the part that breaks on rank 1 if it is wrong."""

    def test_positives_come_first_and_the_index_finds_them(self) -> None:
        positives = torch.arange(3.0).reshape(3, 1)
        negatives = torch.arange(10.0, 16.0).reshape(6, 1)

        item_emb, _, item_ids, index = assemble_pool(
            positives,
            negatives,
            torch.arange(1, 4),
            torch.arange(11, 17),
            torch.zeros(3),
            torch.ones(6),
        )

        assert item_emb.shape == (9, 1)
        assert torch.equal(item_emb[index], positives)
        assert torch.equal(item_ids[index], torch.arange(1, 4))

    def test_every_column_carries_its_own_log_q(self) -> None:
        """Positives and negatives come from different distributions, so the two
        pieces must stay aligned with the columns they describe."""
        _, log_q, _, _ = assemble_pool(
            torch.zeros(2, 1),
            torch.zeros(4, 1),
            torch.arange(2),
            torch.arange(4),
            torch.full((2,), -1.0),
            torch.full((4,), -9.0),
        )

        assert torch.equal(log_q, torch.tensor([-1.0, -1.0, -9.0, -9.0, -9.0, -9.0]))

    def test_the_gradient_reaches_both_halves(self) -> None:
        """At world size 1 the gather is an identity, and an identity that broke
        the graph would make the negatives untrainable."""
        positives = torch.zeros(2, 1, requires_grad=True)
        negatives = torch.zeros(4, 1, requires_grad=True)

        item_emb, _, _, _ = assemble_pool(
            positives, negatives, torch.arange(2), torch.arange(4), torch.zeros(2), torch.zeros(4)
        )
        item_emb.sum().backward()  # type: ignore[no-untyped-call]

        assert positives.grad is not None and negatives.grad is not None
