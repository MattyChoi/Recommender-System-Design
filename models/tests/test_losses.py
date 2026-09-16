"""The logQ correction, and the three ways it fails without erroring.

The first is orientation. ``log_q`` indexes COLUMNS -- items -- and broadcasting
it across rows instead subtracts a constant per row, which log-softmax cancels
exactly. The loss would be numerically identical to the uncorrected one, the
ablation would report no effect, and the conclusion drawn would be about the
correction rather than about the transpose. ``test_a_per_row_shift_is_invisible``
is the reason that mistake is detectable at all: it establishes that a per-row
shift changes nothing, so a correction that changes nothing is a per-row shift.

The second is the positive column. ``arange(B)`` is correct in one process and
on rank 0, and wrong on every other rank once G5 gathers item embeddings across
workers -- each row then trains against another rank's item as its positive.
Single-GPU development never sees it.

The third is the duplicate mask. Once the positive can sit off-diagonal, "the
same item elsewhere in the batch" and "not on the diagonal" stop being the same
set, and masking the wrong one removes a row's own positive.

Everything here runs on CPU by construction; ``tests/test_torch_env.py`` fails
the build if anything under ``models/`` names a device.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional as f

from common.torch_env import select_device
from models.retrieval.losses import sampled_softmax_loss

DIM = 4
TEMP = 0.05


@pytest.fixture
def pair() -> tuple[torch.Tensor, torch.Tensor]:
    """Three users and three items, normalised as the towers would leave them."""
    device = select_device("cpu")
    generator = torch.Generator(device=device).manual_seed(0)
    users = f.normalize(torch.randn(3, DIM, generator=generator, device=device), dim=-1)
    items = f.normalize(torch.randn(3, DIM, generator=generator, device=device), dim=-1)
    return users, items


@pytest.fixture
def other_items() -> torch.Tensor:
    """A second block of item vectors, genuinely distinct from ``pair``'s.

    Distinct matters: imitating a second worker by repeating the same item
    tensor makes column j and column j + 3 numerically identical, so which one
    is named the positive cannot change the loss and the DDP test below has
    nothing to detect.
    """
    device = select_device("cpu")
    generator = torch.Generator(device=device).manual_seed(1)
    return f.normalize(torch.randn(3, DIM, generator=generator, device=device), dim=-1)


def _flat(n: int) -> torch.Tensor:
    """A uniform log_q over n columns."""
    return torch.full((n,), -math.log(n))


class TestTheCorrection:
    def test_a_per_row_shift_is_invisible(self, pair: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Softmax cancels a constant, so uniform log_q must equal no correction.

        This is the control the orientation test below depends on.
        """
        users, items = pair
        flat = sampled_softmax_loss(users, items, _flat(3), TEMP)
        none = sampled_softmax_loss(users, items, torch.zeros(3), TEMP)
        assert torch.allclose(flat, none, atol=1e-6)

    def test_a_per_column_shift_is_not(self, pair: tuple[torch.Tensor, torch.Tensor]) -> None:
        """The one that fails if log_q is broadcast down rows instead of across."""
        users, items = pair
        skewed = torch.tensor([-0.1, -5.0, -5.0])
        assert not torch.allclose(
            sampled_softmax_loss(users, items, skewed, TEMP),
            sampled_softmax_loss(users, items, torch.zeros(3), TEMP),
            atol=1e-6,
        )

    def test_a_popular_negative_is_penalised_less(
        self, pair: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """The whole point: down-weight items that turn up as negatives too often.

        Row 0's positive is column 0, so raising column 1's log_q lowers its
        logit and takes probability mass off it -- which must reduce the loss
        of the row it was competing with.
        """
        users, items = pair
        popular = torch.tensor([-2.0, -0.05, -2.0])
        flat = torch.full((3,), -2.0)
        assert sampled_softmax_loss(users[:1], items, popular, TEMP) < sampled_softmax_loss(
            users[:1], items, flat, TEMP
        )


class TestThePositiveColumn:
    def test_the_default_is_the_diagonal(self, pair: tuple[torch.Tensor, torch.Tensor]) -> None:
        users, items = pair
        explicit = torch.arange(3)
        assert torch.allclose(
            sampled_softmax_loss(users, items, torch.zeros(3), TEMP),
            sampled_softmax_loss(users, items, torch.zeros(3), TEMP, positive_index=explicit),
        )

    def test_an_offset_positive_is_honoured(
        self, pair: tuple[torch.Tensor, torch.Tensor], other_items: torch.Tensor
    ) -> None:
        """G5's gathered case, reproduced in one process.

        An all-gather concatenates every worker's items, so rank 1's positives
        land at columns 3..5 while rank 0's occupy 0..2. Softmax does not care
        what order the columns arrive in, so the SAME candidate set with the
        positives at the back and ``positive_index`` saying so must score
        exactly what it scores with the positives at the front.
        """
        users, items = pair
        front = sampled_softmax_loss(users, torch.cat([items, other_items]), torch.zeros(6), TEMP)
        back = sampled_softmax_loss(
            users,
            torch.cat([other_items, items]),
            torch.zeros(6),
            TEMP,
            positive_index=3 + torch.arange(3),
        )
        assert torch.allclose(front, back, atol=1e-6)

    def test_the_default_is_wrong_once_the_positives_move(
        self, pair: tuple[torch.Tensor, torch.Tensor], other_items: torch.Tensor
    ) -> None:
        """The silent failure itself: on any rank but 0 the default ``arange``
        scores each row against another worker's item as its answer."""
        users, items = pair
        gathered = torch.cat([other_items, items])
        assert not torch.allclose(
            sampled_softmax_loss(
                users, gathered, torch.zeros(6), TEMP, positive_index=3 + torch.arange(3)
            ),
            sampled_softmax_loss(users, gathered, torch.zeros(6), TEMP),
            atol=1e-6,
        )

    def test_a_mismatched_positive_index_raises(
        self, pair: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        users, items = pair
        with pytest.raises(ValueError, match="positive_index"):
            sampled_softmax_loss(users, items, torch.zeros(3), TEMP, positive_index=torch.arange(2))

    def test_a_mismatched_log_q_raises(self, pair: tuple[torch.Tensor, torch.Tensor]) -> None:
        users, items = pair
        with pytest.raises(ValueError, match="log_q"):
            sampled_softmax_loss(users, items, torch.zeros(2), TEMP)


class TestTheDuplicateMask:
    def test_a_repeat_of_the_positive_is_masked(
        self, pair: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Item 0 twice: column 2 must stop competing with row 0's positive.

        Masking it can only lower row 0's loss, since the mass it held returns
        to the positive.
        """
        users, items = pair
        repeated = torch.stack([items[0], items[1], items[0]])
        ids = torch.tensor([7, 8, 7])

        masked = sampled_softmax_loss(users, repeated, torch.zeros(3), TEMP, item_ids=ids)
        unmasked = sampled_softmax_loss(users, repeated, torch.zeros(3), TEMP)
        assert masked < unmasked

    def test_a_rows_own_positive_survives_the_mask(
        self, pair: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Masking on identity alone would blank the positive too, and the loss
        would be the log of nothing."""
        users, items = pair
        ids = torch.tensor([7, 7, 7])
        loss = sampled_softmax_loss(users, items, torch.zeros(3), TEMP, item_ids=ids)
        assert torch.isfinite(loss)

    def test_the_mask_follows_an_offset_positive(
        self, pair: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """With every positive off-diagonal, a diagonal-based mask would remove
        the wrong column and leave each row's true positive masked instead."""
        users, items = pair
        doubled = torch.cat([items, items])
        ids = torch.tensor([7, 8, 9, 7, 8, 9])
        loss = sampled_softmax_loss(
            users,
            doubled,
            torch.zeros(6),
            TEMP,
            item_ids=ids,
            positive_index=3 + torch.arange(3),
        )
        assert torch.isfinite(loss)
        # Each row's positive has exactly one twin among the six columns, so the
        # mask removes one competitor per row and the loss can only fall.
        assert loss < sampled_softmax_loss(
            users, doubled, torch.zeros(6), TEMP, positive_index=3 + torch.arange(3)
        )


class TestTheObjectiveItself:
    def test_alignment_drives_the_loss_down(self) -> None:
        """A sanity floor: identical towers should be nearly free to separate."""
        users = f.normalize(torch.eye(3), dim=-1)
        loss = sampled_softmax_loss(users, users.clone(), torch.zeros(3), TEMP)
        assert loss < 1e-3

    def test_gradients_reach_both_towers(self) -> None:
        users = f.normalize(torch.randn(3, DIM), dim=-1).requires_grad_(True)
        items = f.normalize(torch.randn(3, DIM), dim=-1).requires_grad_(True)
        # torch ships no annotation for Tensor.backward; the call is correct.
        loss = sampled_softmax_loss(users, items, torch.zeros(3), TEMP)
        loss.backward()  # type: ignore[no-untyped-call]
        assert users.grad is not None and torch.any(users.grad != 0)
        assert items.grad is not None and torch.any(items.grad != 0)
