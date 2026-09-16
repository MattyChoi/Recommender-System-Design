"""The sampling-frequency estimate that feeds the logQ correction."""

from __future__ import annotations

import math

import torch
from torch import nn

TRAIN_EXAMPLES = 236_100.0


class StreamingLogQ(nn.Module):
    """Exponentially-decayed frequency estimate over the training stream.

    An ``nn.Module`` for the buffer rather than for a forward pass: the counts
    have to follow ``.to(device)`` and land in the checkpoint. Carrying them as
    a bare attribute is how a resumed run silently restarts with a flat prior,
    and how the counts end up on a different device from the batch.

    Args:
        n_items: Catalogue size. The table is ``n_items + 1`` for the reserved
            OOV row.
        half_life: Examples after which an observation carries half its weight.
        prior: Laplace pseudo-count, held OUT of the decay. The manual seeds the
            counts at one and decays everything, so the smoothing mass
            evaporates and a never-sampled item drifts toward log q = -inf,
            earning an enormous correction on no evidence.
    """

    observed: torch.Tensor

    def __init__(
        self,
        n_items: int,
        half_life: float = TRAIN_EXAMPLES,
        prior: float = 1.0,
    ) -> None:
        super().__init__()
        if half_life <= 0 or prior <= 0:
            raise ValueError(f"half_life and prior must be positive; got {half_life}, {prior}")
        self.register_buffer("observed", torch.zeros(n_items + 1))
        self.half_life = float(half_life)
        self.prior = float(prior)

    def log_q(self, item_idx: torch.Tensor) -> torch.Tensor:
        """Log sampling probability for each index.

        Call this BEFORE :meth:`update` for a given batch, or the batch
        conditions its own correction on its own labels.
        """
        total = self.prior * self.observed.numel() + self.observed.sum()
        return torch.log((self.prior + self.observed[item_idx]) / total)

    def update(self, item_idx: torch.Tensor) -> None:
        """Decay by the batch's size in examples, then count it."""
        self.observed *= 0.5 ** (item_idx.numel() / self.half_life)
        self.observed.index_add_(
            0, item_idx, torch.ones(item_idx.numel(), device=self.observed.device)
        )


def uniform_negatives(
    shape: tuple[int, ...],
    n_items: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Item indices drawn uniformly from ``1..n_items``, never the reserved 0.

    Used to baseline EASY negatives against HARD negatives

    Args:
        shape: Output shape, e.g. ``(batch, k)``.
        n_items: Catalogue size, excluding the reserved index.
        device: Where to allocate. From ``common.torch_env.select_device``.
        generator: Seeded, for a reproducible run.

    Returns:
        Long indices of the given shape.
    """
    return torch.randint(1, n_items + 1, shape, device=device, generator=generator)


def uniform_log_q(shape: tuple[int, ...], n_items: int, device: torch.device) -> torch.Tensor:
    """The exact log sampling probability of a uniform draw."""
    return torch.full(shape, -math.log(n_items), device=device)
