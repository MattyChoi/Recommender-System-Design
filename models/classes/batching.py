"""The batch record and the row-index Dataset it is collated from."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Batch:
    """One training batch.

    Attributes:
        user_feats: ``[B, F]``.
        history_ids: ``[B, L]``, 0 where padded or dropped.
        history_mask: ``[B, L]``.
        item_ids: ``[B]`` the clicked item -- the positives.
        neg_ids: ``[B, N]`` candidate negatives. Never 0: a slate short of N
            entries is topped up with uniform draws rather than padded, so the
            reserved OOV row never becomes a column.
        neg_is_slate: ``[B, N]`` True where the negative came from the
            impression, False where it was drawn uniformly. This is the switch
            deciding which ``log_q`` each column gets.
        impression_ids: ``[B]``, kept so slates can be regrouped at evaluation.
        user_ids: ``[B]``, kept so a per-row result can be attributed to a user
            without the reader having to assume the loader preserved row order.
    """

    user_feats: torch.Tensor
    history_ids: torch.Tensor
    history_mask: torch.Tensor
    item_ids: torch.Tensor
    neg_ids: torch.Tensor
    neg_is_slate: torch.Tensor
    impression_ids: torch.Tensor
    user_ids: torch.Tensor

    def to(self, device: torch.device) -> Batch:
        """Move every field, keeping the dataclass."""
        moved = {
            field.name: getattr(self, field.name).to(device, non_blocking=True)
            for field in fields(self)
        }
        return replace(self, **moved)


class RowIndices(Dataset[int]):
    """Yields row numbers, not rows.

    The collate does one fancy-index into the resident tensors instead of B
    separate lookups and a stack, which is an order of magnitude cheaper at
    B=8192. The Dataset exists so ``DistributedSampler`` can shard the rows.
    """

    def __init__(self, rows: int) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return self.rows

    def __getitem__(self, index: int) -> int:
        return index
