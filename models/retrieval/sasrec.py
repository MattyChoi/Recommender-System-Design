"""Causal self-attention over the click sequence, as a POOLER"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as f

from data_pipeline.features.user_history import MAX_HISTORY


def _normalise(block: torch.Tensor) -> torch.Tensor:
    """Zero-mean unit-variance across a block's last dimension, no parameters.

    An all-zero block normalises to zeros (variance 0 divides by ``sqrt(eps)``),
    which is what keeps a padded position and an empty history at zero.
    """
    normalised: torch.Tensor = f.layer_norm(block, block.shape[-1:])
    return normalised


class SASRec(nn.Module):
    """Reduce a click sequence to one vector with causal self-attention.

    Args:
        dim: Width of the incoming vectors. Set by the caller's table, not
            chosen here -- ``id_dim`` when IDs are on, the projected width when
            they are off.
        n_heads: Attention heads. Must divide ``dim``.
        n_blocks: Stacked encoder layers.
        max_len: Longest sequence, for the positional table. At least the
            loader's ``MAX_HISTORY`` or a full row indexes past the end.
        dropout: Attention weights, residuals and the feed-forward.

    Note:
        Positions are LEARNED rather than sinusoidal: the sequence is at most 50
        long and non-stationary, so sinusoidal extrapolation buys nothing.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 2,
        n_blocks: int = 2,
        max_len: int = MAX_HISTORY,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if dim % n_heads:
            raise ValueError(f"n_heads={n_heads} must divide dim={dim}")

        self.max_len = max_len
        self.pos_emb = nn.Embedding(max_len, dim)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    dim,
                    n_heads,
                    dim * 4,
                    dropout,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(self, vectors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Pool ``[B, L, D]`` down to ``[B, D]``.

        Args:
            vectors: Per-position embeddings, most-recent-first, zero where padded.
            mask: ``[B, L]``, 1 for a real entry, in the same order.

        Returns:
            ``[B, D]``, NOT normalised -- the caller's ``_normalise`` owns that,
            the same as it does for the mean-pool.

        Raises:
            ValueError: If the sequence is longer than the positional table.
                Silent truncation would drop the OLDEST clicks on the reversed
                layout -- the defensible half to lose, and so the kind of bug
                nobody notices.
        """
        _, length, _ = vectors.shape
        if length > self.max_len:
            raise ValueError(f"sequence of {length} exceeds max_len={self.max_len}")

        # Oldest -> newest, padding moved to the FRONT. Index -1 is now the most
        # recent click for every non-empty row, whatever its length.
        state = _normalise(torch.flip(vectors, dims=[1])) + _normalise(
            self.pos_emb(torch.arange(length, device=vectors.device))
        )
        padding = torch.flip(mask, dims=[1]) == 0

        empty = mask.sum(dim=1) == 0
        padding = padding & ~empty.unsqueeze(-1)

        causal = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=vectors.device), diagonal=1
        )
        for block in self.blocks:
            state = block(state, src_mask=causal, src_key_padding_mask=padding)

        pooled: torch.Tensor = state[:, -1]
        zeroed: torch.Tensor = pooled.masked_fill(empty.unsqueeze(-1), 0.0)
        return zeroed

    def extra_repr(self) -> str:
        return f"max_len={self.max_len}, blocks={len(self.blocks)}"
