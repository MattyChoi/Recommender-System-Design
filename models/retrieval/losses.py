"""Sampled softmax over in-batch negatives, with the logQ correction.

In-batch negatives are drawn by whatever the loader hands you, which on an
impression log means proportional to popularity. A popular item therefore serves
as a negative far more often than its share of the catalogue warrants, and an
uncorrected softmax learns to push it down. Subtracting ``log q`` restores the
full-catalogue softmax in expectation (Yi et al., 2019).
"""

from __future__ import annotations

import torch
from torch.nn import functional as f


def sampled_softmax_loss(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    log_q: torch.Tensor,
    temperature: torch.Tensor | float,
    item_ids: torch.Tensor | None = None,
    positive_index: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy over in-batch negatives, corrected for sampling bias.

    Args:
        user_emb: ``[B, D]``, L2-normalised by the tower.
        item_emb: ``[N, D]``. ``N == B`` in a single process; under DDP it is
            ``B * world_size`` after the gradient-carrying all-gather.
        log_q: ``[N]``, the log sampling probability of each COLUMN. Per item,
            not per user -- a transposed broadcast here is a constant shift per
            row, which softmax ignores, so the correction silently does nothing.
        temperature: The model's learned temperature.
        item_ids: ``[N]``. When given, a repeat of a row's positive elsewhere in
            the batch is masked out: it is a false negative and trains against
            the objective.
        positive_index: ``[B]``, the column holding each row's positive.
            Defaults to ``arange(B)``, which is correct only when ``N == B``.
            Under DDP the caller passes ``rank * B + arange(B)``; the default is
            right on rank 0 and wrong everywhere else, with no error and a loss
            that still decreases.

    Returns:
        Scalar loss.

    Raises:
        ValueError: If ``log_q``, ``item_ids`` or ``positive_index`` disagree
            with the embedding shapes.
    """
    n_users, n_items = len(user_emb), len(item_emb)

    if len(log_q) != n_items:
        raise ValueError(f"log_q has {len(log_q)} entries for {n_items} item rows")

    if positive_index is None:
        positive_index = torch.arange(n_users, device=user_emb.device)
    elif len(positive_index) != n_users:
        raise ValueError(f"positive_index has {len(positive_index)} entries for {n_users} users")

    # penalize popular items using log q
    logits = (user_emb @ item_emb.T) / temperature - log_q.unsqueeze(0)

    if item_ids is not None:
        # Mask the duplicate items in the batch so they don't get penalized as the wrong answer
        # despite being the right answer
        if len(item_ids) != n_items:
            raise ValueError(f"item_ids has {len(item_ids)} entries for {n_items} item rows")
        duplicate = item_ids[positive_index].unsqueeze(1) == item_ids.unsqueeze(0)
        duplicate[torch.arange(n_users, device=logits.device), positive_index] = False
        # Fill these in with a score that won't affect the softmax score
        logits = logits.masked_fill(duplicate, torch.finfo(logits.dtype).min)

    return f.cross_entropy(logits, positive_index)
