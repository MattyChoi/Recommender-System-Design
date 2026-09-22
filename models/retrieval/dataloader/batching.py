"""Batches, and the candidate pool the loss scores against.

Three things happen per batch that cannot happen once up front: the history is
randomly truncated, the slate negatives are assembled into columns, and each
column is tagged with which distribution it was drawn from so G2's correction
can use the right ``log_q``.

No DataLoader workers. ``SplitTensors`` is already resident, so a worker process
would pickle ~100 MB across a process boundary to produce data that is already
in RAM. ``num_workers=0`` and one fancy-index per batch is strictly faster.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
from torch.utils.data import DataLoader, DistributedSampler

from common.torch_env import dataloader_kwargs
from models.classes.batching import Batch, RowIndices
from models.classes.dataset import SplitTensors
from models.retrieval.distributed import (
    all_gather_detached,
    all_gather_with_grad,
    is_distributed,
    positive_index,
)
from models.retrieval.sampling import uniform_negatives

HISTORY_DROPOUT = 0.5


def truncate_history(
    history_ids: torch.Tensor,
    history_mask: torch.Tensor,
    dropout: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly shorten histories to a PREFIX.

    Length is uniform over ``[0, current]`` INCLUSIVE of zero. An empty history
    is the single most common serving state -- 88% of dev's users are new -- so
    the model should practise it rather than never see it.

    Args:
        history_ids: ``[B, L]``.
        history_mask: ``[B, L]``, a prefix of ones per row.
        dropout: Probability a given row is truncated at all. 0.0 disables,
            which is what the validation loader passes.
        generator: Seeded, so batches do not depend on what else drew from the
            global RNG first.

    Returns:
        ``(ids, mask)`` of the same shape, ids zeroed past the kept length.
    """
    if dropout <= 0.0:
        return history_ids, history_mask

    rows, width = history_ids.shape
    device = history_ids.device
    lengths = history_mask.sum(dim=1)

    chosen = torch.rand(rows, generator=generator, device=device) < dropout
    sampled = (torch.rand(rows, generator=generator, device=device) * (lengths + 1)).long()
    keep = torch.where(chosen, sampled, lengths)

    mask = (torch.arange(width, device=device).unsqueeze(0) < keep.unsqueeze(1)).long()
    return history_ids * mask, mask


def make_collate(
    split: SplitTensors,
    n_items: int,
    *,
    history_dropout: float = HISTORY_DROPOUT,
    uniform_negs: int = 0,
    generator: torch.Generator | None = None,
) -> Callable[[list[int]], Batch]:
    """A collate closure over one split's resident tensors.

    Args:
        split: From :func:`models.retrieval.dataset.load_split`.
        n_items: Catalogue size, for the uniform draws.
        history_dropout: See :func:`truncate_history`. **The validation loader
            passes 0.0.** Making this a construction argument rather than
            reading ``model.training`` keeps it explicit: two loaders, two
            settings, nothing to remember at the call site.
        uniform_negs: Extra uniformly-drawn negatives per row, ON TOP of the
            slate ones.
        generator: Seeded.

    Returns:
        A function from row indices to a :class:`Batch`.
    """

    def collate(indices: list[int]) -> Batch:
        rows = torch.tensor(indices, dtype=torch.long)
        device = split.item_ids.device

        ids, mask = truncate_history(
            split.history_ids[rows], split.history_mask[rows], history_dropout, generator
        )

        slate = split.neg_ids[rows]
        is_slate = split.neg_mask[rows].bool()
        # Where the slate ran short, substitute a uniform draw rather than
        # padding: a padded 0 flattened into the pool makes the reserved OOV row
        # a live candidate for every user, and its embedding is exactly zero.
        negatives = torch.where(
            is_slate, slate, uniform_negatives(slate.shape, n_items, device, generator)
        )

        if uniform_negs > 0:
            extra = uniform_negatives((len(rows), uniform_negs), n_items, device, generator)
            negatives = torch.cat([negatives, extra], dim=1)
            is_slate = torch.cat(
                [is_slate, torch.zeros(len(rows), uniform_negs, dtype=torch.bool, device=device)],
                dim=1,
            )

        return Batch(
            user_feats=split.user_feats[rows],
            history_ids=ids,
            history_mask=mask,
            item_ids=split.item_ids[rows],
            neg_ids=negatives,
            neg_is_slate=is_slate,
            impression_ids=split.impression_ids[rows],
            user_ids=split.user_ids[rows],
        )

    return collate


def assemble_pool(
    positive_emb: torch.Tensor,
    negative_emb: torch.Tensor,
    positive_ids: torch.Tensor,
    negative_ids: torch.Tensor,
    positive_log_q: torch.Tensor,
    negative_log_q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Every rank's candidates as one column set, positives first.

    **Two gathers, not one, and the order is the reason.** Gathering a single
    per-rank ``[positives | negatives]`` block concatenates to
    ``[r0 pos | r0 neg | r1 pos | r1 neg]``, which puts rank 1's positives at
    ``pool_size + i`` rather than ``rank * B + i`` -- so
    :func:`~models.retrieval.distributed.positive_index` would be wrong on every
    rank but 0, which is precisely the failure it exists to prevent. Gathering
    the two pieces separately gives ``[all pos | all neg]`` and the offset stays
    ``rank * B``. The extra collective is microseconds against megabytes.

    Args:
        positive_emb: ``[B, D]``, gradient-carrying.
        negative_emb: ``[B * N, D]``, gradient-carrying, flattened row-major.
        positive_ids: ``[B]``.
        negative_ids: ``[B * N]``.
        positive_log_q: ``[B]``, from the positives' own frequency counter.
        negative_log_q: ``[B * N]``, per source -- the slate counter where the
            negative came from an impression, the closed-form uniform value
            where it was drawn.

    Returns:
        ``(item_emb, log_q, item_ids, positive_index)``, ready for
        :func:`~models.retrieval.losses.sampled_softmax_loss`.
    """
    item_emb = torch.cat(
        [all_gather_with_grad(positive_emb), all_gather_with_grad(negative_emb)], dim=0
    )
    item_ids = torch.cat(
        [all_gather_detached(positive_ids), all_gather_detached(negative_ids)], dim=0
    )
    log_q = torch.cat(
        [all_gather_detached(positive_log_q), all_gather_detached(negative_log_q)], dim=0
    )
    return item_emb, log_q, item_ids, positive_index(len(positive_emb), positive_emb.device)


def make_loader(
    split: SplitTensors,
    n_items: int,
    batch_size: int,
    device: torch.device,
    *,
    training: bool,
    history_dropout: float,
    uniform_negs: int = 0,
    generator: torch.Generator | None = None,
) -> DataLoader[Batch]:
    """One split's loader.

    ``num_workers`` is 0 deliberately: ``split`` is already resident, so a worker
    would pickle it across a process boundary to hand back what the main process
    holds.

    **``drop_last`` is True for training and False for validation.** Training
    gathers across ranks, and ``_check_uniform_rows`` refuses a ragged gather --
    a final short batch would differ between ranks. Validation never gathers, so
    dropping rows there would just discard held-out requests.
    """
    rows = RowIndices(len(split.item_ids))
    sampler: DistributedSampler[int] | None = (
        DistributedSampler(rows, shuffle=training, drop_last=training)
        if training and is_distributed()
        else None
    )
    # The Dataset yields ints and the collate returns a Batch, so what iteration
    # produces is not what the Dataset's parameter says. One cast here beats a
    # passthrough generator at every consumer.
    return cast(
        "DataLoader[Batch]",
        DataLoader(
            rows,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=training and sampler is None,
            drop_last=training,
            num_workers=0,
            collate_fn=make_collate(
                split,
                n_items,
                history_dropout=history_dropout,
                uniform_negs=uniform_negs,
                generator=generator,
            ),
            **dataloader_kwargs(device),
        ),
    )
