"""Cross-rank collectives for in-batch negatives.

With in-batch negatives, **batch size is the negative count**. Splitting a batch
of 8192 across two GPUs leaves each rank contrasting against 4096 columns, so
adding hardware weakens the objective. Gathering every rank's item embeddings
restores it -- each rank scores its own 4096 users against all 8192 items.

**No device is named here.** The backend comes from
``common.torch_env.distributed_backend``; these functions act on whatever device
the tensors already live on.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    """Whether a process group with more than one rank is live.

    A single-rank run is not a degenerate distributed run: every gather below is
    an identity, so the wrappers short-circuit rather than paying a collective to
    return what they were given.
    """
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def world_size() -> int:
    """Ranks in the group, or 1 when there is no group."""
    return dist.get_world_size() if is_distributed() else 1


def rank() -> int:
    """This process's rank, or 0 when there is no group."""
    return dist.get_rank() if is_distributed() else 0


def _check_uniform_rows(tensor: torch.Tensor) -> None:
    """Refuse a ragged gather instead of hanging or silently misaligning.

    Every rank allocates receive buffers shaped like its OWN tensor, so unequal
    row counts either hang the collective or fill the buffers with the wrong
    number of rows -- and the ``rank * rows`` arithmetic in
    :meth:`AllGatherWithGrad.backward` and :func:`positive_index` then points at
    the wrong slice on every rank but 0.
    """
    local = torch.tensor([tensor.shape[0]], device=tensor.device)
    counts = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(counts, local)
    sizes = [int(c.item()) for c in counts]
    if len(set(sizes)) != 1:
        raise ValueError(
            f"all_gather needs the same row count on every rank; got {sizes}. "
            "Set drop_last=True on the sampler."
        )


class AllGatherWithGrad(torch.autograd.Function):
    """``all_gather`` that gradients flow back through.

    Forward concatenates every rank's tensor in rank order. Backward sums the
    incoming gradient across ranks and returns this rank's slice of it: each
    rank's loss touches every column of the gathered tensor, so the true
    gradient for my rows is the sum of what all ranks want from them.

    Note:
        ``torch.distributed.nn.functional.all_gather`` does this in the box, and
        its backward is cheaper -- reduce_scatter delivers each rank only its own
        slice, where the all_reduce below moves ``world_size`` times the data and
        then discards most of it. It is not used because **it branches on
        backend**: reduce_scatter on NCCL/XCCL, all-to-all plus a sum elsewhere
        (verified against torch 2.13). Tests here run gloo on CPU and training
        runs NCCL, so the builtin's tested path would not be its trained path.
        This one has a single path on both.
    """

    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor) -> torch.Tensor:
        _check_uniform_rows(tensor)
        ctx.rank = dist.get_rank()
        ctx.rows = tensor.shape[0]
        gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
        # contiguous() because the collective reads a flat buffer, and a sliced
        # or transposed tensor would send the wrong bytes without complaining.
        dist.all_gather(gathered, tensor.contiguous())
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> torch.Tensor:
        # Cloned because all_reduce writes in place and this buffer belongs to
        # autograd, which may hand the same one to another consumer.
        summed = grad.contiguous().clone()
        dist.all_reduce(summed)
        start = ctx.rank * ctx.rows
        return summed[start : start + ctx.rows]


def all_gather_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    """Every rank's ``tensor``, concatenated, with gradients flowing back.

    For the ITEM EMBEDDINGS, which are the negatives and must be trained.
    Returns the input unchanged at world size 1, so the single-GPU path is the
    same code rather than a branch in the training loop.
    """
    if not is_distributed():
        return tensor
    gathered: torch.Tensor = AllGatherWithGrad.apply(tensor)  # type: ignore[no-untyped-call]
    return gathered


def all_gather_detached(tensor: torch.Tensor) -> torch.Tensor:
    """Every rank's ``tensor``, concatenated, with no gradient path.

    For the things that carry no gradient and never should -- ``item_ids`` and
    ``log_q``. §9's G5 sketch wraps both in the grad-carrying gather, which is
    harmless (an integer tensor has no grad, and ``log_q`` is a constant) but
    says something untrue about where gradients go.
    """
    if not is_distributed():
        return tensor
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor.contiguous())
    return torch.cat(gathered, dim=0)


def positive_index(rows: int, device: torch.device) -> torch.Tensor:
    """Which gathered column holds each local row's positive.

    ``rank * rows + arange(rows)``, because the gather concatenates in rank
    order. This is the one line the loss cannot infer and the one that is
    correct-by-accident on rank 0, so it lives here rather than being spelled out
    at the call site.
    """
    return rank() * rows + torch.arange(rows, device=device)
