"""The gather, tested by actually running two processes.

There is no way to check this with mocks. The property that matters -- that a
gradient produced by rank 1's loss arrives at rank 0's tensor -- only exists
when two real processes are exchanging real buffers, and it is exactly the
property a plain ``dist.all_gather`` lacks while looking identical from one
process. So these spawn gloo workers on CPU, which takes a second or two and is
worth it.

``test_the_detached_gather_loses_the_gradient`` is the red version: it asserts
the defect, so the reason :class:`AllGatherWithGrad` exists stays executable.

Gloo and CPU throughout. The backend is named rather than defaulted for the
reason ``common/torch_env.py`` gives, and nothing here needs a GPU: the bug is
in the autograd graph, not in the hardware.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from models.retrieval.distributed import (
    all_gather_detached,
    all_gather_with_grad,
    is_distributed,
    positive_index,
    rank,
    world_size,
)
from models.retrieval.two_tower import TwoTower

ROWS = 2
DIM = 3
WORLD = 2

# A minimum-size tower, so the DDP tests below are about synchronisation rather
# than about the model.
N_ITEMS = 3
CONTENT_DIM = 2
N_USER_FEATS = 2


def _tiny_tower() -> TwoTower:
    return TwoTower(
        content=torch.zeros(N_ITEMS + 1, CONTENT_DIM),
        item_category=torch.tensor([0, 1, 1, 2]),
        item_subcategory=torch.tensor([0, 1, 2, 3]),
        n_user_feats=N_USER_FEATS,
        n_categories=2,
        n_subcategories=3,
        out_dim=4,
    )


def _one_step(module: torch.nn.Module, local_rank: int) -> None:
    """One optimiser step on data that DIFFERS per rank.

    Identical data steps identically with or without synchronisation, so both
    DDP tests below would pass whatever DDP did.

    The difference is in the INDICES, not in ``user_feats``. ``user_norm`` is a
    LayerNorm, and LayerNorm is invariant to a per-row shift and scale, so
    handing rank r a constant feature row of ``r + 1`` normalises to the same
    zero vector on every rank and the ranks see identical input after all. An
    earlier version of this helper did exactly that, and both tests were
    vacuous. Different indices select different embedding rows, which nothing
    downstream can normalise away.
    """
    optimiser = torch.optim.SGD(module.parameters(), lr=1.0)
    user_emb, item_emb, temperature = module(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.tensor([[1, 2], [3, 0]] if local_rank == 0 else [[3, 1], [2, 0]]),
        torch.tensor([[1, 1], [1, 0]]),
        torch.tensor([1, 2]) + local_rank,
    )
    # The temperature stands in for the real loss's use of it. Leave it out and
    # log_temp gets no gradient, DDP never finalises, and NOTHING synchronises --
    # which is what these two tests measured before it was included.
    loss = user_emb.sum() + item_emb.sum() + temperature
    loss.backward()
    optimiser.step()


def _ranks_agree(module: torch.nn.Module, world: int) -> bool:
    for parameter in module.parameters():
        gathered = [torch.empty_like(parameter) for _ in range(world)]
        dist.all_gather(gathered, parameter.detach().contiguous())
        if not all(torch.allclose(gathered[0], other, atol=1e-6) for other in gathered):
            return False
    return True


def _run(worker: Callable[[int, int, str], None], tmp_path: Path, world: int = WORLD) -> None:
    """Spawn ``world`` gloo processes and re-raise whatever they assert."""
    # torch.multiprocessing re-exports spawn without listing it in __all__, and
    # ships no annotation for it. Both are stub gaps, not a wrong call.
    mp.spawn(  # type: ignore[attr-defined, no-untyped-call]
        worker,
        args=(world, f"file://{tmp_path / 'store'}"),
        nprocs=world,
        join=True,
    )


def _join(local_rank: int, world: int, init: str) -> None:
    dist.init_process_group("gloo", init_method=init, rank=local_rank, world_size=world)


# --------------------------------------------------------------- workers
# Module level so multiprocessing can pickle them.


def _worker_gradient_arrives(local_rank: int, world: int, init: str) -> None:
    _join(local_rank, world, init)
    try:
        local = torch.full((ROWS, DIM), float(local_rank + 1), requires_grad=True)
        gathered = all_gather_with_grad(local)

        assert gathered.shape == (ROWS * world, DIM)
        # Rank r weights the WHOLE gathered tensor by (r + 1), so every rank's
        # local tensor should end up with gradient 1 + 2 = 3: its own rank's
        # contribution plus the other's. A detached gather gives each rank only
        # its own weight, or no gradient at all.
        # torch ships no annotation for Tensor.backward; the call is correct.
        (gathered.sum() * (local_rank + 1)).backward()  # type: ignore[no-untyped-call]

        assert local.grad is not None
        assert torch.allclose(local.grad, torch.full((ROWS, DIM), 3.0))
    finally:
        dist.destroy_process_group()


def _worker_detached_gather_loses_it(local_rank: int, world: int, init: str) -> None:
    _join(local_rank, world, init)
    try:
        local = torch.full((ROWS, DIM), float(local_rank + 1), requires_grad=True)
        buffers = [torch.empty_like(local) for _ in range(world)]
        dist.all_gather(buffers, local.contiguous())
        gathered = torch.cat(buffers, dim=0)

        # The defect, stated: the copies are plain data, so the concatenation is
        # not attached to anything and cannot be backpropagated at all.
        assert not gathered.requires_grad
        with pytest.raises(RuntimeError):
            gathered.sum().backward()  # type: ignore[no-untyped-call]
    finally:
        dist.destroy_process_group()


def _worker_order_is_rank_order(local_rank: int, world: int, init: str) -> None:
    _join(local_rank, world, init)
    try:
        local = torch.full((ROWS, DIM), float(local_rank))
        gathered = all_gather_detached(local)

        # Rank r's rows occupy [r * ROWS, (r + 1) * ROWS). positive_index and
        # the backward slice both depend on this and neither could detect it
        # being otherwise.
        for source in range(world):
            block = gathered[source * ROWS : (source + 1) * ROWS]
            assert torch.allclose(block, torch.full((ROWS, DIM), float(source)))
    finally:
        dist.destroy_process_group()


def _worker_positive_index_points_at_my_rows(local_rank: int, world: int, init: str) -> None:
    _join(local_rank, world, init)
    try:
        local = torch.full((ROWS, DIM), float(local_rank))
        gathered = all_gather_detached(local)
        picked = gathered[positive_index(ROWS, local.device)]

        assert rank() == local_rank and world_size() == world
        # The whole point: on any rank but 0 the default arange would select
        # rank 0's rows, and the loss would train every row against a stranger's
        # item as its positive, silently.
        assert torch.allclose(picked, local)
    finally:
        dist.destroy_process_group()


def _worker_ragged_rows_raise(local_rank: int, world: int, init: str) -> None:
    _join(local_rank, world, init)
    try:
        local = torch.zeros(ROWS + local_rank, DIM, requires_grad=True)
        with pytest.raises(ValueError, match="same row count"):
            all_gather_with_grad(local)
    finally:
        dist.destroy_process_group()


def _worker_ddp_syncs_through_forward(local_rank: int, world: int, init: str) -> None:
    _join(local_rank, world, init)
    try:
        torch.manual_seed(0)  # identical initialisation on every rank
        wrapped = DistributedDataParallel(_tiny_tower())
        _one_step(wrapped, local_rank)

        assert _ranks_agree(wrapped.module, world)
    finally:
        dist.destroy_process_group()


def _worker_reaching_past_ddp_diverges(local_rank: int, world: int, init: str) -> None:
    _join(local_rank, world, init)
    try:
        torch.manual_seed(0)
        wrapped = DistributedDataParallel(_tiny_tower())
        # What §9's G5 sketch does. DDP arms its all-reduce inside its own
        # forward, so going straight to the module skips the bookkeeping and the
        # gradient hooks return early.
        _one_step(wrapped.module, local_rank)

        assert not _ranks_agree(wrapped.module, world)
    finally:
        dist.destroy_process_group()


# --------------------------------------------------------------- tests


class TestWithoutAProcessGroup:
    """The single-GPU path, which is how this is developed and must not differ."""

    def test_it_reports_a_world_of_one(self) -> None:
        assert not is_distributed()
        assert world_size() == 1 and rank() == 0

    def test_the_gather_returns_the_same_tensor(self) -> None:
        """Identity, not a copy: at world size 1 the collective would be pure
        overhead, and the training loop should not need a branch to avoid it."""
        local = torch.zeros(ROWS, DIM)

        assert all_gather_with_grad(local) is local
        assert all_gather_detached(local) is local

    def test_the_positive_index_is_the_plain_arange(self) -> None:
        got = positive_index(4, torch.device("cpu"))

        assert torch.equal(got, torch.arange(4))


@pytest.mark.slow
class TestWithTwoRanks:
    def test_a_gradient_from_the_other_rank_arrives(self, tmp_path: Path) -> None:
        """The property the whole module exists for."""
        _run(_worker_gradient_arrives, tmp_path)

    def test_the_detached_gather_loses_the_gradient(self, tmp_path: Path) -> None:
        """Red by construction: this is what the plain primitive does."""
        _run(_worker_detached_gather_loses_it, tmp_path)

    def test_the_concatenation_is_in_rank_order(self, tmp_path: Path) -> None:
        _run(_worker_order_is_rank_order, tmp_path)

    def test_the_positive_index_selects_this_ranks_rows(self, tmp_path: Path) -> None:
        _run(_worker_positive_index_points_at_my_rows, tmp_path)

    def test_uneven_row_counts_raise_rather_than_hang(self, tmp_path: Path) -> None:
        """Without the check this deadlocks, which in CI is a timeout with no
        message rather than a failure with one."""
        _run(_worker_ragged_rows_raise, tmp_path)


@pytest.mark.slow
class TestDdpNeedsTheForward:
    """Why ``TwoTower.forward`` exists, settled by running it rather than argued.

    These live here rather than in ``test_two_tower.py`` because the property is
    a distributed one and the spawn machinery is here.
    """

    def test_going_through_forward_keeps_the_ranks_identical(self, tmp_path: Path) -> None:
        """Each rank steps on different data, so agreement afterwards can only
        come from DDP having averaged the gradients."""
        _run(_worker_ddp_syncs_through_forward, tmp_path)

    def test_reaching_past_ddp_lets_the_ranks_drift(self, tmp_path: Path) -> None:
        """Red by construction, and the reason the model grew a ``forward``.

        Two GPUs quietly training two different models is not a failure anything
        reports: the loss falls on both, and they only disagree in the weights.
        """
        _run(_worker_reaching_past_ddp_diverges, tmp_path)
