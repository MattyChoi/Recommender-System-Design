"""The sharded collection under two real ranks, because a plan is not a run.

``make shard-plan`` reports what ``EmbeddingShardingPlanner`` *decides*, on
``meta`` tensors, with no kernel executed and no collective performed. Every
number it prints is TorchRec's cost model's opinion. That is a legitimate thing
to report and it is not a measurement, so something has to establish that the
plan it produces corresponds to a module that runs at all -- otherwise G4's whole
deliverable is a spreadsheet.

The property worth testing is the one that only exists with two processes: **a
rank that does not hold a table still gets the right vector back.** Under
table-wise sharding the ``item`` table lives on one rank and ``user`` on the
other, so every lookup crosses an all-to-all. If that collective were broken,
each rank would return only what it holds -- and from inside one process that
looks exactly like success.

Gloo and CPU, matching ``test_distributed.py``. The cost is that the CPU sharder
offers four sharding types where an accelerator offers seven, so **row-wise is
not exercised here** -- the split that ``make shard-plan`` reports at the
crossover is the one this cannot reach. Stated rather than papered over: what is
verified is that a sharded collection forwards and backwards through a real
collective, not that every sharding type does.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytest.importorskip(
    "torchrec",
    reason="`sharded` is an optional extra; run `uv sync --extra sharded` to include it",
)

from torchrec import KeyedJaggedTensor
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder

from models.layers.sharded_embeddings import (
    FEATURE_NAMES,
    ITEM_FEATURE,
    build_ebc,
    placements,
)

WORLD = 2
N_ITEMS = 64
N_USERS = 32
DIM = 8
BATCH = 4


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


def _batch() -> KeyedJaggedTensor:
    """Identical input on every rank, carrying EVERY declared feature.

    A collection's forward walks all its tables and indexes the input by each
    table's feature name, so a KJT missing ``user_id`` raises even when only the
    item embedding is wanted. Identical across ranks is what makes "the ranks
    agree" a statement about the collective rather than about the data.
    """
    values = torch.cat(
        [torch.arange(1, BATCH + 1, dtype=torch.long) for _ in FEATURE_NAMES]
    )
    return KeyedJaggedTensor.from_lengths_sync(
        keys=list(FEATURE_NAMES),
        values=values,
        lengths=torch.ones(BATCH * len(FEATURE_NAMES), dtype=torch.long),
    )


def _sharded() -> DistributedModelParallel:
    """An EBC built on ``meta`` and materialised by the sharder, not by us."""
    return DistributedModelParallel(
        module=build_ebc(N_ITEMS, N_USERS, dim=DIM),
        device=torch.device("cpu"),
        sharders=[EmbeddingBagCollectionSharder()],
    )


# --------------------------------------------------------------- workers
# Module level so multiprocessing can pickle them.


def _worker_the_plan_uses_both_ranks(local_rank: int, world: int, init: str) -> None:
    _join(local_rank, world, init)
    try:
        model = _sharded()
        ranks = {rank for row in placements(model.plan) for rank in row.ranks}

        # Without this the rest of the file is vacuous: a plan that put both
        # tables on rank 0 would still forward correctly, and would exercise no
        # collective at all while looking exactly like a passing test.
        assert len(ranks) > 1, f"nothing was distributed; every table sits on {ranks}"
    finally:
        dist.destroy_process_group()


def _worker_every_rank_gets_the_same_vectors(local_rank: int, world: int, init: str) -> None:
    _join(local_rank, world, init)
    try:
        model = _sharded()
        output = model(_batch()).to_dict()[ITEM_FEATURE]

        assert output.shape == (BATCH, DIM)

        # THE test. Under table-wise sharding one of these ranks does not hold
        # the item table at all; it has to receive these rows over the wire. A
        # broken redistribution gives each rank only what it owns, which from
        # inside one process is indistinguishable from working.
        gathered = [torch.empty_like(output) for _ in range(world)]
        dist.all_gather(gathered, output.detach().contiguous())

        assert all(torch.allclose(gathered[0], other, atol=1e-6) for other in gathered)
    finally:
        dist.destroy_process_group()


def _worker_the_backward_reaches_the_table(local_rank: int, world: int, init: str) -> None:
    _join(local_rank, world, init)
    try:
        model = _sharded()
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}

        optimiser = torch.optim.SGD(model.parameters(), lr=1.0)
        # torch ships no annotation for Tensor.backward; the call is correct.
        model(_batch()).values().sum().backward()
        optimiser.step()

        after = model.state_dict()
        moved = [
            name
            for name, value in after.items()
            if name in before and not torch.allclose(before[name], value, atol=1e-9)
        ]

        # The fused kernel owns its own optimiser, so the SGD above may touch
        # nothing -- what is asserted is that SOMETHING updated, not which
        # mechanism did it. A sharded table whose weights never move has a
        # backward that stops at the collective.
        assert moved, "no parameter changed: the backward did not reach a table"
    finally:
        dist.destroy_process_group()


# --------------------------------------------------------------- tests


@pytest.mark.slow
class TestTheShardedCollectionActuallyRuns:
    def test_the_plan_distributes_across_ranks(self, tmp_path: Path) -> None:
        """The guard that stops the other two being vacuous."""
        _run(_worker_the_plan_uses_both_ranks, tmp_path)

    def test_every_rank_receives_the_same_embeddings(self, tmp_path: Path) -> None:
        """A rank that does not hold the table still gets the right vector."""
        _run(_worker_every_rank_gets_the_same_vectors, tmp_path)

    def test_the_backward_updates_a_table(self, tmp_path: Path) -> None:
        """Forward crossing a collective proves routing; only a backward that
        changes a weight proves the collective is differentiable."""
        _run(_worker_the_backward_reaches_the_table, tmp_path)
