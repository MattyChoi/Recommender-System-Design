"""``--workers N`` -- the Ray Train launcher for the neural rankers.

For the neural rankers only. Wrapping LightGBM in DDP is a category error:
averaging gradients across replicas is a statement about a model that has
gradients, and a booster's distributed mode is a different mechanism with a
different partitioning story. ``--workers`` with ``--model lgbm`` is refused at
the CLI rather than quietly ignored.

**What crosses the process boundary, and what does not.** The driver has already
started Spark once, loaded the tower and replayed retrieval into a candidate
table. A worker repeating that would be N JVMs assembling N copies of one table
-- and two reads that could disagree if gold were rebuilt between them. So the
rows travel out through the object store and one checkpoint travels back.

**MLflow stays in the driver.** Rank 0 owns the checkpoint here, as it does for
the two-tower, but not the run: every metric this stage reports is computed
after training, over the held-out rows, in the driver. Opening the run inside a
worker would put the measurement and the thing it measures in different
processes for no gain.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

import ray
import ray.train
import ray.train.torch
import torch
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchConfig, TorchTrainer
from torch import nn

from common.torch_env import backend_for_accelerator, set_seed
from models.ranking.dataset import RankingRows, shard_by_request
from models.ranking.torch_fit import (
    Factory,
    Preparation,
    Standardiser,
    fit,
    prepare,
    split_columns,
)


def train_loop_per_worker(config: dict[str, Any]) -> None:
    """One Ray worker: fit on this rank's shard, and on rank 0 write the result.

    Returns nothing. Ray can return a value per worker, but the model has to be
    written to a file for the driver to score it anyway, so a second transport
    would be a second thing that can be stale.

    Args:
        config: ``args`` as a plain dict, the model ``factory``, the shared
            ``preparation``, an absolute ``checkpoint`` path, and two
            ``ObjectRef``s for the row tables. The tables go through the object
            store rather than the task spec so the workers share one immutable
            copy instead of each deserialising its own.
    """
    args = argparse.Namespace(**config["args"])
    factory: Factory = config["factory"]
    preparation: Preparation = config["preparation"]

    rows: RankingRows = ray.get(config["rows"])
    validation: RankingRows = ray.get(config["validation"])

    context = ray.train.get_context()
    rank, world = context.get_world_rank(), context.get_world_size()
    device = ray.train.torch.get_device()

    # One seed on every rank, which is the OPPOSITE of the two-tower's rule and
    # for a reason worth stating. There the seed drew negative item ids, so a
    # shared seed made every rank sample the identical negatives. Here it seeds
    # the initial weights and the shuffle: the weights are broadcast from rank 0
    # by DDP regardless, and the shuffle is already per-rank because each rank
    # holds a different shard.
    set_seed(args.seed)

    trace: list[dict[str, float]] = []

    def on_epoch(record: dict[str, float]) -> None:
        trace.append(record)
        # EVERY rank reports. `ray.train.report` is a barrier, so reporting on
        # rank 0 alone hangs the others at the next epoch boundary.
        ray.train.report({name: value for name, value in record.items() if name != "epoch"})

    model, scaler = fit(
        factory,
        # Sharded. The validation rows are NOT: every rank scores all of them,
        # so after the gradients synchronise the ranks hold identical weights,
        # compute an identical held-out loss and stop on the same epoch. Ranks
        # disagreeing about when to stop is a hang, not a wrong number.
        shard_by_request(rows, rank, world),
        validation,
        device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        seed=args.seed,
        preparation=preparation,
        wrap=ray.train.torch.prepare_model,
        on_epoch=on_epoch,
    )

    if rank != 0:
        return

    destination = Path(config["checkpoint"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Tensors, not the numpy arrays the standardiser holds, so the driver can
    # read this back under `weights_only=True`. `torch.tensor` rather than
    # `from_numpy`: these arrays arrived through Ray's object store, which hands
    # out read-only zero-copy views, and `from_numpy` would wrap one in a tensor
    # torch believes it may write to.
    torch.save(
        {
            "model": model.state_dict(),
            "mean": torch.tensor(scaler.mean),
            "scale": torch.tensor(scaler.scale),
            "trace": trace,
        },
        destination,
    )


def fit_distributed(
    factory: Factory,
    rows: RankingRows,
    validation: RankingRows,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, Standardiser, list[dict[str, float]]]:
    """Fit ``factory`` across ``args.workers`` Ray workers and bring it home.

    Args:
        factory: Called with ``(n_dense, cardinalities)`` on every rank.
        rows: The fitting rows, unsharded. Each worker takes its own stride.
        validation: The held-out rows, scored whole by every rank.
        args: Parsed CLI, with ``workers > 1``.
        device: The driver's device, which the returned model is placed on. It
            need not be the workers' device: a CPU driver reading back a model
            trained on two GPUs is the ordinary case.

    Returns:
        ``(model, scaler, trace)`` -- the same triple the single-process path
        produces, so everything downstream of the fit is one code path.

    Raises:
        FileNotFoundError: If rank 0 wrote no checkpoint. Returning an untrained
            model instead would produce a complete, plausible, meaningless
            results table.
    """
    # Fitted ONCE here, and passed down. See `Preparation`: every rank must
    # standardise identically and size its embedding tables identically, and a
    # shard cannot be trusted to agree with its peers about either.
    preparation = prepare(rows)
    dense_columns, _ = split_columns(rows)

    handoff = Path(tempfile.mkdtemp(prefix="rank-dist-")) / "rank0.pt"
    use_gpu = bool(args.gpu_workers)

    trainer = TorchTrainer(
        train_loop_per_worker,
        train_loop_config={
            "args": vars(args),
            "factory": factory,
            "preparation": preparation,
            # Absolute: it is resolved in another process, whose working
            # directory Ray chooses.
            "checkpoint": str(handoff.resolve()),
            "rows": ray.put(rows),
            "validation": ray.put(validation),
        },
        torch_config=TorchConfig(backend=backend_for_accelerator(use_gpu)),
        scaling_config=ScalingConfig(num_workers=args.workers, use_gpu=use_gpu),
        run_config=RunConfig(name=f"rank-{args.model}"),
    )
    trainer.fit()

    if not handoff.is_file():
        raise FileNotFoundError(f"the workers finished but rank 0 wrote no model at {handoff}")

    payload = torch.load(handoff, weights_only=True)
    model = factory(len(dense_columns), preparation.cardinalities)
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    scaler = Standardiser(
        mean=payload["mean"].numpy(),
        scale=payload["scale"].numpy(),
    )
    trace: list[dict[str, float]] = payload["trace"]
    return model, scaler, trace
