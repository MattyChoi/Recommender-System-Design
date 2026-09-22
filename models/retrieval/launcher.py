"""``--workers N`` -- the Ray Train launcher for the two-tower."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import ray
import ray.train
import ray.train.torch
import torch
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchConfig, TorchTrainer

from common.config import Settings
from common.torch_env import backend_for_accelerator, grad_scaler_for, set_seed
from common.tracking import track
from models.classes.dataset import ItemTables, SplitTensors
from models.classes.train import Counters, TrainingRun
from models.retrieval.dataloader.batching import make_loader
from models.retrieval.sampling import StreamingLogQ
from models.retrieval.train import _unwrap, fit
from models.retrieval.two_tower import TwoTower


def train_loop_per_worker(config: dict[str, Any]) -> None:
    """One Ray worker: build, train, and on rank 0 record the run.

    Returns nothing, deliberately. The history travels back the way everything else 
    durable in this project does -- written to the checkpoint by rank 0 -- rather 
    than over a transport whose availability is a flag.

    Args:
        config: ``args`` as a plain dict, ``settings``, ``marks``, ``run_name``,
            and three ``ObjectRef``s for the resident tensors. The refs are
            fetched here rather than inlined because ``SplitTensors`` is ~100 MB
            and Ray would otherwise serialise a copy into every worker's task
            spec instead of sharing one immutable copy in the object store.
    """
    args = argparse.Namespace(**config["args"])
    settings: Settings = config["settings"]
    marks: Mapping[str, str] = config["marks"]
    run_name: str = config["run_name"]

    items: ItemTables = ray.get(config["items"])
    train_split: SplitTensors = ray.get(config["train_split"])
    val_split: SplitTensors = ray.get(config["val_split"])

    context = ray.train.get_context()
    rank = context.get_world_rank()
    device = ray.train.torch.get_device()

    set_seed(args.seed)
    n_items = items.content.shape[0] - 1

    model = ray.train.torch.prepare_model(
        TwoTower(
            content=items.content,
            item_category=items.category,
            item_subcategory=items.subcategory,
            n_user_feats=train_split.user_feats.shape[1],
            n_categories=items.n_categories,
            n_subcategories=items.n_subcategories,
            use_id=args.use_id,
            use_content=args.use_content,
        )
    )
    counters = Counters(StreamingLogQ(n_items), StreamingLogQ(n_items), corrected=args.logq).to(
        device
    )

    # Seeded PER RANK. With one shared seed every rank draws the IDENTICAL
    # uniform negative ids.
    generator = torch.Generator().manual_seed(args.seed + rank)
    train_loader = make_loader(
        train_split,
        n_items,
        args.batch_size,
        device,
        training=True,
        history_dropout=args.history_dropout,
        uniform_negs=args.uniform_negs,
        generator=generator,
    )
    # No sampler: every rank scores every validation row. Duplicated work, and
    # it is what keeps the ranks from deadlocking -- after DDP has synchronised
    # they hold identical weights, so they compute identical recall and reach
    # the same early-stop decision on the same epoch. Ranks disagreeing about
    # when to stop is a hang, not a wrong number.
    val_loader = make_loader(
        val_split, n_items, args.batch_size, device, training=False, history_dropout=0.0
    )

    scaler = grad_scaler_for(device)
    assert not scaler.is_enabled(), "bf16 autocast has nothing to scale; see torch_env"

    def train(recorder: Any = None) -> TrainingRun:
        def on_epoch(record: dict[str, float]) -> None:
            # EVERY rank reports. `ray.train.report` is a barrier, so reporting
            # on rank 0 alone hangs the rest at the next epoch boundary.
            ray.train.report({name: value for name, value in record.items() if name != "epoch"})
            if recorder is not None:
                recorder.log_metrics(
                    {name: value for name, value in record.items() if name != "epoch"},
                    step=int(record["epoch"]),
                )

        return fit(
            model,
            train_loader,
            val_loader,
            counters,
            device,
            n_items,
            epochs=args.epochs,
            learning_rate=args.lr,
            patience=args.patience,
            k=args.k,
            geometry_every=args.geometry_every,
            on_epoch=on_epoch,
        )

    if rank != 0:
        train()
        return

    # Rank 0 alone owns the run and the checkpoint. Every rank opening a run
    # gives one training job N MLflow runs, which is the exact provenance lie
    # `--workers` was refused over before this module existed.
    params = {**marks, **config["args"], "workers": context.get_world_size()}
    with track(settings, run_name, params) as run:
        result = train(run)
        for record in result.trace:
            run.log_metrics(
                {f"step_{name}": value for name, value in record.items() if name != "step"},
                step=int(record["step"]),
            )
        destination = Path(config["checkpoint"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": _unwrap(model).state_dict(),
                "counters": counters.state_dict(),
                "history": result.history,
                **marks,
            },
            destination,
        )
        run.log_artifact(str(destination))
        print(f"  checkpoint -> {destination}")


def fit_distributed(
    args: argparse.Namespace,
    settings: Settings,
    items: ItemTables,
    train_split: SplitTensors,
    val_split: SplitTensors,
    marks: Mapping[str, str],
    run_name: str,
) -> TrainingRun:
    """Run ``args.workers`` Ray workers over the splits the driver already read.

    Spark runs ONCE, in the driver. A worker reading its own session would be
    ``N`` JVMs assembling ``N`` identical copies of one table -- and, worse, two
    reads that could disagree if the gold layer were rebuilt between them.

    Args:
        args: Parsed CLI, with ``workers > 1``.
        settings: Passed rather than re-loaded per worker, so one training job
            cannot straddle two versions of ``conf/config.yml``.
        items: Content vectors and categorical indices.
        train_split: Everything before the validation boundary.
        val_split: The carved tail.
        marks: Provenance, recorded by rank 0.
        run_name: MLflow run name and checkpoint stem.

    Returns:
        A :class:`TrainingRun` read back out of rank 0's checkpoint. ``trace`` is
        empty here: the step-level geometry is logged to MLflow by rank 0 and is
        not carried across the process boundary.

    Raises:
        FileNotFoundError: If rank 0 wrote no checkpoint. Returning an empty
            history instead would print "no epochs recorded" for a run that may
            have trained perfectly and failed only at the save.
    """
    checkpoint = (
        args.checkpoint or Path(settings.paths.gold).parent / "checkpoints" / f"{run_name}.pt"
    ).resolve()
    use_gpu = bool(args.gpu_workers)

    # env_file = Path(".env").resolve()
    # runtime_env = {"env_vars": {"UV_ENV_FILE": str(env_file)}} if env_file.is_file() else {}
    # ray.init(runtime_env=runtime_env, ignore_reinit_error=True)

    trainer = TorchTrainer(
        train_loop_per_worker,
        train_loop_config={
            "args": vars(args),
            "settings": settings,
            "marks": dict(marks),
            "run_name": run_name,
            "checkpoint": str(checkpoint),
            "items": ray.put(items),
            "train_split": ray.put(train_split),
            "val_split": ray.put(val_split),
        },
        torch_config=TorchConfig(backend=backend_for_accelerator(use_gpu)),
        scaling_config=ScalingConfig(num_workers=args.workers, use_gpu=use_gpu),
        run_config=RunConfig(name=run_name),
    )
    trainer.fit()

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"the workers finished but rank 0 wrote no checkpoint at {checkpoint}"
        )
    history: list[dict[str, float]] = torch.load(checkpoint, weights_only=True)["history"]
    return TrainingRun(history=history, trace=[])
