"""The two-tower training loop."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from common.config import Settings, load_settings
from common.spark import get_spark
from common.torch_env import (
    autocast_for,
    describe,
    grad_scaler_for,
    select_device,
    set_seed,
)
from common.tracking import provenance, require_reachable, track
from data_pipeline.features.user_history import MAX_HISTORY
from evaluation.offline.geometry import _geometry
from models.classes.batching import Batch
from models.classes.dataset import ItemTables, SplitTensors
from models.classes.train import Counters, Hits, TrainingRun
from models.retrieval.dataloader.batching import (
    HISTORY_DROPOUT,
    assemble_pool,
    make_loader,
)
from models.retrieval.dataloader.dataset import (
    CONTENT_VARIANTS,
    load_item_tables,
    load_train_and_validation,
)
from models.retrieval.losses import sampled_softmax_loss
from models.retrieval.sampling import StreamingLogQ
from models.retrieval.two_tower import TwoTower


def _unwrap(model: torch.nn.Module) -> TwoTower:
    """The TwoTower inside, whether or not DDP is wrapping it."""

    inner = model.module if isinstance(model, DistributedDataParallel) else model
    assert isinstance(inner, TwoTower)
    return inner


def _arm(args: argparse.Namespace) -> str:
    """A short name for what this run is, for the MLflow run and the checkpoint."""
    towers = "both" if args.use_id and args.use_content else "id" if args.use_id else "content"
    return (
        f"{towers}-{'logq' if args.logq else 'nologq'}-n{args.max_negs}u{args.uniform_negs}"
        f"-b{args.batch_size}e{args.epochs}lr{args.lr:g}"
    )


@torch.no_grad()
def retrieval_hits(
    model: torch.nn.Module,
    loader: DataLoader[Batch],
    device: torch.device,
    k: int = 100,
    items: torch.Tensor | None = None,
) -> Hits:
    """Per-row Recall@k over the FULL catalogue.

    Args:
        items: Precomputed item embeddings with row 0 already dropped. Passing
            them lets a caller measure the same table two ways -- recall and
            geometry -- without encoding the catalogue twice, and guarantees the
            two numbers describe one table rather than two.
    """
    tower = _unwrap(model)
    was_training = tower.training
    tower.eval()
    try:
        # Row 0 is the reserved OOV bucket, not an article, so it cannot be a
        # correct answer and must not occupy a slot in the top k.
        items = tower.precompute_items()[1:] if items is None else items
        hit: list[torch.Tensor] = []
        item_ids: list[torch.Tensor] = []
        user_ids: list[torch.Tensor] = []
        for batch in loader:
            batch = batch.to(device)
            user_emb = tower.encode_user(batch.user_feats, batch.history_ids, batch.history_mask)
            top = (user_emb @ items.T).topk(k, dim=1).indices + 1  # back to 1-based
            hit.append((top == batch.item_ids.unsqueeze(1)).any(dim=1).cpu())
            item_ids.append(batch.item_ids.cpu())
            user_ids.append(batch.user_ids.cpu())
    finally:
        tower.train(was_training)

    if not hit:
        empty = torch.zeros(0, dtype=torch.long)
        return Hits(torch.zeros(0, dtype=torch.bool), empty, empty)
    return Hits(torch.cat(hit), torch.cat(item_ids), torch.cat(user_ids))


def train_step(
    model: torch.nn.Module,
    batch: Batch,
    counters: Counters,
    optimiser: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    n_items: int,
) -> torch.Tensor:
    """One optimiser step. Returns the loss, detached."""
    batch = batch.to(device)
    rows = len(batch.item_ids)
    negatives = batch.neg_ids.reshape(-1)
    is_slate = batch.neg_is_slate.reshape(-1)

    with torch.no_grad():
        positive_log_q, negative_log_q = counters.log_q_for(
            batch.item_ids, negatives, is_slate, n_items, device
        )

    with autocast_for(device):
        user_emb, item_emb, temperature = model(
            batch.user_feats,
            batch.history_ids,
            batch.history_mask,
            torch.cat([batch.item_ids, negatives]),
        )
        candidates, log_q, item_ids, index = assemble_pool(
            item_emb[:rows],
            item_emb[rows:],
            batch.item_ids,
            negatives,
            positive_log_q,
            negative_log_q,
        )
        loss = sampled_softmax_loss(
            user_emb, candidates, log_q, temperature, item_ids=item_ids, positive_index=index
        )

    scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
    scaler.step(optimiser)
    scaler.update()
    optimiser.zero_grad(set_to_none=True)

    with torch.no_grad():
        counters.observe(batch.item_ids, negatives, is_slate)

    return loss.detach()


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader[Batch],
    counters: Counters,
    optimiser: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    n_items: int,
    epoch: int,
    on_step: Callable[[], None] | None = None,
) -> float:
    """One pass. Returns the mean loss.

    Args:
        on_step: Called after every optimiser step. A hook rather than a
            geometry argument so this function keeps knowing only how to train;
            the caller owns the step counter and decides what is worth
            measuring and how often.
    """
    if isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(epoch)

    model.train()
    total = torch.zeros((), device=device)
    steps = 0
    for batch in loader:
        total += train_step(model, batch, counters, optimiser, scaler, device, n_items)
        steps += 1
        if on_step is not None:
            on_step()
    return float(total / max(steps, 1))


def validate(
    model: torch.nn.Module,
    loader: DataLoader[Batch],
    device: torch.device,
    k: int = 100,
) -> dict[str, float]:
    """Recall and the geometry it came out of, from one pass over the catalogue."""
    tower = _unwrap(model)
    was_training = tower.training
    tower.eval()
    try:
        items = tower.precompute_items()[1:]
        hit = retrieval_hits(model, loader, device, k, items=items).hit
    finally:
        tower.train(was_training)

    return {f"recall@{k}": float(hit.float().mean()) if len(hit) else 0.0, **_geometry(items)}


def item_geometry(model: torch.nn.Module) -> dict[str, float]:
    """The item table's spread, with nothing scored against it.

    Cheaper than :func:`validate` by the whole validation pass, which is what
    makes it affordable every few optimiser steps.
    """
    tower = _unwrap(model)
    was_training = tower.training
    tower.eval()
    try:
        return _geometry(tower.precompute_items()[1:])
    finally:
        tower.train(was_training)


def fit(
    model: torch.nn.Module,
    train_loader: DataLoader[Batch],
    val_loader: DataLoader[Batch],
    counters: Counters,
    device: torch.device,
    n_items: int,
    *,
    epochs: int = 10,
    learning_rate: float = 1e-3,
    patience: int = 2,
    k: int = 100,
    geometry_every: int = 0,
    on_epoch: Callable[[dict[str, float]], None] | None = None,
) -> TrainingRun:
    """Train until validation recall stops improving.

    Args:
        geometry_every: Sample the item geometry every N optimiser steps. 0
            disables it. Each sample encodes the whole catalogue, so this is
            for a diagnostic run rather than for every run.
        on_epoch: Called with each epoch's record as it is produced.

    Returns:
        A :class:`TrainingRun`. The caller logs it; this function does not know
        about MLflow.
    """
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scaler = grad_scaler_for(device)

    trace: list[dict[str, float]] = []
    taken = 0

    def sample() -> None:
        nonlocal taken
        taken += 1
        if geometry_every and taken % geometry_every == 0:
            trace.append({"step": float(taken), **item_geometry(model)})

    history: list[dict[str, float]] = []
    best = -1.0
    best_model: dict[str, torch.Tensor] | None = None
    best_counters: dict[str, dict[str, torch.Tensor]] | None = None
    stale = 0

    for epoch in range(epochs):
        loss = run_epoch(
            model,
            train_loader,
            counters,
            optimiser,
            scaler,
            device,
            n_items,
            epoch,
            on_step=sample if geometry_every else None,
        )
        measured = validate(model, val_loader, device, k)
        recall = measured[f"recall@{k}"]
        record = {"epoch": float(epoch), "loss": loss, **measured}
        history.append(record)
        if on_epoch is not None:
            on_epoch(record)

        if recall > best:
            best, stale = recall, 0
            # The counters ride along: a resume without them restarts from a flat
            # prior and mis-corrects for thousands of steps.
            best_model = {
                name: value.detach().clone() for name, value in _unwrap(model).state_dict().items()
            }
            best_counters = counters.state_dict()
        else:
            stale += 1
            if stale >= patience:
                break

    if best_model is not None and best_counters is not None:
        _unwrap(model).load_state_dict(best_model)
        counters.load_state_dict(best_counters)
    return TrainingRun(history=history, trace=trace)


def _fit_locally(
    args: argparse.Namespace,
    settings: Settings,
    items: ItemTables,
    train_split: SplitTensors,
    val_split: SplitTensors,
    device: torch.device,
    marks: Mapping[str, str],
    run_name: str,
) -> TrainingRun:
    """Build everything, train, and record the run."""
    n_items = items.content.shape[0] - 1
    model = TwoTower(
        content=items.content,
        item_category=items.category,
        item_subcategory=items.subcategory,
        n_user_feats=train_split.user_feats.shape[1],
        n_categories=items.n_categories,
        n_subcategories=items.n_subcategories,
        use_id=args.use_id,
        use_content=args.use_content,
    ).to(device)
    counters = Counters(StreamingLogQ(n_items), StreamingLogQ(n_items), corrected=args.logq).to(
        device
    )

    generator = torch.Generator().manual_seed(args.seed)
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
    val_loader = make_loader(
        val_split, n_items, args.batch_size, device, training=False, history_dropout=0.0
    )

    with track(settings, run_name, {**marks, **vars(args)}) as run:
        result = fit(
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
        )
        for record in result.history:
            run.log_metrics(
                {name: value for name, value in record.items() if name != "epoch"},
                step=int(record["epoch"]),
            )
        for record in result.trace:
            run.log_metrics(
                {f"step_{name}": value for name, value in record.items() if name != "step"},
                step=int(record["step"]),
            )

        destination = args.checkpoint or Path(settings.paths.gold).parent / "checkpoints" / (
            f"{run_name}.pt"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        # The counters ride along: a resume without them restarts from a flat
        # prior and mis-corrects for thousands of steps.
        torch.save(
            {"model": model.state_dict(), "counters": counters.state_dict(), **marks}, destination
        )
        run.log_artifact(str(destination))
        print(f"  checkpoint -> {destination}")

    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the two-tower retriever.")
    parser.add_argument("--workers", type=int, default=1, help="1 runs in-process, no Ray.")
    parser.add_argument(
        "--gpu-workers",
        action="store_true",
        help="Give each Ray worker a GPU. NCCL cannot put two ranks on one device, "
        "so on a single-GPU box --workers 2 --gpu-workers will not start.",
    )
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--k", type=int, default=100, help="Recall@k, the early-stop metric.")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--max-history", type=int, default=MAX_HISTORY)
    parser.add_argument("--max-negs", type=int, default=4, help="Slate negatives per row.")
    parser.add_argument("--uniform-negs", type=int, default=0, help="Mixed-uniform arm.")
    parser.add_argument("--history-dropout", type=float, default=HISTORY_DROPOUT)
    parser.add_argument(
        "--holdout-hours", type=int, default=12, help="Validation window, off the end of TRAIN."
    )
    parser.add_argument("--variant", choices=CONTENT_VARIANTS, default=CONTENT_VARIANTS[0])

    parser.add_argument("--no-logq", dest="logq", action="store_false")
    parser.add_argument("--no-use-id", dest="use_id", action="store_false")
    parser.add_argument("--no-use-content", dest="use_content", action="store_false")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=0,
        help="SMOKE TEST ONLY: keep the first N training rows. Any recall from a "
        "limited run is meaningless and must never be quoted. 0 disables.",
    )
    parser.add_argument(
        "--geometry-every",
        type=int,
        default=0,
        help=("Sample item_rank every N optimiser steps. 0 disables."),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    settings = load_settings()
    device = select_device()
    set_seed(args.seed)

    tables_read = ["training_examples", "user_history", "impression_negatives", "item_content"]
    marks = provenance(settings, vars(args), tables_read)
    run_name = f"{_arm(args)}-{marks['config_hash']}"
    if args.workers > 1:
        run_name = f"{run_name}-w{args.workers}"
    if args.limit_rows:
        run_name = f"{run_name}-limit{args.limit_rows}"
    print(f"{run_name}: {describe(device)}")
    if marks["git_dirty"] == "true":
        print("  WARNING: the working tree is dirty; this run's git_sha does not describe it")

    # Before the Spark read, not after: the alternative is discovering a missing
    # server several minutes in, having done all the expensive work.
    require_reachable(settings)

    spark = get_spark(settings, app="two-tower")
    try:
        items = load_item_tables(settings, args.variant)
        train_split, val_split = load_train_and_validation(
            spark, settings, args.max_history, args.max_negs, args.holdout_hours
        )
    finally:
        spark.stop()

    if args.limit_rows:
        # BOTH splits. Limiting only training leaves validation scoring every
        # held-out row against the whole catalogue -- a [batch, 65238] temporary
        # per step, on every rank, which is the expensive half and was the half
        # the first version of this flag did not touch.
        train_split = train_split.head(args.limit_rows)
        val_split = val_split.head(args.limit_rows)
        print(f"  ** --limit-rows {args.limit_rows}: this run's numbers are NOT a result **")

    print(f"  {len(train_split.item_ids):,} train rows, {len(val_split.item_ids):,} validation")

    if args.workers > 1:
        from models.retrieval.launcher import fit_distributed

        result = fit_distributed(args, settings, items, train_split, val_split, marks, run_name)
    else:
        result = _fit_locally(
            args, settings, items, train_split, val_split, device, marks, run_name
        )

    if not result.history:
        print("  no epochs recorded")
        return 1

    best = max(row[f"recall@{args.k}"] for row in result.history)
    print(f"  best recall@{args.k} = {best:.4f} over {len(result.history)} epochs")
    ranks = [row["item_rank"] for row in result.history if "item_rank" in row]
    print("  item rank by epoch: " + " ".join(f"{value:.1f}" for value in ranks))
    if result.trace:
        print(
            "  item rank by step : "
            + " ".join(f"{row['item_rank']:.1f}" for row in result.trace[:40])
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
