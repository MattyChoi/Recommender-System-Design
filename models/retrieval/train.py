"""The two-tower training loop."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
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
from models.classes.batching import Batch
from models.classes.dataset import ItemTables, SplitTensors
from models.classes.train import Counters, Hits
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
    return f"{towers}-{'logq' if args.logq else 'nologq'}-n{args.max_negs}u{args.uniform_negs}"


@torch.no_grad()
def retrieval_hits(
    model: torch.nn.Module,
    loader: DataLoader[Batch],
    device: torch.device,
    k: int = 100,
) -> Hits:
    """Per-row Recall@k over the FULL catalogue.

    Every rank scores the whole validation set rather than a shard, so all ranks
    agree without a collective. Validation is cheap next to a training epoch.

    The user's history is deliberately NOT filtered out of the candidates. It is
    common practice, but C4 measured 5,704 items clicked both before and during
    the window, so filtering would discard real positives and move the
    denominator in a way that needs its own argument.
    """
    tower = _unwrap(model)
    was_training = tower.training
    tower.eval()
    try:
        # Row 0 is the reserved OOV bucket, not an article, so it cannot be a
        # correct answer and must not occupy a slot in the top k.
        items = tower.precompute_items()[1:]
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
) -> float:
    """One pass. Returns the mean loss."""
    if isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(epoch)

    model.train()
    total = torch.zeros((), device=device)
    steps = 0
    for batch in loader:
        total += train_step(model, batch, counters, optimiser, scaler, device, n_items)
        steps += 1
    return float(total / max(steps, 1))


def validate(
    model: torch.nn.Module,
    loader: DataLoader[Batch],
    device: torch.device,
    k: int = 100,
) -> float:
    """Recall@k over the full catalogue, aggregated -- the early-stop signal.

    A thin wrapper over :func:`retrieval_hits` rather than its own loop, so the
    number that selects the checkpoint and the number the per-band report
    aggregates cannot come apart.
    """
    hit = retrieval_hits(model, loader, device, k).hit
    return float(hit.float().mean()) if len(hit) else 0.0


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
) -> list[dict[str, float]]:
    """Train until validation recall stops improving.

    Returns:
        One record per epoch: ``loss`` and ``recall@k``. The caller logs it;
        this function does not know about MLflow.
    """
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scaler = grad_scaler_for(device)

    history: list[dict[str, float]] = []
    best = -1.0
    best_model: dict[str, torch.Tensor] | None = None
    best_counters: dict[str, dict[str, torch.Tensor]] | None = None
    stale = 0

    for epoch in range(epochs):
        loss = run_epoch(model, train_loader, counters, optimiser, scaler, device, n_items, epoch)
        recall = validate(model, val_loader, device, k)
        history.append({"epoch": float(epoch), "loss": loss, f"recall@{k}": recall})

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
    return history


def _fit_locally(
    args: argparse.Namespace,
    settings: Settings,
    items: ItemTables,
    train_split: SplitTensors,
    val_split: SplitTensors,
    device: torch.device,
    marks: Mapping[str, str],
    run_name: str,
) -> list[dict[str, float]]:
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
        history = fit(
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
        )
        for record in history:
            run.log_metrics(
                {name: value for name, value in record.items() if name != "epoch"},
                step=int(record["epoch"]),
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

    return history


def _parser() -> argparse.ArgumentParser:
    """The command line. Every remaining Part G deliverable is one invocation of it.

    ``--no-logq`` produces G2's gate, the ``--no-use-*`` pair produces G1's three
    arms, and ``--max-negs``/``--uniform-negs`` produce G3's table rows.
    """
    parser = argparse.ArgumentParser(description="Train the two-tower retriever.")
    parser.add_argument("--workers", type=int, default=1, help="1 runs in-process, no Ray.")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--k", type=int, default=100, help="Recall@k, the early-stop metric.")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--max-history", type=int, default=MAX_HISTORY)
    parser.add_argument("--max-negs", type=int, default=4, help="Slate negatives per row.")
    parser.add_argument("--uniform-negs", type=int, default=0, help="G3's mixed-uniform arm.")
    parser.add_argument("--history-dropout", type=float, default=HISTORY_DROPOUT)
    parser.add_argument(
        "--holdout-hours", type=int, default=12, help="Validation window, off the end of TRAIN."
    )
    parser.add_argument("--variant", choices=CONTENT_VARIANTS, default=CONTENT_VARIANTS[0])

    parser.add_argument("--no-logq", dest="logq", action="store_false")
    parser.add_argument("--no-use-id", dest="use_id", action="store_false")
    parser.add_argument("--no-use-content", dest="use_content", action="store_false")
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.workers != 1:
        parser.error("--workers > 1 needs the Ray launcher, which is not built yet")

    settings = load_settings()
    device = select_device()
    set_seed(args.seed)

    tables_read = ["training_examples", "user_history", "impression_negatives", "item_content"]
    marks = provenance(settings, vars(args), tables_read)
    run_name = f"{_arm(args)}-{marks['config_hash']}"
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

    print(f"  {len(train_split.item_ids):,} train rows, {len(val_split.item_ids):,} validation")

    history = _fit_locally(args, settings, items, train_split, val_split, device, marks, run_name)
    best = max(row[f"recall@{args.k}"] for row in history)
    print(f"  best recall@{args.k} = {best:.4f} over {len(history)} epochs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
