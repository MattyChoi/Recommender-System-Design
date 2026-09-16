"""The two-tower training loop."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from common.config import Settings, load_settings
from common.spark import get_spark
from common.torch_env import (
    autocast_for,
    dataloader_kwargs,
    describe,
    grad_scaler_for,
    select_device,
    set_seed,
)
from common.tracking import provenance, require_reachable, track
from data_pipeline.features.user_history import MAX_HISTORY
from models.retrieval.batching import (
    HISTORY_DROPOUT,
    Batch,
    RowIndices,
    assemble_pool,
    make_collate,
)
from models.retrieval.dataset import (
    CONTENT_VARIANTS,
    ItemTables,
    SplitTensors,
    load_item_tables,
    load_train_and_validation,
)
from models.retrieval.distributed import all_gather_detached, is_distributed
from models.retrieval.losses import sampled_softmax_loss
from models.retrieval.sampling import StreamingLogQ, uniform_log_q
from models.retrieval.two_tower import TwoTower


@dataclass
class Counters:
    """One frequency estimator per negative source, and the ablation switch.

    The correction assumes every softmax column was drawn from the distribution
    ``q`` describes, and the pool mixes two observed distributions -- positives
    drawn by popularity, slate negatives drawn by exposure. Uniform draws need
    no estimator; their probability is exact.

    Attributes:
        positives: Frequency of the clicked items.
        slate: Frequency of the impression negatives.
        corrected: False makes every ``log_q`` zero, which is G2's gate --
            "train with and without the correction". A per-row constant is
            invisible to softmax, so zeros are exactly "no correction". Living
            here rather than as a flag threaded through ``fit``, ``run_epoch``
            and ``train_step`` keeps the thing being ablated in one place.
    """

    positives: StreamingLogQ
    slate: StreamingLogQ
    corrected: bool = True

    def to(self, device: torch.device) -> Counters:
        return Counters(self.positives.to(device), self.slate.to(device), self.corrected)

    def log_q_for(
        self,
        item_ids: torch.Tensor,
        negatives: torch.Tensor,
        is_slate: torch.Tensor,
        n_items: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The correction for the positive and negative columns.

        Call BEFORE :meth:`observe` for a given batch, or it conditions its own
        correction on its own labels.
        """
        if not self.corrected:
            return (
                torch.zeros(item_ids.shape, device=device),
                torch.zeros(negatives.shape, device=device),
            )
        return (
            self.positives.log_q(item_ids),
            torch.where(
                is_slate,
                self.slate.log_q(negatives),
                uniform_log_q((negatives.numel(),), n_items, device),
            ),
        )

    def observe(
        self, item_ids: torch.Tensor, negatives: torch.Tensor, is_slate: torch.Tensor
    ) -> None:
        """Count what appeared as a column, across every rank.

        The GATHERED ids, because the distribution being modelled is the whole
        pool -- and every rank sees the identical gathered set, so the counters
        stay in step without a collective of their own.

        Gather the fixed-width tensors and mask afterwards: the masked subset has
        a different length per rank, and a ragged gather either hangs or
        misaligns.
        """
        if not self.corrected:
            return
        self.positives.update(all_gather_detached(item_ids))
        drawn = all_gather_detached(negatives)
        self.slate.update(drawn[all_gather_detached(is_slate.long()).bool()])

    def state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        return {"positives": self.positives.state_dict(), "slate": self.slate.state_dict()}

    def load_state_dict(self, state: dict[str, dict[str, torch.Tensor]]) -> None:
        self.positives.load_state_dict(state["positives"])
        self.slate.load_state_dict(state["slate"])


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


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader[Batch],
    device: torch.device,
    k: int = 100,
) -> float:
    """Recall@k over the FULL catalogue.

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
        hits = 0
        seen = 0
        for batch in loader:
            batch = batch.to(device)
            user_emb = tower.encode_user(batch.user_feats, batch.history_ids, batch.history_mask)
            top = (user_emb @ items.T).topk(k, dim=1).indices + 1  # back to 1-based
            hits += int((top == batch.item_ids.unsqueeze(1)).any(dim=1).sum())
            seen += len(batch.item_ids)
        return hits / max(seen, 1)
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


def _arm(args: argparse.Namespace) -> str:
    """A short name for what this run is, for the MLflow run and the checkpoint."""
    towers = "both" if args.use_id and args.use_content else "id" if args.use_id else "content"
    return f"{towers}-{'logq' if args.logq else 'nologq'}-n{args.max_negs}u{args.uniform_negs}"


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


def _unwrap(model: torch.nn.Module) -> TwoTower:
    """The TwoTower inside, whether or not DDP is wrapping it."""

    inner = model.module if isinstance(model, DistributedDataParallel) else model
    assert isinstance(inner, TwoTower)
    return inner


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
