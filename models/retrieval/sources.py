"""Five retrieval sources behind one interface.

| source | what it knows | the failure mode it is meant to cover |
| --- | --- | --- |
| ``two_tower`` | learned embeddings | general personalisation, long tail |
| ``trending`` | clicks in the prior 12h | cold users, breaking news |
| ``recent`` | the user's own history | immediate intent, re-reads |
| ``covisit`` | item-item co-occurrence | "because you read X" |
| ``content`` | frozen sentence vectors | brand-new items with no interactions |
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from pyspark.sql import DataFrame
from pyspark.sql import functions as f

from common.config import load_settings
from common.spark import get_spark
from common.torch_env import describe, select_device
from common.utils import read_gold
from data_pipeline.features.user_history import MAX_HISTORY
from models.classes.dataset import SplitTensors
from models.retrieval.baselines.covisit import build_covisitation
from models.retrieval.dataloader.dataset import (
    CONTENT_VARIANTS,
    _boundary,
    _clicked_rows,
    load_item_tables,
    load_train_and_validation,
    prior_window_counts,
)
from models.retrieval.evaluate import BAND_EDGES, band_of, load_tower
from models.retrieval.two_tower import TwoTower

DEFAULT_K = 100
SOURCE_NAMES = ("two_tower", "trending", "recent", "covisit", "content")
RESULTS = Path("evaluation/results/sources")

# Requests per scoring chunk. A dense [rows, 65238] score buffer is the binding
# cost, so this is a memory knob and not a modelling one.
SCORE_ROWS = 1024


@dataclass(frozen=True)
class Retrieved:
    """One source's candidates for every validation request.

    Attributes:
        source: Which source produced this.
        top: ``[R, k]`` item indices, best first, **0 where the source ran out
            of candidates**. Never padded with filler.
        item_ids: ``[R]`` the item actually clicked.
        user_ids: ``[R]`` who made the request, so a comparison can pair on the
            user rather than on the row.
        k: The cutoff.
    """

    source: str
    top: torch.Tensor
    item_ids: torch.Tensor
    user_ids: torch.Tensor
    k: int

    @property
    def hit(self) -> torch.Tensor:
        """``[R]`` whether the clicked item is anywhere in the top k."""
        found: torch.Tensor = (self.top == self.item_ids.unsqueeze(1)).any(dim=1)
        return found

    @property
    def pool(self) -> torch.Tensor:
        """``[R]`` real candidates returned, which is <= k for a short source."""
        sizes: torch.Tensor = (self.top > 0).sum(dim=1)
        return sizes

    @property
    def recall(self) -> float:
        return float(self.hit.float().mean()) if len(self.hit) else float("nan")

    @property
    def reach(self) -> float:
        """Share of requests the source could answer at all."""
        return float((self.pool > 0).float().mean()) if len(self.top) else float("nan")

    def save(self, directory: Path = RESULTS) -> Path:
        """Keep the candidates, not just the hit -- the blend needs the lists."""
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"{self.source}.npz"
        np.savez_compressed(
            destination,
            top=self.top.numpy(),
            item_ids=self.item_ids.numpy(),
            user_ids=self.user_ids.numpy(),
            k=np.asarray(self.k),
            source=np.asarray(self.source),
        )
        return destination


def load_retrieved(source: str, directory: Path = RESULTS) -> Retrieved:
    """Read one source's candidates back."""
    with np.load(directory / f"{source}.npz", allow_pickle=False) as data:
        return Retrieved(
            source=str(data["source"]),
            top=torch.from_numpy(data["top"]),
            item_ids=torch.from_numpy(data["item_ids"]),
            user_ids=torch.from_numpy(data["user_ids"]),
            k=int(data["k"]),
        )


def top_k_from_scores(
    scores: torch.Tensor,
    k: int,
    *,
    positive_only: bool = False,
    valid_rows: torch.Tensor | None = None,
) -> torch.Tensor:
    """``[B, n_items]`` scores to ``[B, k]`` 1-based indices, 0 for "nothing".

    Args:
        scores: Scores over items ``1..n``, so column ``j`` is item ``j + 1``.
        k: Cutoff.
        positive_only: Drop candidates whose score is not above zero. For a
            counting source a zero score means *the item was never counted*,
            which is an absence of evidence rather than weak evidence -- and
            without this the top k fills up with arbitrary unscored items and
            the source looks like it has full reach.
        valid_rows: ``[B]`` mask of requests the source can answer at all. A
            request it cannot answer returns an empty list rather than whatever
            an all-zero score vector happens to sort to.

    Returns:
        ``[B, k]`` long.
    """
    width = min(k, scores.shape[1])
    values, indices = scores.topk(width, dim=1)
    top = indices + 1

    if positive_only:
        top = torch.where(values > 0, top, torch.zeros_like(top))
    if valid_rows is not None:
        top = top * valid_rows.unsqueeze(1).long()
    if width < k:
        top = torch.cat([top, torch.zeros(len(top), k - width, dtype=top.dtype)], dim=1)
    return top


def _chunks(rows: int, size: int = SCORE_ROWS) -> Sequence[tuple[int, int]]:
    return [(start, min(start + size, rows)) for start in range(0, rows, size)]


def two_tower_source(
    tower: TwoTower, split: SplitTensors, device: torch.device, k: int = DEFAULT_K
) -> torch.Tensor:
    """Exact top-k over the catalogue, which an ANN index later approximates.

    Deliberately NOT routed through ``make_loader``: a loader carries history
    dropout, shuffling and ``drop_last``, all of which are training concerns, and
    two of them would silently change which rows are scored.
    """
    with torch.no_grad():
        items = tower.precompute_items()[1:]
        users = encode_users(tower, split, device)
        out = [
            top_k_from_scores(users[start:end].to(device) @ items.T, k).cpu()
            for start, end in _chunks(len(users))
        ]
    return torch.cat(out)


def encode_users(tower: TwoTower, split: SplitTensors, device: torch.device) -> torch.Tensor:
    """``[R, out_dim]`` unit-norm request vectors, one per row of the split.

    Split out because the ANN benchmark needs the same queries the exact search
    uses. Encoding them twice from two expressions is how an index gets measured
    against a slightly different set of requests than the thing it approximates.
    """
    with torch.no_grad():
        return torch.cat(
            [
                tower.encode_user(
                    split.user_feats[start:end].to(device),
                    split.history_ids[start:end].to(device),
                    split.history_mask[start:end].to(device),
                ).cpu()
                for start, end in _chunks(len(split.item_ids))
            ]
        )


def trending_source(prior: torch.Tensor, rows: int, k: int = DEFAULT_K) -> torch.Tensor:
    """Point-in-time popularity: the same list for every request."""
    counts = prior[1:].to(torch.float32).unsqueeze(0)
    return top_k_from_scores(counts, k, positive_only=True).expand(rows, k).clone()


def recent_source(
    history_ids: torch.Tensor, history_mask: torch.Tensor, k: int = DEFAULT_K
) -> torch.Tensor:
    """The user's own click history, newest first.

    Bounded above by the re-click rate, already measured at **1.1%** --
    200 of 19,006 validation clicks are for an article
    already in that user's history. So this source's ceiling is known before it
    runs, and the measurement is a check on the plumbing rather than a discovery.
    """
    width = min(k, history_ids.shape[1])
    top = history_ids[:, :width] * history_mask[:, :width]
    if width < k:
        top = torch.cat([top, torch.zeros(len(top), k - width, dtype=top.dtype)], dim=1)
    return top


def covisit_edges(edges: DataFrame, n_rows: int) -> tuple[torch.Tensor, ...]:
    """The co-visitation matrix as CSR over item indices.

    Returns:
        ``(ptr [n_rows + 1], neighbour [E], weight [E])``.
    """
    collected = edges.select("item_id", "related_item_id", "weight").collect()
    source = torch.tensor([row["item_id"] for row in collected], dtype=torch.long)
    target = torch.tensor([row["related_item_id"] for row in collected], dtype=torch.long)
    weight = torch.tensor([row["weight"] for row in collected], dtype=torch.float32)

    order = torch.argsort(source, stable=True)
    source, target, weight = source[order], target[order], weight[order]

    counts = torch.bincount(source, minlength=n_rows)
    ptr = torch.zeros(n_rows + 1, dtype=torch.long)
    ptr[1:] = counts.cumsum(0)
    return ptr, target, weight


def covisit_source(
    ptr: torch.Tensor,
    neighbour: torch.Tensor,
    weight: torch.Tensor,
    history_ids: torch.Tensor,
    history_mask: torch.Tensor,
    n_rows: int,
    k: int = DEFAULT_K,
) -> torch.Tensor:
    """Summed edge weight from everything the user clicked, to each candidate."""
    out = []
    for start, end in _chunks(len(history_ids)):
        ids = history_ids[start:end]
        mask = history_mask[start:end]
        buffer = torch.zeros(len(ids), n_rows, dtype=torch.float32)

        request, slot = torch.nonzero(mask, as_tuple=True)
        seeds = ids[request, slot]
        degree = ptr[seeds + 1] - ptr[seeds]
        if int(degree.sum()) > 0:
            # Expand each (request, seed) into that seed's neighbour run.
            offset = torch.arange(int(degree.sum())) - torch.repeat_interleave(
                degree.cumsum(0) - degree, degree
            )
            edge = torch.repeat_interleave(ptr[seeds], degree) + offset
            buffer.index_put_(
                (torch.repeat_interleave(request, degree), neighbour[edge]),
                weight[edge],
                accumulate=True,
            )
        out.append(top_k_from_scores(buffer[:, 1:], k, positive_only=True))
    return torch.cat(out)


def content_source(
    content: torch.Tensor,
    history_ids: torch.Tensor,
    history_mask: torch.Tensor,
    device: torch.device,
    k: int = DEFAULT_K,
) -> torch.Tensor:
    """Cosine between the mean of the user's read articles and every article.

    The frozen sentence vectors are unit-norm, so a dot product is a cosine. This
    is the only source that can reach an item with **zero** training clicks,
    which is 28.5% of validation rows -- so if blending is worth anything on this
    corpus, this is where it shows up.
    """
    table = content.to(device)
    out = []
    for start, end in _chunks(len(history_ids)):
        ids = history_ids[start:end].to(device)
        mask = history_mask[start:end].to(device).unsqueeze(-1).to(table.dtype)
        pooled = (table[ids] * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        valid = history_mask[start:end].sum(dim=1).to(device) > 0
        out.append(top_k_from_scores(pooled @ table[1:].T, k, valid_rows=valid).cpu())
    return torch.cat(out)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="The two-tower arm to put in the blend.")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--variant", choices=CONTENT_VARIANTS, default=CONTENT_VARIANTS[0])
    parser.add_argument("--max-history", type=int, default=MAX_HISTORY)
    parser.add_argument("--max-negs", type=int, default=4)
    parser.add_argument("--holdout-hours", type=int, default=12)
    parser.add_argument("--covisit-max-gap", type=float, default=86400.0)
    parser.add_argument("--out", type=Path, default=RESULTS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build every source's candidates over one split and keep them."""
    args = _parser().parse_args(argv)
    settings = load_settings()
    device = select_device()
    print(f"sources: {describe(device)}")

    spark = get_spark(settings, app="retrieval-sources")
    try:
        items = load_item_tables(settings, args.variant)
        n_rows = int(items.content.shape[0])
        train_split, val_split = load_train_and_validation(
            spark, settings, args.max_history, args.max_negs, args.holdout_hours
        )
        prior = prior_window_counts(spark, settings, args.holdout_hours, n_rows)

        # TRAIN ONLY, and the same boundary the split was carved on: an edge
        # built over the validation window carries the click being predicted.
        examples = read_gold(spark, settings, "training_examples/train")
        boundary = _boundary(_clicked_rows(examples), args.holdout_hours)
        window = examples.where(f.col("ts") < f.lit(boundary)).select(
            f.col("user_idx").alias("user_id"),
            f.col("item_idx").alias("item_id"),
            "impression_id",
            "ts",
            "clicked",
        )
        matrix = build_covisitation(window, max_gap_seconds=args.covisit_max_gap)
        ptr, neighbour, weight = covisit_edges(matrix, n_rows)
        seeded = int((ptr[1:] > ptr[:-1]).sum())
        print(f"  covisitation: {len(neighbour):,} edges over {seeded:,} items")
    finally:
        spark.stop()

    tower = load_tower(args.checkpoint, items, train_split.user_feats.shape[1], device)
    rows = len(val_split.item_ids)

    built = {
        "two_tower": two_tower_source(tower, val_split, device, args.k),
        "trending": trending_source(prior, rows, args.k),
        "recent": recent_source(val_split.history_ids, val_split.history_mask, args.k),
        "covisit": covisit_source(
            ptr, neighbour, weight, val_split.history_ids, val_split.history_mask, n_rows, args.k
        ),
        "content": content_source(
            items.content, val_split.history_ids, val_split.history_mask, device, args.k
        ),
    }

    train_counts = np.bincount(train_split.item_ids.numpy(), minlength=n_rows).astype("int64")
    band = band_of(train_counts[val_split.item_ids.numpy()])
    long_tail = band < int(np.searchsorted(np.asarray(BAND_EDGES), 26, side="left"))

    header = (
        f"{'source':>10}  {'reach':>7}  {'mean pool':>9}  "
        f"{'recall@' + str(args.k):>11}  {'long tail':>9}"
    )
    print(f"\n{header}\n{'-' * len(header)}")
    for name in SOURCE_NAMES:
        got = Retrieved(name, built[name], val_split.item_ids, val_split.user_ids, args.k)
        got.save(args.out)
        tail = got.hit.numpy()[long_tail]
        print(
            f"{name:>10}  {got.reach:>7.4f}  {float(got.pool.float().mean()):>9.1f}  "
            f"{got.recall:>11.4f}  {tail.mean():>9.4f}"
        )
    print(f"\n  candidates -> {args.out}/<source>.npz over {rows:,} requests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
