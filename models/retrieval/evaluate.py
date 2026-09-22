"""Recall@k by item popularity band.

The aggregate hides the thing the logQ correction is supposed to change. The
correction adjusts each softmax column for how often that item is drawn as a
candidate, so its effect should land on items drawn rarely and be close to
invisible on the ones drawn constantly. One number averages the two together.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch

from common.config import load_settings
from common.spark import get_spark
from common.torch_env import describe, select_device
from data_pipeline.features.user_history import MAX_HISTORY
from models.classes.dataset import ItemTables
from models.classes.train import Hits
from models.retrieval.dataloader.batching import make_loader
from models.retrieval.dataloader.dataset import (
    CONTENT_VARIANTS,
    load_item_tables,
    load_train_and_validation,
    prior_window_counts,
)
from models.retrieval.train import retrieval_hits
from models.retrieval.two_tower import TwoTower

# Upper-inclusive edges. The first band is the single value 0 -- cold items,
# which no counting retriever can reach and which are a quarter of the traffic.
BAND_EDGES = (0, 2, 5, 10, 25, 50, 100, 500)


@dataclass(frozen=True)
class BandRow:
    """One row of the report.

    Attributes:
        label: The band's click range.
        rows: Validation requests in the band.
        items: Distinct clicked items in the band. Alongside ``rows`` this says
            whether a band is many items clicked once or few clicked often,
            which decides whether it is worth splitting further.
        recall: The model's Recall@k on those rows.
        popularity: The counting baseline's Recall@k on the same rows.
    """

    label: str
    rows: int
    items: int
    recall: float
    popularity: float

    @property
    def rows_per_item(self) -> float:
        return self.rows / self.items if self.items else 0.0


def band_labels(edges: Sequence[int] = BAND_EDGES) -> tuple[str, ...]:
    """Labels DERIVED from the edges, never written alongside them.

    Written by hand the two drift: someone moves an edge, the label keeps
    claiming the old range, and the plot misreports with no test failing.
    """
    labels: list[str] = []
    low = 0
    for edge in edges:
        labels.append(str(low) if low == edge else f"{low}-{edge}")
        low = edge + 1
    labels.append(f"{low}+")
    return tuple(labels)


def band_of(
    counts: npt.NDArray[np.int64], edges: Sequence[int] = BAND_EDGES
) -> npt.NDArray[np.int64]:
    """Band index per row, upper-inclusive, with anything past the last edge on top."""
    indices: npt.NDArray[np.int64] = np.searchsorted(np.asarray(edges), counts, side="left")
    return indices


def popularity_hits(
    prior: torch.Tensor, item_ids: npt.NDArray[np.int64], k: int
) -> npt.NDArray[np.bool_]:
    """Whether each row's clicked item is in the top k by recent clicks.

    The reference line: no learning, just counting. Ties at the cut are broken
    by item index, which is arbitrary -- but a top-k of unstable size is worse,
    and the cut on this corpus falls clear of the tie mass.
    """
    top = torch.argsort(prior, descending=True, stable=True)[:k]
    hits: npt.NDArray[np.bool_] = np.isin(item_ids, top.numpy())
    return hits


def summarise(
    hits: Hits,
    band: npt.NDArray[np.int64],
    popularity: npt.NDArray[np.bool_],
    edges: Sequence[int] = BAND_EDGES,
) -> list[BandRow]:
    """One :class:`BandRow` per band, plus an ``overall`` row last.

    The overall row is not decoration: it must reproduce the aggregate the
    training loop reported, and if it does not, the bands are being computed
    over a different row set than the model was scored on.
    """
    hit = hits.hit.numpy()
    items = hits.item_ids.numpy()
    labels = band_labels(edges)

    rows: list[BandRow] = []
    for index, label in enumerate(labels):
        mask = band == index
        rows.append(
            BandRow(
                label=label,
                rows=int(mask.sum()),
                items=len(np.unique(items[mask])),
                recall=float(hit[mask].mean()) if mask.any() else float("nan"),
                popularity=float(popularity[mask].mean()) if mask.any() else float("nan"),
            )
        )
    rows.append(
        BandRow(
            label="overall",
            rows=len(hit),
            items=len(np.unique(items)),
            recall=float(hit.mean()) if len(hit) else float("nan"),
            popularity=float(popularity.mean()) if len(hit) else float("nan"),
        )
    )
    return rows


def render(rows: Sequence[BandRow], k: int) -> str:
    """The report, as a fixed-width table."""
    head = f"{'band':>9}  {'rows':>7}  {'items':>6}  {'rows/item':>9}  "
    head += f"{f'recall@{k}':>10}  {f'pop@{k}':>8}"
    lines = [head, "-" * len(head)]
    for row in rows:
        if row.label == "overall":
            lines.append("-" * len(head))
        lines.append(
            f"{row.label:>9}  {row.rows:>7,}  {row.items:>6,}  {row.rows_per_item:>9.1f}  "
            f"{row.recall:>10.4f}  {row.popularity:>8.4f}"
        )
    return "\n".join(lines)


def load_tower(path: Path, items: ItemTables, n_user_feats: int, device: torch.device) -> TwoTower:
    """Rebuild the tower a checkpoint was saved from.

    ``use_id`` and ``use_content`` are READ OFF the state dict rather than
    passed in. They are the ablation axis, so a mismatched flag would score one
    arm's rows through the other arm's architecture -- and the run name on the
    output would still look right. Every other dimension is left at its default
    and guarded by ``strict=True``, which raises on any shape it did not expect.
    """
    state = torch.load(path, map_location=device, weights_only=True)["model"]
    model = TwoTower(
        content=items.content,
        item_category=items.category,
        item_subcategory=items.subcategory,
        n_user_feats=n_user_feats,
        n_categories=items.n_categories,
        n_subcategories=items.n_subcategories,
        use_id="item_id_emb.weight" in state,
        use_content="content.weight" in state,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def save(
    destination: Path,
    hits: Hits,
    band: npt.NDArray[np.int64],
    popularity: npt.NDArray[np.bool_],
    train_counts: npt.NDArray[np.int64],
) -> None:
    """Per-row results, so two arms can be compared without rescoring either.

    Scoring is the expensive half; the comparison is arithmetic. Keeping the
    rows also means the banding rule can change later without another forward
    pass over the catalogue.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        hit=hits.hit.numpy(),
        item_ids=hits.item_ids.numpy(),
        user_ids=hits.user_ids.numpy(),
        band=band,
        popularity=popularity,
        train_counts=train_counts,
        band_edges=np.asarray(BAND_EDGES),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recall@k by item popularity band.")
    parser.add_argument("checkpoint", type=Path, help="A .pt written by models.retrieval.train.")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--variant", choices=CONTENT_VARIANTS, default=CONTENT_VARIANTS[0])
    parser.add_argument("--max-history", type=int, default=MAX_HISTORY)
    parser.add_argument("--max-negs", type=int, default=4)
    parser.add_argument(
        "--holdout-hours",
        type=int,
        default=12,
        help="MUST match the training run, or this scores a different window.",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Defaults under evaluation/results/retrieval."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Score one checkpoint, band it, print the table, keep the rows."""
    args = _parser().parse_args(argv)
    settings = load_settings()
    device = select_device()
    print(f"{args.checkpoint.stem}: {describe(device)}")

    spark = get_spark(settings, app="retrieval-bands")
    try:
        items = load_item_tables(settings, args.variant)
        train_split, val_split = load_train_and_validation(
            spark, settings, args.max_history, args.max_negs, args.holdout_hours
        )
        recent = prior_window_counts(spark, settings, args.holdout_hours, items.content.shape[0])
    finally:
        spark.stop()

    n_items = items.content.shape[0] - 1
    model = load_tower(args.checkpoint, items, train_split.user_feats.shape[1], device)
    loader = make_loader(
        val_split, n_items, args.batch_size, device, training=False, history_dropout=0.0
    )

    hits = retrieval_hits(model, loader, device, args.k)
    train_counts = np.bincount(
        train_split.item_ids.numpy(), minlength=items.content.shape[0]
    ).astype("int64")
    clicked = hits.item_ids.numpy()
    band = band_of(train_counts[clicked])
    popularity = popularity_hits(recent, clicked, args.k)

    print()
    print(render(summarise(hits, band, popularity), args.k))

    destination = args.out or Path("evaluation/results/retrieval") / f"{args.checkpoint.stem}.npz"
    save(destination, hits, band, popularity, train_counts)
    print(f"\n  rows -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
