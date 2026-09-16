"""Turning gold tables into the tensors the two-tower trains on.

Four tables meet here: ``training_examples`` (labels and point-in-time user
features), ``user_history`` (the sequence, keyed on ``impression_id``),
``impression_negatives`` (the rest of the slate, also keyed on
``impression_id``), and ``item_content`` (the frozen sentence vectors and the
categorical indices).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq
import torch
from pyspark.sql import SparkSession
from pyspark.sql import functions as f

from common.config import Settings
from common.utils import read_gold

# Static user features, in a fixed order. NOT a set: the order is the column
# order of the tensor the model sees..
USER_FEATURES = (
    "user_impressions_24h",
    "user_clicks_24h",
    "user_ctr_smoothed",
    "user_tenure_hours",
)
# Unbounded counts and durations. log1p before the tower's LayerNorm, or the
# first Linear sees a column ranging over four orders of magnitude next to one
# bounded in [0, 1].
LOG1P_FEATURES = frozenset({"user_impressions_24h", "user_clicks_24h", "user_tenure_hours"})
USER_FLAGS = ("has_user_features",)
CONTENT_VARIANTS = ("vec_title_abstract", "vec_title")


@dataclass(frozen=True)
class ItemTables:
    """Item-indexed inputs, sized ``n_items + 1`` with row 0 reserved.

    Attributes:
        content: ``[n_items + 1, dim]``, row 0 all-zero. ``TwoTower`` copies it
            into a frozen embedding and zeroes row 0 on the way in.
        category: ``[n_items + 1]`` category index per item.
        subcategory: ``[n_items + 1]`` subcategory index per item.
        n_categories: Distinct categories, excluding the reserved 0.
        n_subcategories: Distinct subcategories, excluding the reserved 0.
    """

    content: torch.Tensor
    category: torch.Tensor
    subcategory: torch.Tensor
    n_categories: int
    n_subcategories: int


@dataclass(frozen=True)
class SplitTensors:
    """One split, ready for a DataLoader.

    Attributes:
        user_feats: ``[R, F]`` static + context features.
        history_ids: ``[R, L]`` item indices, 0 where padded.
        history_mask: ``[R, L]`` 1 for a real entry.
        item_ids: ``[R]`` the clicked item.
        impression_ids: ``[R]`` kept so slates can be regrouped at evaluation.
        neg_ids: ``[R, K]`` items shown in the same slate and not clicked, 0
            where padded. Hard negatives.
        neg_mask: ``[R, K]`` 1 for a real negative. All-zero for a request whose
            every item was clicked, and for one whose slate held fewer than K
            others.
    """

    user_feats: torch.Tensor
    history_ids: torch.Tensor
    history_mask: torch.Tensor
    item_ids: torch.Tensor
    impression_ids: torch.Tensor
    neg_ids: torch.Tensor
    neg_mask: torch.Tensor


def _cyclic(values: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray]:
    """Sin/cos encoding of a cyclic integer feature."""
    radians = 2.0 * math.pi * values / period
    return np.sin(radians), np.cos(radians)


def load_item_tables(settings: Settings, variant: str = CONTENT_VARIANTS[0]) -> ItemTables:
    """Read ``gold/item_content`` into item-indexed tensors.

    Args:
        settings: The root configuration object.
        variant: Which cached vectors to use. ``vec_title_abstract`` is the
            better representation; ``vec_title`` is what F3's content baseline
            tokenises, so it is the one to pass when the comparison needs the
            input held fixed.

    Returns:
        An :class:`ItemTables`.

    Raises:
        ValueError: If ``variant`` is unknown, or if the cached indices are not
            dense over ``1..N``. A gap would silently leave a zero row that
            looks exactly like the reserved one, so an item with no content
            would be indistinguishable from OOV.
    """
    if variant not in CONTENT_VARIANTS:
        raise ValueError(f"variant must be one of {CONTENT_VARIANTS}; got {variant!r}")

    root = Path(settings.paths.gold) / "item_content"
    table = pq.read_table(
        root / "part-00000.parquet",
        columns=["item_idx", "category_idx", "subcategory_idx", variant],
    )

    idx = torch.from_numpy(table["item_idx"].to_numpy().astype("int64"))
    n_rows = int(idx.max()) + 1
    if len(idx) != n_rows - 1 or int(idx.min()) != 1:
        raise ValueError(
            f"item_content indices are not dense over 1..{n_rows - 1}: got "
            f"{len(idx)} rows spanning {int(idx.min())}..{int(idx.max())}. A gap "
            "leaves a zero row indistinguishable from the reserved OOV row."
        )

    vectors = np.stack(table[variant].to_numpy(zero_copy_only=False)).astype("float32")

    content = torch.zeros(n_rows, vectors.shape[1], dtype=torch.float32)
    content[idx] = torch.from_numpy(vectors)

    category = torch.zeros(n_rows, dtype=torch.long)
    category[idx] = torch.from_numpy(table["category_idx"].to_numpy().astype("int64"))

    subcategory = torch.zeros(n_rows, dtype=torch.long)
    subcategory[idx] = torch.from_numpy(table["subcategory_idx"].to_numpy().astype("int64"))

    return ItemTables(
        content=content,
        category=category,
        subcategory=subcategory,
        n_categories=int(category.max()),
        n_subcategories=int(subcategory.max()),
    )


def load_split(
    spark: SparkSession,
    settings: Settings,
    split: str,
    max_len: int,
    max_negs: int = 4,
) -> SplitTensors:
    """Read one split's CLICKED rows and its histories into tensors.

    Only clicked rows: the two-tower is trained on (user, positive) pairs and
    takes its negatives in-batch or from the slate. On train that is ~236k rows
    rather than 5.8M.

    Args:
        spark: Active session.
        settings: The root configuration object.
        split: Which split to load.
        max_len: History width. Sequences are truncated to the most recent
            ``max_len``; the table is already capped, so this only binds if it
            is set lower than the table was built with.
        max_negs: Slate negatives per row. A PREFIX of the stored array, which
            is a stable sample rather than an arbitrary one because the stored
            order is a hash -- so this is swept without rebuilding gold.

    Returns:
        A :class:`SplitTensors`.
    """
    examples = read_gold(spark, settings, f"training_examples/{split}")
    history = read_gold(spark, settings, f"user_history/{split}")
    negatives = read_gold(spark, settings, f"impression_negatives/{split}")

    columns = [*USER_FEATURES, *USER_FLAGS, "hour_of_day", "day_of_week"]
    rows = (
        examples.where(f.col("clicked"))
        .select("impression_id", "item_idx", *columns)
        .join(history.select("impression_id", "history_idx"), on="impression_id", how="left")
        .join(negatives.select("impression_id", "neg_idx"), on="impression_id", how="left")
        .toPandas()
    )

    numeric = []
    for name in USER_FEATURES:
        column = rows[name].fillna(0.0).to_numpy(dtype="float32")
        numeric.append(np.log1p(column) if name in LOG1P_FEATURES else column)
    for name in USER_FLAGS:
        numeric.append(rows[name].fillna(False).to_numpy(dtype="float32"))
    # Spark's dayofweek is 1..7, hour is 0..23; both wrap, so both get sin/cos.
    numeric.extend(_cyclic(rows["hour_of_day"].to_numpy(dtype="float32"), 24))
    numeric.extend(_cyclic(rows["day_of_week"].to_numpy(dtype="float32") - 1.0, 7))

    ids, mask = _pad_ragged(rows["history_idx"], max_len)
    neg_ids, neg_mask = _pad_ragged(rows["neg_idx"], max_negs)

    return SplitTensors(
        user_feats=torch.from_numpy(np.stack(numeric, axis=1)),
        history_ids=ids,
        history_mask=mask,
        item_ids=torch.from_numpy(rows["item_idx"].to_numpy(dtype="int64")),
        impression_ids=torch.from_numpy(rows["impression_id"].to_numpy(dtype="int64")),
        neg_ids=neg_ids,
        neg_mask=neg_mask,
    )


def _pad_ragged(
    column: Iterable[npt.NDArray[np.int64] | None], max_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ragged item-index arrays to ``[R, max_len]`` ids and mask.

    Serves both per-request sequences the loader assembles: the click history
    and the slate negatives. Both are variable-length item indices keyed on
    ``impression_id`` and both need the same two guarantees, so they share one
    implementation rather than two that drift.

    Padding is index 0 AND mask 0, both. Either alone would be enough -- the
    mask zeroes the term, and row 0 of the content table is zero -- but a caller
    that relied on only one would break silently the moment the other changed.

    Truncation keeps a PREFIX, and what that means is the caller's business:
    the history table is most-recent-first, so a prefix keeps the freshest
    clicks; the negatives table is hash-ordered, so a prefix is a stable
    pseudo-random sample. Both are deliberate, and neither is a property of
    this function.

    Args:
        column: Ragged int arrays, one per request, possibly containing nulls
            where the left join found no row at all. Typed as an iterable
            rather than a Series so the tests can pass a plain list -- the
            padding rules are the part worth exercising, and they do not need
            pandas to do it.
        max_len: Output width.

    Returns:
        ``(ids, mask)``, both ``[R, max_len]``.
    """
    entries = list(column)
    ids = torch.zeros(len(entries), max_len, dtype=torch.long)
    mask = torch.zeros(len(entries), max_len, dtype=torch.long)

    for row, value in enumerate(entries):
        if value is None or len(value) == 0:
            continue
        keep = np.asarray(value[:max_len], dtype="int64")
        ids[row, : len(keep)] = torch.from_numpy(keep)
        mask[row, : len(keep)] = 1

    return ids, mask
