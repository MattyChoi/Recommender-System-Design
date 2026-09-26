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
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq
import torch
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as f

from common.config import Settings
from common.utils import read_gold
from models.classes.dataset import ItemTables, SplitTensors

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

#: Spark's ``dayofweek`` is 1..7 with **Sunday = 1**, which is why
#: :func:`build_user_features` subtracts one before the cyclic encoding.
#: Python's ``datetime.weekday()`` is 0..6 with Monday = 0 -- a caller that
#: passes that directly produces a correctly-shaped, correctly-named vector
#: rotated by a constant, which no width or range check can see.
DAY_OF_WEEK_ORIGIN = 1

USER_TOWER_COLUMNS: tuple[str, ...] = (
    *USER_FEATURES,
    *USER_FLAGS,
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
)


def _cyclic(values: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray]:
    """Sin/cos encoding of a cyclic integer feature."""
    radians = 2.0 * math.pi * values / period
    return np.sin(radians), np.cos(radians)


def build_user_features(
    static: dict[str, npt.NDArray[np.float32]],
    has_user_features: npt.NDArray[np.float32],
    hour_of_day: npt.NDArray[np.float32],
    day_of_week: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """The tower's ``[R, 9]`` static input, from raw columns.

    Args:
        static: The four :data:`USER_FEATURES`, already null-filled. Each
            ``[R]``. ``log1p`` is applied here, to :data:`LOG1P_FEATURES` only,
            so callers pass RAW counts.
        has_user_features: ``[R]`` 1.0 where the feature store had a row. At
            serving this is ``GetUserResponse.found``; offline it is the
            missingness flag the as-of join wrote.
        hour_of_day: ``[R]`` 0..23.
        day_of_week: ``[R]`` in **Spark's 1..7, Sunday = 1** convention. See
            :data:`DAY_OF_WEEK_ORIGIN`.

    Returns:
        ``[R, 9]`` float32 in :data:`USER_TOWER_COLUMNS` order.

    Raises:
        KeyError: If ``static`` is missing one of :data:`USER_FEATURES`. Raised
            rather than zero-filled: a silently absent column shifts every
            later column left by one.
    """
    numeric: list[npt.NDArray[np.float32]] = []
    for name in USER_FEATURES:
        column = static[name]
        numeric.append(np.log1p(column) if name in LOG1P_FEATURES else column)
    numeric.append(has_user_features)
    numeric.extend(_cyclic(hour_of_day, 24))
    numeric.extend(_cyclic(day_of_week - DAY_OF_WEEK_ORIGIN, 7))

    stacked: npt.NDArray[np.float32] = np.stack(numeric, axis=1).astype("float32")
    return stacked


def load_item_tables(settings: Settings, variant: str = CONTENT_VARIANTS[0]) -> ItemTables:
    """Read ``gold/item_content`` into item-indexed tensors.

    Args:
        spark: Active session.
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
    rows = pq.read_table(
        root / "part-00000.parquet",
        columns=["item_idx", "category_idx", "subcategory_idx", variant],
    )

    idx = torch.from_numpy(rows["item_idx"].to_numpy().astype("int64"))
    n_rows = int(idx.max()) + 1
    if len(idx) != n_rows - 1 or int(idx.min()) != 1:
        raise ValueError(
            f"item_content indices are not dense over 1..{n_rows - 1}: got "
            f"{len(idx)} rows spanning {int(idx.min())}..{int(idx.max())}. A gap "
            "leaves a zero row indistinguishable from the reserved OOV row."
        )

    vectors = np.stack(rows[variant].to_numpy(zero_copy_only=False)).astype("float32")

    content = torch.zeros(n_rows, vectors.shape[1], dtype=torch.float32)
    content[idx] = torch.from_numpy(vectors)

    category = torch.zeros(n_rows, dtype=torch.long)
    category[idx] = torch.from_numpy(rows["category_idx"].to_numpy().astype("int64"))

    subcategory = torch.zeros(n_rows, dtype=torch.long)
    subcategory[idx] = torch.from_numpy(rows["subcategory_idx"].to_numpy().astype("int64"))

    return ItemTables(
        content=content,
        category=category,
        subcategory=subcategory,
        n_categories=int(category.max()),
        n_subcategories=int(subcategory.max()),
    )


def _clicked_rows(examples: DataFrame) -> DataFrame:
    """The positives, with ``ts`` kept so a window can still be cut."""
    columns = [*USER_FEATURES, *USER_FLAGS, "hour_of_day", "day_of_week"]
    return examples.where(f.col("clicked")).select(
        "impression_id", "user_idx", "item_idx", "ts", *columns
    )


def _tensors_from(
    examples: DataFrame,
    spark: SparkSession,
    settings: Settings,
    split: str,
    max_len: int,
    max_negs: int,
) -> SplitTensors:
    """Join the two per-impression tables onto ``examples`` and collect.

    Split out so ``load_split`` and :func:`load_train_and_validation` assemble
    tensors the same way; only which rows reach here differs.
    """
    history = read_gold(spark, settings, f"user_history/{split}")
    negatives = read_gold(spark, settings, f"impression_negatives/{split}")

    rows = (
        examples.join(
            history.select("impression_id", "history_idx"), on="impression_id", how="left"
        )
        .join(negatives.select("impression_id", "neg_idx"), on="impression_id", how="left")
        .toPandas()
    )

    numeric = build_user_features(
        static={name: rows[name].fillna(0.0).to_numpy(dtype="float32") for name in USER_FEATURES},
        has_user_features=rows[USER_FLAGS[0]].fillna(False).to_numpy(dtype="float32"),
        hour_of_day=rows["hour_of_day"].to_numpy(dtype="float32"),
        day_of_week=rows["day_of_week"].to_numpy(dtype="float32"),
    )

    ids, mask = _pad_ragged(rows["history_idx"], max_len)
    neg_ids, neg_mask = _pad_ragged(rows["neg_idx"], max_negs)

    return SplitTensors(
        user_feats=torch.from_numpy(numeric),
        history_ids=ids,
        history_mask=mask,
        item_ids=torch.from_numpy(rows["item_idx"].to_numpy(dtype="int64")),
        impression_ids=torch.from_numpy(rows["impression_id"].to_numpy(dtype="int64")),
        user_ids=torch.from_numpy(rows["user_idx"].to_numpy(dtype="int64")),
        neg_ids=neg_ids,
        neg_mask=neg_mask,
    )


def _boundary(examples: DataFrame, holdout_hours: int) -> datetime:
    """Where training stops and the validation window opens.

    One definition, because :func:`prior_window_counts` has to land on the same
    instant as the carve or its "recent" counts would be measured against a
    different window than the one being predicted.
    """
    end: datetime = examples.agg(f.max("ts").alias("end")).collect()[0]["end"]
    return end - timedelta(hours=holdout_hours)


def prior_window_counts(
    spark: SparkSession, settings: Settings, holdout_hours: int, n_rows: int
) -> torch.Tensor:
    """Clicks per item in the window immediately BEFORE validation.

    What a point-in-time popularity retriever standing at the boundary would
    have counted. The window is the same width as the validation window and
    adjacent to it, so "the top 100 by recent clicks" means the same thing on
    both sides of the cut.

    Distinct from the training-window counts used to BAND items: that axis is
    how often the model saw an item, this is what a counting baseline would
    have ranked. On news the two diverge sharply and conflating them is how a
    reference line turns into a strawman.

    Returns:
        ``[n_rows]`` counts, index 0 reserved and always zero.
    """
    examples = _clicked_rows(read_gold(spark, settings, "training_examples/train"))
    boundary = _boundary(examples, holdout_hours)
    window = examples.where(
        (f.col("ts") >= f.lit(boundary - timedelta(hours=holdout_hours)))
        & (f.col("ts") < f.lit(boundary))
    )

    counts = torch.zeros(n_rows, dtype=torch.long)
    for row in window.groupBy("item_idx").count().collect():
        counts[row["item_idx"]] = row["count"]
    return counts


def first_seen_hours(
    spark: SparkSession, settings: Settings, holdout_hours: int, n_rows: int
) -> torch.Tensor:
    """Hours between an item's FIRST IMPRESSION and the validation boundary.

    **First IMPRESSION, not first click.** An article is shown before anyone
    clicks it, and popular articles are clicked sooner, so dating items by their
    first click would make popularity look like youth -- and a freshness policy
    scored on it would be a popularity policy wearing a different name. This
    reads every row, not :func:`_clicked_rows`.

    Returns:
        ``[n_rows]`` ages in hours, index 0 reserved. An item absent from
        training is 0.0: brand new at the boundary, which is the same case as
        one first shown at the boundary and is not distinguished from it.
    """
    examples = read_gold(spark, settings, "training_examples/train")
    boundary = _boundary(_clicked_rows(examples), holdout_hours)

    ages = torch.zeros(n_rows, dtype=torch.float32)
    first = examples.groupBy("item_idx").agg(f.min("ts").alias("first_ts"))
    for row in first.where(f.col("first_ts") < f.lit(boundary)).collect():
        ages[row["item_idx"]] = (boundary - row["first_ts"]).total_seconds() / 3600.0
    return ages


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
    return _tensors_from(
        _clicked_rows(read_gold(spark, settings, f"training_examples/{split}")),
        spark,
        settings,
        split,
        max_len,
        max_negs,
    )


def load_train_and_validation(
    spark: SparkSession,
    settings: Settings,
    max_len: int,
    max_negs: int = 4,
    holdout_hours: int = 24,
) -> tuple[SplitTensors, SplitTensors]:
    """Train, and a TEMPORAL tail of train to early-stop on.

    **Never dev.** Early-stopping on dev selects the checkpoint that scores best
    on the test set, which contaminates every reported dev number -- the model
    would have been chosen using the thing it is about to be judged by.

    Args:
        spark: Active session.
        settings: The root configuration object.
        max_len: History width.
        max_negs: Slate negatives per row.
        holdout_hours: Width of the validation window, taken off the end of
            train. MIND's train week is short, so this is in hours rather than
            ``SplitConfig.holdout_days``; 24 leaves six days to train on.

    Returns:
        ``(train, validation)``.

    Raises:
        ValueError: If the window would leave no training rows.
    """
    examples = _clicked_rows(read_gold(spark, settings, "training_examples/train"))
    boundary = _boundary(examples, holdout_hours)

    before = examples.where(f.col("ts") < f.lit(boundary))
    after = examples.where(f.col("ts") >= f.lit(boundary))
    if before.limit(1).count() == 0:
        raise ValueError(
            f"holdout_hours={holdout_hours} leaves no training rows; the window opens at {boundary}"
        )

    return (
        _tensors_from(before, spark, settings, "train", max_len, max_negs),
        _tensors_from(after, spark, settings, "train", max_len, max_negs),
    )
