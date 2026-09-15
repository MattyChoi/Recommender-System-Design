"""The temporal split protocol (guide 5.6).

Scope. This carves up ONE split's timeline for ablations. It does not produce
the project's primary evaluation, which is MIND's official train/dev boundary
-- that already satisfies the protocol.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta

from pyspark.sql import DataFrame
from pyspark.sql import functions as f

from common.config import SplitConfig

_ANCHOR = "_anchor_ts"


def split_boundaries(impressions: DataFrame, config: SplitConfig) -> tuple[datetime, datetime]:
    """Derive ``(t1, t2)`` from the corpus's own end rather than from literals.

    Two windows of ``holdout_days`` are carved off the end: test last,
    validation immediately before it. One knob rather than two, because a
    validation window of a different length to the test window measures a
    different thing and then gets compared to it anyway.

    Args:
        impressions: The timeline to cut. Triggers a Spark job: one max().
        config: Supplies ``holdout_days``.

    Returns:
        ``(t1, t2)`` for :func:`temporal_split`.

    Raises:
        ValueError: If the corpus is empty, or too short to leave any training
            data once both windows are removed.
    """
    if config.holdout_days < 1:
        raise ValueError(f"holdout_days must be >= 1, got {config.holdout_days}")

    bounds = impressions.agg(f.min("ts").alias("lo"), f.max("ts").alias("hi")).collect()[0]
    if bounds["hi"] is None:
        raise ValueError("cannot derive boundaries from an empty timeline")

    # Midnight AFTER the last day with data, so the final day is whole.
    end = datetime.combine(bounds["hi"].date(), time.min) + timedelta(days=1)
    t2 = end - timedelta(days=config.holdout_days)
    t1 = t2 - timedelta(days=config.holdout_days)

    if t1 <= bounds["lo"]:
        raise ValueError(
            f"holdout_days={config.holdout_days} leaves no training data: the corpus "
            f"starts {bounds['lo']} and the validation window would open {t1}"
        )
    return t1, t2


def _anchored(impressions: DataFrame) -> DataFrame:
    """Attach each impression's earliest timestamp to all of its rows."""
    first_seen = impressions.groupBy("impression_id").agg(f.min("ts").alias(_ANCHOR))
    # Broadcast: one row per impression, ~157K on MIND train, comfortably
    # inside the default threshold and far cheaper than shuffling the rows.
    return impressions.join(f.broadcast(first_seen), on="impression_id", how="inner")


def temporal_split(
    impressions: DataFrame,
    t1: datetime,
    t2: datetime,
    min_user_impressions: int = 0,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Cut one timeline into ``train < t1 <= val < t2 <= test``.

    Args:
        impressions: Silver rows, carrying ``impression_id``, ``user_id`` and
            ``ts``.
        t1: Start of the validation window. Used for early stopping and
            hyperparameter selection.
        t2: Start of the test window. Touched once, at the end.
        min_user_impressions: Impressions a test user must have in TRAIN to be
            kept. Guards against measuring cold-start and calling it
            personalization -- a user with no history cannot be personalized
            to, so scoring them measures the popularity fallback.

            **Zero disables the filter, and zero is the right default here.**
            The threshold is only meaningful when users recur across the
            boundary, and how often they do is a property of the corpus, not of
            the protocol. Carving up MIND's train week, 70.9% of test users
            appear in train and a threshold of 3 keeps 31.1% of them. Applied
            to the official train/dev boundary the same threshold keeps 7.1%,
            because 88% of dev's users are new -- so a filter meant to protect
            the metric would instead discard the evaluation set. Measure before
            setting it.

    Returns:
        ``(train, val, test)``. Row counts sum to the input unless
        ``min_user_impressions`` drops test users.

    Raises:
        ValueError: If the boundaries are not strictly ordered.
    """
    if not t1 < t2:
        raise ValueError(f"boundaries must be ordered: t1={t1!r} is not before t2={t2!r}")

    anchored = _anchored(impressions)
    anchor = f.col(_ANCHOR)

    train = anchored.filter(anchor < f.lit(t1))
    val = anchored.filter((anchor >= f.lit(t1)) & (anchor < f.lit(t2)))
    test = anchored.filter(anchor >= f.lit(t2))

    if min_user_impressions > 0:
        eligible = (
            train.groupBy("user_id")
            .agg(f.count_distinct("impression_id").alias("_n"))
            .filter(f.col("_n") >= min_user_impressions)
            .select("user_id")
        )
        test = test.join(f.broadcast(eligible), on="user_id", how="inner")

    return train.drop(_ANCHOR), val.drop(_ANCHOR), test.drop(_ANCHOR)


def label_cohorts(evaluation: DataFrame, train: DataFrame) -> DataFrame:
    """Mark each evaluation row as cold or warm, by user and by item.

    Cold is defined against TRAIN ALONE, never train plus val.
    Item cold-start is "absent from train" rather than "launched after t1".

    Args:
        evaluation: Rows to label, carrying ``user_id`` and ``item_id``.
        train: The training split from :func:`temporal_split`.

    Returns:
        ``evaluation`` plus boolean ``is_cold_user`` and ``is_cold_item``.
        Row count is unchanged.
    """
    warm_users = train.select("user_id").distinct().withColumn("_warm_user", f.lit(True))
    # Items are broadcast, users are not: a catalogue stays small enough to
    # ship to every executor, whereas a user table is the one that grows with
    # the product and is exactly what you do not want silently broadcast.
    warm_items = train.select("item_id").distinct().withColumn("_warm_item", f.lit(True))

    return (
        evaluation.join(f.broadcast(warm_items), on="item_id", how="left")
        .join(warm_users, on="user_id", how="left")
        .withColumn("is_cold_user", f.col("_warm_user").isNull())
        .withColumn("is_cold_item", f.col("_warm_item").isNull())
        .drop("_warm_user", "_warm_item")
    )


def cohort_summary(labelled: DataFrame) -> DataFrame:
    """Count the four cohorts, so the slicing is reported rather than assumed.

    Args:
        labelled: Output of :func:`label_cohorts`.

    Returns:
        One row per ``(is_cold_user, is_cold_item)`` with row, impression and
        user counts.
    """
    return (
        labelled.groupBy("is_cold_user", "is_cold_item")
        .agg(
            f.count("*").alias("rows"),
            f.count_distinct("impression_id").alias("impressions"),
            f.count_distinct("user_id").alias("users"),
        )
        .orderBy("is_cold_user", "is_cold_item")
    )
