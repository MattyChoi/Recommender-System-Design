"""Popularity baselines: the ones that are shockingly hard to beat.

Three models, in increasing order of how much they know:

* **random** -- no information at all. Lives in the eval harness, because its
  job is to check the ruler rather than to compete.
* **most popular** -- total clicks in train. Ignores time entirely.
* **time-decayed popular** -- clicks weighted by recency. On a news corpus this
  is the real production fallback, and the half-life matters enormously: three
  days is already long when an article's whole life is measured in hours.

**Fitted on TRAIN ONLY.** A popularity vector computed over train plus dev sees
the evaluation window introducing feature leakage

**Decayed relative to each impression's own timestamp**, not to one global
``now``. A single ``now`` gives every dev impression the same scores regardless
of when it happened.
"""

from __future__ import annotations

import math

from pyspark.sql import DataFrame
from pyspark.sql import functions as f

_SECONDS_PER_DAY = 86_400.0


def click_counts(train: DataFrame) -> DataFrame:
    """Total clicks per item, from train.

    Clicks rather than impressions, deliberately. An item's impression count on
    MIND is how often Microsoft's recommender chose to show it -- a measure of
    the incumbent system, not of what users wanted. Ranking by it would build a
    baseline that imitates MSN.

    Args:
        train: Training rows with ``item_id`` and ``clicked``.

    Returns:
        ``item_id`` and ``clicks``.
    """
    return train.filter(f.col("clicked")).groupBy("item_id").agg(f.count("*").alias("clicks"))


def decayed_click_weights(train: DataFrame, half_life_days: float) -> DataFrame:
    """Per-item click history, kept as (timestamp, weight=1) rows for later decay.

    The decay cannot be folded in here: the weight of a click depends on how old
    it is *at scoring time*, and scoring time differs per impression. So this
    returns the raw click timestamps and :func:`score_decayed` applies the decay
    against each label's own instant.

    Args:
        train: Training rows with ``item_id``, ``ts`` and ``clicked``.
        half_life_days: Days for a click's weight to halve. Unused here; present
            so the caller's intent is visible at the call site and so a future
            pre-aggregation has somewhere to land.

    Returns:
        ``item_id`` and ``click_ts``, one row per click.
    """
    del half_life_days  # see docstring
    return train.filter(f.col("clicked")).select("item_id", f.col("ts").alias("click_ts"))


def score_most_popular(labels: DataFrame, train: DataFrame) -> DataFrame:
    """Score each row by its item's total train clicks.

    Items absent from train score 0.0 rather than null. That is honest and it is
    the point: this baseline is structurally blind to cold items, and on dev 32%
    of slates have a cold clicked item. The cold-item cohort is where that shows
    up, and patching it here would hide the finding.

    Args:
        labels: Rows to score, carrying ``item_id``.
        train: Training rows.

    Returns:
        ``labels`` plus ``score``.
    """
    counts = click_counts(train)
    return (
        labels.join(f.broadcast(counts), on="item_id", how="left")
        .withColumn("score", f.coalesce(f.col("clicks").cast("double"), f.lit(0.0)))
        .drop("clicks")
    )


def score_decayed_popular(
    labels: DataFrame, train: DataFrame, half_life_days: float = 3.0
) -> DataFrame:
    """Score each row by its item's recency-weighted train clicks.

    Each label's score sums ``exp(-lambda * age_days)`` over that item's train
    clicks, where age is measured from THAT label's timestamp. A click that
    happened after the label contributes nothing -- it was not knowable.

    Args:
        labels: Rows to score, carrying ``item_id`` and ``ts``.
        train: Training rows with ``item_id``, ``ts`` and ``clicked``.
        half_life_days: Days for a click's weight to halve.

    Returns:
        ``labels`` plus ``score``.

    Raises:
        ValueError: If the half-life is not positive; a zero or negative
            half-life is a configuration error, not an edge case worth encoding.
    """
    if half_life_days <= 0:
        raise ValueError(f"half_life_days must be positive, got {half_life_days}")

    decay = math.log(2) / half_life_days
    clicks = decayed_click_weights(train, half_life_days)

    age_days = (f.col("ts").cast("long") - f.col("click_ts").cast("long")) / _SECONDS_PER_DAY

    scored = (
        labels.join(f.broadcast(clicks), on="item_id", how="left")
        # A click at or after the label instant is in the future for that label.
        # Dropping it here is the same <= rule the as-of join uses.
        .withColumn(
            "weight",
            f.when(
                f.col("click_ts").isNotNull() & (f.col("click_ts") < f.col("ts")),
                f.exp(-decay * age_days),
            ).otherwise(f.lit(0.0)),
        )
    )

    keys = [c for c in labels.columns]
    return (
        scored.groupBy(*keys)
        .agg(f.sum("weight").alias("score"))
        .withColumn("score", f.coalesce(f.col("score"), f.lit(0.0)))
    )


def score_most_recent(labels: DataFrame, train: DataFrame) -> DataFrame:
    """Rank by how recently an item was last clicked -- the half-life -> 0 limit.

    Scored as ``1 / (1 + age_days)``, not as the raw timestamp. Any strictly
    decreasing function of age induces the identical ranking, and this one is
    bounded in (0, 1], keeps the "cold scores 0.0" convention the other
    baselines use, and needs no sentinel for an item nobody has clicked --
    which a negative-age score would.

    Args:
        labels: Rows to score, carrying ``item_id`` and ``ts``.
        train: Training rows with ``item_id``, ``ts`` and ``clicked``.

    Returns:
        ``labels`` plus ``score``.
    """
    # The half-life argument is ignored by design; this is reused so that "a
    # click" has one definition across every baseline in this module.
    clicks = decayed_click_weights(train, 1.0)

    age_days = (f.col("ts").cast("long") - f.col("click_ts").cast("long")) / _SECONDS_PER_DAY
    # Same "<" rule as the as-of join: a click at or after the label instant was
    # not knowable when the label was served.
    knowable = f.col("click_ts").isNotNull() & (f.col("click_ts") < f.col("ts"))

    keys = list(labels.columns)
    return (
        labels.join(f.broadcast(clicks), on="item_id", how="left")
        .groupBy(*keys)
        .agg(f.min(f.when(knowable, age_days)).alias("age_days"))
        .withColumn(
            "score",
            f.coalesce(f.lit(1.0) / (f.lit(1.0) + f.col("age_days")), f.lit(0.0)),
        )
        .select(*keys, "score")
    )
