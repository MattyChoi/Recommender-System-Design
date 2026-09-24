"""Hourly item aggregates of CTR, impression count, age in hours"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as f

from data_pipeline.features.ctr import _MIN_PRIOR_IMPRESSIONS, category_prior, smoothed_ctr


def item_hourly_features(events: DataFrame) -> DataFrame:
    """One row per (item, hour): rolling 24h counts, cumulative counts, and age.

    **The 24h and cumulative columns are different features and both are
    needed.** The rolling pair describes what is happening to an article now;
    the cumulative pair is its whole history, which is what
    ``models/ranking/dataset.py`` fits ``prior_clicks`` and ``train_clicks``
    from and what ``is_cold_item`` is derived from. Serving a window where
    training used a total is training/serving skew that nothing downstream can
    detect -- both columns are non-negative integers of a plausible size.
    """

    # For each item and hour, aggregate the number of impressions and clicks in that hour.
    hourly = (
        events.withColumn("feature_ts", f.date_trunc("hour", "ts") + f.expr("INTERVAL 1 HOUR"))
        .groupBy("item_id", "feature_ts")
        .agg(
            f.count("*").alias("item_impressions_1h"),
            f.sum(f.col("clicked").cast("int")).alias("item_clicks_1h"),
            f.first("category", ignorenulls=True).alias("category"),
        )
    )

    w24 = (
        Window.partitionBy("item_id")
        .orderBy(f.col("feature_ts").cast("long"))
        .rangeBetween(-24 * 3600, 0)
    )
    # Everything up to and including this hour. rangeBetween on the SAME cast
    # key as w24, and inclusive of the current bucket for the same reason: both
    # columns must mean "as of the end of this hour", because feature_ts is
    # stamped at hour-END above and the as-of join trusts that. An exclusive
    # cumulative sitting beside an inclusive 24h window would put the two
    # columns an hour out of step with each other, which reads as noise.
    cumulative = (
        Window.partitionBy("item_id")
        .orderBy(f.col("feature_ts").cast("long"))
        .rangeBetween(Window.unboundedPreceding, 0)
    )
    first_seen = Window.partitionBy("item_id")

    return (
        hourly.withColumn("item_impressions_24h", f.sum("item_impressions_1h").over(w24))
        .withColumn("item_clicks_24h", f.sum("item_clicks_1h").over(w24))
        .withColumn("item_impressions_cum", f.sum("item_impressions_1h").over(cumulative))
        .withColumn("item_clicks_cum", f.sum("item_clicks_1h").over(cumulative))
        .withColumn("item_first_seen", f.min("feature_ts").over(first_seen))
        .withColumn(
            "item_age_hours",
            (f.col("feature_ts").cast("long") - f.col("item_first_seen").cast("long")) / 3600.0,
        )
        .drop("item_first_seen")
    )


def smoothed_ctr_by_category(
    item_hourly: DataFrame,
    prior_strength: float = 20.0,
    min_prior_impressions: int = _MIN_PRIOR_IMPRESSIONS,
) -> DataFrame:
    """Shrink each item's 24h CTR toward its CATEGORY prior, as of that hour.

    Toward the category rather than the global mean, because MIND gives category
    free and a new sports article should regress toward sports behaviour rather
    than toward an average dominated by whichever vertical is largest. A
    category thinner than ``min_prior_impressions`` has not earned its own rate
    and borrows the global one instead -- see :func:`category_prior`.

    Args:
        item_hourly: Hourly item aggregates from ``item_hourly_features``.
        prior_strength: Pseudo-count for the prior, in impressions.
        min_prior_impressions: See :func:`category_prior`.

    Returns:
        ``item_hourly`` with ``cat_expanding_ctr`` and ``item_ctr_smoothed`` attached,
        and nothing else -- the rolling prior is dropped. Row count is unchanged:
        the prior joins one-to-one on ``(category, feature_ts)``.
    """
    prior = category_prior(item_hourly, min_prior_impressions)

    return (
        item_hourly.join(f.broadcast(prior), on=["category", "feature_ts"], how="left")
        .withColumn(
            "item_ctr_smoothed",
            smoothed_ctr(
                f.col("item_clicks_24h"),
                f.col("item_impressions_24h"),
                f.col("cat_expanding_ctr"),
                prior_strength,
            ),
        )
        # The rolling prior is computed but deliberately not shipped.
        .drop("cat_rolling_ctr")
    )
