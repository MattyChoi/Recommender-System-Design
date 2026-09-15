"""The user x category cross feature: how this reader behaves in this vertical.

D1 groups features by refresh cadence and gives "cross" its own row --
user-category affinity multiplied by the item's category is its example. It is
the first feature here that belongs to neither entity alone: the item timeline
knows nothing about who is reading, the user timeline nothing about what they
are reading, and the interesting signal is in the pair.

Which makes (user, category) a timeline in its own right, joined as-of on a
composite key. That is not a special case of the as-of join, just a finer
partition of the same window.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window, WindowSpec
from pyspark.sql import functions as f

from data_pipeline.features.ctr import smoothed_ctr


def _expanding(*partition: str) -> WindowSpec:
    """Every closed bucket at or before this one, for the given partition.

    Expanding rather than the 24h frame the item and user series use, and the
    difference is deliberate. Those measure ACTIVITY, which is bursty and
    should decay. This measures TASTE, which is stable -- and the counts are
    tiny, roughly 994,000 (user, category) pairs spread over the corpus, so
    throwing away history would leave most cells with nothing in them.
    """
    return (
        Window.partitionBy(*partition)
        .orderBy(f.col("feature_ts").cast("long"))
        .rangeBetween(Window.unboundedPreceding, 0)
    )


def user_category_hourly_features(events: DataFrame) -> DataFrame:
    """One row per (user, category, hour the user was active in it).

    Args:
        events: Silver rows carrying ``user_id``, ``category``, ``ts`` and
            ``clicked``.

    Returns:
        Cumulative impressions and clicks per (user, category), plus the
        smoothed affinity. Every column is prefixed ``user_cat_``: this frame
        lands on the same label rows as the item and user series, and an
        unprefixed ``impressions_cum`` would collide with neither today and
        both eventually.
    """
    hourly = (
        # The same close-stamped bucket as every other series. They must agree
        # or one of them leaks past the `<=` boundary in asof_join.
        events.withColumn("feature_ts", f.date_trunc("hour", "ts") + f.expr("INTERVAL 1 HOUR"))
        .groupBy("user_id", "category", "feature_ts")
        .agg(
            f.count("*").alias("_imps_1h"),
            f.sum(f.col("clicked").cast("int")).alias("_clicks_1h"),
        )
    )

    pair = _expanding("user_id", "category")
    by_user = _expanding("user_id")

    return (
        hourly.withColumn("user_cat_impressions_cum", f.sum("_imps_1h").over(pair))
        .withColumn("user_cat_clicks_cum", f.sum("_clicks_1h").over(pair))
        # The PRIOR is the same reader's overall rate, as of the same hour
        .withColumn("_user_imps_cum", f.sum("_imps_1h").over(by_user))
        .withColumn("_user_clicks_cum", f.sum("_clicks_1h").over(by_user))
        .withColumn(
            "user_cat_affinity",
            smoothed_ctr(
                f.col("user_cat_clicks_cum"),
                f.col("user_cat_impressions_cum"),
                f.col("_user_clicks_cum") / f.col("_user_imps_cum"),
                # Lower than the item series' 20. A (user, category) cell holds
                # tens of impressions where an item holds hundreds, so a
                # pseudo-count of 20 would pin almost every cell to the prior
                # and the feature would carry no signal at all.
                prior_strength=5.0,
            ),
        )
        .drop("_imps_1h", "_clicks_1h", "_user_imps_cum", "_user_clicks_cum")
    )
