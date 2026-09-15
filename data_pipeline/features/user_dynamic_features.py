"""Hourly user aggregates of CTR, impression count, age in hours"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as f

from data_pipeline.features.ctr import smoothed_ctr


def user_hourly_features(events: DataFrame) -> DataFrame:
    """One row per (user, hour), emitted as a TIME SERIES.

    A table of "current activity" cannot be joined as-of anything: the history
    needed to answer what a user looked like last Tuesday is already gone.

    Args:
        events: Silver rows carrying ``user_id``, ``ts`` and ``clicked``.

    Returns:
        One row per (user, hour the user was active), with rolling 24h counts
        and tenure. Hours in which a user did nothing emit no row, which is
        why the frame below is RANGE.
    """
    hourly = (
        events.withColumn("feature_ts", f.date_trunc("hour", "ts") + f.expr("INTERVAL 1 HOUR"))
        .groupBy("user_id", "feature_ts")
        .agg(
            f.count("*").alias("user_impressions_1h"),
            f.sum(f.col("clicked").cast("int")).alias("user_clicks_1h"),
        )
    )

    w24 = (
        Window.partitionBy("user_id")
        .orderBy(f.col("feature_ts").cast("long"))
        .rangeBetween(-24 * 3600, 0)
    )
    by_user = Window.partitionBy("user_id")

    return (
        hourly.withColumn("user_impressions_24h", f.sum("user_impressions_1h").over(w24))
        .withColumn("user_clicks_24h", f.sum("user_clicks_1h").over(w24))
        # Safe despite scanning the whole partition: a MINIMUM over all time
        # equals the minimum over the past. max or avg here would leak badly.
        .withColumn("user_first_seen", f.min("feature_ts").over(by_user))
        .withColumn(
            "user_tenure_hours",
            (f.col("feature_ts").cast("long") - f.col("user_first_seen").cast("long")) / 3600.0,
        )
        .drop("user_first_seen")
    )


def smoothed_user_ctr(user_hourly: DataFrame, prior_strength: float = 20.0) -> DataFrame:
    """Shrink each user's 24h click rate toward the global rate, as of that hour.

    Toward the GLOBAL rate rather than a group rate, because MIND ships no
    demographics -- there is no user segment to regress a quiet reader toward.
    An item at least has a category.
    """
    expanding = Window.orderBy(f.col("feature_ts").cast("long")).rangeBetween(
        Window.unboundedPreceding, 0
    )
    global_series = (
        user_hourly.groupBy("feature_ts")
        .agg(
            f.sum("user_impressions_1h").alias("_all_imps"),
            f.sum("user_clicks_1h").alias("_all_clicks"),
        )
        .withColumn("_cum_imps", f.sum("_all_imps").over(expanding))
        .withColumn("_cum_clicks", f.sum("_all_clicks").over(expanding))
        .withColumn("global_expanding_ctr", f.col("_cum_clicks") / f.col("_cum_imps"))
        .select("feature_ts", "global_expanding_ctr")
    )

    return (
        user_hourly.join(f.broadcast(global_series), on="feature_ts", how="left")
        .withColumn(
            "user_ctr_smoothed",
            smoothed_ctr(
                f.col("user_clicks_24h"),
                f.col("user_impressions_24h"),
                f.col("global_expanding_ctr"),
                prior_strength,
            ),
        )
        .drop("global_expanding_ctr")
    )
