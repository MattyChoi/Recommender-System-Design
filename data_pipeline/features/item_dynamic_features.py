"""Hourly item aggregates of CTR, impression count, age in hours"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as f


def item_hourly_features(events: DataFrame) -> DataFrame:
    """One row per (item, hour) with cumulative counts up to that hour."""

    # For each item and hour, aggregate the number of impressions and clicks in that hour.
    hourly = (
        events.withColumn("feature_ts", f.date_trunc("hour", "ts") + f.expr("INTERVAL 1 HOUR"))
        .groupBy("item_id", "feature_ts")
        .agg(
            f.count("*").alias("impressions_1h"),
            f.sum(f.col("clicked").cast("int")).alias("clicks_1h"),
        )
    )

    w24 = (
        Window.partitionBy("item_id")
        .orderBy(f.col("feature_ts").cast("long"))
        .rangeBetween(-24 * 3600, 0)
    )
    first_seen = Window.partitionBy("item_id")

    return (
        hourly.withColumn("impressions_24h", f.sum("impressions_1h").over(w24))
        .withColumn("clicks_24h", f.sum("clicks_1h").over(w24))
        .withColumn("first_seen", f.min("feature_ts").over(first_seen))
        .withColumn(
            "age_hours",
            (f.col("feature_ts").cast("long") - f.col("first_seen").cast("long")) / 3600.0,
        )
    )
