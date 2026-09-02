"""Stable string -> contiguous-int mappings, written once and read forever."""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as f

from common.config import Settings


def build_id_maps(events: DataFrame, news: DataFrame, settings: Settings) -> None:
    """Write user_map and item_map to bronze.

    Note:
        Do NOT use monotonically_increasing_id(). It is unique but not contiguous
        -- it encodes the partition number in the high bits, blowing up the range
        of IDs and making them non-contiguous.

        row_number() over an unpartitioned window collapses to a single
        partition. That is acceptable at ~1M users and ~160K items, and would
        not be at 100x that; the replacement there is RDD.zipWithIndex().
    """
    user_map = (
        events.select("user_id")
        .distinct()
        .withColumn("user_idx", f.row_number().over(Window.orderBy("user_id")) - 1)
    )
    item_map = (
        news.select("item_id")
        .distinct()
        .withColumn("item_idx", f.row_number().over(Window.orderBy("item_id")) - 1)
    )

    user_map.write.mode("overwrite").parquet(str(settings.paths.bronze / "user_map"))
    item_map.write.mode("overwrite").parquet(str(settings.paths.bronze / "item_map"))
