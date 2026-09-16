"""The non-clicked items of each slate, as hard negatives."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as f

# Stored per impression, not the number used per training row. The loader takes
# a prefix of this, which is a stable sample precisely because the stored order
# is deterministic -- so sweeping the training-time count costs nothing, where
# baking it in here would mean a gold rebuild per value.
MAX_NEGATIVES = 20


def slate_negatives(impressions: DataFrame, max_negs: int = MAX_NEGATIVES) -> DataFrame:
    """Per impression, the items shown and not clicked, in a deterministic order.

    Args:
        impressions: Silver impressions, with ``impression_id``, ``item_idx``
            and ``clicked``.
        max_negs: Cap per impression.

    Returns:
        One row per impression that has at least one non-clicked item, with
        ``neg_idx`` (array) and ``neg_len``. Impressions whose every item was
        clicked produce NO row -- they genuinely have no slate negative to
        offer, and the loader's left join turns that into an all-padding row
        rather than a lie.
    """
    # The struct's FIRST field is the sort key, because sort_array on structs
    # orders field by field. item_idx second is the tie-break, which only fires
    # on a hash collision.
    keyed = f.struct(
        f.xxhash64("impression_id", "item_idx").alias("_key"),
        f.col("item_idx").alias("item_idx"),
    )

    return (
        impressions.where(~f.col("clicked"))
        .groupBy("impression_id")
        .agg(f.sort_array(f.collect_list(keyed)).alias("_ordered"))
        .withColumn("neg_idx", f.slice(f.col("_ordered.item_idx"), 1, max_negs))
        .withColumn("neg_len", f.size("neg_idx"))
        .drop("_ordered")
    )
