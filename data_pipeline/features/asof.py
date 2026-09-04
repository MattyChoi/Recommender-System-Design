"""Point-in-time correct feature attachment to prevent feature leakage.

Attaches features to a pySpark DataFrame of labels, ensuring that only
features observable at the time of the label are used.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as f


def asof_join(
    labels: DataFrame,
    features: DataFrame,
    join_key: str,
    label_ts: str = "ts",
    feat_ts: str = "feature_ts",
) -> DataFrame:
    """Attach the latest feature row STRICTLY BEFORE each label's timestamp.

    Args:
        labels: Rows to enrich; must contain ``join_key`` and ``label_ts``.
        features: Time series from D1; must contain ``join_key`` and ``feat_ts``.
        join_key: Join key, e.g. "item_id" or "user_id".

    Returns:
        ``labels`` with one column per feature, carrying the most recent value
        observable at that label's timestamp.
    """
    feat_cols = [c for c in features.columns if c not in {join_key, feat_ts}]

    lab = labels.withColumn("_ts", f.col(label_ts)).withColumn("_is_label", f.lit(1))
    feat = features.withColumn("_ts", f.col(feat_ts)).withColumn("_is_label", f.lit(0))

    # Null out the feature columns on the label side so unionByName aligns.
    for c in feat_cols:
        lab = lab.withColumn(c, f.lit(None).cast(features.schema[c].dataType))

    unioned = lab.unionByName(feat, allowMissingColumns=True)

    # must set _is_label as secondary sort key to ensure that the feature rows are
    # always before the label rows at the same timestamp
    w = (
        Window.partitionBy(join_key)
        .orderBy(f.col("_ts").asc(), f.col("_is_label").asc())
        .rowsBetween(Window.unboundedPreceding, 0)
    )

    # Setting ignorenulls=True ensures that the last() function will skip the
    # label rows and only return the last feature row before the label's timestamp.
    filled = unioned.select(
        *[c for c in unioned.columns if c not in feat_cols],
        *[f.last(c, ignorenulls=True).over(w).alias(c) for c in feat_cols],
    )

    return filled.filter(f.col("_is_label") == 1).drop("_ts", "_is_label", feat_ts)
