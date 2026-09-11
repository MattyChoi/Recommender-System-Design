"""Point-in-time correct feature attachment to prevent feature leakage.

Attaches features to a pySpark DataFrame of labels, ensuring that only
features observable at the time of the label are used.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as f

# Features the as-of join attaches the item and user features, category is separated
# to give every cold start item a prior
_ITEM_FEATURES = (
    "item_impressions_24h",
    "item_clicks_24h",
    "item_ctr_smoothed",
    "item_age_hours",
)
_USER_FEATURES = (
    "user_impressions_24h",
    "user_clicks_24h",
    "user_ctr_smoothed",
    "user_tenure_hours",
)
_CATEGORY_FEATURES = ("cat_expanding_ctr",)

# Counts, where zero is the honest answer for an item with no closed bucket
# before the label: nothing was knowable yet. The RATE is different and must
# never be zero-filled -- see build_training_examples.
_ZERO_FILLED = (
    "item_impressions_24h",
    "item_clicks_24h",
    "item_age_hours",
    "user_impressions_24h",
    "user_clicks_24h",
    "user_tenure_hours",
)


def asof_join(
    labels: DataFrame,
    features: DataFrame,
    join_key: str,
    label_ts: str = "ts",
    feat_ts: str = "feature_ts",
) -> DataFrame:
    """Attach the latest feature row OBSERVABLE at each label's timestamp.

    The mechanism: stack labels and features into one frame, order by time, and
    carry the last non-null feature value forward. Label rows are nulled in
    every feature column and ``last(ignorenulls=True)`` skips them, so a label
    can only ever RECEIVE a value, never supply one to a later label.

    Args:
        labels: Rows to enrich; must contain ``join_key`` and ``label_ts``, and
            must NOT already carry any of the feature columns.
        features: Time series from D1; must contain ``join_key`` and ``feat_ts``.
        join_key: Join key, e.g. "item_id" or "user_id". Run this once per key:
            each has its own independent timeline.

    Returns:
        ``labels`` with one column per feature, carrying the most recent value
        observable at that label's timestamp. Rows preceding the key's first
        feature bucket get NULL, which is the honest answer and must not be
        filled with zero -- see the caller's cold-start handling.

    Raises:
        ValueError: If a feature column already exists on ``labels``.
    """
    feat_cols = [c for c in features.columns if c not in {join_key, feat_ts}]

    # Prevent overwriting a label column with a feature column of the same name. Biggest
    # risk would be the category feature being nulled if it's a cold item
    clash = set(feat_cols) & set(labels.columns)
    if clash:
        raise ValueError(
            f"feature columns already on labels: {sorted(clash)}; "
            "rename or drop them before joining"
        )

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


def attach_point_in_time_features(
    labels: DataFrame, item_features: DataFrame, user_features: DataFrame
) -> DataFrame:
    """Attach features knowable at each label's timestamp.

    Cold items get nulls, and what happens next depends on the column:

      * The RATE is imputed from the category prior. Filling it with 0.0 would
        tell the ranker these articles have a MEASURED click rate of zero,
        which is a lie about evidence rather than a missing value.
      * The COUNTS are zero-filled, because zero is true: no bucket had closed,
        so nothing was knowable. Same for age, which is 0 at first sight.
      * ``has_item_features`` records which happened, so the model can learn
        that absence is itself a signal instead of inferring it from an
        imputed number.

    Args:
        labels: Silver rows, carrying ``item_id``, ``category``, ``ts``.
        item_features: ``item_hourly_features`` output.
        user_features: ``user_hourly_features`` output.

    Returns:
        ``labels`` with the item features, user features the category prior,
        the cold-start flag and the context columns. Rows for which nothing
        was knowable -- neither item nor category had a closed bucket -- are
        DROPPED, so the row count can be lower than the input; the joins themselves
        never multiply it.
    """
    category_features = item_features.select(
        "category", "feature_ts", *_CATEGORY_FEATURES
    ).distinct()
    item_features = item_features.select("item_id", "feature_ts", *_ITEM_FEATURES)
    user_features = user_features.select("user_id", "feature_ts", *_USER_FEATURES)

    examples = asof_join(labels, item_features, join_key="item_id")
    examples = asof_join(examples, user_features, join_key="user_id")
    examples = asof_join(examples, category_features, join_key="category")

    examples = (
        # Create flagsfor whether the item and user
        examples.withColumn("has_item_features", f.col("item_ctr_smoothed").isNotNull())
        .withColumn("has_user_features", f.col("user_ctr_smoothed").isNotNull())
        .withColumn(
            "item_ctr_smoothed",
            f.coalesce(f.col("item_ctr_smoothed"), f.col("cat_expanding_ctr")),
        )
        .withColumn("hour_of_day", f.hour("ts"))
        .withColumn("day_of_week", f.dayofweek("ts"))
    )
    for column in _ZERO_FILLED:
        examples = examples.withColumn(column, f.coalesce(f.col(column), f.lit(0)))

    # A null rate here means BOTH fallbacks were empty: no bucket had closed
    # for this item, and none had closed in its category either. Measured on
    # train, 6,257 of 6,262 such rows sit in the corpus's opening hour, before
    # the first bucket anywhere closes at 01:00.
    #
    # These are dropped rather than kept.
    # The count is logged at build time so the loss is never silent, and the
    # same warm-up applies to the streaming path when it starts cold in Part Q.
    return examples.filter(f.col("item_ctr_smoothed").isNotNull())
