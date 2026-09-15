"""Bayesian shrinkage toward a point-in-time prior."""

from __future__ import annotations

from collections.abc import Sequence

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as f
from pyspark.sql.window import WindowSpec

# Below this many impressions a category has not earned its own prior, and
# shrinking toward it does more harm than not shrinking at all. Measured: at
# the first hour of MIND's train week, `kids` has impressions but zero clicks,
# so an unguarded category prior is exactly 0.0 -- which drags every kids
# article's smoothed CTR toward zero on the strength of no evidence.
_MIN_PRIOR_IMPRESSIONS = 1_000

_ROLLING_SECONDS = 24 * 3600


def _frame(partition: Sequence[str], lookback_seconds: int | None = None) -> WindowSpec:
    """A time-ordered frame over ``feature_ts``, expanding or rolling.

    24 hr rolling frame: .rangeBetween(-24*3600, 0)
    Expanding frame: no lower bound, .rangeBetween(Window.unboundedPreceding, 0)
    """
    order = f.col("feature_ts").cast("long")
    ordered = Window.partitionBy(*partition).orderBy(order) if partition else Window.orderBy(order)
    lower = Window.unboundedPreceding if lookback_seconds is None else -lookback_seconds
    return ordered.rangeBetween(lower, 0)


def smoothed_ctr(
    clicks: Column, impressions: Column, prior_ctr: Column, prior_strength: float = 20.0
) -> Column:
    """Shrink an empirical rate toward a prior.

    prior_strength is a pseudo-count: the number of prior "impressions" the
    prior is worth. Tune it on validation -- it matters more than people
    expect for cold and long-tail items, which on this corpus is most of them.
    """
    return (clicks + prior_ctr * prior_strength) / (impressions + prior_strength)


def category_prior(
    item_hourly: DataFrame, min_impressions: int = _MIN_PRIOR_IMPRESSIONS
) -> DataFrame:
    """The category CTR as it was knowable at each hour. Caclulated to be
    point-in-time by using a PySparks window in the ``_frame`` function.

    Args:
        item_hourly: Output of ``item_hourly_features``, carrying ``category``,
            ``feature_ts``, ``item_impressions_1h`` and ``item_clicks_1h``.
        min_impressions: Below this cumulative count the category falls back to
            the global prior, computed the same way.

    Both expanding and rolling 24 hour ctr windows are computed:

    ``cat_expanding_ctr`` is expanding -- every bucket at or before this one -- and is the
    one that becomes a feature.
    ``cat_rolling_ctr`` is the rolling 24-hour equivalent, carried for comparison and
    kept OUT of the feature table, so the choice between them stays measurable on real data
    rather than re-argued from first principles.

    Measured on MIND's train week, the two priors differ by 0.0070 absolute CTR on
    average (~20% relative), the rolling one is 1.7x noisier hour to hour, and
    category CTR drifts only +0.8% from the first 36 hours to the last. Rolling's
    whole advantage is tracking regime change; which on our stationary corpus it
    buys nothing and costs noise, on exactly the thin-sample population. Revisit
    if the corpus grows long enough for early data to stop resembling late data,
    or if the prior ever has to be computed in the streaming path -- expanding needs
    unbounded state there, rolling is a bounded window.

    Returns:
        One row per ``(category, feature_ts)`` with ``cat_expanding_ctr``
        (expanding) and ``cat_rolling_ctr`` (rolling).
    """
    per_hour = item_hourly.groupBy("category", "feature_ts").agg(
        f.sum("item_impressions_1h").alias("_imps"),
        f.sum("item_clicks_1h").alias("_clicks"),
    )

    expanding, rolling = _frame(["category"]), _frame(["category"], _ROLLING_SECONDS)
    by_category = (
        per_hour.withColumn("_cum_imps", f.sum("_imps").over(expanding))
        .withColumn("_cum_clicks", f.sum("_clicks").over(expanding))
        .withColumn("_roll_imps", f.sum("_imps").over(rolling))
        .withColumn("_roll_clicks", f.sum("_clicks").over(rolling))
    )

    all_expanding, all_rolling = _frame([]), _frame([], _ROLLING_SECONDS)
    overall = (
        per_hour.groupBy("feature_ts")
        # aggregate 1hr clicks and impressions over all categories to get global numbers
        .agg(f.sum("_imps").alias("_all_imps"), f.sum("_clicks").alias("_all_clicks"))
        .withColumn("_all_cum_imps", f.sum("_all_imps").over(all_expanding))
        .withColumn("_all_cum_clicks", f.sum("_all_clicks").over(all_expanding))
        .withColumn("_all_roll_imps", f.sum("_all_imps").over(all_rolling))
        .withColumn("_all_roll_clicks", f.sum("_all_clicks").over(all_rolling))
        .drop("_all_imps", "_all_clicks")
    )

    # Calculate the category CTR, falling back to global CTR if the category has too few
    # impressions.
    def _rate(imps: str, clicks: str, fallback_imps: str, fallback_clicks: str) -> Column:
        return f.when(f.col(imps) >= min_impressions, f.col(clicks) / f.col(imps)).otherwise(
            f.col(fallback_clicks) / f.col(fallback_imps)
        )

    return (
        by_category.join(f.broadcast(overall), on="feature_ts", how="left")
        .withColumn(
            "cat_expanding_ctr",
            _rate("_cum_imps", "_cum_clicks", "_all_cum_imps", "_all_cum_clicks"),
        )
        .withColumn(
            "cat_rolling_ctr",
            _rate("_roll_imps", "_roll_clicks", "_all_roll_imps", "_all_roll_clicks"),
        )
        .select("category", "feature_ts", "cat_expanding_ctr", "cat_rolling_ctr")
    )
