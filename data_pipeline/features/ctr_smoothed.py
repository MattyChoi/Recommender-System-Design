"""Bayesian shrinkage toward a category prior."""

from __future__ import annotations

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as f


def smoothed_ctr(
    clicks: Column, impressions: Column, prior_ctr: Column, prior_strength: float = 20.0
) -> Column:
    """Shrink an empirical rate toward a prior.

    prior_strength is a pseudo-count: the number of prior "impressions" the
    prior is worth. Tune it on validation -- it matters more than people
    expect for cold and long-tail items, which on this corpus is most of them.
    """
    return (clicks + prior_ctr * prior_strength) / (impressions + prior_strength)


def smoothed_ctr_by_category(item_hourly: DataFrame, prior_strength: float = 20.0) -> DataFrame:
    """Shrink each item's 24h CTR toward its CATEGORY mean, not the global mean.

    Args:
        item_hourly: Hourly item aggregates carrying ``clicks_24h``,
            ``impressions_24h`` and ``category``.
        prior_strength: Pseudo-count for the prior, in impressions.

    Returns:
        ``item_hourly`` with ``cat_ctr`` and ``ctr_24h_smoothed`` attached.

    Note:
        ``cat_ctr`` is aggregated over the WHOLE timeline, not as of each
        hour, so the prior is not strictly point-in-time.
    """
    cat = item_hourly.groupBy("category").agg(
        (f.sum("clicks_24h") / f.sum("impressions_24h")).alias("cat_ctr")
    )

    return item_hourly.join(f.broadcast(cat), on="category", how="left").withColumn(
        "ctr_24h_smoothed",
        smoothed_ctr(
            f.col("clicks_24h"), f.col("impressions_24h"), f.col("cat_ctr"), prior_strength
        ),
    )
