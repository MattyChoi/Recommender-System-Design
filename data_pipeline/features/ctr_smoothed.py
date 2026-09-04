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


def smoothed_ctr_by_category(
    item_hourly: DataFrame, news: DataFrame, prior_strength: float = 20.0
) -> DataFrame:
    """Use the CATEGORY mean as the prior, not the global mean. when calculating smoothed CTR."""
    cat = (
        item_hourly.join(f.broadcast(news.select("item_id", "category")), on="item_id", how="left")
        .groupBy("category")
        .agg((f.sum("clicks_24h") / f.sum("impressions_24h")).alias("cat_ctr"))
    )

    return (
        item_hourly.join(f.broadcast(news.select("item_id", "category")), on="item_id", how="left")
        .join(f.broadcast(cat), on="category", how="left")
        .withColumn(
            "ctr_24h_smoothed",
            smoothed_ctr(
                f.col("clicks_24h"), f.col("impressions_24h"), f.col("cat_ctr"), prior_strength
            ),
        )
    )
