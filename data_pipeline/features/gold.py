"""Silver -> gold: the point-in-time feature tables the feature store reads.

Gold differs from bronze and silver in one important way: it is NOT built per
split. Features are a property of the timeline, not of a train/dev label, so
this job unions every split and computes one continuous series.

That is safe in both directions:

  * A train label cannot see a dev-derived feature row. MIND's dev week
    follows the train week, and the as-of join only ever reads backwards.
  * A dev label CAN see train-derived rows, which is exactly right -- the
    production system serving that impression had those statistics.

Building the series from train alone would instead leave every dev-only item
with null features, and a third of dev's catalogue never appears in train.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from pyspark.sql import SparkSession

from common.config import Settings, load_settings
from common.spark import get_spark
from common.utils import SPLITS, _is_built, read_news, read_silver
from data_pipeline.features.ctr_smoothed import smoothed_ctr_by_category
from data_pipeline.features.item_dynamic_features import item_hourly_features


def build_item_hourly(spark: SparkSession, settings: Settings, splits: Sequence[str]) -> None:
    """Write the hourly item feature series to ``paths.gold/item_hourly_features``.

    Deliberately unpartitioned. The table is small -- one row per (item, hour)
    the item was actually shown in, not per (item, hour) in the corpus -- and
    Feast's file offline store reads a flat Parquet directory most
    predictably. A ``dt=`` partition column would also appear on read as a
    column that no FeatureView declares, which is a needless divergence
    between what is on disk and what is registered.
    """
    events = read_silver(spark, settings, splits).select("item_id", "ts", "clicked")
    news = read_news(spark, settings, splits)
    features = smoothed_ctr_by_category(item_hourly_features(events), news)
    features.write.mode("overwrite").parquet(str(settings.paths.gold / "item_hourly_features"))


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Argument list, or None to read ``sys.argv``.

    Returns:
        A process exit code.
    """
    parser = argparse.ArgumentParser(description="Build the gold layer from silver.")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=["train", "dev"],
        help="Splits to fold into the feature timeline. Use every split you "
        "will ever score against: a split left out here gets null features.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Rebuild tables that are already built."
    )
    args = parser.parse_args(argv)

    settings = load_settings()

    missing = [s for s in args.splits if not _is_built(settings, "silver", s)]
    if missing:
        print(f"error: silver not built for {missing}. Run `make silver` first.")
        return 1

    if _is_built(settings, "gold", "item_hourly_features") and not args.force:
        print("item_hourly_features: already built, skipping")
        return 0

    spark = get_spark(settings, app="gold")
    try:
        print(
            f"item_hourly_features: {settings.paths.silver} -> \
            {settings.paths.gold / 'item_hourly_features'}"
        )
        build_item_hourly(spark, settings, args.splits)
    finally:
        spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
