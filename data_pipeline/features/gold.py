"""Silver -> gold: the point-in-time feature tables the feature store reads.

The TRAINING EXAMPLES built here are per split, because a label belongs to
exactly one side of the split protocol even though the features it reads do
not. So gold holds two shapes: one global feature series, and one label table
per split pointing into it.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from pyspark.sql import SparkSession
from pyspark.sql import functions as f

from common.config import Settings, load_settings
from common.spark import get_spark
from common.utils import SPLITS, _is_built, read_gold, read_silver
from data_pipeline.features.asof import attach_point_in_time_features
from data_pipeline.features.item_dynamic_features import (
    item_hourly_features,
    smoothed_ctr_by_category,
)
from data_pipeline.features.user_dynamic_features import smoothed_user_ctr, user_hourly_features


def build_item_hourly(spark: SparkSession, settings: Settings, splits: Sequence[str]) -> None:
    """Write the hourly item feature series to ``paths.gold/item_hourly_features``.

    Deliberately unpartitioned. The table is small -- one row per (item, hour)
    the item was actually shown in, not per (item, hour) in the corpus -- and
    Feast's file offline store reads a flat Parquet directory most
    predictably. A ``dt=`` partition column would also appear on read as a
    column that no FeatureView declares, which is a needless divergence
    between what is on disk and what is registered.
    """
    events = read_silver(spark, settings, splits).select("item_id", "ts", "clicked", "category")
    features = smoothed_ctr_by_category(item_hourly_features(events))
    features.write.mode("overwrite").parquet(str(settings.paths.gold / "item_hourly_features"))


def build_user_hourly(spark: SparkSession, settings: Settings, splits: Sequence[str]) -> None:
    """Write the hourly user feature series to ``paths.gold/user_hourly_features``.

    Unioned across splits for the same reason the item series is: features
    belong to the timeline, not to a label. A dev impression by a user who read
    during the train week should see that history, because the production
    system serving it would have.
    """
    events = read_silver(spark, settings, splits).select("user_id", "ts", "clicked")
    features = smoothed_user_ctr(user_hourly_features(events))
    features.write.mode("overwrite").parquet(str(settings.paths.gold / "user_hourly_features"))


def build_training_examples(spark: SparkSession, settings: Settings, split: str) -> None:
    """Read one split's labels and the global series, join, write."""
    labels = spark.read.parquet(str(settings.paths.silver / "impressions" / split))
    item_features = read_gold(spark, settings, "item_hourly_features")
    user_features = read_gold(spark, settings, "user_hourly_features")

    (
        attach_point_in_time_features(labels, item_features, user_features)
        .write.mode("overwrite")
        .partitionBy("dt")
        .parquet(str(settings.paths.gold / "training_examples" / split))
    )


def _example_counts(spark: SparkSession, settings: Settings, split: str) -> dict[str, float]:
    """Row counts, warm-up loss and both cold-start shares, for the build log."""
    labels = spark.read.parquet(str(settings.paths.silver / "impressions" / split))
    written = spark.read.parquet(str(settings.paths.gold / "training_examples" / split))
    rows = written.count()
    shares = written.agg(
        f.avg(f.col("has_item_features").cast("double")).alias("item"),
        f.avg(f.col("has_user_features").cast("double")).alias("user"),
    ).collect()[0]
    return {
        "rows": rows,
        "dropped": labels.count() - rows,
        "with_item": shares["item"],
        "with_user": shares["user"],
    }


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

    series_built = _is_built(settings, "gold", "user_hourly_features")
    todo = [
        s
        for s in args.splits
        if args.force or not _is_built(settings, "gold", f"training_examples/{s}")
    ]
    if series_built and not todo and not args.force:
        print("gold: already built, skipping")
        return 0

    spark = get_spark(settings, app="gold")
    try:
        print(f"item_hourly_features: {settings.paths.silver} -> {settings.paths.gold}")
        build_item_hourly(spark, settings, args.splits)
        print(f"user_hourly_features: {settings.paths.silver} -> {settings.paths.gold}")
        build_user_hourly(spark, settings, args.splits)

        for split in todo:
            dest = settings.paths.gold / "training_examples" / split
            print(f"{split}: labels + point-in-time features -> {dest}")
            build_training_examples(spark, settings, split)
            counts = _example_counts(spark, settings, split)
            print(
                f"  {counts['rows']:,} examples, "
                f"{counts['with_item']:.1%} with item history, "
                f"{counts['with_user']:.1%} with user history, "
                f"{counts['dropped']:,} dropped as unknowable"
            )
    finally:
        spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
