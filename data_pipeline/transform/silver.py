"""Bronze + metadata + integer indices -> silver."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as f

from common.config import Settings, load_settings
from common.spark import get_spark
from common.utils import SPLITS, _is_built
from data_pipeline.transform.sessionize import sessionize


def join_events_with_news(
    events: DataFrame,
    news: DataFrame,
    item_map: DataFrame,
    user_map: DataFrame,
    settings: Settings,
    split: str,
) -> None:
    """Join, sessionize, and write."""
    joined = (
        events.join(
            f.broadcast(news.select("item_id", "category", "subcategory", "title", "abstract")),
            on="item_id",
            how="left",
        )
        .join(f.broadcast(item_map), on="item_id", how="left")
        .join(user_map, on="user_id", how="left")  # ~1M rows: shuffle join
    )
    silver = (
        sessionize(joined, settings.session.gap_minutes)
        .withColumn("dt", f.to_date("ts"))
        .select(
            "impression_id",
            "user_id",
            "user_idx",
            "session_id",
            "item_id",
            "item_idx",
            "clicked",
            "slot",
            "ts",
            "dt",
            "category",
            "subcategory",
            "title",
        )
    )
    (
        silver.write.mode("overwrite")
        .partitionBy("dt")
        .parquet(str(settings.paths.silver / "impressions" / split))
    )


def build_silver(spark: SparkSession, settings: Settings, split: str) -> None:
    """Read one split's bronze tables and write its silver table."""
    events = spark.read.parquet(str(settings.paths.bronze / "events" / split))
    news = spark.read.parquet(str(settings.paths.bronze / "news" / split))
    item_map = spark.read.parquet(str(settings.paths.bronze / "item_map"))
    user_map = spark.read.parquet(str(settings.paths.bronze / "user_map"))

    join_events_with_news(events, news, item_map, user_map, settings, split)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Argument list, or None to read ``sys.argv``.

    Returns:
        A process exit code.
    """
    parser = argparse.ArgumentParser(description="Build the silver layer from bronze.")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=["train", "dev"])
    parser.add_argument(
        "--force", action="store_true", help="Rebuild splits that are already built."
    )
    args = parser.parse_args(argv)

    settings = load_settings()

    missing = [
        s for s in args.splits if not (settings.paths.bronze / "events" / s / "_SUCCESS").is_file()
    ]
    if missing:
        print(f"error: bronze not built for {missing}. Run `make bronze` first.")
        return 1

    todo = [s for s in args.splits if args.force or not _is_built(settings, "silver", s)]
    for skipped in [s for s in args.splits if s not in todo]:
        print(f"{skipped}: already built, skipping")

    if not todo:
        return 0

    spark = get_spark(settings, app="silver")
    try:
        for split in todo:
            dest = settings.paths.silver / "impressions" / split
            print(f"{split}: {settings.paths.bronze} -> {dest}")
            build_silver(spark, settings, split)
    finally:
        spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
