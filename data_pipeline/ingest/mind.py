"""Build the bronze layer from the raw MIND TSVs.

Reads ``paths.raw/<split>/`` as downloaded by
:mod:`data_pipeline.ingest.download` and writes typed Parquet under
``paths.bronze``: ``events/`` (one row per item shown, partitioned by date),
``history/`` and ``news/``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as f

from common.config import Settings, load_settings
from common.schemas import BEHAVIORS_RAW, NEWS_RAW
from common.spark import get_spark

SPLITS = ("train", "dev", "test")
BRONZE_TABLES = ("events", "history", "news")


def _is_built(settings: Settings, split: str) -> bool:
    """Whether every bronze table for ``split`` holds a committed write.

    Spark drops ``_SUCCESS`` into an output directory only after the write job
    commits, so a run that died midway leaves part-files behind but no marker
    and is correctly reported as unbuilt.
    """
    return all(
        (settings.paths.bronze / table / split / "_SUCCESS").is_file() for table in BRONZE_TABLES
    )


MIND_TS_FORMAT = "M/d/yyyy h:mm:ss a"


def read_behaviors(spark: SparkSession, settings: Settings, split: str) -> DataFrame:
    return (
        spark.read.option("sep", "\t")
        .option("header", "false")
        .schema(BEHAVIORS_RAW)  # declared, never inferred
        .csv(str(settings.paths.raw / split / "behaviors.tsv"))
    )


def read_news(spark: SparkSession, settings: Settings, split: str) -> DataFrame:
    return (
        spark.read.option("sep", "\t")
        .option("header", "false")
        .schema(NEWS_RAW)
        .csv(str(settings.paths.raw / split / "news.tsv"))
    )


def to_events(behaviors: DataFrame) -> DataFrame:
    """One row per (impression, item shown).

    posexplode gives the element AND its index in one pass. A plain explode
    would discard the ordering; we keep it as `slot` purely so the shuffling
    is documented in the data rather than assumed away.
    """
    return (
        behaviors.withColumn("ts", f.to_timestamp("time", MIND_TS_FORMAT))
        .select(
            "impression_id",
            "user_id",
            "ts",
            f.posexplode(f.split(f.col("impressions"), " ")).alias("slot", "shown"),
        )
        .withColumn("item_id", f.split(f.col("shown"), "-").getItem(0))
        .withColumn("clicked", f.split(f.col("shown"), "-").getItem(1) == f.lit("1"))
        .drop("shown")
    )


def ingest(spark: SparkSession, settings: Settings, split: str, *, force: bool = False) -> None:
    """Write one split's raw TSVs to bronze Parquet.

    Args:
        spark: An active session.
        settings: The root configuration object.
        split: ``train``, ``dev`` or ``test``.
        force: Rebuild even when the split's bronze tables are already present.
    """
    if _is_built(settings, split) and not force:
        print(f"{split}: bronze already present, skipping")
        return

    behaviors = read_behaviors(spark, settings, split)

    events = to_events(behaviors).withColumn("dt", f.to_date("ts"))
    (
        events.write.mode("overwrite")
        .partitionBy("dt")
        .parquet(str(settings.paths.bronze / "events" / split))
    )

    # `history` is your sequence-model input, but exploding it alongside the
    # impressions multiplies row count catastrophically. Keep it in its own
    # table, still space-separated, keyed by impression.
    (
        behaviors.select("impression_id", "user_id", "history")
        .write.mode("overwrite")
        .parquet(str(settings.paths.bronze / "history" / split))
    )

    news = read_news(spark, settings, split)
    news.write.mode("overwrite").parquet(str(settings.paths.bronze / "news" / split))


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Argument list, or None to read ``sys.argv``.

    Returns:
        A process exit code; 1 if a split's raw inputs are missing.
    """
    parser = argparse.ArgumentParser(description="Build the bronze layer from raw MIND TSVs.")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=["train", "dev"],
        help="Splits to ingest. Each must already exist under paths.raw.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild a split even when its bronze tables are already present.",
    )
    args = parser.parse_args(argv)

    settings = load_settings()

    missing = [s for s in args.splits if not (settings.paths.raw / s / "behaviors.tsv").is_file()]
    if missing:
        print(f"error: no raw data for {missing} under {settings.paths.raw}; run `make download`")
        return 1

    build_split = [s for s in args.splits if args.force or not _is_built(settings, s)]
    for split in args.splits:
        if split not in build_split:
            print(f"{split}: bronze already present, skipping")
    # Starting a session costs seconds of JVM boot, so decide before paying for it.
    if not build_split:
        return 0

    spark = get_spark(settings, app="mind-ingest")
    try:
        for split in build_split:
            print(f"{split}: {settings.paths.raw / split} -> {settings.paths.bronze}")
            ingest(spark, settings, split, force=args.force)
    finally:
        spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
