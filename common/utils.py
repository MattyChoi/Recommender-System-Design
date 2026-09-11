from __future__ import annotations

from collections.abc import Sequence
from functools import reduce

from pyspark.sql import DataFrame, SparkSession

from common.config import Settings

SPLITS = ("train", "dev", "test")
BRONZE_TABLES = ("events", "history", "news")
GOLD_TABLES = (
    "item_hourly_features",
    "user_hourly_features",
    "user_category_cross_features",
    "training_examples",
)
MIND_TS_FORMAT = "M/d/yyyy h:mm:ss a"


def _is_built(settings: Settings, layer: str, name: str) -> bool:
    """Whether the given data layer holds a committed write for ``name``.

    Spark drops ``_SUCCESS`` into an output directory only after the write job
    commits, so a run that died midway leaves part-files behind but no marker
    and is correctly reported as unbuilt.

    Args:
        settings: The root configuration object.
        layer: One of ``bronze``, ``silver`` or ``gold``.
        name: A split for bronze and silver, which are built per split; a
            TABLE for gold, which is not. Gold features are computed over the
            whole timeline at once -- see ``data_pipeline.transform.gold``.
    """
    if layer not in ("bronze", "silver", "gold"):
        raise ValueError(f"unknown layer {layer!r}")

    if layer == "bronze":
        return all(
            (settings.paths.bronze / table / name / "_SUCCESS").is_file() for table in BRONZE_TABLES
        )
    elif layer == "silver":
        return (settings.paths.silver / "impressions" / name / "_SUCCESS").is_file()
    elif layer == "gold":
        return (settings.paths.gold / name / "_SUCCESS").is_file()
    return False


def read_news(spark: SparkSession, settings: Settings, splits: Sequence[str]) -> DataFrame:
    """Union every split's article metadata into one catalogue.

    Items shown in both splits appear in both files. Deduplicating on
    ``item_id`` is safe because the rows agree: of the 28,460 items present in
    both train and dev, none differ in any metadata column.
    """
    frames = [spark.read.parquet(str(settings.paths.bronze / "news" / split)) for split in splits]
    return reduce(lambda a, b: a.unionByName(b), frames).dropDuplicates(["item_id"])


def read_silver(spark: SparkSession, settings: Settings, splits: Sequence[str]) -> DataFrame:
    """Union every split's silver impressions into one timeline"""
    frames = [
        spark.read.parquet(str(settings.paths.silver / "impressions" / split)) for split in splits
    ]
    return reduce(lambda a, b: a.unionByName(b), frames)


def read_gold(spark: SparkSession, settings: Settings, name: str) -> DataFrame:
    """Read a gold table by name."""
    return spark.read.parquet(str(settings.paths.gold / name))
