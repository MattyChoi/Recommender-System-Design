"""Data contracts for the bronze layer.

Skipped when no bronze layer exists, so CI stays green without the dataset.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as f

BRONZE = Path("data/bronze/events/train")

pytestmark = pytest.mark.skipif(
    not BRONZE.exists(), reason="no bronze layer; run `make data` first"
)

# PySpark needs a JVM, which a bare CI runner does not have. Skip the whole
# module rather than fail, the same way the contract tests skip without data.
_HAS_JVM = bool(os.environ.get("JAVA_HOME")) or shutil.which("java") is not None
requires_jvm = pytest.mark.skipif(
    not _HAS_JVM, reason="no JVM on PATH; PySpark tests need Java 17 or 21"
)


@pytest.fixture(scope="session")
def spark() -> Iterator[object]:
    # session scope matters: a SparkSession costs seconds to build
    session = (
        SparkSession.builder.master("local[1]")
        .appName("mind-tests")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture(scope="session")
def events(spark: SparkSession) -> DataFrame:
    return spark.read.parquet(str(BRONZE)).cache()


def test_timestamps_all_parsed(events: DataFrame) -> None:
    """The single most likely ingest bug: a wrong format string nulls the column."""
    assert events.filter(f.col("ts").isNull()).count() == 0


def test_no_null_keys(events: DataFrame) -> None:
    for col in ("impression_id", "user_id", "item_id"):
        assert events.filter(f.col(col).isNull()).count() == 0, col


def test_slots_start_at_zero(events: DataFrame) -> None:
    assert events.agg(f.min("slot")).collect()[0][0] == 0


def test_no_future_timestamps(events: DataFrame) -> None:
    assert events.agg(f.max("ts")).collect()[0][0] <= datetime.now()


def test_no_duplicate_rows(events: DataFrame) -> None:
    keys = ["impression_id", "item_id"]
    assert events.select(keys).distinct().count() == events.count()


def test_ctr_is_plausible(events: DataFrame) -> None:
    """The highest-value test in the file.

    A label parse that silently inverts produces a pipeline that runs clean
    and models that are confidently wrong. Nothing else in the stack tells you.
    """
    ctr = events.agg(f.avg(f.col("clicked").cast("double"))).collect()[0][0]
    # MIND click-through sits in the low single digits. Near 0.5 means the
    # "-1"/"-0" parse inverted; near 0 or 1 means it collapsed.
    assert 0.01 < ctr < 0.15, f"implausible CTR {ctr}"


def test_every_impression_has_at_least_two_items(events: DataFrame) -> None:
    """A single-item impression cannot be ranked, and skews per-impression AUC."""
    sizes = events.groupBy("impression_id").count()
    assert sizes.filter(f.col("count") < 2).count() == 0
