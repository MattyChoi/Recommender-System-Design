"""Shared fixtures for data_pipeline tests.

The SparkSession is session-scoped because building one costs several seconds
and every test here needs the same configuration.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator

import pytest

# PySpark needs a JVM, which a bare CI runner does not have. Skip the whole
# module rather than fail, the same way the contract tests skip without data.
_HAS_JVM = bool(os.environ.get("JAVA_HOME")) or shutil.which("java") is not None
requires_jvm = pytest.mark.skipif(
    not _HAS_JVM, reason="no JVM on PATH; PySpark tests need Java 17 or 21"
)


@pytest.fixture(scope="session")
def spark() -> Iterator[object]:
    """A minimal local session.

    Deliberately built by hand rather than via get_spark(load_settings()), so
    these tests do not depend on conf/config.yml existing or being correct.

    local[1] removes concurrency, which makes failures reproducible; the UTC
    timezone is not optional -- MIND timestamps are naive local time, and the
    assertions below would shift with the machine's offset without it.
    """
    from pyspark.sql import SparkSession

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
