"""Fixtures shared by every test module in this package.

The JVM guard lives on the ``spark`` fixture rather than on a module-level
``pytestmark``: importing pyspark is pure Python and works fine on a bare
runner, so there is nothing to skip until a test actually asks for a session.
Skipping here cascades to every fixture built on top of it.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator

import pytest
from pyspark.sql import SparkSession

_HAS_JVM = bool(os.environ.get("JAVA_HOME")) or shutil.which("java") is not None


@pytest.fixture(scope="session")
def spark() -> Iterator[SparkSession]:
    """A local SparkSession reused across the whole test session.

    Session scope matters: building one costs seconds, and a per-test fixture
    turns a fast suite into a slow one.
    """
    if not _HAS_JVM:
        pytest.skip("no JVM on PATH; PySpark tests need Java 17 or 21")

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
