"""Version pins and S3A options, checked without a JVM or a running MinIO.

The pins are the point. They resolve cleanly from Maven whatever they say, so a
stale pin is invisible until a write fails deep inside the S3A client with a
NoSuchMethodError -- an hour of debugging MinIO for a dependency problem.
"""

from __future__ import annotations

import pyspark
import pytest

from common.config import Settings, load_settings
from common.spark import (
    HADOOP_AWS_VERSION,
    SCALA_BINARY_VERSION,
    _packages,
    _s3a_options,
    bundled_jar_version,
)
from tests.test_config import REPO_ROOT


@pytest.fixture
def s3_settings() -> Settings:
    settings = load_settings(REPO_ROOT).model_copy(deep=True)
    settings.storage.backend = "s3"
    return settings


def test_hadoop_aws_pin_matches_the_bundled_hadoop() -> None:
    """The pin that goes stale on a pyspark bump, with no other signal."""
    assert bundled_jar_version("hadoop-client-api") == HADOOP_AWS_VERSION


def test_scala_binary_version_matches_the_bundled_scala() -> None:
    assert bundled_jar_version("scala-library").startswith(SCALA_BINARY_VERSION + ".")


def test_hadoop_cloud_is_not_bundled_and_so_must_be_resolved() -> None:
    """The reason spark-hadoop-cloud is in the package list at all.

    If a future pyspark ships it, resolving it again is harmless but the
    coordinate is then dead weight -- this test is where that gets noticed.
    """
    with pytest.raises(FileNotFoundError):
        bundled_jar_version("spark-hadoop-cloud_" + SCALA_BINARY_VERSION)

    assert f"spark-hadoop-cloud_{SCALA_BINARY_VERSION}:{pyspark.__version__}" in _packages()


def test_missing_credentials_name_the_variable(
    s3_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MINIO_ROOT_USER", raising=False)
    monkeypatch.setenv("MINIO_ROOT_PASSWORD", "secret")

    with pytest.raises(RuntimeError, match="MINIO_ROOT_USER"):
        _s3a_options(s3_settings)


def test_the_three_options_minio_cannot_do_without(
    s3_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MINIO_ROOT_USER", "minioadmin")
    monkeypatch.setenv("MINIO_ROOT_PASSWORD", "minioadmin")

    opts = _s3a_options(s3_settings)

    # No DNS for virtual-hosted buckets.
    assert opts["spark.hadoop.fs.s3a.path.style.access"] == "true"
    # The endpoint is plaintext http.
    assert opts["spark.hadoop.fs.s3a.connection.ssl.enabled"] == "false"
    assert opts["spark.hadoop.fs.s3a.endpoint"] == s3_settings.storage.endpoint_url


def test_the_magic_committer_is_bound_on_both_sides(
    s3_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Naming the committer is not enough; Spark's own protocol must bind to it."""
    monkeypatch.setenv("MINIO_ROOT_USER", "minioadmin")
    monkeypatch.setenv("MINIO_ROOT_PASSWORD", "minioadmin")

    opts = _s3a_options(s3_settings)

    assert opts["spark.hadoop.fs.s3a.committer.name"] == "magic"
    assert opts["spark.sql.sources.commitProtocolClass"].endswith("PathOutputCommitProtocol")
    assert opts["spark.sql.parquet.output.committer.class"].endswith(
        "BindingParquetOutputCommitter"
    )
