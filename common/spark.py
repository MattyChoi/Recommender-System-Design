"""The one SparkSession factory, so tuning lives in config rather than scripts."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import pyspark
from pyspark.sql import SparkSession

from common.config import Settings

# Must equal the hadoop-client-api version bundled with pyspark. S3A and the
# rest of hadoop-client-api share internal interfaces that are not stable
# across minor versions, so a mismatch resolves cleanly from Maven and then
# fails at write time with a NoSuchMethodError from inside the S3A client --
# which reads as a MinIO problem rather than as a dependency one.
HADOOP_AWS_VERSION = "3.5.0"

# spark-hadoop-cloud is NOT in the pyspark wheel; it carries the committer
# classes the magic committer binds to, so it has to be resolved alongside
# hadoop-aws or fs.s3a.committer.name=magic fails with ClassNotFoundException.
SCALA_BINARY_VERSION = "2.13"


def _packages() -> str:
    """Maven coordinates resolved at JVM start, as a comma-separated list."""
    return ",".join(
        (
            f"org.apache.hadoop:hadoop-aws:{HADOOP_AWS_VERSION}",
            f"org.apache.spark:spark-hadoop-cloud_{SCALA_BINARY_VERSION}:{pyspark.__version__}",
        )
    )


def _s3a_options(settings: Settings) -> dict[str, str]:
    """Hadoop S3A configuration pointing at the MinIO endpoint.

    Args:
        settings: The root configuration object.

    Returns:
        Spark config keys and values to apply before the session is built.

    Raises:
        RuntimeError: If the MinIO credentials are absent from the environment.
            Without them S3A falls through its provider chain and fails with a
            403 on the first write, which looks like a bucket permission
            problem rather than two unset variables.
    """
    try:
        key = os.environ["MINIO_ROOT_USER"]
        secret = os.environ["MINIO_ROOT_PASSWORD"]
    except KeyError as exc:
        raise RuntimeError(
            f"storage.backend is 's3' but {exc.args[0]} is unset. The Makefile "
            "exports it from .env; a bare shell needs "
            "`set -a && source .env && set +a`."
        ) from None

    return {
        "spark.jars.packages": _packages(),
        "spark.hadoop.fs.s3a.endpoint": settings.storage.endpoint_url,
        # MinIO serves no DNS for virtual-hosted buckets, so without path-style
        # access S3A resolves recsys.localhost:9000 and the connection fails.
        "spark.hadoop.fs.s3a.path.style.access": "true",
        # The endpoint is http://; leaving SSL enabled produces a TLS handshake
        # failure against a plaintext port.
        "spark.hadoop.fs.s3a.connection.ssl.enabled": "false",
        "spark.hadoop.fs.s3a.access.key": key,
        "spark.hadoop.fs.s3a.secret.key": secret,
        "spark.hadoop.fs.s3a.aws.credentials.provider": (
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider"
        ),
        # The magic committer writes task output straight to its final key via
        # multipart uploads, completed only on commit. The default committer
        # instead writes to a temporary directory and renames -- and an object
        # store has no rename, so S3A emulates it as copy-then-delete: O(data)
        # rather than O(1), and NOT atomic, so an interrupted overwrite can
        # leave a half-replaced table behind.
        "spark.hadoop.fs.s3a.committer.name": "magic",
        "spark.sql.sources.commitProtocolClass": (
            "org.apache.spark.internal.io.cloud.PathOutputCommitProtocol"
        ),
        "spark.sql.parquet.output.committer.class": (
            "org.apache.spark.internal.io.cloud.BindingParquetOutputCommitter"
        ),
    }


def get_spark(settings: Settings, app: str = "recsys") -> SparkSession:
    """Create a Spark session with the configured tuning.

    Note:
        ``spark.jars.packages`` is read only when the JVM starts. If a session
        already exists in this process, ``getOrCreate`` returns it and every
        option below is silently discarded -- the failure then surfaces much
        later as ``No FileSystem for scheme: s3a``. The test suite's
        session-scoped ``spark`` fixture builds a JVM without these, so a test
        that calls this function afterwards gets a session that cannot reach
        MinIO. The guard below makes that audible rather than mysterious.

    Args:
        settings: The root configuration object.
        app: Name for the Spark application.

    Returns:
        A Spark session.
    """
    builder = (
        SparkSession.builder.appName(app)
        .master(settings.spark.master)
        .config("spark.driver.memory", settings.spark.driver_memory)
        .config("spark.sql.shuffle.partitions", settings.spark.shuffle_partitions)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.driver.maxResultSize", "2g")
    )

    if settings.storage.backend == "s3":
        if SparkSession.getActiveSession() is not None:
            warnings.warn(
                "a SparkSession already exists in this process, so the s3a jars "
                "and configuration will be ignored; writes to "
                f"{settings.storage.endpoint_url} will fail with "
                "'No FileSystem for scheme: s3a'",
                RuntimeWarning,
                stacklevel=2,
            )
        for name, value in _s3a_options(settings).items():
            builder = builder.config(name, value)

    return builder.getOrCreate()


def bundled_jar_version(prefix: str) -> str:
    """Version of a jar shipped inside the installed pyspark wheel.

    Args:
        prefix: Filename prefix up to the version, e.g. ``hadoop-client-api``.

    Returns:
        The version substring between the prefix and ``.jar``.

    Raises:
        FileNotFoundError: If no jar matches, which means the wheel's layout
            changed and the version pins can no longer be checked.
    """
    jars = Path(pyspark.__file__).parent / "jars"
    matches = sorted(jars.glob(f"{prefix}-*.jar"))
    if not matches:
        raise FileNotFoundError(f"no {prefix}-*.jar under {jars}")
    return matches[0].name.removeprefix(f"{prefix}-").removesuffix(".jar")
