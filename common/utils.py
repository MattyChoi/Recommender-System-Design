from __future__ import annotations

from collections.abc import Sequence
from functools import reduce
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession

from common.config import Settings

SPLITS = ("train", "dev", "test")
BRONZE_TABLES = ("events", "history", "news")
GOLD_TABLES = (
    "item_hourly_features",
    "user_hourly_features",
    "user_category_cross_features",
    "training_examples",
    "user_history",
)

# The gold tables that follow storage.backend. These three are the feature
# store's sources, so they have to be reachable from wherever serving runs.
FEATURE_TABLES = (
    "item_hourly_features",
    "user_hourly_features",
    "user_category_cross_features",
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
        location = gold_location(settings, name)
        if _is_remote(location):
            return _remote_is_built(settings, location)
        return (Path(location) / "_SUCCESS").is_file()
    return False


def _is_remote(location: str) -> bool:
    """Whether a resolved gold location is an object-store URI rather than a path."""
    return location.startswith(("s3://", "s3a://"))


def _remote_is_built(settings: Settings, location: str) -> bool:
    """Whether a committed write exists at an object-store location.

    Spark drops ``_SUCCESS`` only after the write job commits, so the marker
    means the same thing in a bucket as on disk: part-files without it are the
    remains of a run that died.

    s3fs is imported here rather than at module scope because ``common.utils``
    is imported by nearly everything and s3fs drags in aiobotocore and an event
    loop -- a cost for the one path that needs it, not for every import.

    Args:
        settings: The root configuration object.
        location: A resolved gold location, in either scheme.

    Returns:
        Whether the committed marker is present.

    Raises:
        RuntimeError: If the store cannot be reached or refuses the request.
            Answering "not built" there would rebuild and then fail inside
            Spark's S3A client a minute later, which is a much worse place to
            learn that MinIO is down.
    """
    import s3fs

    fs = s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": settings.storage.endpoint_url},
        # A listing cached from before a rebuild would outlive it.
        skip_instance_cache=True,
    )
    # BOTH schemes: gold_location hands back s3a:// by default, which is
    # Spark's registration and means nothing to s3fs.
    key = location.removeprefix("s3a://").removeprefix("s3://")

    try:
        return bool(fs.exists(f"{key}/_SUCCESS"))
    except OSError as exc:
        raise RuntimeError(
            f"cannot reach {settings.storage.endpoint_url} to check whether "
            f"{location} is built -- is MinIO running? (`make up`)"
        ) from exc


def gold_location(settings: Settings, name: str, scheme: str = "s3a") -> str:
    """Resolve a gold table to wherever that particular table lives.

    :meth:`Settings.gold_uri` knows how to build a URI; this knows which tables
    get one. Splitting them keeps the per-table decision in one place instead
    of at every call site.

    Args:
        settings: The root configuration object.
        name: A gold table name, optionally with a suffix --
            ``training_examples/train`` is matched on ``training_examples``.
        scheme: Passed through for remote tables. ``s3a`` for Spark, ``s3``
            for pyarrow and Feast.

    Returns:
        A local filesystem path, or a URI for a table that follows the backend.
    """
    if name.split("/", 1)[0] in FEATURE_TABLES:
        return settings.gold_uri(name, scheme=scheme)
    return str(settings.paths.gold / name)


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
    """Read a gold table by name, from wherever that table lives."""
    return spark.read.parquet(gold_location(settings, name))
