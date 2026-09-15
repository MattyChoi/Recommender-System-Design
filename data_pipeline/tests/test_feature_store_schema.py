"""Every declared FeatureView field must exist, with the right type, in the
table its source points at.

Nothing else checks this. `feast apply` reads the Parquet schema and infers
what it can, so a field declared here that the gold builder never emits fails
at apply or materialise time -- or worse, materialises as null and reaches the
ranker as a silently missing feature. The rename of ctr_24h_smoothed to
ctr_smoothed is the shape of the mistake: one edit in the builder, one in the
FeatureView, and no test to say they matched.

The frames are built by the REAL builders over a tiny inline corpus, so this
needs no built dataset and no MinIO -- only a JVM, which the root conftest
already skips without.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from feast import FeatureView
from feast.types import FeastType, Float64, Int64, String
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as f

from data_pipeline.features.item_dynamic_features import (
    item_hourly_features,
    smoothed_ctr_by_category,
)
from data_pipeline.features.recsys_store.feature_repo.definition import (
    context_features,
    derived_features,
    item_stats,
    ranker_v1,
    user_category_stats,
    user_stats,
)
from data_pipeline.features.user_category_features import user_category_hourly_features
from data_pipeline.features.user_dynamic_features import (
    smoothed_user_ctr,
    user_hourly_features,
)

T0 = datetime(2019, 11, 14, 12, 0, 0)

# Feast declares logical types; Spark writes physical ones. A mismatch here is
# not cosmetic: Int64 against a Spark double materialises as a truncated value.
_SPARK_TYPE_FOR: dict[FeastType, set[str]] = {
    Int64: {"bigint", "int"},
    Float64: {"double", "float"},
    String: {"string"},
}

_EVENTS = "user_id string, item_id string, category string, clicked boolean, ts timestamp"


@pytest.fixture
def events(spark: SparkSession) -> DataFrame:
    """Two users, two items, two categories, spread over three hours.

    Enough for every window and rate to be defined; the values are irrelevant,
    only the schema is under test.
    """
    rows = [
        ("U1", "N1", "sports", True, T0 - timedelta(hours=2)),
        ("U1", "N2", "news", False, T0 - timedelta(hours=2)),
        ("U2", "N1", "sports", False, T0 - timedelta(hours=1)),
        ("U2", "N2", "news", True, T0 - timedelta(hours=1)),
        ("U1", "N1", "sports", False, T0),
    ]
    return spark.createDataFrame(rows, _EVENTS)


@pytest.fixture
def built(events: DataFrame) -> dict[str, DataFrame]:
    """The three series exactly as data_pipeline.features.gold writes them."""
    item_df = smoothed_ctr_by_category(
        item_hourly_features(events.select("item_id", "ts", "clicked", "category"))
    )
    user_df = smoothed_user_ctr(user_hourly_features(events.select("user_id", "ts", "clicked")))
    user_cat_df = user_category_hourly_features(
        events.select("user_id", "category", "ts", "clicked")
    )
    # gold.py stamps this on all three after building.
    return {
        "item_stats": item_df.withColumn("created_ts", f.current_timestamp()),
        "user_stats": user_df.withColumn("created_ts", f.current_timestamp()),
        "user_category_stats": user_cat_df.withColumn("created_ts", f.current_timestamp()),
    }


@pytest.mark.parametrize("view", [item_stats, user_stats, user_category_stats])
def test_declared_fields_exist_with_compatible_types(
    view: FeatureView, built: dict[str, DataFrame]
) -> None:
    actual = dict(built[view.name].dtypes)

    for field in view.schema:
        assert field.name in actual, (
            f"{view.name} declares '{field.name}', which the gold builder does "
            f"not emit. Built columns: {sorted(actual)}"
        )
        allowed = _SPARK_TYPE_FOR[field.dtype]
        assert actual[field.name] in allowed, (
            f"{view.name}.{field.name} is declared {field.dtype} but Spark "
            f"writes {actual[field.name]}"
        )


@pytest.mark.parametrize("view", [item_stats, user_stats, user_category_stats])
def test_the_timestamp_columns_a_point_in_time_join_needs(
    view: FeatureView, built: dict[str, DataFrame]
) -> None:
    """feature_ts orders the as-of join; created_ts breaks ties within it.

    Neither is declared as a Field -- Feast reads them from the source config --
    so nothing above would notice if a builder stopped emitting one.
    """
    actual = dict(built[view.name].dtypes)

    assert actual.get("feature_ts") == "timestamp"
    assert actual.get("created_ts") == "timestamp"


@pytest.mark.parametrize("view", [item_stats, user_stats, user_category_stats])
def test_join_keys_are_present_and_are_strings(
    view: FeatureView, built: dict[str, DataFrame]
) -> None:
    """A missing join key is the one failure that yields zero rows, not an error.

    Resolved through the entities rather than read off ``view.join_keys``.
    That property returns the join keys of ``entity_columns``, which Feast
    populates during ``feast apply`` -- it infers the entity columns from the
    source's own schema. On a view imported straight from the definition
    module it is empty, so iterating it asserts nothing and the test passes on
    a frame with no entity column at all.
    """
    actual = dict(built[view.name].dtypes)
    for entity_column in view.join_keys:
        assert actual.get(entity_column) == "string", (
            f"{view.name} joins on '{entity_column}', which the builder emits "
            f"as {actual.get(entity_column)!r}"
        )


def test_the_feature_service_covers_every_batch_view() -> None:
    """The ranker's contract must not silently lose a view.

    A FeatureService is what serving requests; a view dropped from it becomes a
    feature the ranker was trained on and is no longer served.
    """
    views = (
        item_stats,
        user_stats,
        user_category_stats,
        context_features,
        derived_features,
    )
    expected = {view.name for view in views}
    served = {projection.name for projection in ranker_v1.feature_view_projections}

    assert served == expected


def test_the_feature_service_excludes_the_unwritten_push_view() -> None:
    """user_realtime has no producer until Part Q's Flink job exists."""
    served = {projection.name for projection in ranker_v1.feature_view_projections}

    assert "user_realtime" not in served
