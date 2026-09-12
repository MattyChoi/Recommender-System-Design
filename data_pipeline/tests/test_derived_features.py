"""The derived ODFV must agree with the Spark block it mirrors.

Same argument as test_context_features: two implementations of one definition,
so the test drives BOTH over the same rows rather than asserting the Python side
against a restatement of the rules.

The cases that matter are the null ones. A warm item exercises nothing.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as f

from data_pipeline.features.recsys_store.feature_repo.definition import derived_features

T0 = datetime(2019, 11, 14, 12, 0, 0)

# (item_ctr_smoothed, cat_expanding_ctr, user_ctr_smoothed, user_cat_affinity)
_ROWS = [
    (0.05, 0.03, 0.04, 0.10),  # everything warm
    (None, 0.03, 0.04, 0.10),  # cold item, category prior available
    (0.05, None, 0.04, 0.10),  # item warm, category thin
    (None, None, 0.04, 0.10),  # unknowable: training drops, serving must decide
    (0.05, 0.03, None, 0.10),  # cold user
    (0.05, 0.03, 0.04, None),  # no affinity for this user-category pair
    (None, 0.03, None, None),  # cold everything except the category
]
_SCHEMA = (
    "item_ctr_smoothed double, cat_expanding_ctr double, "
    "user_ctr_smoothed double, user_cat_affinity double"
)


@pytest.fixture
def spark_answer(spark: SparkSession) -> list[tuple[object, ...]]:
    """The Spark block from attach_point_in_time_features, in isolation.

    Copied deliberately rather than imported: attach_point_in_time_features
    needs four joined frames to reach this block, and the point is to compare
    the EXPRESSIONS, not to re-test the joins.
    """
    frame: DataFrame = spark.createDataFrame(_ROWS, _SCHEMA)
    out = (
        frame.withColumn("has_item_features", f.col("item_ctr_smoothed").isNotNull())
        .withColumn("has_user_features", f.col("user_ctr_smoothed").isNotNull())
        .withColumn("has_user_category_features", f.col("user_cat_affinity").isNotNull())
        .withColumn(
            "item_ctr_effective",
            f.coalesce(f.col("item_ctr_smoothed"), f.col("cat_expanding_ctr")),
        )
    )
    return [
        (
            row["has_item_features"],
            row["has_user_features"],
            row["has_user_category_features"],
            row["item_ctr_effective"],
        )
        for row in out.collect()
    ]


@pytest.fixture
def odfv_answer() -> list[tuple[object, ...]]:
    ft = derived_features.feature_transformation
    if not ft:
        return []
    got = ft.udf(
        {
            "item_ctr_smoothed": [row[0] for row in _ROWS],
            "cat_expanding_ctr": [row[1] for row in _ROWS],
            "user_ctr_smoothed": [row[2] for row in _ROWS],
            "user_cat_affinity": [row[3] for row in _ROWS],
        }
    )
    return list(
        zip(
            got["has_item_features"],
            got["has_user_features"],
            got["has_user_category_features"],
            got["item_ctr_effective"],
            strict=True,
        )
    )


def test_the_two_implementations_agree(
    spark_answer: list[tuple[object, ...]], odfv_answer: list[tuple[object, ...]]
) -> None:
    assert odfv_answer == spark_answer


def test_the_flag_reads_the_raw_value_not_the_fallback(
    odfv_answer: list[tuple[object, ...]],
) -> None:
    """Row 1 is a cold item with a category prior.

    has_item_features must be False even though item_ctr_effective is populated.
    Computing the flag after the coalesce makes it True for every row, which is
    the same as not having the feature -- and it would still agree with a Spark
    version that made the identical mistake, so this is pinned separately.
    """
    has_item, _, _, effective = odfv_answer[1]

    assert has_item is False
    assert effective == 0.03


def test_the_unknowable_row_yields_none(
    odfv_answer: list[tuple[object, ...]],
) -> None:
    """No item history and no category prior. Training drops these -- 6,262 rows.

    Serving cannot drop a request, so Part M must decide what the ranker gets.
    This test exists to keep that decision visible rather than defaulted.
    """
    assert odfv_answer[3][3] is None
