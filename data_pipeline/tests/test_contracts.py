"""Data contracts for the bronze layer.

Skipped when no bronze layer exists, so CI stays green without the dataset.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as f

from common.schemas import EVENT_SCHEMA, OOV_IDX
from data_pipeline.features.asof import asof_join
from data_pipeline.features.item_dynamic_features import item_hourly_features

BRONZE = Path("data/bronze/events/train")

# The `spark` fixture, and the JVM skip that guards it, live in conftest.py.


@pytest.fixture(scope="session")
def events(spark: SparkSession) -> DataFrame:
    if not BRONZE.exists():
        pytest.skip("no bronze layer; run `make data` first")
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


def test_bronze_events_matches_the_declared_contract(events: DataFrame) -> None:
    # `dt` is a partition column derived from ts at write time, not part of
    # the event contract itself.
    actual = {
        (field.name, field.dataType.simpleString())
        for field in events.schema.fields
        if field.name != "dt"
    }
    expected = {(field.name, field.dataType.simpleString()) for field in EVENT_SCHEMA.fields}
    assert actual == expected, (
        f"missing from bronze: {sorted(expected - actual)}; "
        f"unexpected in bronze: {sorted(actual - expected)}"
    )


# ---------------------------------------------------------------------------
# Point-in-time correctness
#
# These tests build their own frames, so they run without data/ and without a
# built pipeline. They encode ONE convention, stated once, here:
#
#     feature_ts is the instant a bucket CLOSED.
#     The row stamped 14:00 covers [13:00, 14:00) and is fully knowable at
#     14:00.
#
# Under that convention asof_join's boundary rule -- features sorting BEFORE
# labels at equal timestamps, i.e. "<=" -- is correct. If feature_ts is ever
# restamped with the bucket's START time, both these tests and the tie-break
# in asof.py must change together. An as-of join's correctness is a property
# of the PAIR (timestamp convention, boundary rule), never of either alone.
# ---------------------------------------------------------------------------

T0 = datetime(2019, 11, 14, 12, 0, 0)

_LABEL_SCHEMA = "impression_id string, item_id string, ts timestamp"
_FEATURE_SCHEMA = "item_id string, feature_ts timestamp, ctr double"


def _labels(spark: SparkSession, rows: list[tuple[str, str, datetime]]) -> DataFrame:
    return spark.createDataFrame(rows, _LABEL_SCHEMA)


def _features(spark: SparkSession, rows: list[tuple[str, datetime, float]]) -> DataFrame:
    return spark.createDataFrame(rows, _FEATURE_SCHEMA)


def test_asof_ignores_features_after_the_label(spark: SparkSession) -> None:
    """The basic leak: a value computed after the event must be invisible.

    If this fails, every offline metric in the project is inflated and no
    other test in the repo will tell you.
    """
    labels = _labels(spark, [("I1", "N1", T0)])
    features = _features(
        spark,
        [
            ("N1", T0 - timedelta(hours=1), 0.10),  # visible
            ("N1", T0 + timedelta(hours=1), 0.90),  # the future -- must not be
        ],
    )

    got = asof_join(labels, features, join_key="item_id").collect()
    assert len(got) == 1
    assert got[0]["ctr"] == 0.10, f"leak: read a feature from the future ({got[0]['ctr']})"


def test_asof_takes_the_most_recent_prior_feature(spark: SparkSession) -> None:
    """Of several eligible rows, the newest one wins -- not the first, not the last read.

    The feature rows are supplied out of chronological order on purpose: the
    result must depend on the window's ORDER BY, never on input row order.
    """
    labels = _labels(spark, [("I1", "N1", T0)])
    features = _features(
        spark,
        [
            ("N1", T0 - timedelta(hours=3), 0.10),
            ("N1", T0 - timedelta(hours=1), 0.20),  # newest before T0
            ("N1", T0 - timedelta(hours=2), 0.30),
        ],
    )

    assert asof_join(labels, features, join_key="item_id").collect()[0]["ctr"] == 0.20


def test_asof_includes_a_bucket_that_closed_at_the_label_instant(spark: SparkSession) -> None:
    """The boundary case, and the reason the tie-break is ascending.

    A bucket stamped exactly at the label's timestamp closed at that instant,
    so every event inside it precedes the label and it is legitimately
    readable. Flip ``_is_label`` to ``desc()`` in asof.py and this test goes
    red -- which is the correct alarm, because that change would silently
    discard an hour of real signal for every label landing on the hour.
    """
    labels = _labels(spark, [("I1", "N1", T0)])
    features = _features(
        spark,
        [
            ("N1", T0 - timedelta(hours=1), 0.10),
            ("N1", T0, 0.90),  # closed AT the label instant: readable
        ],
    )

    assert asof_join(labels, features, join_key="item_id").collect()[0]["ctr"] == 0.90


def test_asof_yields_null_before_any_feature_exists(spark: SparkSession) -> None:
    """A cold item gets nulls, not zeros.

    This is not an edge case on MIND: a third of dev's catalogue never appears
    in train. Null means "nothing was knowable"; 0.0 would tell the ranker
    these articles have a measured click rate of zero, which is a much
    stronger and quite false claim.
    """
    labels = _labels(spark, [("I1", "N1", T0)])
    features = _features(spark, [("N1", T0 + timedelta(hours=1), 0.90)])

    assert asof_join(labels, features, join_key="item_id").collect()[0]["ctr"] is None


def test_asof_does_not_reach_across_keys(spark: SparkSession) -> None:
    """One article's history must never enrich another's impression."""
    labels = _labels(spark, [("I1", "N1", T0)])
    features = _features(spark, [("N2", T0 - timedelta(hours=1), 0.90)])

    assert asof_join(labels, features, join_key="item_id").collect()[0]["ctr"] is None


def test_asof_preserves_the_label_row_count(spark: SparkSession) -> None:
    """Enrichment, not a join: same rows out as in.

    A plain join can fan out on duplicate keys or drop on a miss. Neither may
    happen here, and a row-count change is the cheapest possible detector.
    """
    labels = _labels(
        spark,
        [
            ("I1", "N1", T0),
            ("I2", "N1", T0 + timedelta(minutes=30)),
            ("I3", "N2", T0),
        ],
    )
    features = _features(
        spark,
        [
            ("N1", T0 - timedelta(hours=1), 0.10),
            ("N1", T0 - timedelta(minutes=10), 0.20),
        ],
    )

    assert asof_join(labels, features, join_key="item_id").count() == labels.count()


def test_hourly_features_do_not_leak_within_the_bucket(spark: SparkSession) -> None:
    """THE GATE: asof_join and item_hourly must agree on what feature_ts means.

    Every test above feeds asof_join hand-written feature rows, so all of them
    pass even when item_hourly stamps its buckets with the wrong instant. This
    one runs the real pair.

    One impression at 13:10, then a hundred CLICKED impressions at 14:50. A
    label at 14:05 sits between them, so it may see the first and none of the
    rest -- not the hundred that follow it, and not its own impression.

    Stamping a bucket with its START time puts all 101 events under a
    feature_ts of 14:00, which reads as "before" 14:05 while the data is not.
    The label then arrives carrying a 99% click rate manufactured from its own
    future, and every model trained on it looks superb offline.
    """
    hour = datetime(2019, 11, 14, 14, 0, 0)
    rows: list[tuple[str, datetime, int, str]] = [("N1", hour - timedelta(minutes=50), 0, "sports")]
    rows += [("N1", hour + timedelta(minutes=50), 1, "sports") for _ in range(100)]
    raw = spark.createDataFrame(rows, "item_id string, ts timestamp, clicked int, category string")

    labels = _labels(spark, [("I1", "N1", hour + timedelta(minutes=5))])
    got = asof_join(labels, item_hourly_features(raw), join_key="item_id").collect()[0]

    assert got["impressions_24h"] == 1, (
        f"leak: the label saw {got['impressions_24h']} impressions, but only 1 "
        "occurred before it -- item_hourly is stamping buckets with their start time"
    )
    assert got["clicks_24h"] == 0, (
        f"leak: the label saw {got['clicks_24h']} clicks, all of which happened "
        "after it. This is the failure that produces beautiful offline metrics."
    )


# ---------------------------------------------------------------------------
# ID mappings
#
# An index is meaningless on its own; it is meaningful relative to the
# checkpoint trained against it. Every assertion below exists because its
# failure mode is SILENT -- a model that loads, runs, returns recommendations,
# and addresses the wrong embedding row.
# ---------------------------------------------------------------------------

ITEM_MAP = Path("data/bronze/item_map")
USER_MAP = Path("data/bronze/user_map")
SILVER = Path("data/silver/impressions/train")


@pytest.fixture(scope="session")
def item_map(spark: SparkSession) -> DataFrame:
    if not ITEM_MAP.exists():
        pytest.skip("no id maps; run `make bronze` first")
    return spark.read.parquet(str(ITEM_MAP)).cache()


@pytest.fixture(scope="session")
def user_map(spark: SparkSession) -> DataFrame:
    if not USER_MAP.exists():
        pytest.skip("no id maps; run `make bronze` first")
    return spark.read.parquet(str(USER_MAP)).cache()


@pytest.fixture(scope="session")
def silver(spark: SparkSession) -> DataFrame:
    if not SILVER.exists():
        pytest.skip("no silver layer; run `make silver` first")
    return spark.read.parquet(str(SILVER)).cache()


@pytest.mark.parametrize("table,column", [("item_map", "item_idx"), ("user_map", "user_idx")])
def test_indices_are_dense_and_start_after_the_oov_slot(
    request: pytest.FixtureRequest, table: str, column: str
) -> None:
    """Contiguous 1..N, with no duplicates and nothing sitting on OOV.

    Density is not cosmetic. A gap is a row in the embedding table that no
    item ever addresses: it receives no gradient, keeps its random
    initialisation for the whole of training, and costs parameters and memory
    for the privilege. A duplicate is worse -- two items sharing one vector,
    which trains to the average of two unrelated things.

    The lower bound is the half people forget: reserving OOV_IDX is only real
    if the maps genuinely start at 1.
    """
    mapping: DataFrame = request.getfixturevalue(table)
    n = mapping.count()
    lo, hi, distinct = mapping.agg(
        f.min(column), f.max(column), f.count_distinct(column)
    ).collect()[0]

    assert lo == OOV_IDX + 1, f"{column} starts at {lo}; {OOV_IDX} must stay reserved"
    assert hi == n, f"{column} runs to {hi} over {n} rows -- gaps mean dead embedding rows"
    assert distinct == n, f"{column} has {n - distinct} duplicate indices"


@pytest.mark.parametrize("table,key", [("item_map", "item_id"), ("user_map", "user_id")])
def test_the_mapping_key_is_unique(request: pytest.FixtureRequest, table: str, key: str) -> None:
    """A repeated key turns the join in silver into a row multiplier.

    Not hypothetical: item_map is built from the news catalogue, which ships
    one row per article PER SPLIT, and items shown in both weeks appear twice.
    Drop the distinct() and every such impression is duplicated in silver --
    inflating counts, CTR denominators and every metric built on them.
    """
    mapping: DataFrame = request.getfixturevalue(table)
    assert mapping.select(key).distinct().count() == mapping.count()


def test_every_event_id_resolves_to_an_index(
    events: DataFrame, item_map: DataFrame, user_map: DataFrame
) -> None:
    """The maps must cover the data, or silver's LEFT joins quietly drop to OOV.

    This is what catches the case the overwrite guard deliberately creates:
    keeping existing indices when a new split has arrived is correct for the
    checkpoint and wrong for the data, and this is where you find out.
    """
    unmapped_items = events.select("item_id").distinct().join(item_map, "item_id", "left_anti")
    unmapped_users = events.select("user_id").distinct().join(user_map, "user_id", "left_anti")

    assert unmapped_items.count() == 0, "items in bronze have no index; rebuild the maps"
    assert unmapped_users.count() == 0, "users in bronze have no index; rebuild the maps"


def test_silver_never_falls_back_to_oov(silver: DataFrame) -> None:
    """OOV exists for serving-time strangers, not for a batch build.

    silver coalesces a missing index to OOV_IDX so the column is never null.
    That is the right behaviour and also a perfect hiding place, so the batch
    rate is pinned at zero here: every id in a historical log was known when
    the maps were built, by construction.
    """
    for column in ("item_idx", "user_idx"):
        assert silver.filter(f.col(column).isNull()).count() == 0, f"{column} is null"
        assert silver.filter(f.col(column) == OOV_IDX).count() == 0, (
            f"{column} fell back to OOV in a batch build -- the maps do not cover silver"
        )
