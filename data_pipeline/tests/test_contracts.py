"""Data contracts for the all layers of the data pipeline.

Most of these tests were created by a Claude agent
"""

from __future__ import annotations

from datetime import datetime, timedelta
from functools import reduce
from pathlib import Path

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as f

from common.schemas import EVENT_SCHEMA, OOV_IDX
from data_pipeline.features.asof import asof_join
from data_pipeline.features.ctr import category_prior
from data_pipeline.features.item_dynamic_features import (
    item_hourly_features,
    smoothed_ctr_by_category,
)

# The `spark` and `data_root` fixtures live in conftest.py. `data_root` is the
# real corpus when it has been built and a freshly-built synthetic fixture
# otherwise, so every path below hangs off it rather than being hardcoded.


@pytest.fixture(scope="session")
def events(spark: SparkSession, data_root: Path) -> DataFrame:
    return spark.read.parquet(str(data_root / "bronze" / "events" / "train")).cache()


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


def test_asof_rejects_feature_columns_already_present_on_labels(spark: SparkSession) -> None:
    """The collision is silent and destructive, so it must be loud instead.

    Silver carries ``category`` and the gold item series emits ``category``, so
    this is the first thing a real call would hit. Unguarded, the label's own
    value is nulled and then backfilled from the feature series by the
    carry-forward: same column name, plausible value, different meaning, no
    error anywhere.
    """
    labels = spark.createDataFrame(
        [("N1", T0, "sports")], "item_id string, ts timestamp, category string"
    )
    features = spark.createDataFrame(
        [("N1", T0 - timedelta(hours=1), "finance")],
        "item_id string, feature_ts timestamp, category string",
    )

    with pytest.raises(ValueError, match="category"):
        asof_join(labels, features, join_key="item_id")


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

    assert got["item_impressions_24h"] == 1, (
        f"leak: the label saw {got['item_impressions_24h']} impressions, but only 1 "
        "occurred before it -- item_hourly is stamping buckets with their start time"
    )
    assert got["item_clicks_24h"] == 0, (
        f"leak: the label saw {got['item_clicks_24h']} clicks, all of which happened "
        "after it. This is the failure that produces beautiful offline metrics."
    )


# ---------------------------------------------------------------------------
# The category prior
#
# Shrinkage is what stops the ranker promoting an article with one click on one
# impression, which on a news corpus is the common case rather than the edge
# case. That makes the prior load-bearing, and a prior that peeks at the future
# or is computed from overlapping windows is worse than a wrong constant --
# it is a wrong constant that looks measured.
# ---------------------------------------------------------------------------

_PRIOR_SCHEMA = (
    "category string, feature_ts timestamp, item_impressions_1h long, item_clicks_1h long"
)


def _hourly(spark: SparkSession, rows: list[tuple[str, datetime, int, int]]) -> DataFrame:
    return spark.createDataFrame(rows, _PRIOR_SCHEMA)


def test_the_category_prior_cannot_see_the_future(spark: SparkSession) -> None:
    """The prior at hour t uses buckets <= t and nothing after.

    The middle hour is the discriminating one: a whole-timeline prior returns
    0.10 for all three rows, because the quiet hour is averaged against a
    future hour that had not happened yet.
    """
    rows = _hourly(
        spark,
        [
            ("sports", T0, 100, 10),  # 10/100  -> 0.10
            ("sports", T0 + timedelta(hours=1), 100, 0),  # 10/200  -> 0.05
            ("sports", T0 + timedelta(hours=2), 100, 20),  # 30/300  -> 0.10
        ],
    )
    got = {
        r["feature_ts"]: r["cat_expanding_ctr"]
        for r in category_prior(rows, min_impressions=0).collect()
    }

    assert got[T0] == pytest.approx(0.10)
    assert got[T0 + timedelta(hours=1)] == pytest.approx(0.05), "the prior saw the future"
    assert got[T0 + timedelta(hours=2)] == pytest.approx(0.10)


def test_a_thin_category_falls_back_to_the_global_prior(spark: SparkSession) -> None:
    """Measured on MIND: `kids` has impressions and zero clicks in hour one.

    Its own prior is therefore exactly 0.0, and shrinking toward zero on the
    strength of no evidence is worse than not shrinking at all. Below the floor
    the category borrows the global rate, which is itself point-in-time.
    """
    rows = _hourly(spark, [("kids", T0, 50, 0), ("news", T0, 10_000, 400)])

    prior = category_prior(rows, min_impressions=1_000)
    got = {r["category"]: r["cat_expanding_ctr"] for r in prior.collect()}

    assert got["kids"] == pytest.approx(400 / 10_050), "a thin category shrank toward zero"
    assert got["news"] == pytest.approx(400 / 10_000)


def test_the_prior_sums_hourly_counts_not_rolling_windows(spark: SparkSession) -> None:
    """item_clicks_24h overlaps between consecutive rows; item_clicks_1h does not.

    Summing the rolling column counts the same click up to 24 times, weighted
    by how many hours an item happened to appear in -- measured at -36.5% on
    `kids`. Here the correct answer is 20/200; a _24h-based implementation
    would produce 20/300.
    """
    rows = _hourly(spark, [("sports", T0, 100, 0), ("sports", T0 + timedelta(hours=1), 100, 20)])

    latest = category_prior(rows, min_impressions=0).orderBy(f.col("feature_ts").desc())
    assert latest.collect()[0]["cat_expanding_ctr"] == pytest.approx(0.10)


def test_both_prior_frames_are_computed_and_they_differ(spark: SparkSession) -> None:
    """Expanding and rolling are carried side by side so the choice stays measurable.

    The series is built so the two must disagree: a busy, high-CTR first day
    followed by a quiet second day more than 24 hours later. The expanding
    prior still remembers the first day; the rolling one has forgotten it.
    """
    rows = _hourly(
        spark,
        [
            ("sports", T0, 1_000, 100),  # day one: 10% CTR
            ("sports", T0 + timedelta(hours=48), 1_000, 10),  # two days later: 1%
        ],
    )
    late = category_prior(rows, min_impressions=0).orderBy(f.col("feature_ts").desc())
    got = late.collect()[0]

    assert got["cat_expanding_ctr"] == pytest.approx(110 / 2_000)  # 0.055, remembers day one
    assert got["cat_rolling_ctr"] == pytest.approx(10 / 1_000)  # 0.010, has forgotten it


def test_only_the_expanding_prior_reaches_the_feature_table(spark: SparkSession) -> None:
    """The rolling prior is computed, compared, and deliberately not shipped.

    gold.py writes this frame straight to Parquet and the Feast FeatureView
    declares its columns, so an undeclared extra column is a divergence between
    what is on disk and what is registered.
    """
    hourly = spark.createDataFrame(
        [("N1", "sports", T0, 100, 10, 100, 10)],
        "item_id string, category string, feature_ts timestamp, "
        "item_impressions_1h long, item_clicks_1h long, "
        "item_impressions_24h long, item_clicks_24h long",
    )
    got = smoothed_ctr_by_category(hourly, prior_strength=20.0, min_prior_impressions=0)

    assert "cat_expanding_ctr" in got.columns
    assert "cat_rolling_ctr" not in got.columns, "the rolling prior leaked into the feature table"
    # smoothed from the EXPANDING prior: (10 + 0.1*20) / (100 + 20)
    assert got.collect()[0]["item_ctr_smoothed"] == pytest.approx(12 / 120)


# ---------------------------------------------------------------------------
# ID mappings
#
# An index is meaningless on its own; it is meaningful relative to the
# checkpoint trained against it. Every assertion below exists because its
# failure mode is SILENT -- a model that loads, runs, returns recommendations,
# and addresses the wrong embedding row.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def item_map(spark: SparkSession, data_root: Path) -> DataFrame:
    return spark.read.parquet(str(data_root / "bronze" / "item_map")).cache()


@pytest.fixture(scope="session")
def user_map(spark: SparkSession, data_root: Path) -> DataFrame:
    return spark.read.parquet(str(data_root / "bronze" / "user_map")).cache()


@pytest.fixture(scope="session")
def silver(spark: SparkSession, data_root: Path) -> DataFrame:
    return spark.read.parquet(str(data_root / "silver" / "impressions" / "train")).cache()


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


# ---------------------------------------------------------------------------
# The news catalogue
#
# news is the sole source of item_map, so a defect here corrupts C3 without
# implicating itself: the ID-map tests would report the symptom while pointing
# at the wrong file. These sit upstream of that.
# ---------------------------------------------------------------------------

# Present but frequently absent, so a blanket non-null assertion would be
# wrong: ~5% of MIND articles ship no abstract at all. That is a property of
# the corpus, not a parse failure -- see the ceiling asserted below.
_REQUIRED_NEWS_COLUMNS = ("item_id", "category", "subcategory", "title")


@pytest.fixture(scope="session")
def news(spark: SparkSession, data_root: Path) -> dict[str, DataFrame]:
    root = data_root / "bronze" / "news"
    available = {s: root / s for s in ("train", "dev") if (root / s / "_SUCCESS").is_file()}
    if not available:
        pytest.skip("no committed news tables")
    return {s: spark.read.parquet(str(p)).cache() for s, p in available.items()}


@pytest.mark.parametrize("split", ["train", "dev"])
def test_news_item_ids_are_unique_within_a_split(news: dict[str, DataFrame], split: str) -> None:
    """What licenses the .distinct() in build_id_maps.

    A duplicated item_id here would survive into item_map as two rows with two
    different indices for one article -- splitting its interactions across two
    embedding rows, each trained on half the evidence, and neither wrong enough
    to look wrong.
    """
    if split not in news:
        pytest.skip(f"{split} news not built")
    catalogue = news[split]
    assert catalogue.select("item_id").distinct().count() == catalogue.count()


def test_news_rows_agree_where_the_splits_overlap(news: dict[str, DataFrame]) -> None:
    """The assumption read_news() states in prose and nothing has checked.

    read_news unions both splits and calls dropDuplicates(["item_id"]), which
    keeps an ARBITRARY row of each duplicate group. That is only safe while
    overlapping rows are identical. Let them diverge -- a retitled article, a
    recategorised one -- and the surviving row becomes whichever Spark happened
    to encounter first, so item_map, silver's category column and every
    category-level CTR prior turn nondeterministic across runs.

    Measured on MIND-small: 28,460 items appear in both weeks, none differing.
    """
    if len(news) < 2:
        pytest.skip("need both splits to compare")
    columns = ["item_id", "category", "subcategory", "title", "abstract", "url"]
    train, dev = news["train"].select(columns), news["dev"].select(columns)

    shared = train.select("item_id").intersect(dev.select("item_id")).count()
    # INTERSECT compares with null-safe equality, so a shared null abstract
    # counts as agreement rather than silently failing the match.
    identical = train.intersect(dev).count()

    assert identical == shared, (
        f"{shared - identical} of {shared} overlapping articles differ between splits; "
        "dropDuplicates in read_news would pick between them arbitrarily"
    )


@pytest.mark.parametrize("split", ["train", "dev"])
@pytest.mark.parametrize("column", _REQUIRED_NEWS_COLUMNS)
def test_news_required_columns_are_never_null(
    news: dict[str, DataFrame], split: str, column: str
) -> None:
    """category and subcategory feed the smoothed-CTR prior; title feeds content.

    A null in any of them is not a missing value to impute around -- it means
    the TSV column offsets have shifted, and every column after it is holding
    someone else's data.
    """
    if split not in news:
        pytest.skip(f"{split} news not built")
    assert news[split].filter(f.col(column).isNull()).count() == 0


@pytest.mark.parametrize("split", ["train", "dev"])
def test_news_abstracts_are_mostly_present(news: dict[str, DataFrame], split: str) -> None:
    """Absent abstracts are normal; an absent COLUMN is a parse failure.

    ~5% of MIND articles genuinely ship without one, so the contract is a
    ceiling rather than zero. The failure this catches is the whole column
    arriving null, which is what a shifted delimiter or a short row produces
    and which no other test in this file would notice.
    """
    if split not in news:
        pytest.skip(f"{split} news not built")
    catalogue = news[split]
    missing = catalogue.filter(f.col("abstract").isNull()).count() / catalogue.count()
    assert missing < 0.20, f"{missing:.1%} of abstracts are null -- suspect the parse"


@pytest.mark.parametrize("split", ["train", "dev"])
def test_news_categories_are_a_small_closed_vocabulary(
    news: dict[str, DataFrame], split: str
) -> None:
    """A cheap canary for the columns having slid sideways.

    MIND ships 17 categories, all single lowercase tokens. If the parse shifts,
    this column fills with titles or URLs -- thousands of distinct values, most
    containing spaces -- and the smoothed-CTR prior silently degenerates to one
    group per article, which is the same as having no prior at all.
    """
    if split not in news:
        pytest.skip(f"{split} news not built")
    categories = {row[0] for row in news[split].select("category").distinct().collect()}

    assert 5 <= len(categories) <= 50, f"{len(categories)} distinct categories"
    assert all(" " not in c for c in categories), "a category contains a space"


# ---------------------------------------------------------------------------
# Referential integrity and the history snapshot
#
# Guide C4's remaining contracts. Two of them are adapted rather than copied,
# and the adaptation is the interesting part -- see
# test_history_predates_the_log_window.
# ---------------------------------------------------------------------------

# Measured on MIND-small: 0.4% of train's in-log clicked pairs appear in
# history, 0.2% of dev's. A ceiling well above both, but far below the ~100%
# that an end-of-window snapshot would produce.
_MAX_HISTORY_OVERLAP = 0.05


@pytest.fixture(scope="session")
def history(spark: SparkSession, data_root: Path) -> dict[str, DataFrame]:
    root = data_root / "bronze" / "history"
    available = {s: root / s for s in ("train", "dev") if (root / s / "_SUCCESS").is_file()}
    if not available:
        pytest.skip("no committed history tables")
    return {s: spark.read.parquet(str(p)).cache() for s, p in available.items()}


def _history_pairs(history: DataFrame) -> DataFrame:
    """Distinct (user_id, item_id) pairs named in the history snapshot."""
    return (
        history.select(
            "user_id",
            f.explode(f.split(f.coalesce(f.col("history"), f.lit("")), " ")).alias("item_id"),
        )
        .filter(f.col("item_id") != "")
        .distinct()
    )


def test_every_click_sits_inside_an_impression(events: DataFrame) -> None:
    """Guide C4's orphan-click contract.

    Structurally guaranteed on this schema and asserted anyway. MIND ships one
    row per (impression, item shown) with the click as a BOOLEAN on that row,
    so a click cannot exist without its impression -- there is nowhere for it
    to live. The guide assumes an event-type model where clicks arrive as
    separate rows and can genuinely be orphaned.

    Kept because the guarantee is a property of the current schema, not of the
    project: Part Q feeds clicks through Kafka as their own events, and on the
    day silver is built from that stream instead, this stops being free and
    starts being the test that catches it.
    """
    clicks = events.filter(f.col("clicked")).select("impression_id", "item_id")
    shown = events.select("impression_id", "item_id")
    orphans = clicks.join(shown, ["impression_id", "item_id"], "left_anti").count()

    assert orphans == 0, f"{orphans} clicks without a matching impression"


def test_silver_items_all_exist_in_the_catalogue(
    silver: DataFrame, news: dict[str, DataFrame]
) -> None:
    """Referential integrity, asserted against the catalogue rather than the map.

    The ID-map tests already prove events resolve to an index, but item_map is
    DERIVED from news -- so checking against it cannot detect the two drifting
    apart. This checks silver against the catalogue directly, which is where
    category, subcategory and title actually come from.
    """
    catalogue = reduce(
        lambda a, b: a.unionByName(b),
        [df.select("item_id") for df in news.values()],
    ).distinct()

    missing = silver.select("item_id").distinct().join(catalogue, "item_id", "left_anti")
    assert missing.count() == 0, "silver holds items absent from the news catalogue"


@pytest.mark.parametrize("split", ["train", "dev"])
def test_history_is_one_snapshot_per_user(history: dict[str, DataFrame], split: str) -> None:
    """A user's history never changes across their impressions.

    Verified at zero exceptions over 100,000 users, and it is the property
    every sequence feature rests on. MIND's `history` is a SNAPSHOT, not a
    running log: the same string is attached to a user's first impression and
    their last. Should that ever stop being true, any code treating history as
    "the sequence as of this impression" quietly changes meaning without
    changing shape, which is the kind of bug that survives a code review.

    The consequence is worth stating plainly: history is STALE for later
    impressions, not leaky. A user's day-6 impression carries their day-1
    history and knows nothing of the five days between. Part H's sequential
    model has to top it up from in-window clicks as of each impression.
    """
    if split not in history:
        pytest.skip(f"{split} history not built")
    coalesced = history[split].withColumn("h", f.coalesce(f.col("history"), f.lit("")))
    per_user = coalesced.groupBy("user_id").agg(f.count_distinct("h").alias("variants"))

    assert per_user.filter(f.col("variants") > 1).count() == 0


def test_history_predates_the_log_window(events: DataFrame, history: dict[str, DataFrame]) -> None:
    """The leak guard, rewritten -- the guide's version fails on this corpus.

    C4 asks for `test_history_predates_its_impression`, asserting that no item
    in a user's history was clicked at or after the impression it hangs off,
    with a hard zero. Run verbatim on MIND-small it reports 5,704 violations,
    and every one of them is benign: an item clicked BEFORE the log window
    (which is what put it in history) and clicked again during it. The test is
    wrong, not the data.

    What the guide is really reaching for is whether the snapshot was taken
    before the window or after it. Taken after, a user's history would contain
    the clicks they made during the window, and training on it would leak the
    future wholesale. That is measurable directly, and it degrades gracefully:
    an end-of-window snapshot pushes this overlap towards 100%, while a
    genuine pre-window snapshot leaves only re-clicks behind.

    Measured: 0.4% on train, 0.2% on dev.
    """
    if "train" not in history:
        pytest.skip("train history not built")
    clicked = events.filter(f.col("clicked")).select("user_id", "item_id").distinct()
    total = clicked.count()
    assert total > 0, "no clicks in bronze -- check the label parse"

    overlap = clicked.join(_history_pairs(history["train"]), ["user_id", "item_id"]).count()
    rate = overlap / total

    assert rate < _MAX_HISTORY_OVERLAP, (
        f"{rate:.1%} of in-log clicks appear in history; a pre-window snapshot "
        "leaves only re-clicks, so this suggests history was captured later"
    )
