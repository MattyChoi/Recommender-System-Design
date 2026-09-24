"""One definition, served offline for training and online for inference.

Feast parses every .py file under the feature repo directory, so this file
must live BESIDE feature_store.yaml -- not up in data_pipeline/features/,
where the CLI will never look at it.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from feast import (
    Entity,
    FeatureService,
    FeatureView,
    Field,
    FileSource,
    Project,
    PushSource,
    RequestSource,
    ValueType,
)
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Array, Bool, Float64, Int64, String, UnixTimestamp

from common.config import load_settings
from common.utils import gold_location

# The CLI chdirs into this directory before importing, so a load relative to the
# working directory fails here. __file__ is stable no matter who calls, and
# load_settings() takes a root -- which is why the bucket and the endpoint are
# read from conf/config.yml below rather than repeated as literals.
#
#   feature_repo/ -> recsys_store/ -> features/ -> data_pipeline/ -> repo root
PROJECT_ROOT = Path(__file__).resolve().parents[4]
SETTINGS = load_settings(PROJECT_ROOT)


def _source(table: str) -> FileSource:
    """A gold feature series as Feast sees it.

    ``scheme="s3"``: Feast reads through pyarrow, which registers ``s3://``
    and has never heard of ``s3a://``. Spark writes the same bytes to the same
    bucket through the Hadoop client, which registers only ``s3a://``. Two
    clients, two schemes, one location -- see ``StorageConfig``.

    Args:
        table: Gold table name.

    Returns:
        A configured :class:`FileSource`.
    """
    endpoint: str | None = None
    if SETTINGS.storage.backend == "s3":
        # Ignored by pyarrow for a local path, but passing it anyway would
        # claim a dependency on MinIO that the local backend does not have.
        endpoint = SETTINGS.storage.endpoint_url

    return FileSource(
        path=gold_location(SETTINGS, table, scheme="s3"),
        s3_endpoint_override=endpoint,
        timestamp_field="feature_ts",  # what makes the as-of join possible
        # Breaks ties between rows sharing an event timestamp. No series can
        # produce one today -- each is unique per (entity, feature_ts) -- so
        # this is a guard on a future backfill that rewrites a bucket rather
        # than a fix for anything current.
        created_timestamp_column="created_ts",
    )


project = Project(
    name="recsys_store",
    description="Point-in-time feature definitions for the recommender system features.",
)

item = Entity(name="item", join_keys=["item_id"], value_type=ValueType.STRING)
user = Entity(name="user", join_keys=["user_id"], value_type=ValueType.STRING)
category = Entity(name="category", join_keys=["category"], value_type=ValueType.STRING)

item_stats_source = _source("item_hourly_features")

item_stats = FeatureView(
    name="item_stats",
    entities=[item],
    ttl=timedelta(hours=2),
    schema=[
        Field(name="item_impressions_24h", dtype=Int64),
        Field(name="item_clicks_24h", dtype=Int64),
        Field(name="item_impressions_cum", dtype=Int64),
        Field(name="item_clicks_cum", dtype=Int64),
        Field(name="item_ctr_smoothed", dtype=Float64),
        Field(name="cat_expanding_ctr", dtype=Float64),
        Field(name="item_age_hours", dtype=Float64),
        Field(name="category", dtype=String),
    ],
    source=item_stats_source,
    online=True,
)

user_stats_source = _source("user_hourly_features")

user_stats = FeatureView(
    name="user_stats",
    entities=[user],
    # Six hours rather than the item view's two. A user's recent activity ages
    # far more slowly than a news article's click rate, and a stale-by-an-hour
    # reader profile is a much smaller error than a stale-by-an-hour trending
    # count. Feast's own default is 24h, which on either would be too long.
    ttl=timedelta(hours=6),
    schema=[
        Field(name="user_impressions_24h", dtype=Int64),
        Field(name="user_clicks_24h", dtype=Int64),
        Field(name="user_ctr_smoothed", dtype=Float64),
        Field(name="user_tenure_hours", dtype=Float64),
    ],
    source=user_stats_source,
    online=True,
)

user_category_source = _source("user_category_cross_features")

user_category_stats = FeatureView(
    name="user_category_stats",
    entities=[user, category],
    ttl=timedelta(hours=24),
    schema=[
        Field(name="user_cat_impressions_cum", dtype=Int64),
        Field(name="user_cat_clicks_cum", dtype=Int64),
        Field(name="user_cat_affinity", dtype=Float64),
    ],
    source=user_category_source,
    online=True,
)

# TODO: Flink job pushes into this
#
# It is excluded from `make feast` materialisation: a push view has no batch
# rows to pull, and the batch_source below exists only because Feast requires
# one for schema inference and offline retrieval.
user_realtime = FeatureView(
    name="user_realtime",
    entities=[user],
    ttl=timedelta(hours=6),
    schema=[
        Field(name="last_50_items", dtype=Array(String)),
        Field(name="session_length", dtype=Int64),
        Field(name="session_categories", dtype=Array(String)),
    ],
    source=PushSource(name="user_rt_push", batch_source=user_stats_source),
    online=True,
)


# Context features: derived from the request timestamp, stored nowhere.
#
# Training computes these in Spark, inside attach_point_in_time_features. Serving
# would otherwise recompute them in Go, from a different clock in a different
# language -- two implementations of one definition, which is how an off-by-one
# hour reaches the ranker with nothing to catch it.
request_time = RequestSource(
    name="request_time",
    schema=[Field(name="request_ts", dtype=UnixTimestamp)],
    description="The instant the recommendation was requested.",
)


def _as_utc(value: datetime | int | float) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    return datetime.fromtimestamp(value, tz=UTC)


@on_demand_feature_view(  # type: ignore[untyped-decorator]
    sources=[request_time],
    schema=[
        Field(name="hour_of_day", dtype=Int64),
        Field(name="day_of_week", dtype=Int64),
    ],
    mode="python",
    description="Calendar features derived from the request timestamp, in UTC.",
)
def context_features(inputs: dict[str, Any]) -> dict[str, Any]:
    """Match Spark's ``hour`` and ``dayofweek`` exactly.

    Two conventions have to be honoured, and neither is the Python default:

    * **UTC.** Spark reads these with ``spark.sql.session.timeZone`` pinned to
      UTC, so a naive local-time conversion here shifts every hour by the
      developer's offset. That is the timezone bug this project already hit
      once, in a test, where it was visible. Here it would not be.
    * **Spark's week numbering.** ``dayofweek`` is 1=Sunday..7=Saturday, while
      Python's ``isoweekday`` is 1=Monday..7=Sunday. ``% 7 + 1`` maps one onto
      the other: Monday 1 -> 2, Sunday 7 -> 1.

    tests/test_context_features.py asserts this against Spark itself rather than
    against a restatement of these rules.
    """
    hours: list[int] = []
    days: list[int] = []

    for value in inputs["request_ts"]:
        moment = _as_utc(value)
        hours.append(moment.hour)
        days.append(moment.isoweekday() % 7 + 1)

    return {"hour_of_day": hours, "day_of_week": days}


@on_demand_feature_view(  # type: ignore[untyped-decorator]
    sources=[item_stats, user_stats, user_category_stats],
    schema=[
        Field(name="item_ctr_effective", dtype=Float64),
        Field(name="has_item_features", dtype=Bool),
        Field(name="has_user_features", dtype=Bool),
        Field(name="has_user_category_features", dtype=Bool),
    ],
    mode="python",
    description=(
        "Cold-start fallback and the missingness flags, matching what "
        "attach_point_in_time_features computes offline."
    ),
)
def derived_features(inputs: dict[str, Any]) -> dict[str, Any]:
    """Mirror the post-join block in ``attach_point_in_time_features``.

    A row where the item has no history AND its category has no prior yields
    None here. Training drops those rows; serving cannot drop a request, so
    we have to decide what to send the ranker later
    """
    item_ctr = inputs["item_ctr_smoothed"]
    category_ctr = inputs["cat_expanding_ctr"]

    return {
        "has_item_features": [value is not None for value in item_ctr],
        "has_user_features": [value is not None for value in inputs["user_ctr_smoothed"]],
        "has_user_category_features": [value is not None for value in inputs["user_cat_affinity"]],
        "item_ctr_effective": [
            item if item is not None else category
            for item, category in zip(item_ctr, category_ctr, strict=True)
        ],
    }


# user_realtime is absent because nothing writes it yet -- the Flink job that
# pushes into it will be implemented later
ranker_v1 = FeatureService(
    name="ranker_v1",
    features=[
        item_stats,
        user_stats,
        user_category_stats,
        context_features,
        derived_features,
    ],
    description=(
        "Features the ranker consumes, matching what build_training_examples "
        "attaches offline. Change this and the ranker's input width changes."
    ),
)
