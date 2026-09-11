"""One definition, served offline for training and online for inference.

Feast parses every .py file under the feature repo directory, so this file
must live BESIDE feature_store.yaml -- not up in data_pipeline/features/,
where the CLI will never look at it.
"""

from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float64, Int64, String

# The CLI chdirs into this directory before running, and load_settings() reads
# "conf/config.yml" relative to the working directory -- so settings cannot be
# used here. Anchor on __file__ instead, which is stable no matter who calls.
#
#   feature_repo/ -> recsys_store/ -> features/ -> data_pipeline/ -> repo root
PROJECT_ROOT = Path(__file__).resolve().parents[4]
GOLD = PROJECT_ROOT / "data" / "gold"

item = Entity(name="item_id", value_type=ValueType.STRING)
user = Entity(name="user_id", value_type=ValueType.STRING)
category = Entity(name="category", value_type=ValueType.STRING)

item_stats_source = FileSource(
    path=str(GOLD / "item_hourly_features"),
    timestamp_field="feature_ts",  # what makes the as-of join possible
)

item_stats = FeatureView(
    name="item_stats",
    entities=[item],
    ttl=timedelta(hours=2),
    schema=[
        Field(name="item_impressions_24h", dtype=Int64),
        Field(name="item_clicks_24h", dtype=Int64),
        Field(name="item_ctr_smoothed", dtype=Float64),
        Field(name="cat_expanding_ctr", dtype=Float64),
        Field(name="item_age_hours", dtype=Float64),
        Field(name="category", dtype=String),
    ],
    source=item_stats_source,
    online=True,
)

user_stats_source = FileSource(
    path=str(GOLD / "user_hourly_features"),
    timestamp_field="feature_ts",
)

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

user_category_source = FileSource(
    path=str(GOLD / "user_category_cross_features"),
    timestamp_field="feature_ts",
)

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
