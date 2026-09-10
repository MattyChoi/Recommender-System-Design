from pyspark.sql import types as t

# Index 0 is RESERVED and appears in no mapping table. It is the slot every
# unmapped string resolves to -- an article published after the maps were
# built, a user seen for the first time at serving time.
OOV_IDX: int = 0

# --- raw, as MIND ships it (tab-separated, no header) ---

BEHAVIORS_RAW = t.StructType(
    [
        t.StructField("impression_id", t.LongType(), False),
        t.StructField("user_id", t.StringType(), False),  # "U12345"
        t.StructField("time", t.StringType(), False),  # "11/11/2019 9:05:58 AM"
        t.StructField("history", t.StringType(), True),  # "N1 N2 N3", prior clicks
        t.StructField("impressions", t.StringType(), False),  # "N4-1 N5-0 N6-0"
    ]
)

NEWS_RAW = t.StructType(
    [
        t.StructField("item_id", t.StringType(), False),  # "N45678"
        t.StructField("category", t.StringType(), True),
        t.StructField("subcategory", t.StringType(), True),
        t.StructField("title", t.StringType(), True),
        t.StructField("abstract", t.StringType(), True),
        t.StructField("url", t.StringType(), True),
        t.StructField("title_ents", t.StringType(), True),  # JSON string
        t.StructField("abstract_ents", t.StringType(), True),  # JSON string
    ]
)


# --- the canonical contract, after exploding the impression list ---
# One row per (impression, item shown).

EVENT_SCHEMA = t.StructType(
    [
        t.StructField("impression_id", t.LongType(), False),
        t.StructField("user_id", t.StringType(), False),
        t.StructField("item_id", t.StringType(), False),
        t.StructField("clicked", t.BooleanType(), False),  # observed, not sampled
        t.StructField("slot", t.IntegerType(), False),  # slot within the list
        t.StructField("ts", t.TimestampType(), False),
    ]
)
