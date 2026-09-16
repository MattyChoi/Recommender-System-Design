"""Point-in-time user click history, for the retrieval user tower.

D1 defers "user sequence" to Part H because MIND's ``history`` is a constant
per-user snapshot (C4: 0 of 100,000 users have it vary) that predates the log
window, so used raw it is stale by up to six days. The two-tower's user tower
needs a sequence anyway -- mean-pooling item embeddings over it is half of what
the tower is -- so this builds the honest version: the snapshot, topped up with
the clicks the user actually made **earlier in the window than this request**.

Transforms only. Every function here takes and returns DataFrames; ``gold.py``
does the reading and the writing, as it does for the three hourly series.

Two properties make this safe, and both are easy to get wrong.

**RANGE, not ROWS.** The window frame is
``rangeBetween(Window.unboundedPreceding, -1)`` over ``ts`` in whole seconds,
which admits rows whose timestamp is at most ``current - 1`` -- strictly
earlier, since MIND stamps to the second. Note this is the *opposite* of
``asof.py``, which is deliberately ROWS. The two want opposite frames for the
same underlying reason: the as-of join needs its feature-before-label tie-break
honoured, so it must exclude peers from the frame and rank them itself; here
peers must not be admitted at all. Getting either backwards is silent.

**Why peers are excluded rather than ranked.** Every row of an impression
shares one ``ts``, so a frame that admitted peers would put an impression's own
clicks into its own history -- direct label leakage, and it would produce a
beautiful, meaningless NDCG with no error anywhere. Measured: 17 (user, ts)
pairs in train and 5 in dev carry *two* impressions (34 and 10 impressions, so
0.022% and 0.014%), so a same-second collision is rare but real, and keying the
timeline on (user, ts) alone would have leaked across exactly those. Equal
timestamps establish no ordering, so neither impression sees the other.

**Grain: one row per impression.**
"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as f

MAX_HISTORY = 50


def snapshot_history(history: DataFrame, item_map: DataFrame) -> DataFrame:
    """The per-user pre-window click snapshot, as item indices.

    ``bronze/history`` is one row per impression, but the string is constant
    per user (C4), so this collapses to one row per user. Any variation would be
    silently dropped, which is why ``test_user_history.py`` asserts the
    invariant rather than trusting the measurement to stay true.

    Note:
        **Ordering inside the snapshot is an assumption, not a measurement.**
        MIND ships the history as a bare space-separated list with no
        timestamps, so "later in the string is more recent" is a convention
        about its serialisation that this corpus gives no way to verify. It
        only affects which items survive the ``max_len`` cut, and the cut
        almost never binds here.

    Args:
        history: ``user_id`` and a space-separated ``history`` string.
        item_map: ``item_id`` to ``item_idx``.

    Returns:
        ``user_id``, ``snapshot_idx`` (most recent first), ``snapshot_len``.
    """
    exploded = (
        history.select("user_id", "history")
        .where(f.col("history").isNotNull() & (f.trim(f.col("history")) != ""))
        .dropDuplicates(["user_id"])
        .select(
            "user_id",
            f.posexplode(f.split(f.trim(f.col("history")), r"\s+")).alias("pos", "item_id"),
        )
    )

    # An inner join DROPS history items absent from news.tsv rather than
    # mapping them to OOV_IDX.
    mapped = exploded.join(item_map, on="item_id", how="inner")

    return (
        mapped.groupBy("user_id")
        .agg(f.sort_array(f.collect_list(f.struct("pos", "item_idx")), asc=False).alias("ordered"))
        .select(
            "user_id",
            f.col("ordered.item_idx").alias("snapshot_idx"),
            f.size("ordered").alias("snapshot_len"),
        )
    )


def in_window_history(clicks: DataFrame, requests: DataFrame) -> DataFrame:
    """Each request's view of the user's earlier in-window clicks.

    Clicks and requests are unioned into one timeline so a single window pass
    answers "what had this user clicked before now" for every request,
    including requests at instants where nothing was clicked. ``collect_list``
    ignores nulls, so request rows contribute nothing to the accumulation they
    read from.

    Args:
        clicks: Clicked rows across every split, with ``user_id``,
            ``item_idx`` and ``ts``.
        requests: Distinct ``impression_id``, ``user_id``, ``ts``.

    Returns:
        ``impression_id``, ``inwindow_idx`` (most recent first), ``inwindow_len``.
    """
    clicked_at = (
        clicks.select("user_id", "item_idx", f.unix_timestamp("ts").alias("ts_sec"))
        .groupBy("user_id", "ts_sec")
        .agg(f.collect_list("item_idx").alias("at_ts"))
        .withColumn("impression_id", f.lit(None).cast("long"))
    )

    asked = requests.select(
        "impression_id",
        "user_id",
        f.unix_timestamp("ts").alias("ts_sec"),
        f.lit(None).cast("array<int>").alias("at_ts"),
    )

    # RANGE, not ROWS -- see the module docstring. `-1` is one SECOND, which is
    # MIND's timestamp resolution, so this is exactly `strictly earlier`.
    earlier = (
        Window.partitionBy("user_id").orderBy("ts_sec").rangeBetween(Window.unboundedPreceding, -1)
    )

    timeline = asked.unionByName(clicked_at).withColumn(
        "prior", f.flatten(f.collect_list("at_ts").over(earlier))
    )

    return timeline.where(f.col("impression_id").isNotNull()).select(
        "impression_id",
        # The window accumulates in frame order, i.e. oldest first.
        f.reverse(f.col("prior")).alias("inwindow_idx"),
        f.size("prior").alias("inwindow_len"),
    )


def assemble_history(
    requests: DataFrame,
    in_window: DataFrame,
    snapshot: DataFrame,
    max_len: int = MAX_HISTORY,
) -> DataFrame:
    """Join the two sources into one capped, most-recent-first sequence.

    Both joins are LEFT: a request with no in-window clicks and no snapshot is a
    real state (88% of dev's users are new), and it must come back with an empty
    history rather than vanish from the table.

    Args:
        requests: ``impression_id``, ``user_id``, ``ts``.
        in_window: ``impression_id``, ``inwindow_idx``, ``inwindow_len``.
        snapshot: ``user_id``, ``snapshot_idx``, ``snapshot_len``.
        max_len: Entries kept, most recent first.

    Returns:
        One row per request with ``history_idx``, ``history_len``,
        ``inwindow_len``, ``snapshot_len`` and ``has_history``.
    """
    empty_int = f.array().cast("array<int>")

    return (
        requests.join(in_window, on="impression_id", how="left")
        .join(snapshot, on="user_id", how="left")
        .select(
            "impression_id",
            f.slice(
                f.concat(
                    f.coalesce(f.col("inwindow_idx"), empty_int),
                    f.coalesce(f.col("snapshot_idx"), empty_int),
                ),
                1,
                max_len,
            ).alias("history_idx"),
            f.coalesce(f.col("inwindow_len"), f.lit(0)).alias("inwindow_len"),
            f.coalesce(f.col("snapshot_len"), f.lit(0)).alias("snapshot_len"),
        )
        .withColumn("history_len", f.size("history_idx"))
        .withColumn("has_history", f.col("history_len") > 0)
    )


def point_in_time_history(
    requests: DataFrame,
    clicks: DataFrame,
    history: DataFrame,
    item_map: DataFrame,
    max_len: int = MAX_HISTORY,
) -> DataFrame:
    """The whole transform, for ``gold.py`` to read into and write out of.

    Args:
        requests: Distinct ``impression_id``, ``user_id``, ``ts`` for the split
            being keyed.
        clicks: Clicked rows across EVERY split. A dev impression by a user who
            read during the train week should see those clicks, because the
            system serving it would have; the official boundary is temporally
            clean (eight seconds of gap), so this cannot reach forward.
        history: Bronze ``history`` rows across every split.
        item_map: ``item_id`` to ``item_idx``.
        max_len: Entries kept per request, most recent first.

    Returns:
        One row per request, keyed on ``impression_id``.
    """
    return assemble_history(
        requests,
        in_window_history(clicks, requests),
        snapshot_history(history, item_map),
        max_len,
    )
