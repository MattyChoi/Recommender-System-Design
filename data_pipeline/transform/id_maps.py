"""Stable string -> contiguous-int mappings, written once and read forever.

"Written once" is load-bearing and is enforced here rather than left to
discipline. An index is only meaningful relative to the checkpoint trained
against it: renumber the map and a saved embedding table silently addresses
the wrong rows. The model still loads, still runs, and still returns
recommendations -- it is simply reading a different item's vector every time,
and the only symptom is metrics you will blame on the model.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as f

from common.config import Settings
from common.schemas import OOV_IDX


def _maps_exist(settings: Settings) -> bool:
    """Whether both mapping tables hold a committed write."""
    return all(
        (settings.paths.bronze / name / "_SUCCESS").is_file() for name in ("user_map", "item_map")
    )


def build_id_maps(
    events: DataFrame, news: DataFrame, settings: Settings, force: bool = False
) -> None:
    """Write user_map and item_map to bronze.

    Indices run from 1, not 0: :data:`common.schemas.OOV_IDX` owns 0 and
    appears in neither table.

    Args:
        events: Union of every split's bronze events. Source of the user ids.
        news: Union of every split's bronze news. Source of the item ids.
        settings: The root configuration object.
        force: Overwrite maps that already exist. Off by default, because
            overwriting them invalidates every checkpoint trained against
            them.

    Note:
        Do NOT use monotonically_increasing_id(). It is unique but not contiguous
        -- it encodes the partition number in the high bits, blowing up the range
        of IDs and making them non-contiguous.

        row_number() over an unpartitioned window collapses to a single
        partition. That is acceptable at ~1M users and ~160K items, and would
        not be at 100x that; the replacement there is RDD.zipWithIndex().
    """
    if _maps_exist(settings) and not force:
        _warn_if_uncovered(events, news, settings)
        print("id maps: already built, keeping existing indices (rebuild: --force)")
        return

    user_map = (
        events.select("user_id")
        .distinct()
        .withColumn("user_idx", f.row_number().over(Window.orderBy("user_id")))
    )
    # From `news`, i.e. the whole catalogue -- not from the items actually
    # shown. ~63% of these indices address articles no impression ever
    # contained, so their embeddings never receive a gradient. Accepted
    # deliberately: a catalogue-derived map changes only when the catalogue
    # does, whereas an events-derived one would be renumbered by any ingest
    # change and invalidate every checkpoint.
    item_map = (
        news.select("item_id")
        .distinct()
        .withColumn("item_idx", f.row_number().over(Window.orderBy("item_id")))
    )

    user_map.write.mode("overwrite").parquet(str(settings.paths.bronze / "user_map"))
    item_map.write.mode("overwrite").parquet(str(settings.paths.bronze / "item_map"))


def _warn_if_uncovered(events: DataFrame, news: DataFrame, settings: Settings) -> None:
    """Say so, loudly, when kept maps do not cover the data now present.

    This is the cost of refusing to overwrite. Add a split and the new ids have
    no index, so every row carrying one collapses to OOV -- which is wrong, but
    wrong in a way the contract tests catch, unlike a silent renumbering.
    """
    spark = events.sparkSession
    users = spark.read.parquet(str(settings.paths.bronze / "user_map"))
    items = spark.read.parquet(str(settings.paths.bronze / "item_map"))

    new_users = events.select("user_id").distinct().join(users, "user_id", "left_anti").count()
    new_items = news.select("item_id").distinct().join(items, "item_id", "left_anti").count()
    if new_users or new_items:
        print(
            f"WARNING: id maps do not cover the current data -- {new_users} users and "
            f"{new_items} items have no index and will resolve to OOV ({OOV_IDX}).\n"
            "         Rebuild with `make bronze FORCE=1`, then RETRAIN: rebuilt maps "
            "renumber everything and invalidate existing checkpoints."
        )
