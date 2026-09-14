"""``make coverage`` -- how much of the split co-visitation can score at all.

A model that ranks badly and a model that has nothing to rank with produce
nearly identical report cards. Co-visitation at the manual's 1-hour window
scored GAUC 0.5007 against random's 0.5007 -- not a weak result, a silent one:
91% of slates had every candidate tied at 0.0, and a slate of ties contributes
exactly 0.5 to GAUC by construction.

This sweeps the pairing window and reports **reach** rather than quality, with
no report cards written. Building eight cards to discover that six of them are
empty is the expensive way to learn the same thing.

Why the window is the parameter that binds: train holds 106,965
inter-impression transitions and only 15.7% of them fall inside an hour, so the
default throws away 84% of the pairing opportunities before any modelling
happens.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as f

from common.config import Settings, load_settings
from common.spark import get_spark
from common.utils import read_gold
from models.retrieval.covisit import build_covisitation, score_covisit

# Needed by score_covisit; kept narrow so the cached frames stay small.
_NEEDED = ("user_id", "item_id", "impression_id", "clicked", "ts")

_HOUR = 3600.0


def measure(labels: DataFrame, train: DataFrame, max_gap_seconds: float) -> dict[str, Any]:
    """Reach of the matrix and of the scores it produces at one window.

    Four numbers, because they fail independently. A matrix can be large while
    reaching few items; it can reach many items that no evaluated user ever
    clicked; and rows can score while whole slates stay flat, which is what
    actually decides GAUC.

    Args:
        labels: Evaluation rows, already narrowed to ``_NEEDED``.
        train: Training rows. The matrix comes from these alone.
        max_gap_seconds: Pairing window under test.

    Returns:
        ``edges``, ``sources``, ``reachable`` (sources also present in the
        evaluated split), ``rows_scored`` and ``slates_scored``.
    """
    matrix = build_covisitation(train, max_gap_seconds=max_gap_seconds)
    edges = matrix.count()
    sources = matrix.select("item_id").distinct()

    # Intersect with the evaluated split's items, never divide by it. The matrix
    # is built from TRAIN, whose item set is the larger one, so a raw source
    # count over the dev catalogue is not a share at all -- it goes past 100%
    # and says nothing. What matters is how much of what we actually evaluate
    # the matrix can speak about.
    evaluated = labels.select("item_id").distinct()
    reachable = sources.join(evaluated, on="item_id", how="inner").count()

    scored = score_covisit(labels, train, max_gap_seconds=max_gap_seconds)
    rows = scored.agg(f.avg((f.col("score") > 0).cast("double")).alias("share")).first()

    # A slate where every candidate ties contributes exactly 0.5 to GAUC no
    # matter what the scores are, so slate reach -- not row reach -- is the
    # number that predicts whether the metric can move.
    slates = (
        scored.groupBy("impression_id")
        .agg(f.max("score").alias("best"))
        .agg(f.avg((f.col("best") > 0).cast("double")).alias("share"))
        .first()
    )

    return {
        "max_gap_seconds": max_gap_seconds,
        "edges": edges,
        "sources": sources.count(),
        "reachable": reachable,
        "rows_scored": float(rows["share"]),  # type: ignore[index]
        "slates_scored": float(slates["share"]),  # type: ignore[index]
    }


def report(rows: Sequence[dict[str, Any]], catalogue_size: int) -> None:
    """Print the reach curve, and the GAUC ceiling it implies.

    The ceiling is the honest headline. If only a fraction ``s`` of slates carry
    any signal, every other slate is a tie at 0.5, so GAUC cannot exceed
    ``0.5 * (1 - s) + s``. A model whose ceiling sits near 0.5 cannot be
    evaluated at that setting -- widening the window is not tuning, it is the
    difference between having a measurement and not.

    ``headroom`` is that ceiling less 0.5: the entire budget a perfect ranker
    would have to work with. Compare a measured GAUC against it rather than
    against 0.5, or a model that used 2% of its budget reads as one that merely
    underperformed.

    Args:
        rows: One measurement per window, in the order measured.
        catalogue_size: Distinct items in the evaluated split. Used only to
            express reach; it is never a denominator for the train-built matrix.
    """
    print(f"\n  {catalogue_size:,} distinct items in the evaluated split\n")
    print("   window     edges   sources   of split   rows   slates   ceiling   headroom")
    for row in rows:
        hours = row["max_gap_seconds"] / _HOUR
        reach = row["reachable"] / catalogue_size if catalogue_size else float("nan")
        ceiling = 0.5 * (1.0 - row["slates_scored"]) + row["slates_scored"]
        print(
            f"   {hours:>5,.0f}h   {row['edges']:>9,}   {row['sources']:>7,}   "
            f"{reach:>7.1%}   {row['rows_scored']:>5.2%}   "
            f"{row['slates_scored']:>6.2%}   {ceiling:>7.4f}   {ceiling - 0.5:>8.4f}"
        )
    print("\n  Reach, not quality: no report cards were written.")
    print("  Build cards only at windows whose ceiling leaves room to measure.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--windows",
        nargs="+",
        type=float,
        default=[3600.0, 21600.0, 86400.0, 259200.0],
        metavar="SECONDS",
        help="pairing windows to measure (default 1h 6h 24h 72h)",
    )
    parser.add_argument("--split", default="dev")
    args = parser.parse_args(argv)

    if any(value <= 0 for value in args.windows):
        parser.error("windows must be positive")

    settings: Settings = load_settings()
    spark: SparkSession = get_spark(settings, app="covisit-coverage")
    try:
        train = read_gold(spark, settings, "training_examples/train").select(*_NEEDED).cache()
        labels = (
            read_gold(spark, settings, f"training_examples/{args.split}").select(*_NEEDED).cache()
        )
        catalogue_size = labels.select("item_id").distinct().count()

        rows = [measure(labels, train, window) for window in args.windows]
    finally:
        spark.stop()

    report(rows, catalogue_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
