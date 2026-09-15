"""``make sweep`` -- the decayed-popularity half-life curve.

Separate from ``run_eval`` because it answers a different question. ``run_eval``
scores one model and produces one comparable record. This produces an
**ablation**: the same model at several settings, where the shape across
settings is the result and no single row is.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

from pyspark.sql import SparkSession

from common.config import Settings, load_settings
from common.spark import get_spark
from evaluation.offline.run_eval import RESULTS, evaluate, write_card

MODEL = "decayed_popularity"


def _slug(half_life: float) -> str:
    """Filename-safe half-life, so 0.25 becomes ``0p25`` rather than a dot.

    A dot in the stem reads as an extension to half the tooling that will ever
    glob this directory.
    """
    return f"{half_life:g}".replace(".", "p")


def sweep_half_life(
    spark: SparkSession,
    settings: Settings,
    values: Sequence[float],
    split: str = "dev",
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Score decayed popularity at each half-life, writing one card per value.

    Each card is written to ``decayed_popularity_hl<value>.json`` and carries
    ``half_life_days`` in its own body -- so a file can never be separated from
    the parameter that produced it, which is what happens the first time someone
    copies one out of the directory.

    The Spark session is shared across values. Building one per half-life would
    add session startup and jar resolution to every point on the curve and
    dominate the actual work.

    Args:
        spark: An active session, reused across values.
        settings: The root configuration object.
        values: Half-lives in days.
        split: Evaluation split.
        seed: Carried through to :func:`evaluate`; unused by this model.

    Returns:
        One summary row per value, in the order given.
    """
    summaries: list[dict[str, Any]] = []

    for half_life in values:
        card = evaluate(spark, settings, MODEL, split, seed, half_life)
        card["half_life_days"] = half_life
        overall = write_card(card, RESULTS / f"{MODEL}_hl{_slug(half_life)}.json")
        summaries.append(
            {
                "half_life_days": half_life,
                "gauc": overall["gauc"],
                "ndcg@10": overall["ndcg@10"],
                "mrr": overall["mrr"],
            }
        )

    return summaries


def print_curve(summaries: Sequence[dict[str, Any]]) -> None:
    """Print the curve, and say what it does and does not license.

    The best row is highlighted but explicitly not offered as the answer: if the
    metric is still climbing at the edge of the grid, the grid is wrong, and
    that is only visible from the whole curve.
    """
    if not summaries:
        return

    print("\n  half-life   GAUC     NDCG@10   MRR")
    for row in summaries:
        print(
            f"  {row['half_life_days']:>7g}d   {row['gauc']:.4f}   "
            f"{row['ndcg@10']:.4f}    {row['mrr']:.4f}"
        )

    best = max(summaries, key=lambda r: r["gauc"])
    edges = (summaries[0]["half_life_days"], summaries[-1]["half_life_days"])
    print(f"\n  best GAUC at {best['half_life_days']:g}d")

    if best["half_life_days"] in edges:
        print(
            "  WARNING: the best value sits at the edge of the grid, so the "
            "optimum may lie outside it. Extend the range before quoting this."
        )
    print("  Report the curve, not just the best row.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--half-lives",
        nargs="+",
        type=float,
        required=True,
        metavar="DAYS",
        help="half-lives to score, in days",
    )
    parser.add_argument("--split", default="dev")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if any(value <= 0 for value in args.half_lives):
        parser.error("half-lives must be positive")

    settings = load_settings()
    spark = get_spark(settings, app="eval-sweep")
    try:
        summaries = sweep_half_life(spark, settings, args.half_lives, args.split, args.seed)
    finally:
        spark.stop()

    print_curve(summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
