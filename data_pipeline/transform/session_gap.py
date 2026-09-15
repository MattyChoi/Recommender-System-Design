"""``make gap`` -- measure the inter-impression gap, so the session threshold is a
number from this corpus rather than a web-analytics convention for calculating the
session gap threshold for the silver data layer.
"""

from __future__ import annotations

import argparse
import itertools
from collections.abc import Sequence

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as f

from common.config import Settings, load_settings
from common.spark import get_spark

# Candidate thresholds, spanning the plausible range from "one sitting" to "one
# day". The elbow is read off the retention curve across these, not picked.
_CANDIDATES = (5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 120.0, 240.0, 720.0, 1440.0)

_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)

# How far the steepest density fall must beat the next-steepest before it is
# called a knee. Two is a judgement; it is named here so it can be argued with.
_DOMINANCE = 2.0


def impression_gaps(events: DataFrame) -> DataFrame:
    """Minutes between each user's consecutive impressions.

    One row per *transition*, so a user with a single impression contributes
    nothing and a user with n impressions contributes n-1. That is the right
    denominator: the threshold decides whether a transition is a boundary, and
    users who never transition cannot inform it.

    Bronze events are one row per item shown, so impressions are deduplicated
    first -- otherwise every impression's row count would weight it, and a
    40-item page view would count forty times more than a 1-item one.

    Args:
        events: Bronze events with ``user_id``, ``impression_id`` and ``ts``.

    Returns:
        ``user_id`` and ``gap_minutes``, one row per transition.
    """
    impressions = events.select("user_id", "impression_id", "ts").dropDuplicates(
        ["user_id", "impression_id"]
    )
    # Ordered by ts, never by impression_id: MIND's impression ids are not
    # chronological, and ordering by one would measure a shuffled timeline.
    ordered = Window.partitionBy("user_id").orderBy("ts")
    return (
        impressions.withColumn("prev_ts", f.lag("ts").over(ordered))
        .filter(f.col("prev_ts").isNotNull())
        .withColumn(
            "gap_minutes",
            (f.unix_timestamp("ts") - f.unix_timestamp("prev_ts")) / 60.0,
        )
        .select("user_id", "gap_minutes")
    )


def describe(gaps: DataFrame) -> dict[str, float]:
    """Quantiles plus, for each candidate threshold, the share of transitions it keeps.

    The retention share is the number that matters. A threshold is a claim about
    which transitions are "the same sitting"; the curve of that share across
    candidates is where the elbow shows, and a quantile table alone hides it.

    Args:
        gaps: Output of :func:`impression_gaps`.

    Returns:
        Quantiles keyed ``p05``..``p99`` and retention keyed ``keep_<n>m``.

    Raises:
        ValueError: If no user has two impressions, which would make every
            session a singleton and the threshold meaningless.
    """
    gaps = gaps.cache()
    total = gaps.count()
    if total == 0:
        raise ValueError("no user has two impressions; sessions would all be singletons")

    # approxQuantile with a small relative error: exact quantiles need a full
    # sort of every transition, and the elbow is not a fourth-decimal question.
    values = gaps.approxQuantile("gap_minutes", list(_QUANTILES), 0.001)
    out = {f"p{int(q * 100):02d}": value for q, value in zip(_QUANTILES, values, strict=True)}

    kept = gaps.select(
        *[
            f.avg((f.col("gap_minutes") <= threshold).cast("double")).alias(f"keep_{threshold:g}m")
            for threshold in _CANDIDATES
        ]
    ).first()
    # A global aggregate always yields exactly one row, even over no input.
    assert kept is not None
    out.update({name: float(kept[name]) for name in kept.asDict()})

    out["transitions"] = float(total)
    out["users_with_gaps"] = float(gaps.select("user_id").distinct().count())
    gaps.unpersist()
    return out


def report(stats: dict[str, float]) -> None:
    """Print the quantiles, the retention curve, and whether it has a knee at all."""
    print(
        f"\n  {stats['transitions']:,.0f} transitions from "
        f"{stats['users_with_gaps']:,.0f} users with 2+ impressions\n"
    )

    print("  gap quantiles (minutes)")
    for q in _QUANTILES:
        print(f"    p{int(q * 100):02d}  {stats[f'p{int(q * 100):02d}']:>10,.1f}")

    # Share per minute of bucket width, NOT raw marginal gain. The candidate
    # thresholds are not evenly spaced -- 720m-1440m is a 720-minute bucket
    # against a 5-minute one -- so raw gain rises with bucket width whatever the
    # data does, and reading an elbow off it finds one that is not there.
    print("\n  threshold   transitions kept   share/min of bucket")
    previous_share, previous_edge = 0.0, 0.0
    densities: dict[float, float] = {}
    for threshold in _CANDIDATES:
        share = stats[f"keep_{threshold:g}m"]
        densities[threshold] = (share - previous_share) / (threshold - previous_edge)
        print(f"    {threshold:>6,.0f}m   {share:>13.1%}   {densities[threshold]:>16.3%}")
        previous_share, previous_edge = share, threshold

    # A knee is a bucket where density falls off a cliff. Carry the index with
    # each drop: the `before > 0` guard can skip an entry, and a positional
    # lookup into _CANDIDATES afterwards would then name the wrong threshold.
    ordered = [densities[t] for t in _CANDIDATES]
    drops = [
        ((before - after) / before, i)
        for i, (before, after) in enumerate(itertools.pairwise(ordered))
        if before > 0
    ]
    if not drops:
        print("\n  Every bucket is empty; nothing to read a threshold off.")
        return

    ranked = sorted(drops, reverse=True)
    steepest, index = ranked[0]
    runner_up, runner_index = ranked[1] if len(ranked) > 1 else (0.0, index)

    # _CANDIDATES[index], not [index + 1]: drops[i] compares the bucket ENDING at
    # _CANDIDATES[i] with the next one, so the collapse happens just PAST
    # _CANDIDATES[i]. The threshold that keeps the behaviour before the cliff is
    # the near side of it; naming the far side puts the boundary inside the
    # regime the cliff separates out.
    print(f"\n  sharpest fall: density drops {steepest:.0%} just past {_CANDIDATES[index]:g}m")

    # Dominance over the RUNNER-UP, not over the mean. A mean is dragged down by
    # the long flat tail, so almost any curve clears a multiple of it; comparing
    # the two steepest falls asks the question that matters -- is there one
    # cliff, or several comparable steps?
    if runner_up > 0:
        ratio = steepest / runner_up
        print(
            f"  that is {ratio:.1f}x the next-sharpest ({runner_up:.0%} past "
            f"{_CANDIDATES[runner_index]:g}m)"
        )
    else:
        ratio = float("inf")
        print("  and it is the only fall in the curve")

    # The rule is printed with the verdict so a reader can disagree with it
    # rather than having to reverse-engineer it from the source.
    if ratio >= _DOMINANCE:
        print(
            f"  Decisive by the {_DOMINANCE:g}x rule: {_CANDIDATES[index]:g}m is the "
            "threshold this data picks out."
        )
    else:
        print(f"  NOT decisive by the {_DOMINANCE:g}x rule -- the decay has no single")
        print("  knee, so no threshold is picked out. Choose one on grounds you can")
        print("  state, and record this measurement beside it.")
    print("  Changing session.gap_minutes in conf/config.yml requires `make silver` again.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", help="measure on train; dev is the holdout")
    args = parser.parse_args(argv)

    settings: Settings = load_settings()
    spark = get_spark(settings, app="session-gap")
    try:
        events = spark.read.parquet(str(settings.paths.bronze / "events" / args.split))
        report(describe(impression_gaps(events)))
    finally:
        spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
