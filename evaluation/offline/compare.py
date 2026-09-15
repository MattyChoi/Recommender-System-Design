"""``make compare`` -- the paired test, on two real models over the same users.

E3 built :func:`paired_bootstrap` and nothing called it. The machinery was
tested and then never pointed at a model, which is how a project ends up
quoting a 0.0006 difference as though someone had checked it.

Paired because variance across users dwarfs the effect size. On this corpus two
adjacent points on the half-life curve differ by ~0.005 NDCG while the per-user
interval on either one is ~0.005 wide, so an unpaired comparison cannot resolve
them. Scoring the SAME users under both systems and bootstrapping each user
against themselves cancels the between-user spread that is doing the hiding.

Both models are scored in ONE pass over one read, so the pairing is by
construction rather than by hoping two separate runs saw the same rows.
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
from evaluation.offline.metrics import group_slates, ndcg_at_k
from evaluation.offline.run_eval import _NEEDED, _score
from evaluation.offline.stats import PairedResult, paired_bootstrap, per_user_means


def parse_spec(spec: str) -> tuple[str, float | None]:
    """Split ``name@value`` into a model and its one tunable.

    The value's unit belongs to the model -- days for ``decayed_popularity``,
    seconds for ``covisit`` -- because a single shared unit would be a lie about
    one of them. ``recency`` and ``popularity`` take no value.

    Args:
        spec: ``name`` or ``name@value``.

    Returns:
        The model name and its parameter, or None.

    Raises:
        ValueError: If the value is present but not a number.
    """
    name, _, value = spec.partition("@")
    if not value:
        return name, None
    try:
        return name, float(value)
    except ValueError as exc:
        raise ValueError(f"{spec!r}: {value!r} is not a number") from exc


def score_spec(spec: str, frame: DataFrame, train: DataFrame, seed: int) -> DataFrame:
    """Score one side of the comparison, routed through the one harness.

    Goes through ``run_eval._score`` rather than calling a model directly, so a
    comparison can never score a model by a path the report card does not use.

    Raises:
        ValueError: If a value is supplied for a model that has no tunable.
            Silently ignoring it would let ``recency@0.02`` read as meaningful.
    """
    name, value = parse_spec(spec)
    if name == "decayed_popularity":
        return _score(name, frame, train, seed, value if value is not None else 3.0)
    if name == "covisit":
        return _score(name, frame, train, seed, 3.0, value if value is not None else 3600.0)
    if value is not None:
        raise ValueError(f"{spec!r}: {name} takes no parameter")
    return _score(name, frame, train, seed, 3.0)


def _user_ndcg(rows: Any, score_column: str, k: int) -> dict[str, float]:
    """Per-user mean NDCG from a collected frame."""
    slates = rows["impression_id"].astype(str).to_numpy()
    by_slate = group_slates(
        rows[score_column].to_numpy(), rows["clicked"].astype(int).to_numpy(), slates
    )

    owner: dict[Any, str] = {}
    for slate, user in zip(slates.tolist(), rows["user_id"].astype(str).tolist(), strict=True):
        owner.setdefault(slate, user)

    return per_user_means(
        [ndcg_at_k(lab, sc, k) for lab, sc in by_slate.values()],
        [owner[slate] for slate in by_slate],
    )


def compare(
    spark: SparkSession,
    settings: Settings,
    baseline: str,
    candidate: str,
    split: str = "dev",
    seed: int = 0,
    k: int = 10,
) -> tuple[PairedResult, int, int]:
    """Score both models over one read and bootstrap the per-user difference.

    Returns:
        The paired result, and each side's user count before pairing -- if those
        differ from the paired count, some users were scorable under one system
        and not the other, which changes what the test is about.
    """
    examples = read_gold(spark, settings, f"training_examples/{split}").select(*_NEEDED)
    train = read_gold(spark, settings, "training_examples/train")

    left = score_spec(baseline, examples, train, seed).select(
        "impression_id", "item_id", "user_id", "clicked", f.col("score").alias("score_a")
    )
    right = score_spec(candidate, examples, train, seed).select(
        "impression_id", "item_id", f.col("score").alias("score_b")
    )

    # Joined in Spark and collected ONCE. Two separate collects of 2.7M rows is
    # how the half-life sweep exhausted the JVM's socket buffers, and two
    # separate runs would pair rows by assumption rather than by key.
    rows = left.join(right, on=["impression_id", "item_id"], how="inner").toPandas()

    a = _user_ndcg(rows, "score_a", k)
    b = _user_ndcg(rows, "score_b", k)
    return paired_bootstrap(a, b, seed=seed), len(a), len(b)


def report(baseline: str, candidate: str, result: PairedResult, n_a: int, n_b: int, k: int) -> None:
    """Print the difference, its interval, and what it does and does not license."""
    print(f"\n  {candidate}  minus  {baseline}")
    print(f"  per-user NDCG@{k} difference: {result.difference:+.5f}")
    print(f"  95% CI: [{result.lo:+.5f}, {result.hi:+.5f}]  over {result.n_users:,} paired users")

    if n_a != result.n_users or n_b != result.n_users:
        print(
            f"  NOTE: {n_a:,} users scorable under the baseline and {n_b:,} under the "
            f"candidate; {result.n_users:,} under both. The test is about those."
        )

    if result.significant:
        better = candidate if result.difference > 0 else baseline
        print(f"  SIGNIFICANT: the interval excludes zero. {better} is ahead.")
    else:
        print("  NOT SIGNIFICANT: the interval spans zero. These two are not")
        print("  distinguishable on this split; report them as tied rather than")
        print("  ranking them by a point estimate.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="model, or model@param")
    parser.add_argument("--candidate", required=True, help="model, or model@param")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args(argv)

    settings = load_settings()
    spark = get_spark(settings, app="eval-compare")
    try:
        result, n_a, n_b = compare(
            spark, settings, args.baseline, args.candidate, args.split, args.seed, args.k
        )
    finally:
        spark.stop()

    report(args.baseline, args.candidate, result, n_a, n_b, args.k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
