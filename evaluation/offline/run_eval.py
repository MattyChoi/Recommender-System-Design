"""``make eval MODEL=<name>`` -- score a model through the one harness.

Reads the gold training examples for a split, scores them, labels cohorts
against train, and writes ``evaluation/results/<name>.json``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as f

from common.config import Settings, load_settings
from common.spark import get_spark
from common.utils import read_gold
from evaluation.offline.report import default_cohorts, report_card
from models.retrieval.baselines.als import score_als, score_als_item
from models.retrieval.baselines.content import score_content
from models.retrieval.baselines.covisit import score_covisit
from models.retrieval.baselines.popularity import (
    score_decayed_popular,
    score_most_popular,
    score_most_recent,
)

RESULTS = Path("evaluation/results")

# ts is needed by the decayed baseline, which measures a click's age from the
# label's own instant rather than from one global "now".
_NEEDED = ("user_id", "item_id", "user_idx", "item_idx", "impression_id", "clicked", "ts")


def _score(
    model: str,
    frame: DataFrame,
    train: DataFrame,
    seed: int,
    half_life: float,
    max_gap_seconds: float = 3600.0,
    catalogue: DataFrame | None = None,
) -> DataFrame:
    """Attach a ``score`` column.

    Every model goes through here, so every model is scored by the same harness
    over the same rows

    Args:
        model: Model name.
        frame: Evaluation rows.
        train: Training rows, for the baselines fitted on them. TRAIN ONLY:
            fitting popularity over train plus dev would let the baseline see
            the evaluation window, and it would look stronger for it.
        seed: Fixes the random scorer so a rerun reproduces the card.
        half_life: Days, for the decayed baseline.
        max_gap_seconds: Pairing window for co-visitation. The matrix is built
            from ``train`` alone; the user's prior clicks come from ``frame``
            too, since those are the request rather than the model.

    Returns:
        ``frame`` plus ``score``.

    Raises:
        NotImplementedError: For a model this harness cannot yet produce scores
            for. Naming it is better than scoring it wrongly.
    """
    if model == "random":
        return frame.withColumn("score", f.rand(seed=seed))
    if model == "popularity":
        return score_most_popular(frame, train)
    if model == "decayed_popularity":
        return score_decayed_popular(frame, train, half_life_days=half_life)
    if model == "content":
        if catalogue is None:
            raise ValueError("content similarity needs a catalogue of item titles")
        return score_content(frame, train, catalogue)
    if model == "als":
        return score_als(frame, train)
    if model == "als_item":
        return score_als_item(frame, train)
    if model == "recency":
        return score_most_recent(frame, train)
    if model == "covisit":
        return score_covisit(frame, train, max_gap_seconds=max_gap_seconds)

    raise NotImplementedError(
        f"no scorer for {model!r}. Available: random, popularity, "
        "decayed_popularity, recency, covisit, als, als_item, content."
    )


def _canonical_order(rows: pd.DataFrame) -> pd.DataFrame:
    """Row order that is a property of the data, not of the cluster.

    ``toPandas()`` returns Spark's partition order, and ``spark.master`` is
    ``local[*]`` -- however many cores this machine happens to have. Every
    consumer downstream resolves ties by input order and by nothing else: the
    three ``np.argsort(..., kind="stable")`` calls in ``metrics.py``,
    ``group_slates`` bucketing in first-seen order, ``_served_items``' stable
    ``list.sort``, and ``per_user_means``' dict, whose insertion order decides
    which index the bootstrap's draws refer to.

    So the same code over the same data disagreed with itself across machines.
    Measured rather than feared: ``local[*]`` against ``local[1]`` on one
    machine moved ``cold_item.ndcg@10`` by 0.0109 and ``cold_item.mrr`` by
    0.0120, where the models in ``docs/baselines.md`` are separated by about
    half that. GAUC and ``gauc_ceiling`` did not move at all, because
    ``impression_auc`` counts ties as half and never sorts.

    Note:
        This makes the metrics reproducible. It does not make them meaningful
        for a model that ties everything -- such a model still scores whatever
        the tie-break hands it, which is what ``gauc_ceiling`` is on the card to
        say.

    Args:
        rows: Collected evaluation rows, carrying ``impression_id`` and
            ``item_idx``.

    Returns:
        The same rows, deterministically ordered.
    """
    tiebreak = pd.util.hash_pandas_object(
        pd.DataFrame(
            {
                "impression_id": rows["impression_id"].astype(str),
                "item_idx": rows["item_idx"].astype("int64"),
            }
        ),
        index=False,
    ).to_numpy()

    return (
        rows.assign(_tiebreak=tiebreak)
        .sort_values(["impression_id", "_tiebreak"], kind="stable", ignore_index=True)
        .drop(columns="_tiebreak")
    )


def evaluate(
    spark: SparkSession,
    settings: Settings,
    model: str,
    split: str,
    seed: int,
    half_life: float = 3.0,
    max_gap_seconds: float = 3600.0,
) -> dict[str, Any]:
    """Build one report card end to end."""
    examples = read_gold(spark, settings, f"training_examples/{split}")
    train = read_gold(spark, settings, "training_examples/train")

    catalogue = (
        examples.select("item_id", "title")
        .unionByName(train.select("item_id", "title"))
        .groupBy("item_id")
        .agg(f.first("title", ignorenulls=True).alias("title"))
    )

    scored = _score(
        model,
        examples.select(*_NEEDED),
        train,
        seed,
        half_life,
        max_gap_seconds,
        catalogue,
    )

    warm_users = train.select("user_id").distinct().withColumn("_wu", f.lit(True))
    warm_items = train.select("item_id").distinct().withColumn("_wi", f.lit(True))
    labelled = (
        scored.join(f.broadcast(warm_items), on="item_id", how="left")
        .join(warm_users, on="user_id", how="left")
        .withColumn("is_cold_user", f.col("_wu").isNull())
        .withColumn("is_cold_item", f.col("_wi").isNull())
        .drop("_wu", "_wi")
    )

    # Ordered here, once, so every metric below inherits a row order that does
    # not depend on this machine's core count. Must precede default_cohorts:
    # the cohort masks are positional.
    rows = _canonical_order(labelled.toPandas())

    # The FULL item map, not the items dev happened to show.
    catalogue_size = spark.read.parquet(str(settings.paths.bronze / "item_map")).count()

    return report_card(
        model_name=model,
        scores=rows["score"].to_numpy(),
        labels=rows["clicked"].astype(int).to_numpy(),
        impression_ids=rows["impression_id"].astype(str).to_numpy(),
        user_ids=rows["user_id"].astype(str).to_numpy(),
        cohorts=default_cohorts(
            rows["is_cold_user"].to_numpy(),
            rows["is_cold_item"].to_numpy(),
            rows["impression_id"].astype(str).to_numpy(),
            rows["clicked"].astype(int).to_numpy(),
        ),
        split=split,
        item_ids=rows["item_id"].astype(str).to_numpy(),
        catalogue_size=catalogue_size,
    )


def write_card(card: dict[str, Any], destination: Path) -> dict[str, Any]:
    """Persist one card and echo its headline. Returns the overall cohort."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(card, indent=2) + "\n")

    overall: dict[str, Any] = card["cohorts"]["overall"]
    print(f"wrote {destination}")
    print(
        f"  overall: GAUC {overall['gauc']} · NDCG@10 {overall['ndcg@10']} "
        f"· {overall['impressions']:,} impressions, "
        f"{overall['skipped_impressions']:,} with no click"
    )

    honest, wrong = overall.get("ci95"), overall.get("ci95_by_impression_DO_NOT_QUOTE")
    if honest and wrong:
        print(
            f"  ndcg@10 95% CI over {overall['ci_users']:,} USERS:       "
            f"[{honest[0]}, {honest[1]}]  width {honest[1] - honest[0]:.4f}"
        )
        print(
            f"  ndcg@10 95% CI over {overall['ci_impressions']:,} IMPRESSIONS: "
            f"[{wrong[0]}, {wrong[1]}]  width {wrong[1] - wrong[0]:.4f}  <- WRONG unit"
        )
        print("  impressions from one user are correlated; the narrower interval is unearned.")
    return overall


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--half-life",
        type=float,
        default=3.0,
        help="days, for decayed_popularity; sweep it, the curve is a free ablation",
    )
    parser.add_argument(
        "--max-gap-seconds",
        type=float,
        default=3600.0,
        help="pairing window for covisit; sweep it, the curve is the deliverable",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    settings = load_settings()
    spark = get_spark(settings, app=f"eval-{args.model}")
    try:
        card = evaluate(
            spark,
            settings,
            args.model,
            args.split,
            args.seed,
            args.half_life,
            args.max_gap_seconds,
        )
    finally:
        spark.stop()

    write_card(card, args.out or RESULTS / f"{args.model}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
