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

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as f

from common.config import Settings, load_settings
from common.spark import get_spark
from common.utils import read_gold
from evaluation.offline.report import default_cohorts, report_card

RESULTS = Path("evaluation/results")

_NEEDED = ("user_id", "item_id", "impression_id", "clicked")


def _score(model: str, frame: DataFrame, seed: int) -> DataFrame:
    """Attach a ``score`` column.

    Args:
        model: Model name. ``random`` is implemented here; every other name is
            expected to come from Part F onwards.
        frame: Evaluation rows.
        seed: Fixes the random scorer so a rerun reproduces the card.

    Returns:
        ``frame`` plus ``score``.

    Raises:
        NotImplementedError: For a model this harness cannot yet produce scores
            for. Naming it is better than scoring it wrongly.
    """
    if model == "random":
        return frame.withColumn("score", f.rand(seed=seed))

    raise NotImplementedError(
        f"no scorer for {model!r}. 'random' is available now; popularity, "
        "co-visitation and the learned models arrive in Part F onwards."
    )


def evaluate(
    spark: SparkSession, settings: Settings, model: str, split: str, seed: int
) -> dict[str, Any]:
    """Build one report card end to end."""
    examples = read_gold(spark, settings, f"training_examples/{split}")
    train = read_gold(spark, settings, "training_examples/train")

    scored = _score(model, examples.select(*_NEEDED), seed)

    warm_users = train.select("user_id").distinct().withColumn("_wu", f.lit(True))
    warm_items = train.select("item_id").distinct().withColumn("_wi", f.lit(True))
    labelled = (
        scored.join(f.broadcast(warm_items), on="item_id", how="left")
        .join(warm_users, on="user_id", how="left")
        .withColumn("is_cold_user", f.col("_wu").isNull())
        .withColumn("is_cold_item", f.col("_wi").isNull())
        .drop("_wu", "_wi")
    )

    rows = labelled.toPandas()

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
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    settings = load_settings()
    spark = get_spark(settings, app=f"eval-{args.model}")
    try:
        card = evaluate(spark, settings, args.model, args.split, args.seed)
    finally:
        spark.stop()

    destination = args.out or RESULTS / f"{args.model}.json"
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
