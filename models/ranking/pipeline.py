"""Everything the two ranking entry points do identically.

There are two trainers -- a booster under ``baselines`` and the neural rankers
beside it -- and exactly one of their jobs differs: fitting. Loading the gold
tables, replaying retrieval into a candidate table, splitting it by user,
scoring the funnel, pairing against the order retrieval already produced, and
recording the run are the same work in both, and **they have to stay the same
work** or the head-to-head that Part K turns on stops being a head-to-head.

So that half lives here. A trainer supplies scores and gets a report; if it
wants a different denominator it has to change this file, where the change is
visible, rather than drift into one of its own.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch

from common.config import Settings, load_settings
from common.spark import get_spark
from common.torch_env import describe, select_device
from common.tracking import provenance, require_reachable, run_label, track
from data_pipeline.features.user_history import MAX_HISTORY
from evaluation.offline.stats import paired_bootstrap, per_user_means
from models.ranking.calibration import calibrate
from models.ranking.calibration import render as render_calibration
from models.ranking.compare import RESULTS as RUNS
from models.ranking.compare import Run
from models.ranking.dataset import (
    FEATURES,
    RankingRows,
    build,
    holdout_mask,
    split_by_user,
)
from models.ranking.evaluate import evaluate_scores, per_request_ndcg, render
from models.retrieval.dataloader.dataset import (
    CONTENT_VARIANTS,
    load_item_tables,
    load_train_and_validation,
    prior_window_counts,
)
from models.retrieval.evaluate import load_tower
from models.retrieval.sources import RESULTS, load_retrieved

CHECKPOINTS = Path("data/checkpoints/ranking")

# Gold tables this stage reads, for the dataset digest. The retrieved candidates
# are NOT in it: they are an artifact of a checkpoint, and that checkpoint's own
# hash is in `retriever`.
TABLES = ["training_examples", "user_history", "impression_negatives", "item_content"]


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """The flags that describe the DATA, which both trainers must share.

    A flag added to one trainer and not the other is a silent change of
    denominator between two arms that are meant to be compared.
    """
    parser.add_argument("checkpoint", type=Path, help="The retriever whose candidates are ranked.")
    parser.add_argument("--sources", type=Path, default=RESULTS)
    parser.add_argument("--names", nargs="+", default=["two_tower", "trending"])
    parser.add_argument("--k", type=int, default=10, help="NDCG cutoff.")
    parser.add_argument("--max-candidates", type=int, default=100)
    parser.add_argument(
        "--quota",
        type=int,
        nargs="+",
        default=None,
        help="Slots per source, in --names order. Default gives them all to the "
        "first; the rest still contribute rank features.",
    )
    parser.add_argument(
        "--drop",
        nargs="+",
        default=(),
        help="Feature names to leave out. A large gain share is a claim; "
        "dropping the column and re-measuring is the test of it.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Save per-request scores under this name so another run can be "
        "PAIRED against it. Two marginal intervals are not a comparison.",
    )
    parser.add_argument("--runs", type=Path, default=RUNS)
    parser.add_argument("--holdout", type=float, default=0.3, help="Share of USERS held out.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variant", choices=CONTENT_VARIANTS, default=CONTENT_VARIANTS[0])
    parser.add_argument("--max-history", type=int, default=MAX_HISTORY)
    parser.add_argument("--max-negs", type=int, default=20, help="Slate width for `observed`.")
    parser.add_argument("--holdout-hours", type=int, default=12)


def arm(model: str, args: argparse.Namespace) -> str:
    """A short name for what this run is.

    Carries the pieces that change the model: which learner, which sources fed
    it candidates, how the slots were divided, and which features were dropped.
    Anything that changes the run has to change the name, or two runs land on
    one artifact and the second silently wins.
    """
    sources = "+".join(args.names)
    quota = "" if not args.quota else "-q" + ".".join(str(value) for value in args.quota)
    dropped = "" if not args.drop else "-no" + ".".join(sorted(args.drop))
    return f"{model}-{sources}-c{args.max_candidates}{quota}{dropped}"


@dataclass(frozen=True)
class Prepared:
    """The candidate table and everything needed to record a run over it."""

    settings: Settings
    device: torch.device
    marks: dict[str, str]
    run_name: str
    rows: RankingRows
    fitting: RankingRows
    held: RankingRows


def prepare(args: argparse.Namespace, model: str) -> Prepared:
    """Load gold, replay retrieval into candidates, and split by user."""
    settings = load_settings()
    device = select_device()

    marks = {**provenance(settings, vars(args), TABLES), "retriever": args.checkpoint.stem}
    run_name = run_label(arm(model, args), marks)
    print(f"{run_name}: {describe(device)}")
    if marks["git_dirty"] == "true":
        print("  WARNING: the working tree is dirty; this run's git_sha does not describe it")

    # Before the Spark read, not after: a missing tracking server should cost a
    # second, not several minutes of loading followed by a connection error.
    require_reachable(settings)

    spark = get_spark(settings, app="ranking")
    try:
        items = load_item_tables(settings, args.variant)
        n_rows = int(items.content.shape[0])
        train_split, val_split = load_train_and_validation(
            spark, settings, args.max_history, args.max_negs, args.holdout_hours
        )
        prior = prior_window_counts(spark, settings, args.holdout_hours, n_rows).numpy()
    finally:
        spark.stop()

    tower = load_tower(args.checkpoint, items, train_split.user_feats.shape[1], device)
    sources = {name: load_retrieved(name, args.sources) for name in args.names}
    train_clicks = np.bincount(train_split.item_ids.numpy(), minlength=n_rows).astype("int64")

    rows = build(
        tower,
        items,
        val_split,
        sources,
        train_clicks,
        prior,
        device,
        args.max_candidates,
        args.quota,
        [name for name in FEATURES if name not in set(args.drop)],
    )
    print(
        f"  {len(rows.groups):,} requests · {len(rows.labels):,} candidate rows · "
        f"retrieval ceiling {rows.recall_ceiling:.4f}"
    )
    print(
        f"  {rows.observed_share:.1%} of negatives were actually shown to the user; "
        "the rest are assumed"
    )

    fitting, held = split_by_user(rows, args.holdout, args.seed)
    print(f"  {len(fitting.groups):,} requests to fit, {len(held.groups):,} held out by user\n")

    return Prepared(settings, device, marks, run_name, rows, fitting, held)


def report(
    args: argparse.Namespace,
    prepared: Prepared,
    model: str,
    scores: npt.NDArray[np.float64],
    ranked_by: list[tuple[str, float]],
    save_model: Callable[[Path], Path],
    extra: Mapping[str, float] | None = None,
    trace: list[dict[str, float]] | None = None,
) -> int:
    """Score, print, log and save. The identical half of both trainers.

    Args:
        model: Names the MLflow experiment, so the experiment list reads as the
            set of models this project has.
        scores: One per candidate row of ``prepared.held``.
        ranked_by: Feature importances, or empty for a learner with none.
        save_model: Writes the fitted model into the directory it is given and
            returns the path. Passed in rather than branched on here, so this
            file never learns what a booster is.
        extra: Additional metrics to log, e.g. an MMoE gate statistic.
        trace: Per-epoch records, logged here rather than from inside the loop.
            Under ``--workers`` that loop runs in another process, and a run
            opened there would wrap a fragment of this measurement.
    """
    held = prepared.held
    result = evaluate_scores(scores, held, args.k)
    print(render(result, ranked_by, args.k))

    # Calibration, on a BY-USER half of the holdout the isotonic fit never saw.
    # A different seed from the train/held split, or this would re-derive the
    # same partition and fit on nothing new.
    calibration = calibrate(
        scores,
        held.labels,
        np.repeat(holdout_mask(held, 0.5, args.seed + 1), held.groups),
    )
    print(
        f"\n  calibration: fitted on {calibration.fitted_rows:,} rows, "
        f"reported on {calibration.report_rows:,}"
    )
    print(render_calibration(calibration.buckets, calibration.ece, calibration.base_rate))
    print(
        f"  ECE          {calibration.ece:.4f} calibrated · "
        f"{calibration.ece_constant:.4f} for a CONSTANT base-rate model · "
        f"{calibration.ece_raw:.4f} raw"
    )
    print(
        f"  log loss     {calibration.log_loss:.4f} calibrated · "
        f"{calibration.log_loss_constant:.4f} constant  "
        f"({1 - calibration.log_loss / calibration.log_loss_constant:+.1%})"
    )
    print(
        f"  The constant model is perfectly calibrated and knows nothing, so ECE\n"
        f"  alone cannot separate them -- log loss is the line with content.\n"
        f"  calibrator resolution: {calibration.resolution} distinct buckets"
    )

    # The order retrieval already produced. If the ranker cannot beat this, it
    # is latency and complexity for nothing.
    inherited = held.features[:, held.names.index("retrieval_score")].astype(float)
    baseline = per_request_ndcg(inherited, held, args.k)
    ranked = per_request_ndcg(scores, held, args.k)
    users = [str(value) for value in held.user_ids]
    delta = paired_bootstrap(
        per_user_means(baseline.tolist(), users), per_user_means(ranked.tolist(), users)
    )

    print(
        f"\n  retrieval order alone, NDCG@{args.k}          {baseline.mean():.4f}\n"
        f"  ranker over the same candidates             {ranked.mean():.4f}\n"
        f"  difference                                  {delta.difference:+.4f} "
        f"[{delta.lo:+.4f}, {delta.hi:+.4f}]"
        f"{' *' if delta.significant else '  (not resolvable)'}"
    )
    print(f"  paired over {delta.n_users:,} users, model held fixed.")

    with track(
        prepared.settings,
        prepared.run_name,
        {**prepared.marks, **vars(args)},
        experiment=model,
    ) as run:
        run.log_metrics(
            {
                f"ndcg@{args.k}": result.ndcg,
                f"ndcg@{args.k}_by_request": result.ndcg_by_request,
                f"ndcg@{args.k}_retrieved": result.ndcg_retrieved,
                "ceiling": result.ceiling,
                "headroom_used": result.headroom_used,
                "gauc": result.gauc.mean,
                "gauc_scored_requests": float(result.gauc.scored),
                "gauc_skipped_requests": float(result.gauc.skipped),
                "auc": result.auc,
                "ece": calibration.ece,
                "ece_uncalibrated": calibration.ece_raw,
                "ece_constant": calibration.ece_constant,
                "log_loss": calibration.log_loss,
                "log_loss_constant": calibration.log_loss_constant,
                "over_retrieval_order": delta.difference,
                "over_retrieval_order_lo": delta.lo,
                "over_retrieval_order_hi": delta.hi,
                "observed_negatives": prepared.rows.observed_share,
                "fitting_requests": float(len(prepared.fitting.groups)),
                "held_requests": float(len(held.groups)),
                **(extra or {}),
            }
        )
        for name, share in ranked_by:
            run.log_metrics({f"gain_{name}": share})
        for record in trace or []:
            run.log_metrics(
                {name: value for name, value in record.items() if name != "epoch"},
                step=int(record["epoch"]),
            )

        CHECKPOINTS.mkdir(parents=True, exist_ok=True)
        artifact = save_model(CHECKPOINTS)
        run.log_artifact(str(artifact))
        print(f"\n  model -> {artifact}")

    if args.label:
        destination = Run(args.label, ranked, held.user_ids, held.found, args.k).save(args.runs)
        print(f"  per-request scores -> {destination}")
        print(f"  compare two runs: uv run python -m models.ranking.compare <a> {args.label}")
    return 0
