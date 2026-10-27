"""Print the metrics MLflow holds for the ranking runs, for the write-up.

Read-only. It exists so a results page quotes the tracking server rather than
somebody's memory of a terminal that has since scrolled away -- the same reason
`baselines.md` is generated from the report cards instead of typed.

Run: ``uv run python -m scripts.dump_ranking_runs``
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import mlflow
from mlflow.tracking import MlflowClient

from common.config import load_settings
from common.tracking import mlflow_name

# Written in the project's own vocabulary and translated on the way out, the
# same as every writer does. Typing the stored spelling instead is how the first
# version of this script matched nothing at all: `mlflow_name` rewrites `@` to
# `_at_`, so the metric is `ndcg_at_10` and `ndcg@10` is not a near miss, it is
# a name that has never existed.
INTERESTING = (
    "ndcg@{k}",
    "ndcg@{k}_by_request",
    "ndcg@{k}_retrieved",
    "ceiling",
    "headroom_used",
    "over_retrieval_order",
    "over_retrieval_order_lo",
    "over_retrieval_order_hi",
    "observed_negatives",
    "fitting_requests",
    "held_requests",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="", help="Only runs whose name starts with this.")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument(
        "--experiment",
        nargs="+",
        default=None,
        help="Restrict to these experiments. Default searches ALL of them: runs "
        "predate the split of the experiment by model, so the same arm can sit "
        "in two places and a default that picked one would silently drop the rest.",
    )
    args = parser.parse_args(argv)

    settings = load_settings()
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    client = MlflowClient()

    found = {item.name: item.experiment_id for item in client.search_experiments()}
    wanted = args.experiment or list(found)
    missing = [name for name in wanted if name not in found]
    if missing:
        raise SystemExit(f"no experiment named {missing}; found: {sorted(found)}")

    runs = client.search_runs(
        [found[name] for name in wanted],
        order_by=["attributes.start_time DESC"],
        max_results=args.limit,
    )

    for run in runs:
        run_name = run.data.tags.get("mlflow.runName", run.info.run_id)
        if args.prefix and not run_name.startswith(args.prefix):
            continue
        # Ranking runs are the ones carrying an NDCG. Filtering on the metric
        # rather than on the experiment is what lets every stage share one.
        # The cutoff is read off the run rather than assumed, because --k is a
        # flag and a run at k=20 is a run this script should still print.
        cutoffs = [
            name.removeprefix("ndcg_at_")
            for name in run.data.metrics
            if name.startswith("ndcg_at_") and name.removeprefix("ndcg_at_").isdigit()
        ]
        if not cutoffs:
            continue
        k = cutoffs[0]

        experiment_name = next(
            (name for name, id_ in found.items() if id_ == run.info.experiment_id), "?"
        )
        print(f"\n{run_name}  [{experiment_name}]")
        params = run.data.params
        # git_dirty is printed beside the config because two runs can share a
        # run_name while differing in code: config_hash digests the settings and
        # the command line, and neither changes when a module is edited.
        for key in (
            "model",
            "names",
            "max_candidates",
            "quota",
            "drop",
            "embeddings",
            "workers",
            "seed",
            "git_sha",
            "git_dirty",
        ):
            if key in params:
                print(f"    {key:<16} {params[key]}")
        for template in INTERESTING:
            key = template.format(k=k)
            stored = mlflow_name(key)
            if stored in run.data.metrics:
                print(f"  {key:<26} {run.data.metrics[stored]:.4f}")
        gains = {
            key.removeprefix("gain_"): value
            for key, value in run.data.metrics.items()
            if key.startswith("gain_")
        }
        for feature, share in sorted(gains.items(), key=lambda pair: -pair[1]):
            print(f"    gain {feature:<22} {share:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
