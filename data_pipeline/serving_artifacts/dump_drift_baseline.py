"""``make drift-baseline`` -- freeze training's feature distribution for serving.

The live drift monitor compares what the orchestrator serves against what the
model trained on. It cannot recompute the training side: that is a Spark read of
gold, and Spark has no business on the serving path. So the comparison's fixed
half is baked here, once per training distribution, and shipped like every other
serving artifact.

What is frozen, per user feature:

* the **bin edges** -- training's deciles, de-duplicated, FINITE. The monitor
  opens the outer pair to infinity itself (``common.drift.histogram_shares``),
  because JSON has no infinity and that convention belongs in one place.
* the **expected share** in each bin. Persisted rather than derived, so the
  monitor needs neither the training sample nor the edges' original quantile
  weights to produce a PSI.

A feature training was CONSTANT on carries ``constant`` and no bins: there is no
bin structure to express a shift against, and inventing one would report a
confident number about nothing.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from common.drift import PSI_BINS, bin_edges, histogram_shares
from common.features import USER_FEATURES
from common.tracking import git_revision
from data_pipeline.features.training_block import load_training


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze training's feature bins for serving.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", type=Path, default=Path("serving/artifacts/drift_baseline.json"))
    parser.add_argument("--version", default=None)
    args = parser.parse_args(argv)

    block, requests = load_training(args.split)
    if block.shape[0] == 0:
        # Refused rather than written empty. A baseline over no rows produces
        # edges that compare everything to nothing, and the monitor would report
        # a PSI for every feature with no way to tell it was meaningless.
        raise SystemExit(
            f"training_examples/{args.split} yielded no found-user rows "
            f"out of {requests:,} requests; nothing to baseline"
        )

    features: dict[str, Any] = {}
    constant = []
    for column, name in enumerate(USER_FEATURES):
        sample = block[:, column]
        edges = bin_edges(sample)
        if edges.size < 2:
            features[name] = {"constant": float(edges[0])}
            constant.append(name)
            continue
        features[name] = {
            "edges": edges.tolist(),
            "expected": histogram_shares(sample, edges).tolist(),
        }

    version = args.version or git_revision()["code_hash"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "version": version,
                "split": args.split,
                "rows": int(block.shape[0]),
                "requests": requests,
                "bins": PSI_BINS,
                # Keyed by feature NAME, not positional like catalogue.json: the
                # monitor labels its gauge by name, and a positional array would
                # make a column reorder in USER_FEATURES silently relabel every
                # series rather than fail.
                "features": features,
            }
        )
        + "\n"
    )

    print(
        f"wrote {args.out}: {len(USER_FEATURES)} features over "
        f"{block.shape[0]:,} found rows of {requests:,} requests, version {version}"
    )
    if constant:
        print(f"  constant in training, no bins: {', '.join(constant)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
