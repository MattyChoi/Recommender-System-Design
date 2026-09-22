"""Fit the gradient-boosted ranker and score it against the order it inherited.

The comparison that decides whether this stage exists at all is not the ranker
against nothing -- it is the ranker against **the order retrieval already
produced**. Retrieval hands over candidates sorted by score; a ranker that
cannot beat that ordering has added a model, a feature pipeline and serving
latency for no gain, and the measurement should say so plainly.

Everything except the fit is in ``models.ranking.pipeline``, shared with the
neural trainer, so the two are compared on one denominator by construction.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from models.ranking.baselines.lgbm import importances, train
from models.ranking.pipeline import add_common_arguments, prepare, report

MODEL = "lgbm"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    prepared = prepare(args, MODEL)

    model = train(prepared.fitting, prepared.held)
    scores = np.asarray(model.predict(prepared.held.features), dtype=float)

    def save(directory: Path) -> Path:
        artifact = directory / f"{prepared.run_name}.txt"
        model.booster_.save_model(str(artifact))
        return artifact

    return report(
        args,
        prepared,
        MODEL,
        scores,
        importances(model, prepared.held.names),
        save,
    )


if __name__ == "__main__":
    raise SystemExit(main())
