"""Pair two ranking runs on the same users.

Three comparisons in Part K ended in "the intervals overlap, so nothing can be
said", which is not a fact about the data -- it is a fact about comparing two
MARGINAL intervals. Between-user variance dwarfs the effect: some users are far
easier to serve than others, and that spread is much larger than the two or
three thousandths separating two feature sets. Comparing each user against
themselves cancels it.

This project has made the mistake once before and written it down: a paired
interval measured 3.5x narrower than either marginal and excluded zero where
the marginals overlapped heavily. So a difference between two runs is reported
from the pairing, never from reading two ranges side by side.

Both runs must have scored the **same requests in the same order** -- which
they do whenever the user split is deterministic and the split seed matches.
The check is refused rather than assumed, because two runs over different
holdouts would still pair row-for-row and still return a confident number about
nothing.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from evaluation.offline.stats import PairedResult, paired_bootstrap, per_user_means

RESULTS = Path("evaluation/results/ranking")


@dataclass(frozen=True)
class Run:
    """One scored ranking run, kept so it can be paired later.

    Attributes:
        label: What this run was.
        ndcg: ``[R]`` per-request NDCG@k, zero where retrieval missed.
        user_ids: ``[R]`` who made each request.
        found: ``[R]`` whether retrieval surfaced the clicked article.
        k: The cutoff.
    """

    label: str
    ndcg: npt.NDArray[np.float64]
    user_ids: npt.NDArray[np.int64]
    found: npt.NDArray[np.bool_]
    k: int

    def save(self, directory: Path = RESULTS) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"{self.label}.npz"
        np.savez_compressed(
            destination,
            ndcg=self.ndcg,
            user_ids=self.user_ids,
            found=self.found,
            k=np.asarray(self.k),
            label=np.asarray(self.label),
        )
        return destination


def load(label: str, directory: Path = RESULTS) -> Run:
    with np.load(directory / f"{label}.npz", allow_pickle=False) as data:
        return Run(
            label=str(data["label"]),
            ndcg=data["ndcg"],
            user_ids=data["user_ids"],
            found=data["found"],
            k=int(data["k"]),
        )


def check_aligned(baseline: Run, candidate: Run) -> None:
    """Refuse two runs that did not score the same requests.

    Raises:
        ValueError: On a length, user or cutoff mismatch.
    """
    if len(baseline.ndcg) != len(candidate.ndcg):
        raise ValueError(
            f"{baseline.label} scored {len(baseline.ndcg):,} requests, "
            f"{candidate.label} scored {len(candidate.ndcg):,}"
        )
    if not np.array_equal(baseline.user_ids, candidate.user_ids):
        raise ValueError(
            "the two runs scored different users, or the same users in a different order"
        )
    if baseline.k != candidate.k:
        raise ValueError(f"cutoffs differ: @{baseline.k} against @{candidate.k}")


def compare(baseline: Run, candidate: Run) -> PairedResult:
    """Per-user NDCG difference, ``candidate - baseline``."""
    check_aligned(baseline, candidate)
    users = [str(value) for value in baseline.user_ids]
    return paired_bootstrap(
        per_user_means(baseline.ndcg.tolist(), users),
        per_user_means(candidate.ndcg.tolist(), users),
    )


def render(baseline: Run, candidate: Run, result: PairedResult) -> str:
    users = [str(value) for value in baseline.user_ids]
    marginals = (
        float(np.mean(list(per_user_means(baseline.ndcg.tolist(), users).values()))),
        float(np.mean(list(per_user_means(candidate.ndcg.tolist(), users).values()))),
    )
    return "\n".join(
        [
            f"  {baseline.label:<44}{marginals[0]:>9.4f}",
            f"  {candidate.label:<44}{marginals[1]:>9.4f}",
            f"  paired difference{'':<27}{result.difference:>+9.4f}"
            f"  [{result.lo:+.4f}, {result.hi:+.4f}]"
            f"{' *' if result.significant else '   (not resolvable)'}",
            f"  over {result.n_users:,} users scored under both.",
            "",
            "  Read the paired line, not the gap between the two above it: the",
            f"  spread BETWEEN users is far larger than {abs(result.difference):.4f}, so two",
            "  marginal ranges can overlap while the paired difference is clear.",
        ]
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--runs", type=Path, default=RESULTS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    baseline, candidate = load(args.baseline, args.runs), load(args.candidate, args.runs)
    print(render(baseline, candidate, compare(baseline, candidate)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
