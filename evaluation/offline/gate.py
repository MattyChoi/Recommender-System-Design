"""The model quality gate: may a candidate replace what is serving?

Deliberately the SAME SHAPE as :mod:`indexing.lifecycle` -- ``(allowed,
reason)``, fail-closed, and the reason recorded when it passes as well as when
it refuses, because a gate that explains only its refusals leaves no evidence
behind when it approves. ``RECALL_TOLERANCE`` is imported from there rather than
restated, so the model gate and the index gate cannot drift apart.

Two thresholds, and a candidate must clear BOTH:

* an **absolute floor**, which catches a drop too small to be significant on one
  run but large enough to matter when it recurs at every retrain;
* a **paired bootstrap over users**, which catches a drop inside the floor that
  is nonetheless real. The per-row cards are kept precisely so this costs no
  rescoring -- ``models.retrieval.evaluate.save`` says exactly that.

Absolute rather than relative, following the index gate's reasoning: recall here
is ~0.38, so the 2% relative figure the guide suggests is 0.0075 -- 1.5x looser
than the gate already running on indexes, and a model is harder to roll back
than an index.

Where per-row data does not exist the paired half is reported UNAVAILABLE rather
than assumed to pass. That is GAUC's situation today: the cards hold one scalar
per cohort. Making GAUC pairable means keeping per-user AUC rows in
``run_eval``, which is a harness change plus a re-score of every card.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from evaluation.offline.stats import PairedResult, paired_bootstrap, per_user_means
from indexing.lifecycle import RECALL_TOLERANCE
from models.ranking.compare import RESULTS as RANKING_RESULTS
from models.ranking.compare import compare as paired_ndcg
from models.ranking.compare import load as load_run

POINTER = Path("evaluation/results/PRODUCTION.json")

# The five cohorts every card carries. Named here rather than discovered from
# the candidate's own keys: a candidate that simply stopped emitting cold_item
# would otherwise pass by omission.
COHORTS = ("overall", "warm_user", "cold_user", "warm_item", "cold_item")

# Provisional, and INHERITED rather than derived. RECALL_TOLERANCE is defended
# by a measured run-to-run spread; GAUC's spread on these cards has never been
# measured, and the ci95 the cards carry is on ndcg@10, NOT on GAUC, so quoting
# it here would be the DO_NOT_QUOTE mistake with a different label. To earn a
# number: score one model twice under different seeds and look at the gap.
GAUC_TOLERANCE = RECALL_TOLERANCE


@dataclass(frozen=True)
class Check:
    """One metric's verdict.

    Attributes:
        metric: What was measured, cohort included where there is one.
        candidate: The candidate's value.
        baseline: What is serving.
        tolerance: How far below baseline the candidate was allowed to fall.
        paired: The paired bootstrap, or None where the rows do not exist. None
            means UNAVAILABLE, never "passed".
        allowed: Whether this metric permits promotion.
        reason: Logged either way.
    """

    metric: str
    candidate: float
    baseline: float
    tolerance: float
    paired: PairedResult | None
    allowed: bool
    reason: str


@dataclass(frozen=True)
class Verdict:
    """Every check, and the conjunction of them."""

    checks: tuple[Check, ...]

    @property
    def allowed(self) -> bool:
        """True only if every check passed. An empty verdict is NOT a pass."""
        return bool(self.checks) and all(check.allowed for check in self.checks)


def check(
    metric: str,
    candidate: float,
    baseline: float,
    tolerance: float,
    paired: PairedResult | None,
) -> Check:
    """Apply the floor, then the paired test. Fail-closed on anything undefined."""
    if math.isnan(candidate) or math.isnan(baseline):
        return Check(
            metric,
            candidate,
            baseline,
            tolerance,
            paired,
            False,
            "undefined on one side; refusing to promote",
        )

    drop = baseline - candidate
    if drop > tolerance:
        return Check(
            metric,
            candidate,
            baseline,
            tolerance,
            paired,
            False,
            f"{candidate:.4f} is {drop:.4f} below {baseline:.4f}, "
            f"past the {tolerance:.4f} tolerance",
        )

    # Inside the floor, but the rows say the drop is real. This is the half a
    # threshold alone cannot see, and the reason the per-row cards are kept.
    if paired is not None and paired.significant and paired.difference < 0:
        return Check(
            metric,
            candidate,
            baseline,
            tolerance,
            paired,
            False,
            f"drop of {drop:.4f} is inside the tolerance, but the paired interval "
            f"[{paired.lo:+.4f}, {paired.hi:+.4f}] over {paired.n_users:,} users "
            f"excludes zero",
        )

    return Check(
        metric,
        candidate,
        baseline,
        tolerance,
        paired,
        True,
        f"{candidate:.4f} against {baseline:.4f} (drop {drop:+.4f})",
    )


def _hits(path: Path) -> tuple[npt.NDArray[np.float64], list[str]]:
    """Per-row hits and their users, from a retrieval card.

    Mirrors the keys ``models.retrieval.evaluate.save`` writes. The honest home
    for this is a ``load`` beside that ``save``; it lives here because reading a
    card should not require editing the writer.
    """
    with np.load(path, allow_pickle=False) as data:
        hit: npt.NDArray[np.float64] = data["hit"].astype(float)
        users = [str(value) for value in data["user_ids"]]
    return hit, users


def _gauc_by_cohort(path: Path) -> dict[str, float]:
    """``cohort -> gauc`` from a JSON result card."""
    card = json.loads(path.read_text())
    cohorts = card.get("cohorts", {})
    return {
        name: float(body["gauc"])
        for name, body in cohorts.items()
        if isinstance(body, dict) and "gauc" in body
    }


def recall_check(candidate: Path, baseline: Path, tolerance: float = RECALL_TOLERANCE) -> Check:
    """Recall@k on the retrieval cards, floor plus paired test.

    The cutoff is whatever the cards were scored at -- ``evaluate.py`` defaults
    to 100 and stores no k, so the label says recall rather than recall@100 and
    the holdout agreement is the caller's problem, as evaluate.py's own
    --holdout-hours help text warns.
    """
    candidate_hit, candidate_users = _hits(candidate)
    baseline_hit, baseline_users = _hits(baseline)

    try:
        paired = paired_bootstrap(
            per_user_means(baseline_hit.tolist(), baseline_users),
            per_user_means(candidate_hit.tolist(), candidate_users),
        )
    except ValueError as exc:
        # Fail-closed: two cards scored over disjoint users cannot be compared,
        # and the means would compare happily and mean nothing.
        return Check(
            "recall",
            float(candidate_hit.mean()),
            float(baseline_hit.mean()),
            tolerance,
            None,
            False,
            f"cannot pair these cards: {exc}",
        )

    return check(
        "recall",
        float(candidate_hit.mean()),
        float(baseline_hit.mean()),
        tolerance,
        paired,
    )


def ndcg_check(
    candidate: str,
    baseline: str,
    runs: Path = RANKING_RESULTS,
    tolerance: float = RECALL_TOLERANCE,
) -> Check:
    """Per-user NDCG@k on the ranking cards, via the existing paired comparison.

    Routed through ``models.ranking.compare`` rather than reimplemented, so the
    gate cannot pair two runs by a path that module would have refused --
    ``check_aligned`` rejects a length, user or cutoff mismatch.
    """
    try:
        candidate_run, baseline_run = load_run(candidate, runs), load_run(baseline, runs)
        paired = paired_ndcg(baseline_run, candidate_run)
    except (OSError, ValueError, KeyError) as exc:
        return Check(
            "ndcg",
            math.nan,
            math.nan,
            tolerance,
            None,
            False,
            f"cannot compare {candidate!r} against {baseline!r}: {exc}",
        )

    users = [str(value) for value in baseline_run.user_ids]
    baseline_mean = float(np.mean(list(per_user_means(baseline_run.ndcg.tolist(), users).values())))
    candidate_mean = float(
        np.mean(list(per_user_means(candidate_run.ndcg.tolist(), users).values()))
    )
    return check(f"ndcg@{baseline_run.k}", candidate_mean, baseline_mean, tolerance, paired)


def gauc_checks(
    candidate: Path,
    baseline: Path,
    tolerance: float = GAUC_TOLERANCE,
    cohorts: Sequence[str] = COHORTS,
) -> list[Check]:
    """One check per cohort, floor only.

    Per cohort rather than on ``overall`` alone: a model can regress badly on
    cold items while the aggregate holds, and cold items are where the spread
    already is -- recency scores 0.4198 there against 0.6029 on warm items.

    A cohort missing from either side REFUSES rather than being skipped. A
    candidate that stopped emitting a cohort would otherwise promote by omission,
    which is the failure this loop exists to prevent.
    """
    candidate_gauc = _gauc_by_cohort(candidate)
    baseline_gauc = _gauc_by_cohort(baseline)

    out: list[Check] = []
    for cohort in cohorts:
        metric = f"gauc[{cohort}]"
        if cohort not in candidate_gauc or cohort not in baseline_gauc:
            missing = "candidate" if cohort not in candidate_gauc else "baseline"
            out.append(
                Check(
                    metric,
                    math.nan,
                    math.nan,
                    tolerance,
                    None,
                    False,
                    f"absent from the {missing} card",
                )
            )
            continue
        out.append(check(metric, candidate_gauc[cohort], baseline_gauc[cohort], tolerance, None))
    return out


@dataclass(frozen=True)
class Pointer:
    """Which cards a side of the comparison is made of.

    PATHS AND LABELS ONLY, never numbers. A pointer carrying the metrics would
    drift from the cards it quotes, which is the argument ``make results`` makes
    about docs/baselines.md.

    Attributes:
        retrieval: A retrieval card, or None if this stage is not declared.
        ranking: A ranking run LABEL, resolved under the runs directory.
        sources: Model name to JSON card, for the per-cohort GAUC checks.
    """

    retrieval: Path | None
    ranking: str | None
    sources: dict[str, Path]


def load_pointer(path: Path) -> Pointer:
    """Read a pointer file. Absent keys mean "not declared", never "passed"."""
    body = json.loads(path.read_text())
    retrieval = body.get("retrieval")
    sources = body.get("sources") or {}
    return Pointer(
        retrieval=Path(retrieval) if retrieval else None,
        ranking=body.get("ranking") or None,
        sources={name: Path(card) for name, card in sources.items()},
    )


def run(candidate: Pointer, baseline: Pointer, runs: Path = RANKING_RESULTS) -> Verdict:
    """Every stage both pointers declare. A stage only one declares is refused."""
    checks: list[Check] = []

    if candidate.retrieval and baseline.retrieval:
        checks.append(recall_check(candidate.retrieval, baseline.retrieval))
    elif candidate.retrieval or baseline.retrieval:
        checks.append(
            Check(
                "recall",
                math.nan,
                math.nan,
                RECALL_TOLERANCE,
                None,
                False,
                "declared on one side only; nothing to compare against",
            )
        )

    if candidate.ranking and baseline.ranking:
        checks.append(ndcg_check(candidate.ranking, baseline.ranking, runs))
    elif candidate.ranking or baseline.ranking:
        checks.append(
            Check(
                "ndcg",
                math.nan,
                math.nan,
                RECALL_TOLERANCE,
                None,
                False,
                "declared on one side only; nothing to compare against",
            )
        )

    for name, card in sorted(candidate.sources.items()):
        against = baseline.sources.get(name)
        if against is None:
            checks.append(
                Check(
                    f"gauc[{name}]",
                    math.nan,
                    math.nan,
                    GAUC_TOLERANCE,
                    None,
                    False,
                    "no baseline card declared for this model",
                )
            )
            continue
        checks.extend(gauc_checks(card, against))

    return Verdict(tuple(checks))


def render(verdict: Verdict) -> str:
    """The table, then the headline. Both sides of every check, always."""
    lines = ["", "  model quality gate", "  " + "-" * 72]
    for item in verdict.checks:
        mark = "PASS" if item.allowed else "FAIL"
        paired = "paired UNAVAILABLE"
        if item.paired is not None:
            verdict_word = "significant" if item.paired.significant else "not resolvable"
            paired = (
                f"paired {item.paired.difference:+.4f} "
                f"[{item.paired.lo:+.4f}, {item.paired.hi:+.4f}] {verdict_word}"
            )
        lines.append(f"  {mark}  {item.metric:<18}{item.reason}")
        lines.append(f"        {'':<18}{paired}")

    if not verdict.checks:
        lines.append("  FAIL  nothing was compared; an empty gate is not a pass")

    lines.append("  " + "-" * 72)
    lines.append("  PROMOTE" if verdict.allowed else "  REFUSED")
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Block a promotion that regresses quality.")
    parser.add_argument("--candidate", type=Path, required=True, help="a pointer file")
    parser.add_argument("--baseline", type=Path, default=POINTER, help="defaults to PRODUCTION")
    parser.add_argument("--runs", type=Path, default=RANKING_RESULTS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Exit 1 on a refusal, so a CI step or a scheduler fails on it."""
    args = _parser().parse_args(argv)
    verdict = run(load_pointer(args.candidate), load_pointer(args.baseline), args.runs)
    print(render(verdict))
    return 0 if verdict.allowed else 1


if __name__ == "__main__":
    raise SystemExit(main())
