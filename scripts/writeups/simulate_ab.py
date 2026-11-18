"""``make simulate`` -- a simulated A/B test and a launch recommendation.

Produces the four artifacts a real launch decision requires: a point estimate
with a 95% interval on the primary metric, guardrails with their own bounds, a
power calculation, and a written recommendation. The decision rule lives in
:mod:`evaluation.simulation.analysis` and is a pure function of its inputs, so
the call can be argued with rather than taken on trust.

**Calibrated from the served impression log, not invented.** The slate size and
the item popularity distribution are read from
``ImpressionContext.item_ids`` -- which means the control arm draws from the
items this policy actually reaches, and only exploration reaches the rest of
the catalogue. That is the behaviour ``docs/exposure.md`` measured, carried
into the simulator rather than assumed away.

Two parameters cannot be read from that log and are flags with documented
defaults:

* **The session click rate.** ``impression_context`` carries no clicks -- there
  is no client sending them -- so the baseline comes from the corpus. MIND's
  per-item CTR is 4.04% (C4), which over a 10-slot slate is a session rate of
  ``1 - 0.9596^10`` = 33.5%.
* **Latency.** Taken from ``docs/design.md``'s measured sweep: 8 ms p50 and
  13.4 ms p99 at 300 rps, which fixes the lognormal sigma at
  ``ln(13.4/8.0) / z(0.99)``.

⚠️ Every number this produces is a statement about the MODEL. See the warning
at the top of :mod:`evaluation.simulation.ab`.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import numpy.typing as npt

from common.tracking import git_revision
from evaluation.simulation.ab import Corpus, Policy, simulate
from evaluation.simulation.analysis import Experiment, Guardrail, analyse
from evaluation.simulation.power import sample_size_per_arm

BEGIN = "<!-- experiment:begin -->"
END = "<!-- experiment:end -->"

#: MIND's per-item CTR, from the C4 contract tests.
ITEM_CTR = 0.0404

#: From docs/design.md's measured sweep at 300 rps.
LATENCY_P50_MS = 8.0
LATENCY_P99_MS = 13.4

#: Share of exposure that defines the "head" for the long-tail guardrail.
HEAD_COVERAGE = 0.80

GUARDRAILS = (
    Guardrail(name="latency (ms)", direction="lower_is_better", tolerance=0.05),
    Guardrail(name="long-tail share", direction="higher_is_better", tolerance=0.02),
)


def _records(lines: Iterable[str]) -> Iterable[dict[str, Any]]:
    for line in lines:
        start = line.find("{")
        if start < 0:
            continue
        try:
            yield json.loads(line[start:])
        except json.JSONDecodeError:
            continue


def read_exposure(path: Path, catalogue: int) -> tuple[npt.NDArray[np.float64], int]:
    """Item exposure weights over the whole catalogue, from the served log.

    Items the policy never served keep a weight of zero, so the control arm
    cannot draw them and only exploration can -- which is the reachability
    gap docs/exposure.md measured, rather than a smoothed-over version of it.

    Returns:
        ``(weights, slate_size)``.
    """
    counts = np.zeros(catalogue, dtype=np.float64)
    slate_sizes: list[int] = []

    for record in _records(path.open()):
        if record.get("degraded"):
            continue  # a degraded slate is not this policy's behaviour
        items = record.get("item_ids") or []
        if not items:
            continue
        slate_sizes.append(len(items))
        for item in items:
            index = int(item)
            if 0 <= index < catalogue:
                counts[index] += 1.0

    total = counts.sum()
    if total == 0.0:
        raise SystemExit(
            f"{path}: no healthy served slates. Is the orchestrator running with -kafka, "
            "and was TOPIC=impression_context consumed into this file?"
        )
    if not slate_sizes:
        raise SystemExit(f"{path}: no slates with items")

    return counts / total, round(float(np.mean(slate_sizes)))


def head_size(weights: npt.NDArray[np.float64], coverage: float = HEAD_COVERAGE) -> int:
    """How many items carry ``coverage`` of the exposure."""
    ordered = np.sort(weights)[::-1]
    return int(np.searchsorted(np.cumsum(ordered), coverage) + 1)


def session_ctr(item_ctr: float, slate_size: int) -> float:
    """P(at least one click) for a slate, from the per-item rate."""
    return 1.0 - (1.0 - item_ctr) ** slate_size


def latency_sigma(p50: float, p99: float) -> float:
    """Lognormal sigma implied by a measured median and 99th percentile."""
    if p99 <= p50:
        raise ValueError(f"p99 ({p99}) must exceed p50 ({p50})")
    return math.log(p99 / p50) / NormalDist().inv_cdf(0.99)


def render(result: Experiment, corpus: Corpus, source: str, seed: int) -> str:
    verdict = {"LAUNCH": "✅", "DO NOT LAUNCH": "🛑", "INCONCLUSIVE": "⚠️"}[result.recommendation]

    lines = [
        f"## {verdict} {result.recommendation}",
        "",
    ]
    lines += [f"> {reason}" for reason in result.reasons]
    lines += [
        "",
        "### Design",
        "",
        f"- **{result.sessions_per_arm:,} sessions per arm**, equal allocation.",
        f"- Powered for a **{result.target_lift:+.1%}** relative lift; this run could have "
        f"detected **{result.minimum_detectable:+.2%}** "
        f"(power {result.power_for_target:.0%} at the target).",
        f"- Catalogue {corpus.item_weights.size:,} items, slate {corpus.slate_size}, "
        f"head {corpus.head_size:,} items carry {HEAD_COVERAGE:.0%} of exposure.",
        f"- Baseline session CTR {corpus.session_ctr:.2%}, seed {seed}.",
        "",
        "### Sample ratio",
        "",
        f"{result.srm.summary()}",
        "",
        "### Primary metric",
        "",
        "| Metric | Control | Treatment | Change | 95% CI on the difference |",
        "|---|---:|---:|---:|---|",
    ]
    for metric in (result.primary, result.primary_cuped):
        lines.append(
            f"| {metric.name} | {metric.control:.4f} | {metric.treatment:.4f} | "
            f"{metric.relative:+.2%} | [{metric.lo:+.5f}, {metric.hi:+.5f}] |"
        )
    lines += [
        "",
        f"CUPED removed **{result.variance_reduction:.1%}** of the variance, which is why the "
        "adjusted interval is the narrower one. The decision reads the adjusted row.",
        "",
        "### Guardrails",
        "",
    ]
    if not result.guardrails:
        lines.append("None configured — nothing could have blocked this launch.")
    else:
        lines += [
            "| Guardrail | Control | Treatment | Change | 95% CI | Tolerance | Verdict |",
            "|---|---:|---:|---:|---|---:|---|",
        ]
        for rail, spec in zip(result.guardrails, GUARDRAILS, strict=True):
            lines.append(
                f"| {rail.name} | {rail.control:.4g} | {rail.treatment:.4g} | "
                f"{rail.relative:+.2%} | [{rail.lo:+.4g}, {rail.hi:+.4g}] | "
                f"{spec.tolerance:.0%} | {'BREACHED' if rail.breached else 'ok'} |"
            )

    lines += [
        "",
        f"_Generated {datetime.now(UTC).isoformat(timespec='seconds')} "
        f"from `{git_revision()['code_hash']}`, calibrated on {source}._",
    ]
    return "\n".join(lines)


def splice(document: str, body: str) -> str:
    head, found_begin, rest = document.partition(BEGIN)
    _, found_end, tail = rest.partition(END)
    if not found_begin or not found_end:
        raise SystemExit(f"docs/experiment.md has no {BEGIN} ... {END} block to replace")
    return f"{head}{BEGIN}\n{body}\n{END}{tail}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A simulated A/B test and a launch call.")
    parser.add_argument(
        "--served",
        type=Path,
        required=True,
        help="ImpressionContext JSON lines: `make consume TOPIC=impression_context N=...`",
    )
    parser.add_argument("--item-map", type=Path, default=Path("serving/artifacts/item_map.json"))
    parser.add_argument("--item-ctr", type=float, default=ITEM_CTR, help="per-item CTR (C4)")
    parser.add_argument("--relevance-lift", type=float, default=0.10)
    parser.add_argument("--epsilon", type=float, default=0.10, help="treatment exploration rate")
    parser.add_argument("--latency-shift-ms", type=float, default=1.5)
    parser.add_argument("--target-lift", type=float, default=0.02)
    parser.add_argument(
        "--sessions",
        type=int,
        default=0,
        help="sessions per arm; 0 uses the power calculation for --target-lift",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("docs/experiment.md"))
    args = parser.parse_args(argv)

    if not args.item_map.exists():
        raise SystemExit(f"{args.item_map}: not found; run `make item-map`")
    items = json.loads(args.item_map.read_text()).get("items") or {}
    if not items:
        raise SystemExit(f"{args.item_map}: no `items`; run `make item-map`")

    weights, slate = read_exposure(args.served, len(items))
    baseline = session_ctr(args.item_ctr, slate)

    sessions = args.sessions or sample_size_per_arm(baseline, args.target_lift).per_arm

    corpus = Corpus(
        slate_size=slate,
        session_ctr=baseline,
        item_weights=weights,
        head_size=head_size(weights),
        latency_p50_ms=LATENCY_P50_MS,
        latency_sigma=latency_sigma(LATENCY_P50_MS, LATENCY_P99_MS),
    )

    control, treatment = simulate(
        Policy(name="control"),
        Policy(
            name="treatment",
            relevance_lift=args.relevance_lift,
            epsilon=args.epsilon,
            latency_ms_shift=args.latency_shift_ms,
        ),
        corpus,
        sessions,
        seed=args.seed,
    )
    result = analyse(control, treatment, corpus, GUARDRAILS, args.target_lift, seed=args.seed)

    body = render(result, corpus, f"`{args.served}`", args.seed)
    args.out.write_text(splice(args.out.read_text(), body))
    print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
