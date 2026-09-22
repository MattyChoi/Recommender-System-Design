"""Two arms, paired per user, banded by item popularity.

``evaluate.py`` scores each arm and saves its per-row results, and this reads
two of those files.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from evaluation.offline.stats import PairedResult, paired_bootstrap
from models.retrieval.evaluate import BAND_EDGES, LONG_TAIL_BELOW, band_labels, long_tail_mask

# Categorical slots 1, 2 and 3 of the validated palette. Validated as a set for
# adjacent pairs in light mode: worst CVD dE 9.2, worst normal-vision dE 27.6.
# Aqua sits below 3:1 on the surface, so it also carries a dashed style and the
# table below is the relief the contrast warning requires.
CANDIDATE_COLOUR = "#2a78d6"
BASELINE_COLOUR = "#eb6834"
REFERENCE_COLOUR = "#1baf7a"
INK = "#0b0b0b"
MUTED = "#52514e"
RULE = "#d8d7d2"


@dataclass(frozen=True)
class Arm:
    """One scored arm, as ``evaluate.save`` wrote it."""

    name: str
    hit: npt.NDArray[np.bool_]
    item_ids: npt.NDArray[np.int64]
    user_ids: npt.NDArray[np.int64]
    band: npt.NDArray[np.int64]
    popularity: npt.NDArray[np.bool_]


@dataclass(frozen=True)
class BandComparison:
    """One band's row of the gate.

    Attributes:
        result: ``None`` when the band holds too few users to bootstrap. Absent
            rather than zero, so an untestable band cannot be read as a null
            result.
    """

    label: str
    rows: int
    users: int
    baseline: float
    candidate: float
    reference: float
    result: PairedResult | None
    summary: bool = False

    @property
    def difference(self) -> float:
        return self.candidate - self.baseline


def load_arm(path: Path) -> Arm:
    with np.load(path) as data:
        return Arm(
            name=path.stem,
            hit=data["hit"],
            item_ids=data["item_ids"],
            user_ids=data["user_ids"],
            band=data["band"],
            popularity=data["popularity"],
        )


def check_aligned(baseline: Arm, candidate: Arm) -> None:
    """Refuse to pair two arms that did not score the same rows in the same order.

    The whole value of a paired test is that per-request variance cancels. Two
    files built from different holdout windows, or from a gold table rebuilt in
    between, still pair row-for-row and still return a narrow interval -- around
    a difference between unrelated requests. Nothing downstream would say so.

    Raises:
        ValueError: If the rows, the users or the banding differ.
    """
    for field in ("item_ids", "user_ids", "band"):
        left: npt.NDArray[np.int64] = getattr(baseline, field)
        right: npt.NDArray[np.int64] = getattr(candidate, field)
        if left.shape != right.shape or not np.array_equal(left, right):
            raise ValueError(
                f"{baseline.name} and {candidate.name} disagree on {field}, so they "
                "did not score the same rows and pairing them is meaningless. "
                "Rescore both with the same --holdout-hours against the same gold "
                "tables."
            )


def per_user(arm: Arm, mask: npt.NDArray[np.bool_]) -> dict[str, float]:
    """Each user's mean hit rate over the masked rows, keyed as `paired_bootstrap` wants."""
    collected: dict[str, list[float]] = defaultdict(list)
    for user, hit in zip(arm.user_ids[mask], arm.hit[mask], strict=True):
        collected[str(user)].append(float(hit))
    return {user: float(np.mean(values)) for user, values in collected.items()}


def _masks(
    band: npt.NDArray[np.int64], edges: Sequence[int]
) -> list[tuple[str, npt.NDArray[np.bool_], bool]]:
    """One mask per band, then the two pooled rows: `<26` and `overall`."""
    per_band = [(label, band == index, False) for index, label in enumerate(band_labels(edges))]
    return [
        *per_band,
        (f"<{LONG_TAIL_BELOW}", long_tail_mask(band, edges), True),
        ("overall", np.ones(len(band), dtype=bool), True),
    ]


def compare(
    baseline: Arm,
    candidate: Arm,
    edges: Sequence[int] = BAND_EDGES,
    resamples: int = 10_000,
    seed: int = 0,
) -> list[BandComparison]:
    """The gate, one row per band. ``difference`` is candidate minus baseline."""
    check_aligned(baseline, candidate)

    rows: list[BandComparison] = []
    for label, mask, summary in _masks(baseline.band, edges):
        before = per_user(baseline, mask)
        after = per_user(candidate, mask)
        shared = set(before) & set(after)
        rows.append(
            BandComparison(
                label=label,
                summary=summary,
                rows=int(mask.sum()),
                users=len(shared),
                baseline=float(baseline.hit[mask].mean()) if mask.any() else float("nan"),
                candidate=float(candidate.hit[mask].mean()) if mask.any() else float("nan"),
                reference=float(baseline.popularity[mask].mean()) if mask.any() else float("nan"),
                result=(
                    paired_bootstrap(before, after, n_resamples=resamples, seed=seed)
                    if len(shared) >= 2
                    else None
                ),
            )
        )
    return rows


def render(rows: Sequence[BandComparison], k: int, baseline: str, candidate: str) -> str:
    """The table. It is also the relief the palette's contrast warning requires."""
    head = f"{'band':>9}  {'rows':>7}  {'users':>6}  {'baseline':>9}  {'candidate':>9}  "
    head += f"{'pop':>7}  {'delta':>8}  {'95% CI':>18}  real"
    lines = [f"baseline = {baseline}", f"candidate = {candidate}", "", head, "-" * len(head)]

    ruled = False
    for row in rows:
        if row.summary and not ruled:
            lines.append("-" * len(head))
            ruled = True
        if row.result is None:
            verdict, interval = "  --", f"{'too few users':>18}"
        else:
            verdict = " yes" if row.result.significant else "  no"
            interval = f"[{row.result.lo:+.4f}, {row.result.hi:+.4f}]"
        lines.append(
            f"{row.label:>9}  {row.rows:>7,}  {row.users:>6,}  {row.baseline:>9.4f}  "
            f"{row.candidate:>9.4f}  {row.reference:>7.4f}  {row.difference:>+8.4f}  "
            f"{interval:>18}  {verdict}"
        )

    lines.append("")
    lines.append(f"recall@{k}; delta = candidate - baseline, paired per USER")
    lines.append("'real' is a 95% paired-bootstrap interval excluding zero")
    return "\n".join(lines)


def plot(
    rows: Sequence[BandComparison],
    destination: Path,
    k: int,
    baseline: str,
    candidate: str,
) -> None:
    """Two panels sharing one x axis -- never two y scales on one panel.

    The top panel is the three recall curves; the bottom is the paired
    difference with its interval. Reading significance off two overlapping
    curves is guesswork, and the difference is the quantity the gate is about,
    so it gets its own axis rather than a caption.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless: this runs over SSH and in CI
    import matplotlib.pyplot as plt

    banded = [row for row in rows if not row.summary and row.rows]
    x = np.arange(len(banded))
    labels = [row.label for row in banded]

    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(9.5, 6.8), sharex=True, height_ratios=[2, 1], dpi=160
    )

    series = (
        ("two-tower, logQ correction", CANDIDATE_COLOUR, "o", "-", [r.candidate for r in banded]),
        ("two-tower, no correction", BASELINE_COLOUR, "s", "-", [r.baseline for r in banded]),
        ("recent-popularity count", REFERENCE_COLOUR, "^", "--", [r.reference for r in banded]),
    )
    for label, colour, marker, style, values in series:
        top.plot(x, values, color=colour, lw=2, marker=marker, ms=6, ls=style, label=label)

    top.set_ylabel(f"Recall@{k}", color=INK)
    top.set_ylim(-0.03, 1.03)
    top.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK)
    top.set_title(
        "The logQ correction rescues the head, not the tail", color=INK, fontsize=12, loc="left"
    )

    difference = np.array([r.difference for r in banded])
    lo = np.array([r.result.lo if r.result else np.nan for r in banded])
    hi = np.array([r.result.hi if r.result else np.nan for r in banded])
    bottom.errorbar(
        x,
        difference,
        yerr=[difference - lo, hi - difference],
        fmt="o",
        ms=6,
        lw=2,
        capsize=4,
        color=CANDIDATE_COLOUR,
    )
    bottom.axhline(0.0, color=MUTED, lw=1)
    # ASCII hyphen: RUF001 rejects the typographic minus, and an axis label is
    # not worth an ignore.
    bottom.set_ylabel("logQ - none", color=INK)
    bottom.set_xlabel("clicks on the item during the training window", color=MUTED)

    for panel in (top, bottom):
        panel.grid(axis="y", color=RULE, lw=0.8)
        panel.set_axisbelow(True)
        panel.tick_params(colors=MUTED)
        for side in ("top", "right"):
            panel.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            panel.spines[side].set_color(RULE)

    bottom.set_xticks(x, labels)
    figure.text(
        0.011,
        0.012,
        f"baseline {baseline} · candidate {candidate} · 95% paired bootstrap over users",
        color=MUTED,
        fontsize=7.5,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    figure.savefig(destination, facecolor="#fcfcfb")
    plt.close(figure)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two scored arms, band by band.")
    parser.add_argument("baseline", type=Path, help="The .npz the candidate is measured against.")
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--k", type=int, default=100, help="Only labels the output.")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Defaults to a path naming both arms, so two comparisons cannot collide.",
    )
    return parser


def _slug(name: str) -> str:
    """An arm's identity without its budget or config hash.

    ``both-logq-n4u0-b8192e10lr0.001-ab7e1d50`` -> ``both-logq-n4u0``. Enough to
    tell two comparisons apart in a filename, short enough to read.
    """
    return re.sub(r"-b\d+e\d+lr.*$", "", name)


def default_plot(baseline: Arm, candidate: Arm) -> Path:
    """A plot path derived from the arms.

    A fixed default silently overwrote G2's plot the first time two comparisons
    were run in a row -- the second finished, said "plot -> ...", and the first
    deliverable was gone with nothing reporting it.
    """
    return Path("docs/img") / f"{_slug(candidate.name)}-vs-{_slug(baseline.name)}.png"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    baseline, candidate = load_arm(args.baseline), load_arm(args.candidate)

    rows = compare(baseline, candidate, resamples=args.resamples, seed=args.seed)
    print(render(rows, args.k, baseline.name, candidate.name))

    destination = args.plot or default_plot(baseline, candidate)
    plot(rows, destination, args.k, baseline.name, candidate.name)
    print(f"\n  plot -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
