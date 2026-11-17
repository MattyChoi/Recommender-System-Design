"""``make exposure`` -- exposure concentration over time, for the feedback loop.

Plots the Gini coefficient of impressions over items in equal-sized windows,
for TWO policies on one axis:

* **logged** -- MIND's own editorial exposure, replayed onto the ``impressions``
  topic. The distribution the model was trained on.
* **served** -- the slates this orchestrator actually produced, from
  ``ImpressionContext.item_ids`` on ``impression_context``.

The comparison is the deliverable. A single curve says only "exposure is
uneven", which is true of every recommender ever built; the pair says whether
THIS policy concentrates exposure more than the log it learned from, which is
the first observable symptom of a feedback loop.

What this does NOT show, and the reason the figure is labelled the way it is:
nothing here retrains on the served log, so no loop closes. These are two fixed
policies measured side by side. A closing loop would be a RISING served curve
across retraining generations, and producing one needs a retrain step this
repo does not have -- see docs/design.md section 8.

Three measurement choices that decide whether the two curves are comparable:

* **Windows are counted in EXPOSURES, not in wall-clock.** The replay spans
  days of 2019 and the served log spans minutes of a bench run; on a time axis
  they could not share a panel. Gini also rises mechanically as a sample
  shrinks against a fixed catalogue, so equal-sized windows are what make the
  two readable against each other at all.
* **The denominator is the whole catalogue**, zeros included, for the reason
  ``metrics.catalog_coverage`` requires it to be passed: measuring against only
  the items that appeared flatters the number badly here, where the item map
  spans 65k articles and most are never shown (ADR 0005). The Gini over just
  the observed items is reported beside it, not plotted, so the decomposition
  into "how many items" and "how evenly among them" stays visible.
* **Served rows are excluded from the logged series by propensity.** Both
  policies write to ``impressions``, so a capture taken after serving traffic
  contains both. MIND carries no propensity (B1: the policy that produced those
  impressions is undocumented), and every slot this orchestrator serves carries
  one -- ``rerank.Deterministic`` on the exploited slots. So a non-zero
  propensity is the discriminator, and the count of excluded rows is printed
  rather than assumed to be zero.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from common.tracking import git_revision
from evaluation.offline.metrics import gini

BEGIN = "<!-- exposure:begin -->"
END = "<!-- exposure:end -->"

# Below this many exposures a window's Gini is dominated by the window size
# rather than by the policy. Same reasoning as skew_report's MIN_SERVED_ROWS.
MIN_WINDOW = 500


@dataclass(frozen=True)
class Window:
    """One equal-sized block of exposures."""

    index: int
    exposures: int
    distinct: int
    gini_catalogue: float
    gini_observed: float


@dataclass(frozen=True)
class Series:
    """One policy's exposure stream."""

    name: str
    windows: list[Window]
    exposures: int
    excluded: int


def parse_served(lines: Iterable[str]) -> tuple[list[str], int]:
    """Flatten ImpressionContext.item_ids into one entry per item shown."""
    items: list[str] = []
    for record in _records(lines):
        items.extend(str(item) for item in (record.get("item_ids") or []))
    return items, 0


def parse_logged(lines: Iterable[str]) -> tuple[list[str], int]:
    """One entry per InteractionEvent row, minus this orchestrator's own.

    Returns the item ids and how many rows were excluded as ours.
    """
    items: list[str] = []
    excluded = 0
    for record in _records(lines):
        if float(record.get("propensity") or 0.0) > 0.0:
            excluded += 1
            continue
        item = record.get("item_id")
        if item:
            items.append(str(item))
    return items, excluded


def _records(lines: Iterable[str]) -> Iterable[dict[str, Any]]:
    """Decode the JSON on each line, skipping decode_events' column prefix."""
    for line in lines:
        start = line.find("{")
        if start < 0:
            continue  # decode_events' "# timed out" and blank lines
        try:
            yield json.loads(line[start:])
        except json.JSONDecodeError:
            continue


def windows(name: str, items: Sequence[str], size: int, catalogue: int, excluded: int) -> Series:
    """Cut the stream into equal blocks and measure each one."""
    out: list[Window] = []
    for index, start in enumerate(range(0, len(items) - size + 1, size)):
        block = items[start : start + size]
        counts = np.array(list(_counts(block)), dtype=np.float64)
        if len(counts) > catalogue:
            raise SystemExit(
                f"{name}: {len(counts):,} distinct items in one window exceeds the "
                f"{catalogue:,}-item catalogue; is --item-map the one this log was served from?"
            )
        padded = np.concatenate([counts, np.zeros(catalogue - len(counts))])
        out.append(
            Window(
                index=index,
                exposures=len(block),
                distinct=len(counts),
                gini_catalogue=gini(padded),
                gini_observed=gini(counts),
            )
        )
    return Series(name=name, windows=out, exposures=len(items), excluded=excluded)


def _counts(items: Iterable[str]) -> Iterable[int]:
    tally: dict[str, int] = {}
    for item in items:
        tally[item] = tally.get(item, 0) + 1
    return tally.values()


def plot(series: list[Series], destination: Path, size: int, catalogue: int) -> None:
    """Two stacked panels: concentration, and the coverage that explains it."""
    import matplotlib

    matplotlib.use("Agg")  # headless: this runs over SSH and in CI
    import matplotlib.pyplot as plt

    figure, (top, bottom) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)

    for line, colour in zip(series, ("#5E6C67", "#26644B"), strict=False):
        x = [w.index for w in line.windows]
        top.plot(x, [w.gini_catalogue for w in line.windows], label=line.name, color=colour, lw=2)
        bottom.plot(x, [w.distinct for w in line.windows], label=line.name, color=colour, lw=2)

    top.set_ylabel("Gini of exposure\n(over the full catalogue)")
    top.set_title(
        f"Exposure concentration, {size:,}-impression windows\n"
        "two fixed policies side by side -- neither retrains on its own log",
        fontsize=11,
    )
    bottom.set_ylabel(f"distinct items shown\n(of {catalogue:,})")
    bottom.set_xlabel("window (equal blocks of impressions, in log order)")
    for axis in (top, bottom):
        axis.grid(alpha=0.25, lw=0.6)
        axis.legend(frameon=False)
        axis.spines[["top", "right"]].set_visible(False)

    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, facecolor="#fcfcfb", dpi=150)


def render(
    series: list[Series], image: Path, out: Path, size: int, catalogue: int, source: str
) -> str:
    # Relative to the DOCUMENT, not to a hardcoded docs/: --img and --out move
    # together or the link in the generated block points at nothing.
    link = os.path.relpath(image, out.parent)
    lines = [f"![Exposure concentration over time]({link})", ""]

    for line in series:
        note = f", {line.excluded:,} rows excluded as this policy's own" if line.excluded else ""
        lines.append(
            f"**{line.name}**: {line.exposures:,} impressions, "
            f"{len(line.windows)} full windows of {size:,}{note}."
        )
    lines.append("")

    if any(w.exposures < MIN_WINDOW for line in series for w in line.windows):
        lines += [
            f"**Windows below {MIN_WINDOW:,} impressions are size-dominated** and are shown, "
            "not interpreted.",
            "",
        ]

    lines += [
        "| Policy | Window | Impressions | Distinct items | Gini (catalogue) | Gini (observed) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for line in series:
        for window in line.windows:
            lines.append(
                f"| {line.name} | {window.index} | {window.exposures:,} | "
                f"{window.distinct:,} | {window.gini_catalogue:.4f} | "
                f"{window.gini_observed:.4f} |"
            )

    lines += [
        "",
        f"Catalogue: {catalogue:,} items. Gini (catalogue) counts every item that "
        "could have been shown, so it falls as coverage rises; Gini (observed) "
        "holds coverage fixed and reports only how evenly the shown items shared "
        "the exposure. Reading them together is what separates *showing fewer "
        "things* from *showing the same things more often*.",
        "",
        f"_Generated {datetime.now(UTC).isoformat(timespec='seconds')} "
        f"from `{git_revision()['code_hash']}`, over {source}._",
    ]
    return "\n".join(lines)


def splice(document: str, body: str) -> str:
    """Replace the text between the markers, refusing a document without them."""
    head, found_begin, rest = document.partition(BEGIN)
    _, found_end, tail = rest.partition(END)
    if not found_begin or not found_end:
        raise SystemExit(f"docs/exposure.md has no {BEGIN} ... {END} block to replace")
    return f"{head}{BEGIN}\n{body}\n{END}{tail}"


def catalogue_size(path: Path) -> int:
    """How many items COULD be shown, from the item map the server loaded.

    Read the same way dump_popular.py reads it -- `{"version", "items"}` -- so
    a change to the artifact's shape breaks both at once rather than leaving
    this one quietly measuring against zero.
    """
    items = (json.loads(path.read_text()).get("items") or {}) if path.exists() else {}
    if not items:
        raise SystemExit(f"{path}: no `items`; run `make item-map` or pass --item-map")
    return len(items)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Exposure concentration over time.")
    parser.add_argument(
        "--served",
        type=Path,
        help="ImpressionContext JSON lines: `make consume TOPIC=impression_context N=...`",
    )
    parser.add_argument(
        "--logged",
        type=Path,
        help="InteractionEvent JSON lines: `make consume TOPIC=impressions N=...`",
    )
    parser.add_argument("--item-map", type=Path, default=Path("serving/artifacts/item_map.json"))
    parser.add_argument("--window", type=int, default=5_000, help="impressions per window")
    parser.add_argument("--out", type=Path, default=Path("docs/exposure.md"))
    parser.add_argument("--img", type=Path, default=Path("docs/img/exposure.png"))
    args = parser.parse_args(argv)

    if not args.served and not args.logged:
        raise SystemExit("pass --served, --logged, or both; neither was given")

    catalogue = catalogue_size(args.item_map)
    series: list[Series] = []
    sources: list[str] = []

    for name, path, parse in (
        ("logged (MIND replay)", args.logged, parse_logged),
        ("served (this policy)", args.served, parse_served),
    ):
        if path is None:
            continue
        with path.open() as handle:
            items, excluded = parse(handle)
        if len(items) < args.window:
            raise SystemExit(
                f"{path}: {len(items):,} impressions is less than one --window of "
                f"{args.window:,}. Consume more (`make consume ... N=`), or lower --window."
            )
        series.append(windows(name, items, args.window, catalogue, excluded))
        sources.append(f"`{path}`")

    plot(series, args.img, args.window, catalogue)
    body = render(series, args.img, args.out, args.window, catalogue, " and ".join(sources))
    args.out.write_text(splice(args.out.read_text(), body))
    print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
