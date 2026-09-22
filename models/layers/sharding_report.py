"""``make shard-plan`` -- where an embedding table stops fitting on one device.

``--compute-device`` is REQUIRED and has no default, which is deliberate twice
over. No module under ``models/`` may name a device (``tests/test_torch_env.py``
fails the build on one), and the argument is load-bearing rather than
bureaucratic: the sharder offers seven sharding types on an accelerator and four
on CPU, so a default that fell back to CPU would silently delete row-wise from
the search space and the report would conclude the planner never picks it.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from models.layers.sharded_embeddings import (
    DENSE_RESERVATION,
    ITEM_TABLE,
    PlacedTable,
    build_ebc,
    placements,
    plan_for,
    synthetic_topology,
)
from models.layers.sizing import DESIGN_TARGET_ITEMS, table_bytes

#: ``item_map``'s size, measured. The manual's G4 says "160K MIND articles",
#: which is wrong in the same way ADR 0004 and ``docs/design.md`` are wrong --
#: and it matters here, because 160K would put the corpus inside the guide's
#: stated 100K-2M ANN band and 65,238 puts it below the floor.
MIND_ITEMS = 65_238
MIND_USERS = 94_057

DIM = 64
TABLE_WISE = "table_wise"

#: Two pods worth naming: the machine this project trains on, scaled out, and a
#: datacentre part. The crossover is a function of per-device HBM, so one row of
#: each is what shows that.
PODS: tuple[tuple[str, int, float], ...] = (
    ("8 x 24 GB (4090-class)", 8, 24.0),
    ("8 x 80 GB (A100-class)", 8, 80.0),
)


@dataclass(frozen=True)
class Scenario:
    """One hypothetical: a catalogue size against a topology."""

    rows: int
    world_size: int
    hbm_gb: float

    @property
    def params_gib(self) -> float:
        return table_bytes(self.rows, DIM) / 1024**3


@dataclass(frozen=True)
class Outcome:
    """What the planner said, or why it would not say anything.

    Attributes:
        scenario: What was asked.
        item: The item table's placement, or ``None`` if the planner refused.
        refusal: The exception type name when it refused. "Refused" is a
            legitimate cell in this table -- a topology that cannot hold the
            table at all is a real answer to "should you shard this" -- so it is
            recorded rather than raised.
    """

    scenario: Scenario
    item: PlacedTable | None
    refusal: str | None

    @property
    def sharding_type(self) -> str:
        return self.item.sharding_type if self.item is not None else f"({self.refusal})"

    @property
    def whole(self) -> bool:
        """Placed entire on one device. ``False`` also when the planner refused,
        which is correct: a refusal is certainly not a whole placement."""
        return self.item is not None and self.item.sharding_type == TABLE_WISE


def run(scenario: Scenario, compute_device: str) -> Outcome:
    """Plan one scenario.

    The collection is built on ``meta``, so a 300M-row table costs nothing to
    ask about. That is what makes the sweep possible on one machine, and it is
    why this is a dry run rather than a benchmark of this GPU.

    Args:
        scenario: Rows and topology.
        compute_device: TorchRec's device string. Never a literal here.

    Returns:
        An :class:`Outcome`, carrying a refusal rather than raising one.
    """
    collection = build_ebc(scenario.rows, MIND_USERS, dim=DIM)
    topology = synthetic_topology(scenario.world_size, compute_device, hbm_gb=scenario.hbm_gb)
    try:
        plan = plan_for(collection, topology)
    except Exception as error:  # broad: any refusal is a cell value, not a crash
        return Outcome(scenario, None, type(error).__name__)

    item = next(row for row in placements(plan) if row.table == ITEM_TABLE)
    return Outcome(scenario, item, None)


def crossover(
    world_size: int,
    hbm_gb: float,
    compute_device: str,
    *,
    low: int,
    high: int,
    tolerance: float = 0.01,
) -> tuple[int, Outcome] | None:
    """The smallest catalogue the planner will not place on one device.

    A bisection, not a scan, because the interesting quantity is a boundary and
    a scan's resolution is whatever step someone picked.

    Args:
        world_size: Ranks.
        hbm_gb: Per-device HBM.
        compute_device: TorchRec's device string.
        low: A size known to fit whole. Checked, not assumed.
        high: A size believed not to. Checked, not assumed.
        tolerance: Stop when the bracket is within this fraction of ``low``.

    Returns:
        ``(rows, outcome)`` for the first size past the boundary, or ``None``
        when the bracket does not contain one -- which is itself reportable and
        is why this does not raise. A bracket whose ends agree means the sweep
        was pointed at the wrong range, and saying so beats returning a number
        that is really just ``high``.
    """
    if not run(Scenario(low, world_size, hbm_gb), compute_device).whole:
        return None
    if run(Scenario(high, world_size, hbm_gb), compute_device).whole:
        return None

    while high - low > max(1, int(low * tolerance)):
        middle = (low + high) // 2
        if run(Scenario(middle, world_size, hbm_gb), compute_device).whole:
            low = middle
        else:
            high = middle

    return high, run(Scenario(high, world_size, hbm_gb), compute_device)


def _row(label: str, outcome: Outcome) -> str:
    item = outcome.item
    shards = f"{item.n_shards}" if item is not None else "-"
    ranks = f"{len(item.ranks)}" if item is not None else "-"
    return (
        f"| {label} | {outcome.scenario.rows:,} | {outcome.scenario.params_gib:.2f} "
        f"| {outcome.sharding_type} | {shards} | {ranks} |"
    )


def render(
    fixed: Sequence[tuple[str, Outcome]],
    boundaries: Sequence[tuple[str, tuple[int, Outcome] | None]],
    compute_device: str,
) -> str:
    """The report, as markdown, ready to paste into ``docs/benchmarks.md``."""
    lines = [
        "# Sharded embeddings -- when the planner starts splitting",
        "",
        f"`EmbeddingShardingPlanner`, dim {DIM}, fp32, batch 8192, "
        f"compute device `{compute_device}`.",
        f"Storage reservation fixed at {DENSE_RESERVATION:.0%} "
        "(TorchRec's default heuristic over-estimates under meta tensors).",
        "",
        "## The catalogues that exist",
        "",
        "| topology | rows | params GiB | sharding | shards | ranks |",
        "| --- | ---: | ---: | --- | ---: | ---: |",
    ]
    lines += [_row(label, outcome) for label, outcome in fixed]
    lines += [
        "",
        "## The crossover -- where a table stops fitting on one device",
        "",
        "| topology | rows | params GiB | sharding | shards | ranks |",
        "| --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for label, found in boundaries:
        if found is None:
            lines.append(f"| {label} | *no crossover in the bracket searched* | | | | |")
            continue
        _, outcome = found
        lines.append(_row(label, outcome))

    lines += [
        "",
        "## Reading this",
        "",
        "**The corpus is not in the regime this machinery is for.** MIND-small's",
        f"`item_map` is {MIND_ITEMS:,} rows -- a {table_bytes(MIND_ITEMS, DIM) / 1024**2:.1f} MiB",
        'table. The manual\'s own G4 text says "160K MIND articles"; the measured',
        "number is less than half that, and below the 100K floor ADR 0004 claims the",
        "corpus sits inside.",
        "",
        "**Nor is the design target.** At 2M items the table is",
        f"{table_bytes(DESIGN_TARGET_ITEMS, DIM) / 1024**2:.0f} MiB and every topology above still",
        "places it whole. Building for the design target demonstrates the mechanism;",
        "it does not exercise the decision.",
        "",
        "**The crossover is a function of per-device HBM, not of the model.** A table",
        "is placed whole whenever it fits on one device after reservation, so the",
        "boundary moves with the card and not with anything a recsys engineer",
        'controls -- which is the actual answer to "when should I reach for this".',
        "",
        "**What is NOT measured here.** These are planner decisions on `meta` tensors:",
        "no kernel ran, no all-to-all happened, no throughput was observed. The plan is",
        "a prediction from TorchRec's cost model. Treating it as a benchmark would be",
        "quoting a model's opinion as a measurement.",
        "",
        "**Optimiser state is rowwise.** The fused kernel's default is rowwise Adagrad",
        "-- one float per ROW, ~8 MB on a 2M-row table, not one per element. An earlier",
        "estimate assumed Adam's two moments per parameter and put the crossover about",
        "three times too low.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compute-device",
        required=True,
        help="TorchRec device string for the hypothetical topology, e.g. the "
        "accelerator name. Required: there is no safe default (see module docstring).",
    )
    parser.add_argument(
        "--search-high",
        type=int,
        default=500_000_000,
        help="Upper bracket for the crossover bisection, in rows.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Write here instead of stdout.")
    args = parser.parse_args(argv)

    fixed: list[tuple[str, Outcome]] = []
    boundaries: list[tuple[str, tuple[int, Outcome] | None]] = []
    for label, world_size, hbm_gb in PODS:
        for rows in (MIND_ITEMS, DESIGN_TARGET_ITEMS):
            fixed.append((label, run(Scenario(rows, world_size, hbm_gb), args.compute_device)))
        boundaries.append(
            (
                label,
                crossover(
                    world_size,
                    hbm_gb,
                    args.compute_device,
                    low=DESIGN_TARGET_ITEMS,
                    high=args.search_high,
                ),
            )
        )

    report = render(fixed, boundaries, args.compute_device)
    if args.out is None:
        print(report)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
