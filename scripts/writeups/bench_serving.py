"""``make bench`` -- the serving design's latency table, measured with ghz.

That table is worth more than another model architecture. It is also the table
most easily faked, so two things are enforced here rather than left to
discipline.

**Every request uses a DIFFERENT user.** ghz's default is to send one payload
over and over, which on this system measures the warm path and nothing else:
the same `user_id` hits the same cached embedding in the retrieval sidecar,
the same seen-list key in Redis, and the same Feast row in the gateway. The
numbers come out beautiful and describe a system nobody runs. The request data
below templates `{{.RequestNumber}}` into the user id so the caches are missed
at roughly the rate production would miss them.

**The sweep runs to SATURATION, not to a chosen number.** A single RPS figure
says nothing without knowing whether it was near the limit. The sweep climbs
until p99 passes the budget, throughput stops tracking the offered rate, or the
service stops doing its job -- and reports where that happened, which is the
number the HPA in `infra/k8s` scales on.

**DEGRADATION IS A SATURATION SYMPTOM, and it was missing here.** The first run
of this script reported "did not reach saturation up to 10,000 rps" for a
service that was serving the popularity fallback to **97% of requests**: the
feature gateway was missing its 8 ms deadline, the orchestrator had no user
vector, the retrieval sidecar refused a feature block of the wrong width, and
the slate fell to a precomputed list. Of 526,980 requests, Triton scored about
6,600 -- 1.3% of the sweep measured the product.

Latency and gRPC errors cannot see any of that. This pipeline degrades rather
than failing, so a degraded request is a 200 with a full slate, returned
FASTER than a healthy one because it skipped the expensive stages. A harness
watching only latency and status codes therefore reports a system that scales
beautifully by not working, and the original saturation test -- p99 past the
budget, or throughput below the offered rate -- could never fire.

So each rate now reads the orchestrator's own counters before and after, and
reports what share of requests actually ran the funnel. `recsys.proto` says it
plainly: "a fallback that is not reported is an outage that looks like a
success." The same is true of a benchmark.

Prerequisites: the orchestrator and both sidecars up, Triton serving the
exported ranker. Every component must be warm before the first measured run,
which is what ``--warmup`` is for: the first request through a cold gRPC
connection pays connection setup, and at these budgets that is most of it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

PROTO = Path("serving/proto/recsys.proto")
CALL = "recsys.v1.Recommender.GetRecommendations"

#: The orchestrator's own scrape endpoint. :9102 and not :9090, which
#: Prometheus itself holds -- see the -metrics flag in cmd/server.
#:
#: Read DIRECTLY rather than through Prometheus. A 15 s scrape interval cannot
#: resolve a 20 s step, and asking the orchestrator itself means the numbers
#: are exact counter deltas rather than rates fitted to samples.
DEFAULT_METRICS = "http://localhost:9102/metrics"

#: Offered rates to sweep, requests per second. docs/design.md assumes a peak
#: of 500, so the sweep brackets it and keeps going: the interesting number is
#: where the service stops keeping up, not whether it survives the assumption.
DEFAULT_RATES = (200, 300, 400, 500)

#: docs/design.md budgets 90ms end to end; the alert rules page at 100ms.
P99_BUDGET_MS = 100.0

#: Share of requests that may degrade before a rate counts as saturated.
#:
#: TIGHTER than the alert rules, which page at 5% fallback and 2% per-source,
#: and deliberately so. Those thresholds answer "is production broken"; this
#: one answers "is this row measuring the product". A table quoting a p99 for
#: a rate at which a tenth of requests skipped the ranker is not a latency
#: measurement of this system.
MAX_DEGRADED = 0.01


@dataclass(frozen=True)
class Counters:
    """The orchestrator's degradation counters at one instant.

    Attributes:
        requests: `recsys_requests_total`, summed over status codes.
        fallback: `recsys_fallback_total` -- retrieval produced nothing at all
            and a popularity slate went out.
        degraded: `recsys_degraded_total` by source. A request can degrade in
            several places at once, so these do NOT sum to a share of traffic
            and are reported per source.
    """

    requests: float = 0.0
    fallback: float = 0.0
    degraded: dict[str, float] = field(default_factory=dict)

    def since(self, earlier: Counters) -> Counters:
        """What happened between ``earlier`` and this reading.

        Counters only climb, and the process is not restarted mid-sweep, so a
        plain subtraction is right. A negative would mean a restart, which
        would invalidate the step rather than needing to be clamped -- so it is
        left visible instead of maxed to zero.
        """
        sources = set(self.degraded) | set(earlier.degraded)
        return Counters(
            requests=self.requests - earlier.requests,
            fallback=self.fallback - earlier.fallback,
            degraded={
                name: self.degraded.get(name, 0.0) - earlier.degraded.get(name, 0.0)
                for name in sources
            },
        )


def _label(series: str, key: str) -> str:
    """One label's value out of a Prometheus series name, or "" if absent."""
    marker = f'{key}="'
    start = series.find(marker)
    if start < 0:
        return ""
    start += len(marker)
    end = series.find('"', start)
    return series[start:end] if end > start else ""


def scrape(url: str) -> Counters | None:
    """Read the orchestrator's counters, or None if it is not reachable.

    None rather than an exception: a sweep that still produces a latency table
    is more useful than one that dies because the metrics port moved, and the
    report says plainly that degradation is unknown. What it must NOT do is
    silently report zero degradation, which is indistinguishable from a healthy
    run and is the exact error this whole addition exists to correct.
    """
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            body = response.read().decode()
    except (urllib.error.URLError, OSError, ValueError):
        return None

    requests_total = 0.0
    fallback = 0.0
    degraded: dict[str, float] = {}
    for line in body.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        series, _, raw = line.rpartition(" ")
        try:
            amount = float(raw)
        except ValueError:
            continue
        if series.startswith("recsys_requests_total"):
            requests_total += amount
        elif series.startswith("recsys_fallback_total"):
            fallback += amount
        elif series.startswith("recsys_degraded_total"):
            source = _label(series, "source") or "(unlabelled)"
            degraded[source] = degraded.get(source, 0.0) + amount

    return Counters(requests=requests_total, fallback=fallback, degraded=degraded)


@dataclass(frozen=True)
class Row:
    """One offered rate and what came back."""

    offered_rps: int
    achieved_rps: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    error_rate: float
    #: None when the metrics endpoint could not be read. Distinct from an
    #: all-zero reading, which would claim a healthy run.
    counters: Counters | None = None

    @property
    def fallback_rate(self) -> float | None:
        if self.counters is None or self.counters.requests <= 0:
            return None
        return self.counters.fallback / self.counters.requests

    @property
    def degraded_rate(self) -> float | None:
        """The WORST single source's share, not the sum.

        A request can degrade in several places at once -- a missed feature
        deadline takes the retriever down with it, because the sidecar refuses
        a feature block of the wrong width -- so summing would double-count one
        failure and read as more than 100%.
        """
        if self.counters is None or self.counters.requests <= 0:
            return None
        if not self.counters.degraded:
            return 0.0
        return max(self.counters.degraded.values()) / self.counters.requests

    @property
    def worst_source(self) -> str:
        if self.counters is None or not self.counters.degraded:
            return ""
        # Bound to a local: mypy cannot carry the narrowing of `self.counters`
        # into the lambda's body.
        degraded = self.counters.degraded
        return max(degraded, key=lambda name: degraded[name])

    @property
    def saturated(self) -> bool:
        """Whether the service stopped keeping up at this rate.

        THREE independent symptoms, any of which counts:

        - Latency past the budget.
        - Throughput that stopped tracking the offered rate.
        - Degradation past MAX_DEGRADED.

        The first two come apart from each other -- a service can hold its
        latency and refuse work, or accept everything and get slower -- which
        is why both were here from the start. The third comes apart from BOTH,
        and its absence is what let this script report 10,000 rps for a service
        serving 97% popularity fallbacks. A degraded request is fast and
        successful; it is simply not the product.
        """
        if self.p99_ms > P99_BUDGET_MS or self.achieved_rps < self.offered_rps * 0.95:
            return True
        worst = self.degraded_rate
        fallback = self.fallback_rate
        return (worst is not None and worst > MAX_DEGRADED) or (
            fallback is not None and fallback > MAX_DEGRADED
        )


def ensure_ghz() -> str:
    path = shutil.which("ghz")
    if path is None:
        raise SystemExit(
            "ghz is not on PATH. Install it: https://ghz.sh/docs/install\n"
            "  go install github.com/bojand/ghz/cmd/ghz@latest"
        )
    return path


def run_one(
    ghz: str,
    target: str,
    rate: int,
    duration: str,
    concurrency: int,
    users: int,
    metrics: str | None = None,
) -> Row:
    """One ghz run at a fixed offered rate, bracketed by a counter read."""
    # {{.RequestNumber}} is ghz's per-request counter. Modulo a user pool so
    # the run touches a realistic number of distinct users rather than one --
    # see the module docstring; this is the difference between measuring the
    # system and measuring its caches.
    data = '{"user_id": "U{{mod .RequestNumber ' + str(users) + '}}", "num_results": 10}'

    # Read immediately before and after, so the delta covers this step and not
    # the gaps between steps. ghz's own warmup requests fall inside the window,
    # which is correct: they are load the service handled.
    before = scrape(metrics) if metrics else None

    completed = subprocess.run(
        [
            ghz,
            "--insecure",
            "--proto",
            str(PROTO),
            "--call",
            CALL,
            "--rps",
            str(rate),
            "--concurrency",
            str(concurrency),
            "--duration",
            duration,
            "--data",
            data,
            "--format",
            "json",
            target,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(f"ghz failed at {rate} rps:\n{completed.stderr.strip()}")

    after = scrape(metrics) if metrics else None
    # Both readings or neither. One of them missing means the window is not a
    # window, and a half-known delta is worse than an admitted gap.
    counters = after.since(before) if (before is not None and after is not None) else None

    report = json.loads(completed.stdout)
    latencies = {entry["percentage"]: entry["latency"] for entry in report["latencyDistribution"]}
    total = int(report.get("count", 0)) or 1
    errors = sum(
        count for code, count in report.get("statusCodeDistribution", {}).items() if code != "OK"
    )

    return Row(
        offered_rps=rate,
        achieved_rps=float(report.get("rps", 0.0)),
        # ghz reports latency in nanoseconds.
        p50_ms=latencies.get(50, 0) / 1e6,
        p95_ms=latencies.get(95, 0) / 1e6,
        p99_ms=latencies.get(99, 0) / 1e6,
        error_rate=errors / total,
        counters=counters,
    )


def _share(value: float | None) -> str:
    """A rate as a percentage, or "?" when the counters could not be read.

    Never "0%" on a failed read. A benchmark that cannot see degradation must
    say so, because a silent zero is indistinguishable from a healthy run and
    is the precise error this column was added to correct.
    """
    return "?" if value is None else f"{value:.2%}"


def render(rows: Sequence[Row], target: str) -> str:
    header = [
        "offered rps",
        "achieved rps",
        "p50 ms",
        "p95 ms",
        "p99 ms",
        "errors",
        "fallback",
        "degraded",
        "worst source",
    ]
    # Every column right-aligned except the last, which is text.
    right = [True] * (len(header) - 1) + [False]
    body = [
        [
            f"{row.offered_rps:,}",
            f"{row.achieved_rps:,.0f}",
            f"{row.p50_ms:.1f}",
            f"{row.p95_ms:.1f}",
            f"{row.p99_ms:.1f}" + (" **\\***" if row.saturated else ""),
            f"{row.error_rate:.2%}",
            _share(row.fallback_rate),
            _share(row.degraded_rate),
            row.worst_source or "--",
        ]
        for row in rows
    ]

    # Padded so the table reads in a terminal as well as rendered Markdown.
    widths = [max(len(cells[i]) for cells in [header, *body]) for i in range(len(header))]

    def line(cells: list[str]) -> str:
        padded = (
            cell.rjust(width) if align else cell.ljust(width)
            for cell, width, align in zip(cells, widths, right, strict=True)
        )
        return "| " + " | ".join(padded) + " |"

    rule = (
        "| "
        + " | ".join(
            "-" * (width - 1) + ":" if align else ":" + "-" * (width - 1)
            for width, align in zip(widths, right, strict=True)
        )
        + " |"
    )
    lines = [line(header), rule, *(line(cells) for cells in body)]

    first = next((row for row in rows if row.saturated), None)
    note = (
        f"\n**\\*** saturated: p99 past {P99_BUDGET_MS:.0f} ms, throughput below the "
        f"offered rate, or more than {MAX_DEGRADED:.0%} of requests degraded.\n"
        "\n`degraded` is the WORST SINGLE SOURCE's share, not a sum: one failure can "
        "degrade several sources at once -- a missed feature deadline takes the "
        "retriever with it, because the sidecar refuses a feature block of the wrong "
        "width -- so summing would double-count it.\n"
    )
    if any(row.counters is None for row in rows):
        note += (
            "\n⚠️ **Degradation could not be read for at least one rate** (`?` above). "
            "Latency alone cannot see a degraded request: this pipeline degrades "
            "rather than failing, so those return 200 with a full slate, FASTER than "
            "a healthy request because they skipped the expensive stages. Treat any "
            "row with `?` as unverified.\n"
        )
    if first is None:
        note += (
            f"\nThe sweep did not reach saturation up to {rows[-1].offered_rps:,} rps. "
            "The ceiling is above the range measured, so this table bounds it from "
            "below and does not locate it -- extend the sweep before quoting a max.\n"
        )
    else:
        cause = "latency or throughput"
        worst, fallback = first.degraded_rate, first.fallback_rate
        if worst is not None and worst > MAX_DEGRADED:
            cause = f"{worst:.1%} of requests degraded ({first.worst_source})"
        elif fallback is not None and fallback > MAX_DEGRADED:
            cause = f"{fallback:.1%} of requests served the popularity fallback"
        note += (
            f"\nSaturation first appears at **{first.offered_rps:,} rps** "
            f"(p99 {first.p99_ms:.1f} ms, achieved {first.achieved_rps:,.0f}) -- "
            f"{cause}. That is the number the HPA in `infra/k8s` scales against.\n"
        )
    return f"Target `{target}`.\n\n" + "\n".join(lines) + "\n" + note


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sweep the orchestrator to saturation.")
    parser.add_argument("--target", default="localhost:50051")
    parser.add_argument("--rates", type=int, nargs="+", default=list(DEFAULT_RATES))
    parser.add_argument("--duration", default="20s")
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument(
        "--users",
        type=int,
        default=10_000,
        help="Distinct user ids to cycle through. One user measures the caches, not the system.",
    )
    parser.add_argument(
        "--warmup",
        default="5s",
        help="An unmeasured run first. A cold gRPC connection pays setup on its first "
        "request, which at a 90ms budget is most of it.",
    )
    parser.add_argument(
        "--metrics",
        default=DEFAULT_METRICS,
        help="The orchestrator's scrape endpoint, read before and after each rate. "
        "Without it this table cannot tell a served recommendation from a fallback. "
        "Empty disables it, and every row is then marked unverified.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Write here instead of stdout.")
    args = parser.parse_args(argv)

    if not PROTO.is_file():
        raise SystemExit(f"{PROTO} not found; run from the repository root")
    ghz = ensure_ghz()

    metrics = args.metrics or None
    if metrics and scrape(metrics) is None:
        # Said ONCE, up front, rather than discovered as a column of "?" after
        # a ten-minute sweep.
        print(f"  warning: {metrics} is unreadable; degradation will not be measured")

    print(f"warming up for {args.warmup}...")
    run_one(ghz, args.target, args.rates[0], args.warmup, args.concurrency, args.users)

    rows: list[Row] = []
    for rate in args.rates:
        print(f"  {rate:>6,} rps for {args.duration}...", flush=True)
        row = run_one(ghz, args.target, rate, args.duration, args.concurrency, args.users, metrics)
        rows.append(row)
        # Kept going past the first saturation point on purpose: one slow
        # sample is noise, and the SHAPE past the knee is what says whether
        # the service degrades or collapses.
        if row.saturated:
            print(
                f"      saturated (p99 {row.p99_ms:.1f} ms, {row.achieved_rps:,.0f} rps, "
                f"{_share(row.degraded_rate)} degraded, {_share(row.fallback_rate)} fallback)"
            )

    report = render(rows, args.target)
    if args.out is None:
        print("\n" + report)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
