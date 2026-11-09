"""``make skew`` -- the training/serving skew report, from the impression log.

Compares the user features the orchestrator ACTUALLY SERVED (``ImpressionContext.
user_feats`` on the ``impression_context`` topic) against what training read for
the same columns (gold ``training_examples``). That is the only comparison that
measures the model's skew rather than the store's drift: a later re-read of the
online store would describe the store at report time, not what the model saw.

Per feature it reports the two means, the two-sample KS statistic and the
Population Stability Index, and calls a status from the PSI on the conventional
bands (<0.1 stable, 0.1-0.25 watch, >0.25 skew). PSI decides because it is
binned on TRAINING's quantiles, so it answers "has the mass moved between the
bins the model learned on"; KS is reported beside it as the bin-free check.

Rows compared like for like, which is the part that is easy to get wrong:

* **One row per request on both sides.** Served rows are one per slate. Training
  is de-duplicated to one row per ``impression_id``, because ``training_examples``
  carries a row per candidate and would weight a request by its slate length.
* **Store misses are a coverage number, not distribution mass.** A miss serves
  zero defaults; training fills a missing row with zeros too. Pooling those
  would let the miss RATE masquerade as a shift in the values. So rows with
  ``user_found`` false (served) or ``has_user_features`` false (training) are
  counted in the coverage line and excluded from the per-feature rows.
* **Requests whose feature fetch failed are dropped.** ``degraded`` containing
  ``features`` means no features were served at all; that is an availability
  incident, reported as a count, not skew.

Input is newline-delimited ``ImpressionContext`` JSON -- the file sink's output,
or ``make consume TOPIC=impression_context N=...`` redirected to a file (its
leading timestamp/kind/key columns are skipped).

The report is written between two markers in ``docs/skew_report.md`` so the
hand-written analysis around it survives regeneration. Generated, never
hand-edited: a hand-maintained skew report is a claim, not a measurement.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from common.features import USER_FEATURES, USER_FLAGS

BEGIN = "<!-- skew-report:begin -->"
END = "<!-- skew-report:end -->"

PSI_WATCH = 0.1
PSI_SKEW = 0.25
PSI_BINS = 10
# Floors an empty bin so PSI stays finite. Small enough that one empty bin out
# of ten moves PSI only when the other side genuinely holds mass there.
PSI_EPSILON = 1e-6
# Below this many served rows the KS and PSI figures are noise, and saying so
# is better than printing a confident SKEW off forty requests.
MIN_SERVED_ROWS = 500

FEATURES_DEGRADED = "features"


@dataclass(frozen=True)
class Served:
    """What the impression log says was served."""

    feats: npt.NDArray[np.float64]  # [found requests, len(USER_FEATURES)]
    requests: int  # parsed requests, before any exclusion
    found: int
    fetch_failed: int
    malformed: int


@dataclass(frozen=True)
class Row:
    feature: str
    train_mean: float
    serve_mean: float
    ks: float
    psi: float

    @property
    def status(self) -> str:
        if self.psi > PSI_SKEW:
            return "SKEW"
        if self.psi > PSI_WATCH:
            return "WATCH"
        return "OK"


def parse_served(lines: Iterable[str]) -> Served:
    """Parse ImpressionContext JSON lines into the found-user feature block."""
    width = len(USER_FEATURES)
    rows: list[list[float]] = []
    requests = found = fetch_failed = malformed = 0

    for line in lines:
        start = line.find("{")
        if start < 0:
            continue  # decode_events' "# timed out" and blank lines
        try:
            record = json.loads(line[start:])
        except json.JSONDecodeError:
            malformed += 1
            continue

        requests += 1
        if FEATURES_DEGRADED in (record.get("degraded") or []):
            fetch_failed += 1
            continue
        if not record.get("user_found", False):
            continue
        feats = record.get("user_feats") or []
        if len(feats) != width:
            # A width change is a contract break between gateway and tower, not
            # a skew; it would also misalign every column after it.
            malformed += 1
            continue
        found += 1
        rows.append([float(value) for value in feats])

    block = np.asarray(rows, dtype=np.float64).reshape(-1, width)
    return Served(block, requests, found, fetch_failed, malformed)


def psi(expected: npt.NDArray[np.float64], actual: npt.NDArray[np.float64]) -> float:
    """Population Stability Index of ``actual`` against ``expected``'s quantile bins.

    Edges are TRAINING's deciles, de-duplicated: a count feature with most mass
    at zero has repeated quantiles, and a zero-width bin would hold nothing on
    either side and contribute only epsilon noise. The outer edges are opened
    to infinity so serving values beyond training's range land in the end bins
    -- which is exactly the mass PSI should see -- instead of being dropped.
    """
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, PSI_BINS + 1)))
    if edges.size < 2:
        # Training was constant. Any serving value elsewhere is a shift.
        return 0.0 if np.all(actual == edges[0]) else float("inf")
    edges[0], edges[-1] = -np.inf, np.inf

    e = np.histogram(expected, edges)[0] / expected.size
    a = np.histogram(actual, edges)[0] / actual.size
    e, a = np.maximum(e, PSI_EPSILON), np.maximum(a, PSI_EPSILON)
    return float(np.sum((a - e) * np.log(a / e)))


def ks_statistic(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
    """Two-sample Kolmogorov-Smirnov D: the largest gap between the two ECDFs.

    In numpy rather than scipy, which is only a transitive dependency here. The
    supremum is attained at a sample point, so evaluating both ECDFs at the
    pooled sorted values is exact -- ties included, via ``side="right"``.
    """
    a, b = np.sort(a), np.sort(b)
    points = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, points, side="right") / a.size
    cdf_b = np.searchsorted(b, points, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def compare(train: npt.NDArray[np.float64], served: npt.NDArray[np.float64]) -> list[Row]:
    rows = []
    for column, name in enumerate(USER_FEATURES):
        t, s = train[:, column], served[:, column]
        rows.append(
            Row(
                feature=f"user.{name}",
                train_mean=float(t.mean()),
                serve_mean=float(s.mean()),
                ks=ks_statistic(t, s),
                psi=psi(t, s),
            )
        )
    return rows


def render(
    rows: list[Row], served: Served, train_found: int, train_requests: int, split: str
) -> str:
    lines = [
        f"Served: {served.requests:,} requests parsed; {served.found:,} with a stored user "
        f"row; {served.fetch_failed:,} excluded because the feature fetch failed; "
        f"{served.malformed:,} malformed. Training: `{split}`, {train_requests:,} "
        f"requests (one per impression).",
        "",
    ]
    if served.found < MIN_SERVED_ROWS:
        lines.append(
            f"**Too few served rows to call skew** ({served.found:,} < {MIN_SERVED_ROWS:,}). "
            "Figures below are shown, not judged."
        )
        lines.append("")

    lines += [
        "| Feature | Train mean | Serve mean | KS-stat | PSI | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    judged = served.found >= MIN_SERVED_ROWS
    for row in rows:
        status = row.status if judged else "—"
        lines.append(
            f"| {row.feature} | {row.train_mean:.4g} | {row.serve_mean:.4g} | "
            f"{row.ks:.3f} | {row.psi:.3f} | {status} |"
        )

    train_rate = train_found / train_requests if train_requests else 0.0
    serve_rate = served.found / (served.requests - served.fetch_failed or 1)
    lines += [
        "",
        f"**Coverage** (`has_user_features` / `user_found`): train {train_rate:.1%}, "
        f"served {serve_rate:.1%}. Rows without a stored user are excluded above, "
        "so a coverage gap shows here and not as a shifted mean.",
    ]
    return "\n".join(lines)


def splice(document: str, body: str) -> str:
    """Replace the text between the markers, refusing a document without them."""
    head, found_begin, rest = document.partition(BEGIN)
    _, found_end, tail = rest.partition(END)
    if not found_begin or not found_end:
        raise SystemExit(f"docs/skew_report.md has no {BEGIN} ... {END} block to replace")
    return f"{head}{BEGIN}\n{body}\n{END}{tail}"


def load_training(split: str) -> tuple[npt.NDArray[np.float64], int]:
    """Training's user block for found users, and the request count.

    Spark, because gold is Parquet written by Spark under the configured storage
    backend; this runs offline, beside `make parity`, never on the serving path.
    """
    from common.config import load_settings
    from common.spark import get_spark
    from common.utils import read_gold

    settings = load_settings()
    spark = get_spark(settings, app="skew-report")
    frame = (
        read_gold(spark, settings, f"training_examples/{split}")
        .select("impression_id", *USER_FEATURES, *USER_FLAGS)
        .dropDuplicates(["impression_id"])
        .toPandas()
    )
    found = frame[USER_FLAGS[0]].fillna(False).astype(bool).to_numpy()
    block = frame.loc[found, list(USER_FEATURES)].to_numpy(dtype=np.float64)
    return block, len(frame)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Training/serving skew from the impression log.")
    parser.add_argument(
        "--served",
        type=Path,
        required=True,
        help="ImpressionContext JSON lines: the file sink's output or `make consume` redirected.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", type=Path, default=Path("docs/skew_report.md"))
    args = parser.parse_args(argv)

    with args.served.open() as handle:
        served = parse_served(handle)
    if served.found == 0:
        raise SystemExit(
            f"{args.served}: no served rows with a stored user ({served.requests} parsed). "
            "Is the orchestrator running with -kafka or -impressions, and was "
            "TOPIC=impression_context consumed?"
        )

    train, train_requests = load_training(args.split)
    rows = compare(train, served.feats)
    body = render(rows, served, len(train), train_requests, args.split)

    args.out.write_text(splice(args.out.read_text(), body))
    print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
