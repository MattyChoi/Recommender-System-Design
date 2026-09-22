"""Turning ranking scores into probabilities, and naming what they are of.

A ranker's raw output is not a probability. A lambdarank model is fitted on
*comparisons within a request*, so its scale is arbitrary -- shifting every
score by a constant changes nothing it was trained on. Isotonic regression can
still map those scores monotonically onto observed click rates, which is what
makes them combinable, but the result needs its question stated.

**What the calibrated number answers here.** The labels are "is this the clicked
article among the candidates retrieval returned for this request". With fifty
candidates and one positive, the base rate is 2% by construction. So a
calibrated 0.04 means *twice as likely as an average retrieved candidate to be
the one clicked* -- NOT "a 4% chance this user clicks this article if shown".
Those differ by the candidate pool, and a pool is a denominator.

Calibration matters the moment scores are compared across models or combined
into a weighted objective, because an uncalibrated score has no units and a
weighted sum of things with no units has no meaning.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Bin:
    """One bucket of a reliability diagram.

    Attributes:
        lo: Lower edge of the predicted-probability bucket.
        hi: Upper edge.
        predicted: Mean predicted probability inside it.
        observed: Share of rows in it that were actually clicked.
        rows: How many rows. **The denominator**: a bucket holding nine rows
            can be wildly off and mean nothing, and a reliability diagram that
            plots points without their weights invites reading exactly that.
    """

    lo: float
    hi: float
    predicted: float
    observed: float
    rows: int


@dataclass(frozen=True)
class Calibration:
    """A calibrator's report, on rows it was not fitted on.

    Attributes:
        buckets: The reliability diagram.
        ece: Row-weighted expected calibration error, calibrated.
        log_loss: Mean NLL under the calibrated probabilities.
        ece_raw: ECE of the UNCALIBRATED scores, squashed into [0, 1] only so
            the bucketing is defined. Shows the raw output is nowhere near a
            probability; it is not a fair rival, because a squashed score is
            not a competing estimate of anything.
        ece_constant: **The control that makes ECE readable.** ECE of a model
            that ignores every feature and predicts the base rate for each row.
            It is ~0 by construction. If the calibrated ECE is not clearly
            better than this, the calibrated ECE is evidence of nothing.
        log_loss_constant: Log loss of that same constant model. **This is the
            comparison that has content**: unlike ECE, log loss punishes a
            predictor for failing to discriminate, so beating it means the
            probabilities carry information about which candidate was clicked.
        base_rate: Share of report rows that were the clicked article. **The
            number every probability here should be read against.**
        fitted_rows: Rows the isotonic fit consumed.
        report_rows: Rows the numbers above are computed on.
    """

    buckets: list[Bin]
    ece: float
    log_loss: float
    ece_raw: float
    ece_constant: float
    log_loss_constant: float
    base_rate: float
    fitted_rows: int
    report_rows: int

    @property
    def resolution(self) -> int:
        """Distinct buckets the calibrator's output supports.

        One bucket means it maps every score to the same probability, which is
        the constant model wearing a calibrator's name.
        """
        return len(self.buckets)


def fit_calibrator(scores: npt.NDArray[np.float64], labels: npt.NDArray[np.int64]) -> Any:
    """Isotonic regression from raw scores to click rates.

    Fitted on the VALIDATION window: never on train, where the model already
    fits the labels and the mapping would be the identity in disguise, and
    never on test, where it would tune the thing being reported.

    Isotonic rather than Platt scaling because a ranker's score distribution is
    not sigmoid-shaped -- Platt assumes one and imposes it. Isotonic assumes
    only monotonicity, which is the single property a ranking score is actually
    guaranteed to have.
    """
    from sklearn.isotonic import IsotonicRegression

    return IsotonicRegression(out_of_bounds="clip").fit(scores, labels)


def bin_edges(
    probabilities: npt.NDArray[np.float64], bins: int, strategy: str = "quantile"
) -> npt.NDArray[np.float64]:
    """Bucket boundaries, by quantile of the predictions or by equal width.

    **Quantile by default, and the corpus forces it.** With fifty to a hundred
    candidates per request and one positive, the base rate is under 1% and a
    calibrated score is almost always a small number. Equal-width buckets put
    99.99% of rows in ``[0.0, 0.1)`` and the diagram becomes a single point --
    which is what the first version of this printed, and it looked like perfect
    calibration rather than like no resolution. The manual asks for the rate
    "by decile", which is this.

    Duplicate edges are collapsed, so a step-function calibrator with few
    distinct outputs yields fewer buckets than requested. That is not a
    degradation to paper over: the bucket count IS the calibrator's resolution,
    and it belongs in the output.
    """
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, bins + 1)
    if strategy != "quantile":
        raise ValueError(f"unknown binning strategy {strategy!r}")
    return np.unique(np.quantile(probabilities, np.linspace(0.0, 1.0, bins + 1)))


def reliability(
    probabilities: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int64],
    bins: int = 10,
    strategy: str = "quantile",
) -> list[Bin]:
    """Predicted against observed, bucketed. Empty buckets are dropped, not zeroed."""
    edges = bin_edges(probabilities, bins, strategy)
    if len(edges) < 2:
        # Every prediction identical: one bucket, and the caller should see it.
        edges = np.array([float(probabilities.min()), float(probabilities.max()) + 1e-12])

    out: list[Bin] = []
    for index, (lo, hi) in enumerate(pairwise(edges)):
        last = index == len(edges) - 2
        inside = (probabilities >= lo) & (probabilities <= hi if last else probabilities < hi)
        if not inside.any():
            continue
        out.append(
            Bin(
                lo=float(lo),
                hi=float(hi),
                predicted=float(probabilities[inside].mean()),
                observed=float(labels[inside].mean()),
                rows=int(inside.sum()),
            )
        )
    return out


def expected_calibration_error(
    probabilities: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int64],
    bins: int = 10,
    strategy: str = "quantile",
) -> float:
    """Row-weighted mean gap between predicted and observed.

    Weighted by bucket occupancy, so a nearly-empty bucket at the top of the
    range cannot dominate. Unweighted ECE is a common variant and reports a much
    worse number on a skewed score distribution -- which ours is, since 49 of
    every 50 rows are negatives clustered near zero.

    ⚠️ **A near-zero ECE is not by itself evidence of anything.** A model that
    ignores its input and predicts the base rate for every row scores an ECE of
    exactly zero: it is perfectly calibrated and perfectly useless. Read this
    next to :attr:`Calibration.ece_constant`, and read log loss for whether the
    probabilities discriminate.
    """
    buckets = reliability(probabilities, labels, bins, strategy)
    total = sum(bucket.rows for bucket in buckets)
    if not total:
        return float("nan")
    return sum(bucket.rows / total * abs(bucket.predicted - bucket.observed) for bucket in buckets)


def log_loss(
    probabilities: npt.NDArray[np.float64], labels: npt.NDArray[np.int64], eps: float = 1e-12
) -> float:
    """Mean negative log likelihood of the labels under the probabilities.

    Clipped away from 0 and 1 because isotonic regression returns EXACTLY 0 and
    1 for the ends of its range -- it is a step function fitted on observed
    rates, not a squashed linear model -- and one confident miss at a hard 0
    makes the whole mean infinite. The clip is a property of the estimator, not
    a fudge, so it is named here rather than hidden in a default.
    """
    clipped = np.clip(probabilities, eps, 1.0 - eps)
    return float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1.0 - clipped)))


def calibrate(
    scores: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int64],
    fit_rows: npt.NDArray[np.bool_],
    bins: int = 10,
) -> Calibration:
    """Fit on one set of rows, report on the rest.

    **The split is the whole point.** Fitting isotonic regression and then
    measuring ECE on the same rows reports how well a monotone step function
    can memorise those rows, which is nearly perfectly, and the number looks
    excellent. The caller supplies the mask rather than a fraction because the
    split must be BY USER: two rows from one request are not independent, and a
    row-wise split would put a request's negatives in the fit and its positive
    in the report.

    Args:
        scores: Raw model output, one per candidate row.
        labels: 1 where the candidate is the clicked article.
        fit_rows: True where a row is for fitting, False where it is reported.
        bins: Buckets in the reliability diagram.

    Raises:
        ValueError: If either side is empty, or the fit side has one class --
            isotonic regression on a single class returns a constant and the
            report would be a flat line that looks like perfect calibration.
    """
    report_rows = ~fit_rows
    if not fit_rows.any() or not report_rows.any():
        raise ValueError("calibration needs rows on both sides of the split")
    if len(np.unique(labels[fit_rows])) < 2:
        raise ValueError("the calibration fold holds one class; isotonic would return a constant")

    calibrator = fit_calibrator(scores[fit_rows], labels[fit_rows])
    probabilities = np.asarray(calibrator.predict(scores[report_rows]), dtype=float)
    held_labels = labels[report_rows]

    # The uncalibrated baseline, min-max squashed ONLY so that bucketing into
    # [0, 1] is defined. It is not a probability and is not claimed to be; it
    # exists so the calibrated ECE is quoted against something.
    raw = scores[report_rows]
    span = float(raw.max() - raw.min())
    squashed = (raw - raw.min()) / span if span else np.zeros_like(raw)

    # The control: predict the report fold's own base rate for every row. It is
    # the best a model with no information can do, and it is what stops a tiny
    # ECE from being mistaken for a result.
    base_rate = float(held_labels.mean())
    constant = np.full_like(probabilities, base_rate)

    return Calibration(
        buckets=reliability(probabilities, held_labels, bins),
        ece=expected_calibration_error(probabilities, held_labels, bins),
        log_loss=log_loss(probabilities, held_labels),
        ece_raw=expected_calibration_error(squashed, held_labels, bins),
        ece_constant=expected_calibration_error(constant, held_labels, bins),
        log_loss_constant=log_loss(constant, held_labels),
        base_rate=base_rate,
        fitted_rows=int(fit_rows.sum()),
        report_rows=int(report_rows.sum()),
    )


def render(buckets: list[Bin], ece: float, base_rate: float) -> str:
    """The reliability table, with the base rate it should be read against."""
    head = f"{'bucket':>14}{'rows':>10}{'predicted':>11}{'observed':>10}{'gap':>9}"
    lines = [head, "-" * len(head)]
    for bucket in buckets:
        gap = bucket.predicted - bucket.observed
        # Three significant figures, not two decimals. Every edge on this corpus
        # sits below 0.01, so a fixed-decimal format prints ten buckets all
        # labelled "[0.00, 0.00)" -- a table that cannot show its own x axis.
        edges = f"[{bucket.lo:.3g}, {bucket.hi:.3g})"
        lines.append(
            f"  {edges:<14}{bucket.rows:>10,}"
            f"{bucket.predicted:>11.4f}{bucket.observed:>10.4f}{gap:>+9.4f}"
        )
    lines += [
        "-" * len(head),
        f"  ECE (row-weighted) {ece:.4f}   base rate {base_rate:.4f}",
        "",
        "  `observed` is the share of rows in the bucket that were the clicked",
        "  article, out of the candidates retrieval returned -- not a click-through",
        "  rate on a shown impression. The pool is part of the number.",
    ]
    return "\n".join(lines)
