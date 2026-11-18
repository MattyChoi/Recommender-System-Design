"""Population Stability Index, and the bands a PSI is read against."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# The conventional bands. <0.1 stable, 0.1-0.25 watch, >0.25 shifted.
PSI_WATCH = 0.1
PSI_SKEW = 0.25

PSI_BINS = 10

# Floors an empty bin so PSI stays finite. Small enough that one empty bin out
# of ten moves PSI only when the other side genuinely holds mass there.
PSI_EPSILON = 1e-6


def band(value: float) -> str:
    """``OK``, ``WATCH`` or ``SKEW``. The report's status and the alert's
    threshold read the same number through this."""
    if value > PSI_SKEW:
        return "SKEW"
    if value > PSI_WATCH:
        return "WATCH"
    return "OK"


def bin_edges(expected: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """TRAINING's deciles, de-duplicated. The edges a PSI is binned on.

    Binned on the expected distribution and never on a fresh quantile of the
    actual one: edges that move with the data cannot see the shift they exist to
    detect.

    De-duplicated because a count feature with most of its mass at zero has
    repeated quantiles, and a zero-width bin holds nothing on either side while
    still contributing epsilon noise.

    Returns the edges FINITE, which is what gets persisted into an artifact --
    JSON has no infinity. :func:`histogram_shares` opens the outer pair, so that
    convention lives in one place.

    A size-1 result means training was constant on this feature; callers decide
    what that means, because there are no bins to compare.
    """
    edges: npt.NDArray[np.float64] = np.unique(
        np.quantile(expected, np.linspace(0, 1, PSI_BINS + 1))
    )
    return edges


def histogram_shares(
    sample: npt.NDArray[np.float64], edges: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Share of ``sample`` in each bin, epsilon-floored.

    The outer edges are opened to infinity here, so a serving value beyond
    training's range lands in an end bin -- which is exactly the mass PSI should
    see -- rather than being dropped and making the shift invisible.
    """
    opened = edges.astype(float).copy()
    opened[0], opened[-1] = -np.inf, np.inf
    counts = np.histogram(sample, opened)[0]
    return np.maximum(counts / max(sample.size, 1), PSI_EPSILON)


def psi_from_shares(expected: npt.NDArray[np.float64], actual: npt.NDArray[np.float64]) -> float:
    """PSI from two per-bin share vectors.

    The monitor's entry point: it keeps rolling counts rather than samples, so
    it never has an array to pass :func:`psi`.
    """
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def psi(expected: npt.NDArray[np.float64], actual: npt.NDArray[np.float64]) -> float:
    """Population Stability Index of ``actual`` against ``expected``'s bins.

    The report's entry point, holding both samples.
    """
    edges = bin_edges(expected)
    if edges.size < 2:
        # Training was constant. Any serving value elsewhere is a shift, and
        # there is no bin structure to express how large one.
        return 0.0 if np.all(actual == edges[0]) else float("inf")

    return psi_from_shares(histogram_shares(expected, edges), histogram_shares(actual, edges))
