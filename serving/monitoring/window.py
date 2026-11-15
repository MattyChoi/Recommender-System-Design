"""The drift monitor's arithmetic: a rolling window of served rows, and its PSI.

Separate from ``drift.py`` so this half imports numpy and ``common`` and nothing
else. ``drift.py`` pulls in Kafka and ``prometheus_client``, both of which live
in the ``serving`` extra -- a test that imported them at module scope would
collect only under ``uv sync --extra serving`` and error everywhere else, which
is precisely the defect ``ci.yml`` carried for weeks on ``onnxruntime``.

So the part worth testing has no serving dependencies, and the part with the
dependencies has nothing worth testing.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from common.drift import histogram_shares, psi_from_shares
from common.features import USER_FEATURES

# Matches MIN_SERVED_ROWS in the skew report, and for the same reason: a
# confident SKEW off forty requests is worse than reporting nothing.
MIN_ROWS = 500
WINDOW_ROWS = 5_000

# Features whose distribution turns over by DESIGN -- observed, never alerted
# on. Empty today, and the reason matters: ImpressionContext carries
# `user_feats` only, so this monitor cannot see an item feature at all. On a
# news corpus `item_age_hours` and `item_impressions_24h` are what churn (46.2%
# of the dev catalogue never appears in train), so this set stops being empty
# the moment the impression log starts carrying them.
EXPECTED_CHURN: frozenset[str] = frozenset()

FEATURES_DEGRADED = "features"


@dataclass(frozen=True)
class Feature:
    """One feature's frozen training distribution.

    Attributes:
        edges: Finite bin edges, or None when training was constant.
        expected: Share of training mass per bin, aligned with ``edges``.
        constant: The single training value, when there was only one.
    """

    edges: npt.NDArray[np.float64] | None
    expected: npt.NDArray[np.float64] | None
    constant: float | None


def load_baseline(path: Path) -> dict[str, Feature]:
    """Read what ``make drift-baseline`` froze.

    Raises:
        SystemExit: If a feature the serving path reports is absent from the
            baseline. Monitoring three of four while reporting nothing about the
            fourth is the failure this refuses -- the gap would be indis-
            tinguishable from a feature that never drifts.
    """
    body = json.loads(path.read_text())
    frozen = body.get("features") or {}

    missing = [name for name in USER_FEATURES if name not in frozen]
    if missing:
        raise SystemExit(
            f"{path} has no bins for {', '.join(missing)}; "
            "regenerate it with `make drift-baseline` against this feature set"
        )

    out: dict[str, Feature] = {}
    for name in USER_FEATURES:
        entry = frozen[name]
        if "constant" in entry:
            out[name] = Feature(edges=None, expected=None, constant=float(entry["constant"]))
            continue
        out[name] = Feature(
            edges=np.asarray(entry["edges"], dtype=np.float64),
            expected=np.asarray(entry["expected"], dtype=np.float64),
            constant=None,
        )
    return out


class Window:
    """A rolling window of served feature rows, and its drift against training."""

    def __init__(self, baseline: dict[str, Feature], size: int = WINDOW_ROWS) -> None:
        self.baseline = baseline
        self.rows: deque[list[float]] = deque(maxlen=size)
        self.requests = 0
        self.found = 0
        self.degraded = 0
        self.malformed = 0

    def observe(self, user_found: bool, degraded: Sequence[str], feats: Sequence[float]) -> None:
        """Apply the report's three like-for-like rules to one served request.

        Primitives rather than a record, so the rules are independent of the
        wire format: the batch report reads JSON lines and the monitor reads
        protobuf, and they must not diverge on what counts.
        """
        self.requests += 1

        # A failed fetch served no features at all -- an availability incident,
        # not a shift. Pooling it would move every mean.
        if FEATURES_DEGRADED in degraded:
            self.degraded += 1
            return

        # A store miss serves zeros, and training fills a missing row with zeros
        # too. Pooling those lets the miss RATE masquerade as a shift in the
        # values, so it is counted as coverage instead.
        if not user_found:
            return

        if len(feats) != len(USER_FEATURES):
            # A width change is a contract break between gateway and tower, and
            # would misalign every column after it.
            self.malformed += 1
            return

        self.found += 1
        self.rows.append([float(value) for value in feats])

    def coverage(self) -> float:
        """Found rate over requests that got as far as a feature fetch.

        NaN, not 0.0, when nothing was eligible. Zero would read as "no request
        found a user", which is a measurement; nothing was measured. Prometheus
        carries NaN and Grafana renders it as No data, so the two states stay
        distinguishable on the panel instead of both showing 0%.
        """
        eligible = self.requests - self.degraded
        return self.found / eligible if eligible else float("nan")

    def drift(self) -> dict[str, float]:
        """PSI per feature, or ``{}`` while the window is too small to judge."""
        if len(self.rows) < MIN_ROWS:
            return {}

        block = np.asarray(self.rows, dtype=np.float64)
        out: dict[str, float] = {}
        for column, name in enumerate(USER_FEATURES):
            feature = self.baseline[name]
            sample = block[:, column]

            if feature.edges is None or feature.expected is None:
                # Training was constant: any other value is a shift, and there
                # is no bin structure to say how large one.
                out[name] = 0.0 if np.all(sample == feature.constant) else float("inf")
                continue

            out[name] = psi_from_shares(feature.expected, histogram_shares(sample, feature.edges))
        return out
