"""The drift monitor's arithmetic and its like-for-like row selection.

No Kafka and no Prometheus: both live in the ``serving`` extra, and importing
them here would make this file collect only under ``--extra serving``. The
window is pure, and it is where a drift monitor quietly measures the wrong
thing -- the same place the skew report's own tests aim.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import yaml

from common.drift import PSI_SKEW, bin_edges, histogram_shares
from common.features import USER_FEATURES
from serving.monitoring.window import MIN_ROWS, Feature, Window, load_baseline

ALERTS = Path(__file__).resolve().parents[1] / "serving/monitoring/alerts.yml"

WIDTH = len(USER_FEATURES)
ROWS = MIN_ROWS + 100


def frozen_bins(sample: npt.NDArray[np.float64]) -> dict[str, Any]:
    """One feature's baseline, built the way the dumper builds it.

    Through the same two functions on purpose: a fixture that computed its own
    edges could agree with the monitor while both disagreed with the artifact.
    """
    edges = bin_edges(sample)
    return {"edges": edges.tolist(), "expected": histogram_shares(sample, edges).tolist()}


def write_baseline(tmp_path: Path, features: dict[str, Any]) -> Path:
    path = tmp_path / "drift_baseline.json"
    path.write_text(json.dumps({"version": "test", "features": features}))
    return path


def flat_baseline(tmp_path: Path, sample: npt.NDArray[np.float64]) -> dict[str, Feature]:
    """The same distribution frozen for every feature."""
    features: dict[str, Any] = dict.fromkeys(USER_FEATURES, frozen_bins(sample))
    return load_baseline(write_baseline(tmp_path, features))


def fill(window: Window, values: npt.NDArray[np.float64]) -> None:
    """Serve `values` on every feature."""
    for value in values:
        window.observe(True, [], [float(value)] * WIDTH)


def alert_expr(name: str) -> str:
    """One rule's PromQL, by alert name."""
    body = yaml.safe_load(ALERTS.read_text())
    for group in body["groups"]:
        for rule in group["rules"]:
            if rule.get("alert") == name:
                return str(rule["expr"])
    raise AssertionError(f"{ALERTS} declares no alert named {name}")


class TestLoadBaseline:
    def test_a_missing_feature_is_refused(self, tmp_path: Path) -> None:
        """Three of four monitored is worse than a hard failure: the gap reads
        as a feature that never drifts."""
        thinner: dict[str, Any] = dict.fromkeys(USER_FEATURES[:-1], frozen_bins(np.arange(100.0)))
        with pytest.raises(SystemExit, match=USER_FEATURES[-1]):
            load_baseline(write_baseline(tmp_path, thinner))

    def test_a_constant_feature_loads_without_bins(self, tmp_path: Path) -> None:
        frozen: dict[str, Any] = {name: {"constant": 7.0} for name in USER_FEATURES}
        loaded = load_baseline(write_baseline(tmp_path, frozen))
        assert loaded[USER_FEATURES[0]].edges is None
        assert loaded[USER_FEATURES[0]].constant == 7.0


class TestRowSelection:
    def test_a_degraded_fetch_is_excluded_from_both(self, tmp_path: Path) -> None:
        """No features were served at all. That is availability, not skew, so it
        must leave neither a row nor a dent in the coverage denominator."""
        window = Window(flat_baseline(tmp_path, np.arange(1000.0)))
        window.observe(True, ["features"], [])

        assert window.degraded == 1
        assert len(window.rows) == 0
        # NaN, not 0.0: nothing was eligible, so there is no rate. Zero would
        # read as "no request found a user", which is a different claim.
        assert math.isnan(window.coverage())

    def test_a_store_miss_counts_as_coverage_not_as_mass(self, tmp_path: Path) -> None:
        """A miss serves zeros and training fills a missing row with zeros, so
        pooling them lets the miss RATE look like a shift in the values."""
        window = Window(flat_baseline(tmp_path, np.arange(1000.0)))
        window.observe(True, [], [1.0] * WIDTH)
        window.observe(False, [], [0.0] * WIDTH)

        assert len(window.rows) == 1
        assert window.coverage() == pytest.approx(0.5)

    def test_a_width_change_is_malformed_not_skew(self, tmp_path: Path) -> None:
        """A contract break between gateway and tower would misalign every
        column after it; averaging it in would report that as drift."""
        window = Window(flat_baseline(tmp_path, np.arange(1000.0)))
        window.observe(True, [], [1.0] * (WIDTH + 1))

        assert window.malformed == 1
        assert len(window.rows) == 0

    def test_an_empty_window_says_which_kind_of_empty(self, tmp_path: Path) -> None:
        """The counters exist so that zero rows has a reason attached.

        Three causes look identical on the PSI panel -- nothing consumed, a
        feature fetch that had already failed, and a user the store does not
        hold -- and the monitor is the only thing that can tell them apart.
        """
        window = Window(flat_baseline(tmp_path, np.arange(1000.0)))
        window.observe(True, ["features"], [])
        window.observe(False, [], [0.0] * WIDTH)
        window.observe(True, [], [1.0] * (WIDTH + 1))

        assert len(window.rows) == 0
        assert (window.requests, window.degraded, window.found, window.malformed) == (3, 1, 0, 1)

    def test_the_window_forgets(self, tmp_path: Path) -> None:
        window = Window(flat_baseline(tmp_path, np.arange(1000.0)), size=10)
        fill(window, np.arange(50.0))

        assert len(window.rows) == 10
        assert window.found == 50  # the counter is cumulative, the window is not


class TestDrift:
    def test_nothing_is_reported_below_the_floor(self, tmp_path: Path) -> None:
        """A PSI off forty requests is noise. Reporting no series at all is how
        recsys_drift_window_rows gets to say which it is."""
        window = Window(flat_baseline(tmp_path, np.arange(1000.0)))
        fill(window, np.arange(float(MIN_ROWS - 1)))

        assert window.drift() == {}

    def test_the_same_distribution_reads_as_no_drift(self, tmp_path: Path) -> None:
        training = np.arange(ROWS, dtype=float)
        window = Window(flat_baseline(tmp_path, training))
        fill(window, training)

        for name, value in window.drift().items():
            assert value < 0.01, f"{name} drifted against its own distribution"

    def test_a_shift_is_caught(self, tmp_path: Path) -> None:
        """Serving one value only: every other bin empties, which is the largest
        shift the statistic can see."""
        training = np.arange(ROWS, dtype=float)
        window = Window(flat_baseline(tmp_path, training))
        fill(window, np.full(ROWS, training.max()))

        for name, value in window.drift().items():
            assert value > 0.25, f"{name} reported {value:.3f} for a collapsed distribution"

    def test_values_beyond_training_land_in_the_end_bins(self, tmp_path: Path) -> None:
        """Not dropped. Mass outside training's range is exactly what PSI should
        see, and silently discarding it would read as no drift at all."""
        training = np.arange(ROWS, dtype=float)
        window = Window(flat_baseline(tmp_path, training))
        fill(window, np.full(ROWS, training.max() * 100))

        assert all(value > 0.25 for value in window.drift().values())

    def test_a_constant_feature_is_zero_or_infinite(self, tmp_path: Path) -> None:
        """There is no bin structure to express how large a shift is, so the only
        honest answers are "no change" and "changed"."""
        frozen: dict[str, Any] = {name: {"constant": 3.0} for name in USER_FEATURES}
        baseline = load_baseline(write_baseline(tmp_path, frozen))

        steady = Window(baseline)
        fill(steady, np.full(ROWS, 3.0))
        assert all(value == 0.0 for value in steady.drift().values())

        moved = Window(baseline)
        fill(moved, np.full(ROWS, 4.0))
        assert all(math.isinf(value) for value in moved.drift().values())


class TestTheAlertAgreesWithTheCode:
    """Two thresholds live in a YAML file that cannot import Python.

    The repo's standing rule is that a red panel and a firing alert must not
    disagree; this extends it to the code the alert is about. Without these, the
    comment in alerts.yml claiming 0.25 IS ``PSI_SKEW`` is an assertion nothing
    checks, and the two drift apart the first time one is tuned.
    """

    def test_the_psi_threshold_is_the_skew_band(self) -> None:
        found = re.search(r">\s*([0-9.]+)", alert_expr("RecsysFeatureDrift"))
        assert found, "RecsysFeatureDrift's expr has no threshold to compare"
        assert float(found.group(1)) == PSI_SKEW

    def test_the_starvation_threshold_is_the_row_floor(self) -> None:
        """Below this the window reports no PSI, so the alert that notices an
        absent PSI has to fire at exactly the same number."""
        found = re.search(r"<\s*([0-9.]+)", alert_expr("RecsysDriftMonitorStarved"))
        assert found, "RecsysDriftMonitorStarved's expr has no threshold to compare"
        assert float(found.group(1)) == MIN_ROWS

    def test_the_psi_alert_selects_only_the_alerting_label(self) -> None:
        """The churn set is a label, not a filter. An alert that dropped the
        selector would page on features that move by design."""
        assert 'alerting="true"' in alert_expr("RecsysFeatureDrift")
