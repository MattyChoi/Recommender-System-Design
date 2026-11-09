"""The skew job's arithmetic and its like-for-like row selection.

No Spark and no Kafka: the parsing and the statistics are pure, and they are
where a skew report quietly measures the wrong thing.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from common.features import USER_FEATURES
from scripts.writeups.skew_report import (
    BEGIN,
    END,
    PSI_SKEW,
    Row,
    ks_statistic,
    parse_served,
    psi,
    splice,
)

WIDTH = len(USER_FEATURES)


def record(**fields: object) -> str:
    base: dict[str, object] = {
        "request_id": "r",
        "user_feats": [1.0] * WIDTH,
        "user_found": True,
        "degraded": [],
    }
    base.update(fields)
    return json.dumps(base)


class TestParsing:
    def test_store_misses_are_counted_not_compared(self) -> None:
        """Zero defaults pooled in would turn the miss RATE into a fake shift."""
        served = parse_served([record(), record(user_found=False, user_feats=[0.0] * WIDTH)])

        assert served.requests == 2
        assert served.found == 1
        assert served.feats.shape == (1, WIDTH)

    def test_a_failed_fetch_is_excluded_from_coverage_too(self) -> None:
        served = parse_served([record(degraded=["features"]), record()])

        assert served.fetch_failed == 1
        assert served.found == 1

    def test_a_wrong_width_vector_is_malformed_not_skew(self) -> None:
        served = parse_served([record(user_feats=[1.0] * (WIDTH - 1))])

        assert served.malformed == 1
        assert served.found == 0

    def test_consume_output_columns_are_skipped(self) -> None:
        """`make consume` prefixes timestamp, kind and key before the JSON."""
        line = "1731150000000\tcreate\tU1\t" + record()
        served = parse_served([line, "# timed out after 1 records"])

        assert served.found == 1


class TestPSI:
    def test_identical_distributions_score_near_zero(self) -> None:
        rng = np.random.default_rng(0)
        sample = rng.normal(size=20_000)

        assert psi(sample, rng.normal(size=20_000)) < 0.01

    def test_a_shifted_distribution_is_called_skew(self) -> None:
        rng = np.random.default_rng(0)

        assert psi(rng.normal(size=20_000), rng.normal(1.0, size=20_000)) > PSI_SKEW

    def test_mass_beyond_trainings_range_is_not_dropped(self) -> None:
        """Open outer edges: serving values past train's max land in the top bin."""
        train = np.linspace(0, 1, 1_000)

        assert psi(train, np.full(1_000, 50.0)) > PSI_SKEW

    def test_a_count_feature_with_repeated_quantiles_stays_finite(self) -> None:
        train = np.concatenate([np.zeros(900), np.arange(1, 101, dtype=float)])

        assert np.isfinite(psi(train, train.copy()))

    @pytest.mark.parametrize(("value", "status"), [(0.05, "OK"), (0.2, "WATCH"), (0.3, "SKEW")])
    def test_status_bands(self, value: float, status: str) -> None:
        assert Row("f", 0, 0, 0, value).status == status


def test_ks_matches_the_textbook_values() -> None:
    a = np.array([1.0, 2.0, 3.0])
    assert ks_statistic(a, a.copy()) == 0.0
    assert ks_statistic(a, a + 10) == 1.0
    # ECDFs at 2: a holds 2/3, b = [2, 3, 4] holds 1/3.
    assert ks_statistic(a, np.array([2.0, 3.0, 4.0])) == pytest.approx(1 / 3)


def test_splice_keeps_the_prose_around_the_table() -> None:
    doc = f"intro\n{BEGIN}\nold\n{END}\noutro\n"

    assert splice(doc, "new") == f"intro\n{BEGIN}\nnew\n{END}\noutro\n"


def test_splice_refuses_a_document_without_markers() -> None:
    with pytest.raises(SystemExit):
        splice("no markers", "new")
