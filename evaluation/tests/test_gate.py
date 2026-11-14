"""The promotion gate's decision logic.

Pure arithmetic over small inputs: no cards, no Spark, no GPU. That is the point
of splitting the gate from the scoring -- the half that decides can be tested
everywhere, and the half that measures needs the corpus.

The cases worth having are the ones where the two thresholds disagree, because
either one alone is a gate that passes a real regression.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from evaluation.offline.gate import (
    COHORTS,
    Check,
    Pointer,
    Verdict,
    check,
    gauc_checks,
    load_pointer,
    render,
)
from evaluation.offline.slates import SlateRows
from evaluation.offline.slates import save as save_slates
from evaluation.offline.stats import PairedResult

TOLERANCE = 0.005


def paired(difference: float, lo: float, hi: float, users: int = 1000) -> PairedResult:
    """A PairedResult whose `significant` follows stats.py's own rule.

    Derived rather than passed in, so a fixture cannot claim a significance the
    real bootstrap would not have reported for the same interval.
    """
    return PairedResult(
        difference=difference,
        lo=lo,
        hi=hi,
        n_users=users,
        significant=bool(lo > 0.0 or hi < 0.0),
    )


def card(path: Path, gauc: dict[str, float]) -> Path:
    """A minimal result card: just the cohorts and their GAUC."""
    body = {"cohorts": {name: {"gauc": value} for name, value in gauc.items()}}
    path.write_text(json.dumps(body))
    return path


def rows(card_path: Path, auc: dict[str, list[float]], users: list[str]) -> Path:
    """Per-slate rows beside a card, one slate per user.

    One slate each on purpose: it makes the per-user mean equal to the slate's
    own AUC, so a test states the per-user number it wants rather than a set of
    slates that happens to average to it.
    """
    return save_slates(
        card_path.stem,
        {
            cohort: SlateRows(auc=values, users=users, weight=[1.0] * len(values))
            for cohort, values in auc.items()
        },
        card_path.parent / "slates",
    )


def test_a_drop_past_the_floor_refuses() -> None:
    got = check("recall", 0.3700, 0.3800, TOLERANCE, None)
    assert not got.allowed
    assert "past the" in got.reason


def test_a_drop_inside_the_floor_promotes_when_the_pairing_cannot_resolve_it() -> None:
    """The ordinary case: a small wobble the rows say nothing about."""
    got = check("recall", 0.3780, 0.3800, TOLERANCE, paired(-0.0020, -0.0051, +0.0012))
    assert got.allowed


def test_a_drop_inside_the_floor_refuses_when_the_pairing_resolves_it() -> None:
    """**The case a threshold alone cannot catch.**

    0.002 is well inside the 0.005 floor, so a relative-or-absolute gate waves
    it through -- but the per-user interval excludes zero, so the drop is real
    and will not average away at the next retrain.
    """
    got = check("recall", 0.3780, 0.3800, TOLERANCE, paired(-0.0020, -0.0031, -0.0009))
    assert not got.allowed
    assert "excludes zero" in got.reason


def test_a_significant_improvement_is_not_mistaken_for_a_regression() -> None:
    """The paired half must fire on direction, not on significance alone."""
    got = check("recall", 0.3850, 0.3800, TOLERANCE, paired(+0.0050, +0.0031, +0.0069))
    assert got.allowed


def test_an_undefined_metric_refuses() -> None:
    """Fail-closed, following the index gate: what cannot be evaluated aborts."""
    assert not check("recall", math.nan, 0.38, TOLERANCE, None).allowed
    assert not check("recall", 0.38, math.nan, TOLERANCE, None).allowed


def test_gauc_is_checked_on_every_cohort(tmp_path: Path) -> None:
    baseline = card(tmp_path / "base.json", dict.fromkeys(COHORTS, 0.60))
    candidate = card(tmp_path / "cand.json", dict.fromkeys(COHORTS, 0.60))
    got = gauc_checks(candidate, baseline, TOLERANCE)
    assert [item.metric for item in got] == [f"gauc[{name}]" for name in COHORTS]
    assert all(item.allowed for item in got)


def test_a_cohort_regression_refuses_even_when_overall_holds(tmp_path: Path) -> None:
    """Why this is per cohort. The aggregate is dominated by warm traffic, so a
    cold-item collapse can hide behind an overall number that barely moves."""
    baseline = card(tmp_path / "base.json", dict.fromkeys(COHORTS, 0.60))
    candidate = card(tmp_path / "cand.json", {**dict.fromkeys(COHORTS, 0.60), "cold_item": 0.40})

    got = {item.metric: item for item in gauc_checks(candidate, baseline, TOLERANCE)}
    assert got["gauc[overall]"].allowed
    assert not got["gauc[cold_item]"].allowed
    assert not Verdict(tuple(got.values())).allowed


def test_a_cohort_the_candidate_stopped_emitting_refuses(tmp_path: Path) -> None:
    """Promotion by omission is the failure the explicit cohort list prevents: a
    candidate that simply drops a cohort must not be read as unchanged on it."""
    baseline = card(tmp_path / "base.json", dict.fromkeys(COHORTS, 0.60))
    thinner = {name: 0.60 for name in COHORTS if name != "cold_item"}
    candidate = card(tmp_path / "cand.json", thinner)

    got = {item.metric: item for item in gauc_checks(candidate, baseline, TOLERANCE)}
    assert not got["gauc[cold_item]"].allowed
    assert "absent from the candidate card" in got["gauc[cold_item]"].reason


def test_gauc_reports_its_pairing_as_unavailable_not_as_passed() -> None:
    """A card with no per-slate rows beside it has no paired test to run. It
    must read as UNAVAILABLE: a blank would be taken for a clean result."""
    got = check("gauc[overall]", 0.60, 0.60, TOLERANCE, None)
    assert got.paired is None
    assert "paired UNAVAILABLE" in render(Verdict((got,)))


def test_gauc_pairs_from_the_rows_beside_the_card(tmp_path: Path) -> None:
    """**The case the floor alone cannot catch, now for GAUC.**

    0.002 is well inside the 0.005 floor, so the floor promotes it -- which is
    what this gate did for GAUC for as long as the cards held nothing but a
    scalar. Every user drops by that same 0.002, so the paired interval cannot
    straddle zero, and the drop is real rather than a wobble.
    """
    users = [f"U{index}" for index in range(200)]
    baseline = card(tmp_path / "base.json", dict.fromkeys(COHORTS, 0.600))
    candidate = card(tmp_path / "cand.json", dict.fromkeys(COHORTS, 0.598))
    rows(baseline, {"overall": [0.600] * len(users)}, users)
    rows(candidate, {"overall": [0.598] * len(users)}, users)

    got = {item.metric: item for item in gauc_checks(candidate, baseline, TOLERANCE)}
    assert not got["gauc[overall]"].allowed
    assert "excludes zero" in got["gauc[overall]"].reason

    # Rows were written for one cohort only, and the other four keep the old
    # behaviour rather than refusing for want of them.
    assert got["gauc[cold_item]"].allowed
    assert got["gauc[cold_item]"].paired is None


def test_gauc_falls_back_to_the_floor_when_no_rows_were_written(tmp_path: Path) -> None:
    """Every card scored before the rows existed is in this state, the one
    PRODUCTION.json points at included. UNAVAILABLE plus the floor, never a
    refusal: refusing would break every comparison the gate can make today."""
    baseline = card(tmp_path / "base.json", dict.fromkeys(COHORTS, 0.600))
    candidate = card(tmp_path / "cand.json", dict.fromkeys(COHORTS, 0.598))

    got = gauc_checks(candidate, baseline, TOLERANCE)
    assert all(item.allowed for item in got)
    assert all(item.paired is None for item in got)


def test_gauc_measures_over_users_not_over_the_cards_scalar(tmp_path: Path) -> None:
    """The floor and the interval have to share a unit.

    The card's GAUC is weighted by impression size; the paired test is a mean
    over users. Where the rows exist BOTH halves come from the rows, so a card
    claiming no change cannot promote a candidate whose per-user rows
    collapsed. Taking the floor from the card and the interval from the rows
    would be ci95_by_impression_DO_NOT_QUOTE wearing a different label.
    """
    users = [f"U{index}" for index in range(200)]
    baseline = card(tmp_path / "base.json", dict.fromkeys(COHORTS, 0.60))
    candidate = card(tmp_path / "cand.json", dict.fromkeys(COHORTS, 0.60))
    rows(baseline, {"overall": [0.60] * len(users)}, users)
    rows(candidate, {"overall": [0.50] * len(users)}, users)

    got = {item.metric: item for item in gauc_checks(candidate, baseline, TOLERANCE)}
    assert not got["gauc[overall]"].allowed
    assert "past the" in got["gauc[overall]"].reason
    assert got["gauc[overall]"].candidate == 0.50


def test_gauc_refuses_cards_scored_over_disjoint_users(tmp_path: Path) -> None:
    """Fail-closed, as recall_check does. Two cards scored over different users
    have per-user means that compare happily and mean nothing."""
    baseline = card(tmp_path / "base.json", dict.fromkeys(COHORTS, 0.60))
    candidate = card(tmp_path / "cand.json", dict.fromkeys(COHORTS, 0.60))
    rows(baseline, {"overall": [0.60] * 50}, [f"U{index}" for index in range(50)])
    rows(candidate, {"overall": [0.60] * 50}, [f"V{index}" for index in range(50)])

    got = {item.metric: item for item in gauc_checks(candidate, baseline, TOLERANCE)}
    assert not got["gauc[overall]"].allowed
    assert "cannot pair these cards" in got["gauc[overall]"].reason


def test_an_empty_verdict_is_not_a_pass() -> None:
    """A gate that compared nothing has not approved anything. Without this, a
    pointer naming no stages would promote everything."""
    assert not Verdict(()).allowed
    assert "REFUSED" in render(Verdict(()))


def test_one_failure_sinks_the_verdict() -> None:
    passing = check("recall", 0.38, 0.38, TOLERANCE, None)
    failing = check("gauc[cold_item]", 0.40, 0.60, TOLERANCE, None)
    assert not Verdict((passing, failing)).allowed
    assert Verdict((passing,)).allowed


def test_a_pointer_leaves_undeclared_stages_as_none(tmp_path: Path) -> None:
    """Absent means not declared. `run` turns that into a refusal when only one
    side declares a stage, which is checked where that decision is made."""
    path = tmp_path / "pointer.json"
    path.write_text(json.dumps({"retrieval": "a.npz"}))

    got = load_pointer(path)
    assert got == Pointer(retrieval=Path("a.npz"), ranking=None, sources={})


def test_the_rendered_table_shows_both_sides_of_every_check() -> None:
    """A gate's output is evidence. A bare PASS is not reviewable later."""
    text = render(Verdict((Check("recall", 0.3780, 0.3800, TOLERANCE, None, True, "ok"),)))
    assert "recall" in text
    assert "PROMOTE" in text
