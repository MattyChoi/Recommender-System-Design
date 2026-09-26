"""The DAG parses, and its shape is what the gate depends on.

A scheduler finds an import error or a broken dependency at parse time, in
production, at 02:00. These run it in CI instead. Nothing here executes a task:
the tasks call Spark and a GPU, and what can be wrong about the *wiring* is
whether it imports, whether the gate sits between the build and the swap, and
whether the schedule is the one that was argued for.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

# Airflow reads both at import time. Without the first it creates ~/airflow and
# a sqlite database there, so a test run leaves state on the machine; without
# the second the bundled example DAGs are parsed too, and their import errors
# become this test's failures.
os.environ.setdefault("AIRFLOW_HOME", tempfile.mkdtemp(prefix="airflow-test-"))
os.environ.setdefault("AIRFLOW__CORE__LOAD_EXAMPLES", "False")

pytest.importorskip("airflow", reason="orchestration extra not installed")

DAG_PATH = Path(__file__).resolve().parents[2] / "orchestration" / "dags" / "hourly_index.py"


# `Any` rather than `DAG`: airflow is in the scoped ignore_missing_imports list,
# so its exports are untyped here and a real annotation would be a fiction.
@pytest.fixture(scope="module")
def dag() -> Any:
    """The DAG object, loaded the way a scheduler loads it -- by executing the file.

    Read off ``bag.dags`` rather than ``bag.get_dag()``: the getter consults the
    metadata database for the DAG's serialised record, so it needs a migrated
    schema and a running deployment. The parse itself needs neither, and the
    parse is what this file is testing.
    """
    from airflow.dag_processing.dagbag import DagBag

    bag = DagBag(dag_folder=str(DAG_PATH.parent))
    assert not bag.import_errors, bag.import_errors
    return bag.dags["hourly_index"]


class TestItParses:
    def test_the_file_has_no_import_errors(self, dag: Any) -> None:
        """The whole point: a moved import is found here, not by the scheduler.

        Airflow 3 relocated ``DAG`` to ``airflow.sdk``, ``PythonOperator`` to a
        provider package and ``DagBag`` to ``airflow.dag_processing``. Written
        from memory against any of the old paths, this file imports fine on the
        version it was written for and fails at parse time inside the scheduler
        on the version that is deployed.
        """
        assert dag is not None

    def test_every_task_is_present(self, dag: Any) -> None:
        assert {task.task_id for task in dag.tasks} == {"rebuild", "validate", "swap"}


class TestTheShape:
    def test_the_gate_sits_between_the_build_and_the_swap(self, dag: Any) -> None:
        """If the swap does not depend on the gate, the gate is decoration."""
        assert "swap" in dag.get_task("validate").downstream_task_ids
        assert "validate" in dag.get_task("rebuild").downstream_task_ids

    def test_the_swap_cannot_run_without_the_gate(self, dag: Any) -> None:
        """A short circuit only skips what is DOWNSTREAM of it."""
        assert "validate" in dag.get_task("swap").upstream_task_ids

    def test_it_runs_hourly(self, dag: Any) -> None:
        """Nightly would leave an article missing for its entire useful life."""
        assert dag.schedule == "0 * * * *"

    def test_runs_do_not_overlap(self, dag: Any) -> None:
        """Two rebuilds writing one pointer is a race with a stale winner."""
        assert dag.max_active_runs == 1

    def test_it_does_not_backfill(self, dag: Any) -> None:
        """Catchup from 2019 would queue tens of thousands of rebuilds."""
        assert dag.catchup is False


class TestParameters:
    def test_paths_are_parameters_rather_than_literals(self, dag: Any) -> None:
        assert set(dag.params) >= {"checkpoint", "artifacts", "pointer", "kind"}

    def test_the_default_index_is_the_exact_one(self, dag: Any) -> None:
        """Measured on this corpus: approximation buys nothing it needs here."""
        assert dag.params["kind"] == "hnsw"
