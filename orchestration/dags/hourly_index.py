"""Rebuild, gate and promote the retrieval index, every hour.

Wiring only. The whole build-gate-promote composition is
:func:`indexing.pipeline.rebuild_and_promote`, which ``make index-promote``
calls too. A DAG that carries its own logic is logic no CI run ever executes --
and a *sequence* only the DAG knows is the same problem one level up, which is
what this file used to be.

**Hourly, not nightly.** A news article can be published, peak and die inside a
day, so an item missing from the index is missing for its whole useful life.
Twenty-four builds a day over a 65k-item catalogue is seconds of work each.

**The gate reads click-recall, not agreement with exact search.** Those come
apart: an index measured on this corpus disagreed with exact search about a
tenth of its candidates and found MORE clicks, while another lost 1.3% of the
candidates for 0.05% of the clicks. Promoting on agreement would block the first
and wave through the second.

**Promotion is fail-closed.** The live index already works, so the risk to
design against is replacing it with a regression, not missing a refresh. A
candidate that cannot be scored does not promote, and the pointer stays put.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pendulum
from airflow.sdk import DAG, task

from indexing.pipeline import rebuild_and_promote


@task
def rebuild(
    stamp: str,
    checkpoint: str,
    artifacts: str,
    pointer: str,
    kind: str,
) -> dict[str, Any]:
    """One rebuild, with the outcome in the log and in XCom.

    ``datetime.fromisoformat`` rather than ``pendulum.parse``: the latter is
    typed as returning a date, a time or a duration depending on what it finds,
    so a malformed stamp becomes a ``Date`` and silently produces a label with
    no hour in it -- which would give every build of the day the same name.
    """
    outcome = rebuild_and_promote(
        checkpoint=Path(checkpoint),
        artifacts=Path(artifacts),
        pointer=Path(pointer),
        kind=kind,
        moment=datetime.fromisoformat(stamp),
    )

    print(f"{outcome.label}: {outcome.reason}")
    for stale in outcome.retired:
        print(f"retire {stale}")
    if not outcome.promoted:
        print(f"{outcome.label}: pointer unchanged")

    # Small enough for XCom -- a label, two floats, a flag and a sentence --
    # unlike the probe table the three-task version was avoiding.
    return {
        "label": outcome.label,
        "promoted": outcome.promoted,
        "candidate_recall": outcome.candidate_recall,
        "live_recall": outcome.live_recall,
        "reason": outcome.reason,
    }


with DAG(
    dag_id="hourly_index",
    schedule="0 * * * *",
    start_date=pendulum.datetime(2019, 11, 9, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    params={
        "checkpoint": "data/checkpoints/current.pt",
        "artifacts": "/srv/recsys/index",
        "pointer": "/srv/recsys/index/CURRENT",
        "kind": "flat",
    },
    tags=["retrieval", "index"],
) as dag:
    rebuild(
        stamp="{{ logical_date }}",
        checkpoint="{{ params.checkpoint }}",
        artifacts="{{ params.artifacts }}",
        pointer="{{ params.pointer }}",
        kind="{{ params.kind }}",
    )
