"""Rebuild, gate and promote the retrieval index, every hour.

Wiring only. Every step is a plain function in ``indexing.pipeline`` or
``indexing.lifecycle``, tested without a scheduler; a DAG that carries its own
logic is logic no CI run ever executes.

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
candidate that cannot be scored does not promote, and the short circuit leaves
the pointer where it is.

**The run's instant arrives as a rendered string, not from the task context.**
Argument templating is stable across releases where the context keys are not --
``execution_date`` became ``logical_date`` between majors -- and a version label
built from the wrong key silently names every hourly build the same thing.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pendulum
from airflow.sdk import DAG, task

from indexing.lifecycle import current, expired, gate, promote, version_label
from indexing.pipeline import build, embed_catalogue, load_version, probe_recall, write_version


@task
def rebuild(stamp: str, checkpoint: str, artifacts: str, kind: str) -> str:
    """Encode the catalogue, build the index, write it under a new version.

    ``datetime.fromisoformat`` rather than ``pendulum.parse``: the latter is
    typed as returning a date, a time or a duration depending on what it finds,
    so a malformed stamp becomes a ``Date`` and silently produces a label with
    no hour in it -- which would give every build of the day the same name.
    """
    label = version_label(datetime.fromisoformat(stamp))
    probe = embed_catalogue(Path(checkpoint))
    written = write_version(build(probe.vectors, kind), Path(artifacts), label)
    print(f"{label}: wrote {written}")
    return label


@task.short_circuit
def validate(label: str, checkpoint: str, artifacts: str, pointer: str) -> bool:
    """Score the candidate and the live index on the SAME probe set.

    Both are re-encoded here rather than carried from ``rebuild`` through XCom:
    XCom is a metadata-database column, and a 65k x 128 float table does not
    belong in one.
    """
    root, marker = Path(artifacts), Path(pointer)
    probe = embed_catalogue(Path(checkpoint))

    candidate = probe_recall(load_version(root, label), probe)
    live_label = current(marker)
    live = probe_recall(load_version(root, live_label), probe) if live_label else None

    decision = gate(candidate, live)
    print(f"{label}: {decision.reason}")
    return decision.allowed


@task
def swap(label: str, artifacts: str, pointer: str) -> None:
    """Atomic pointer swap, then retire anything past the retention depth."""
    root = Path(artifacts)
    promote(Path(pointer), label)
    for stale in expired([path.name for path in root.glob("v=*")]):
        print(f"retire {stale}")


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
    CHECKPOINT = "{{ params.checkpoint }}"
    ARTIFACTS = "{{ params.artifacts }}"
    POINTER = "{{ params.pointer }}"

    built = rebuild(
        stamp="{{ logical_date }}",
        checkpoint=CHECKPOINT,
        artifacts=ARTIFACTS,
        kind="{{ params.kind }}",
    )
    # `built` is an XComArg standing in for the string the task will return.
    # The tasks are typed for the values they receive at run time, which is the
    # useful contract; the placeholder cannot be expressed in those types.
    gated = validate(built, CHECKPOINT, ARTIFACTS, POINTER)  # type: ignore[arg-type]
    promoted = swap(built, ARTIFACTS, POINTER)  # type: ignore[arg-type]

    gated >> promoted
