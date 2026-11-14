"""Per-slate AUC rows, written beside each report card so the gate can pair GAUC.

A card holds one GAUC scalar per cohort, and a scalar cannot be paired: the
paired bootstrap in ``evaluation.offline.stats`` needs one number per USER on
each side. These rows are the step before that -- one AUC per slate, with the
user who saw it and its GAUC weight -- and ``gate`` reduces them per user.

Layout: ``<cards dir>/slates/<card stem>.parquet``, one file per card, long
format (``cohort, auc, user, weight``). Keyed by the card's STEM so the pairing
is by name, the same way ``PRODUCTION.json`` already names cards.

Undefined AUCs (all-click or no-click slates) are stored as NaN rather than
dropped, so a file round-trips to exactly the rows ``report`` produced.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

import pandas as pd

# Beside the cards in evaluation/results/ (run_eval.RESULTS), not imported from
# there: run_eval pulls in Spark, and the gate must not need it just to read
# rows. Callers use ``SLATES.name`` to place rows beside whichever card they
# hold, so cards written elsewhere keep their rows next to them.
SLATES = Path("evaluation/results/slates")

_COLUMNS = ["cohort", "auc", "user", "weight"]


class SlateRows(NamedTuple):
    """One cohort's slates, index-aligned: ``auc[i]``, ``users[i]`` and
    ``weight[i]`` describe the same slate.

    Attributes:
        auc: AUC per slate, NaN where it is undefined.
        users: The user each slate was served to.
        weight: The slate's size, which is GAUC's weight.
    """

    auc: list[float]
    users: list[str]
    weight: list[float]


def path_for(stem: str, directory: Path) -> Path:
    """Where the rows for the card named ``stem`` live in ``directory``."""
    return directory / f"{stem}.parquet"


def save(stem: str, rows: Mapping[str, SlateRows], directory: Path) -> Path:
    """Write every cohort's rows for one card, replacing any earlier file.

    Args:
        stem: The card's file stem, ``ranker`` for ``ranker.json``.
        rows: Cohort name to its rows. Cohorts absent here are absent on load,
            which the gate reads as unavailable.
        directory: Usually ``<card dir>/slates``.

    Returns:
        The file written.

    Raises:
        ValueError: If a cohort's three columns differ in length -- they would
            pair the wrong slates with the wrong users.
    """
    frames = []
    for cohort, slate_rows in rows.items():
        lengths = {len(slate_rows.auc), len(slate_rows.users), len(slate_rows.weight)}
        if len(lengths) != 1:
            raise ValueError(
                f"cohort {cohort!r}: auc, users and weight have lengths "
                f"{len(slate_rows.auc)}, {len(slate_rows.users)}, {len(slate_rows.weight)}"
            )
        frames.append(
            pd.DataFrame(
                {
                    "cohort": cohort,
                    "auc": pd.Series(slate_rows.auc, dtype="float64"),
                    "user": pd.Series(slate_rows.users, dtype="string"),
                    "weight": pd.Series(slate_rows.weight, dtype="float64"),
                }
            )
        )
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_COLUMNS)

    directory.mkdir(parents=True, exist_ok=True)
    destination = path_for(stem, directory)
    # Via a temporary name, so a crash mid-write never leaves a truncated file
    # that the gate would load as a short, wrong set of rows.
    partial = destination.with_name(destination.name + ".partial")
    table.to_parquet(partial, index=False)
    partial.replace(destination)
    return destination


def load(stem: str, directory: Path) -> dict[str, SlateRows]:
    """Read back what :func:`save` wrote for the card named ``stem``.

    Returns:
        Cohort name to its rows, in the order they were saved.

    Raises:
        FileNotFoundError: If no rows were ever written for this card. The gate
            relies on this to tell "scored before rows existed" from an error.
    """
    source = path_for(stem, directory)
    if not source.is_file():
        raise FileNotFoundError(f"no per-slate rows at {source}")

    table = pd.read_parquet(source, columns=_COLUMNS)
    out: dict[str, SlateRows] = {}
    for cohort, group in table.groupby("cohort", sort=False):
        out[str(cohort)] = SlateRows(
            auc=group["auc"].astype(float).tolist(),
            users=group["user"].astype(str).tolist(),
            weight=group["weight"].astype(float).tolist(),
        )
    return out
