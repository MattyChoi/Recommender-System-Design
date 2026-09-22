"""The work an index rebuild actually does, with no scheduler in sight.

Every step a DAG task calls lives here as a plain function: encode the
catalogue, build an index, score it on a probe set. The scheduler file wires
these together and does nothing else, so the part that can be wrong is the part
CI runs.

**The probe set is the validation window, and it is fixed.** A gate that scores
the candidate index on fresher data than the live one measured would compare a
model against a different question and call the difference a regression.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from common.config import Settings, load_settings
from common.spark import get_spark
from common.torch_env import select_device
from data_pipeline.features.user_history import MAX_HISTORY
from indexing.build_index import build_flat, build_hnsw, build_ivfpq, save, search
from models.retrieval.dataloader.dataset import (
    CONTENT_VARIANTS,
    load_item_tables,
    load_train_and_validation,
)
from models.retrieval.evaluate import load_tower
from models.retrieval.sources import encode_users

INDEX_FILENAME = "index.faiss"


@dataclass(frozen=True)
class Probe:
    """Everything needed to build an index and to judge it.

    Attributes:
        vectors: ``[n_items, dim]`` unit-norm item embeddings, row 0 dropped.
        queries: ``[R, dim]`` request vectors from the validation window.
        clicked: ``[R]`` the item each request actually clicked.
    """

    vectors: npt.NDArray[np.float32]
    queries: npt.NDArray[np.float32]
    clicked: npt.NDArray[np.int64]


def embed_catalogue(
    checkpoint: Path,
    settings: Settings | None = None,
    variant: str = CONTENT_VARIANTS[0],
    holdout_hours: int = 12,
    max_negs: int = 4,
) -> Probe:
    """Run the towers once: every item, and every probe request."""
    resolved = settings or load_settings()
    device = select_device()

    spark = get_spark(resolved, app="index-rebuild")
    try:
        items = load_item_tables(resolved, variant)
        train_split, val_split = load_train_and_validation(
            spark, resolved, MAX_HISTORY, max_negs, holdout_hours
        )
    finally:
        spark.stop()

    tower = load_tower(checkpoint, items, train_split.user_feats.shape[1], device)
    from indexing.build_index import item_vectors

    return Probe(
        vectors=item_vectors(tower),
        queries=encode_users(tower, val_split, device).numpy().astype("float32"),
        clicked=val_split.item_ids.numpy(),
    )


def build(vectors: npt.NDArray[np.float32], kind: str = "flat") -> Any:
    """One of the three index kinds, by name.

    Raises:
        ValueError: On an unknown kind. A typo would otherwise fall through to
            a default and promote an index nobody asked for.
    """
    if kind == "flat":
        return build_flat(vectors)
    if kind == "hnsw":
        return build_hnsw(vectors)
    if kind == "ivfpq":
        return build_ivfpq(vectors)
    raise ValueError(f"unknown index kind {kind!r}; expected flat, hnsw or ivfpq")


def probe_recall(index: Any, probe: Probe, k: int = 100) -> float:
    """Share of probe requests whose clicked item the index returns.

    Click-recall, not overlap with exact search. The two come apart: an index
    can disagree with exact search about a tenth of its candidates and find more
    of the answers, so overlap is the wrong thing to promote on.
    """
    got = search(index, probe.queries, k)
    return float((got == probe.clicked[:, None]).any(axis=1).mean())


def write_version(index: Any, root: Path, label: str) -> Path:
    """Write the index under its own version directory. Never in place."""
    return save(index, root / label / INDEX_FILENAME)


def load_version(root: Path, label: str) -> Any:
    from indexing.build_index import load

    return load(root / label / INDEX_FILENAME)
