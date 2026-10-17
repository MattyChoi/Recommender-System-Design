from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import numpy.typing as npt
import torch

from common.config import load_settings
from common.features import (
    CONTENT_VARIANTS,
    USER_FEATURES,
    USER_TOWER_COLUMNS,
    build_user_features,
    spark_day_of_week,
)
from common.utils import load_tower
from data_pipeline.features.user_history import MAX_HISTORY
from models.retrieval.dataloader.dataset import load_item_tables
from models.retrieval.two_tower import TwoTower

# ``torch.nn.functional.normalize``'s own floor.
_NORM_EPS = 1e-12


@dataclass(frozen=True)
class LoadedIndex:
    tower: TwoTower
    index: Any  # faiss.Index; faiss ships no usable type stubs
    kind: str
    version: str
    ef_search: int
    content: Any  # np.ndarray[np.float32, Any]


def load_index(checkpoint: Path, artifacts: Path, pointer: Path, ef_search: int) -> LoadedIndex:
    """Read CURRENT, load what it names, and report what was actually loaded."""
    if not pointer.exists():
        raise SystemExit(f"{pointer} does not exist -- run `make index-promote` first")

    version = pointer.read_text().strip()
    if not version:
        raise SystemExit(f"{pointer} is empty -- promotion wrote a pointer and no version")

    path = artifacts / version / "index.faiss"
    if not path.exists():
        raise SystemExit(f"{pointer} names '{version}' but {path} is missing")

    index = faiss.read_index(str(path))

    kind = "flat"
    if isinstance(index, faiss.IndexHNSWFlat):
        kind = "hnsw"
        index.hnsw.efSearch = ef_search

    settings = load_settings()
    items = load_item_tables(settings, CONTENT_VARIANTS[0])

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
    if "user_norm.weight" not in state:
        raise SystemExit(f"{checkpoint} is not a two-tower checkpoint")
    n_user_feats = int(state["user_norm.weight"].shape[0])
    if n_user_feats != len(USER_TOWER_COLUMNS):
        raise ValueError(
            f"checkpoint expects {n_user_feats} user columns, "
            f"USER_TOWER_COLUMNS has {len(USER_TOWER_COLUMNS)}: {USER_TOWER_COLUMNS}"
        )

    tower = load_tower(checkpoint, items, n_user_feats, torch.device("cpu"))

    # The same [n_items + 1, dim] table `models.ranking.dataset.build` pools
    # over: item-indexed, row 0 reserved
    content = np.ascontiguousarray(items.content.numpy(), dtype="float32")

    if index.ntotal + 1 != len(content):
        raise SystemExit(
            f"index {version} holds {index.ntotal} vectors but the "
            f"catalogue has {len(content) - 1} items -- rebuild with "
            "`make index-promote` from the checkpoint this sidecar loads"
        )

    return LoadedIndex(
        tower=tower,
        index=index,
        kind=kind,
        version=version,
        ef_search=ef_search,
        content=content,
    )


def real_history(history: Sequence[int], n_items: int) -> list[int]:
    """The history entries that name a real article, bounded to the tower's window.

    Truncate THEN filter, in that order.
    """
    return [item for item in history[:MAX_HISTORY] if 0 < item < n_items]


@torch.inference_mode()
def encode_user(
    tower: TwoTower, feats: npt.NDArray[np.float32], history: list[int]
) -> npt.NDArray[np.float32]:
    items = torch.zeros((1, MAX_HISTORY), dtype=torch.long)
    mask = torch.zeros((1, MAX_HISTORY), dtype=torch.bool)
    if history:
        items[0, : len(history)] = torch.tensor(history, dtype=torch.long)
        mask[0, : len(history)] = True

    vector = tower.encode_user(torch.from_numpy(feats), items, mask)
    # float32 and C-contiguous, because FAISS takes neither on faith.
    return np.ascontiguousarray(vector.numpy(), dtype="float32")


def content_similarity(
    content: npt.NDArray[np.float32], history: Sequence[int], items: Sequence[int]
) -> npt.NDArray[np.float32]:
    """``content_similarity`` per candidate, as the ranker was trained on it.

        The definition lives in ``models.ranking.dataset.build``: mean-pool the
        content vectors of the user's history, normalise, and dot it against each
        candidate's content vector.

    Args:
        content: ``[n_items + 1, dim]`` item-indexed content table, row 0
            reserved, NOT normalised.
        history: The caller's history, already through :func:`real_history`.
        items: Candidate item ids, in response order.

    Returns:
        One similarity per candidate, aligned with ``items``.
    """
    rows = list(history)
    if rows:
        pooled = np.asarray(content[rows].sum(axis=0) / len(rows), dtype="float32")
    else:
        pooled = np.zeros(content.shape[1], dtype="float32")
    pooled /= max(float(np.linalg.norm(pooled)), _NORM_EPS)

    # One [k, dim] gather and a matvec. Candidates are item ids and the table is
    # item-indexed, so this needs no offset -- which is only true because
    # `Retrieve` converted the FAISS positions first.
    similarity: npt.NDArray[np.float32] = content[list(items)] @ pooled
    return similarity


def build_user_tower_columns(
    raw: list[float],
    has_user_features: bool,
    when: datetime,
) -> npt.NDArray[np.float32]:
    static = {name: np.asarray([raw[i]], dtype="float32") for i, name in enumerate(USER_FEATURES)}
    return build_user_features(
        static,
        np.asarray([float(has_user_features)], dtype="float32"),
        np.asarray([when.hour], dtype="float32"),
        np.asarray([spark_day_of_week(when)], dtype="float32"),
    )
