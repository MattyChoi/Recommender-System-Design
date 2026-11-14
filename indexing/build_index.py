"""FAISS indexes over the item tower's output, and the invariant they rest on.

**Inner product is cosine only because the towers L2-normalise.

**Row 0 never enters an index.** It is the reserved OOV bucket, not an article;
it cannot be a correct answer and must not occupy a slot. The vectors handed to
FAISS are therefore items ``1..n`` at positions ``0..n-1``, and every id coming
back out is shifted by one.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import numpy.typing as npt
import torch

# FAISS's own floor for training a coarse quantiser. Below it the centroids are
# fitted on too little data and it warns; the warning is usually ignored and the
# recall loss is then attributed to quantisation rather than to undertraining.
POINTS_PER_CENTROID = 39

INDEX_KINDS = ("flat", "hnsw", "ivfpq")


def assert_unit_norm(vectors: npt.NDArray[np.float32], tolerance: float = 1e-3) -> None:
    """Refuse vectors an inner-product index would rank by the wrong metric.

    Raises:
        ValueError: If any vector's norm is off 1.0 by more than ``tolerance``.
            An un-normalised table still builds, still searches and still
            returns plausible neighbours -- it ranks by ``|u||v|cos`` instead of
            ``cos``, so long vectors win, and nothing in the output says so.
    """
    norms = np.linalg.norm(vectors, axis=1)
    worst = float(np.abs(norms - 1.0).max()) if len(norms) else 0.0
    if worst > tolerance:
        raise ValueError(
            f"vectors are not unit-norm (worst deviation {worst:.4g}); inner "
            "product would not be cosine and the index would rank by magnitude"
        )


def item_vectors(tower: torch.nn.Module) -> npt.NDArray[np.float32]:
    """Every article's embedding as FAISS wants it, with row 0 dropped."""
    with torch.no_grad():
        table = tower.precompute_items()[1:]  # type: ignore[operator]
    vectors: npt.NDArray[np.float32] = np.ascontiguousarray(
        table.detach().cpu().numpy(), dtype=np.float32
    )
    assert_unit_norm(vectors)
    return vectors


def to_item_ids(ids: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """FAISS positions to item indices.

    Position ``p`` holds item ``p + 1``. FAISS returns ``-1`` when it finds
    fewer than k neighbours, which becomes 0 -- the reserved index, which is how
    every other module in this project spells "no candidate".
    """
    return np.where(ids < 0, 0, ids + 1).astype(np.int64)


def recommended_nlist(n_vectors: int, ceiling: int = 4096) -> int:
    """Coarse cells this corpus can actually train.

    ``4 * sqrt(n)`` is the usual starting point, capped so the quantiser gets at
    least :data:`POINTS_PER_CENTROID` vectors per cell. On a 65k-item catalogue
    that cap binds hard and is the reason a copied-in ``nlist=4096`` trains on
    16 points per centroid and blames the quantiser for the result.
    """
    if n_vectors <= 0:
        raise ValueError("no vectors to index")
    trainable = max(1, n_vectors // POINTS_PER_CENTROID)
    return max(1, min(ceiling, int(4 * math.sqrt(n_vectors)), trainable))


def build_flat(vectors: npt.NDArray[np.float32]) -> Any:
    """Exact search. Not a baseline to beat -- the ground truth to measure against."""
    assert_unit_norm(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def build_hnsw(vectors: npt.NDArray[np.float32], m: int = 32, ef_construction: int = 200) -> Any:
    """Graph index: fast, memory-hungry, no training step."""
    assert_unit_norm(vectors)
    index = faiss.IndexHNSWFlat(vectors.shape[1], m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.add(vectors)
    return index


def build_ivfpq(
    vectors: npt.NDArray[np.float32],
    nlist: int | None = None,
    m: int = 32,
    nbits: int = 8,
) -> Any:
    """Quantised index: small, trained, and lossy in a way HNSW is not.

    Two different approximations stack here and they are worth keeping apart.
    ``nlist``/``nprobe`` decide how much of the catalogue is *looked at*;
    ``m``/``nbits`` decide how precisely each vector is *stored*. Only the first
    is tunable at query time, so the second is a build-time commitment.

    Raises:
        ValueError: If ``m`` does not divide the dimension. PQ splits each
            vector into ``m`` sub-vectors and a remainder is not expressible.
    """
    assert_unit_norm(vectors)
    dim = int(vectors.shape[1])
    if dim % m:
        raise ValueError(f"m={m} must divide dim={dim}")

    cells = recommended_nlist(len(vectors)) if nlist is None else nlist
    quantiser = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(quantiser, dim, cells, m, nbits, faiss.METRIC_INNER_PRODUCT)
    index.train(vectors)
    index.add(vectors)
    return index


def search(index: Any, queries: npt.NDArray[np.float32], k: int) -> npt.NDArray[np.int64]:
    """Top-k item indices per query, already shifted off FAISS's positions."""
    _, ids = index.search(np.ascontiguousarray(queries, dtype="float32"), k)
    return to_item_ids(ids)


def index_bytes(index: Any) -> int:
    """Serialised size, which is what a serving node has to hold and ship."""
    return int(faiss.serialize_index(index).nbytes)


def save(index: Any, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(destination))
    return destination


def load(source: Path) -> Any:
    return faiss.read_index(str(source))
