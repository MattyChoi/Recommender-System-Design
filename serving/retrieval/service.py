"""The retrieval sidecar: embed a user, search the index, return candidates."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from common.config import Settings, load_settings
from common.pb import retrieval_pb2, retrieval_pb2_grpc
from data_pipeline.features.user_history import MAX_HISTORY
from indexing.build_index import to_item_ids
from indexing.lifecycle import current
from indexing.pipeline import load_version
from models.retrieval.dataloader.dataset import CONTENT_VARIANTS, load_item_tables
from models.retrieval.evaluate import load_tower
from models.retrieval.two_tower import TwoTower
from serving.retrieval.cache import UserEmbeddingCache

#: Where the hourly DAG writes versions and points CURRENT. Matches
#: orchestration/dags/hourly_index.py's params; the two must agree or the
#: service serves an index nobody promoted.
DEFAULT_ARTIFACTS = Path("/srv/recsys/index")
DEFAULT_POINTER = DEFAULT_ARTIFACTS / "CURRENT"

#: Fallback when a request sends 0. ADR 0002 ships 512 over the
#: throughput-optimal 128: that is where the recall cost stopped reproducing
#: across checkpoints, and a config drift back to 128 is invisible in every
#: system metric, which is why HealthResponse reports what actually loaded.
DEFAULT_EF_SEARCH = 512


def _index_kind(index: Any) -> str:
    """What kind of index this actually is, asked of the OBJECT.

    Not read from config and not inferred from the version label. A rebuild
    that wrote a flat index under a name saying hnsw is exactly the failure
    `index_kind` exists to surface, and trusting the label would report the
    intention rather than the fact.
    """
    if hasattr(index, "hnsw"):
        return "hnsw"
    if hasattr(index, "nprobe"):
        return "ivfpq"
    return "flat"


def _supports_search_params() -> bool:
    """Whether this FAISS build takes per-search parameters.

    Feature-detected, never assumed -- see scripts/probe_faiss_search_params.py.
    Without it the only way to honour a per-request efSearch is to mutate
    `index.hnsw.efSearch` on the shared index, and FAISS releases the GIL during
    a search, so two concurrent requests would race and both searches would run
    at whichever value won. Nothing raises; the slates are just built at a
    parameter neither request asked for.

    **Measured on faiss 1.15.1: present, accepted, and honoured** -- efSearch=1
    and efSearch=256 agreed on only 0.4981 of neighbours, with a poisoned
    `index.hnsw.efSearch = 999` on the shared index not leaking into either
    call. So the refusal path below does not fire on this build. It stays
    because "the installed version supports it" is a fact about today's lock
    file, and the failure it guards is silent.
    """
    import faiss

    return hasattr(faiss, "SearchParametersHNSW")


def recent(history: Sequence[int]) -> list[int]:
    """The history both model-derived columns are computed from.

    One function, called by both, because they MUST agree. The tower pools it
    to make the query embedding and the content column pools it to make
    `content_similarity`; offline, `split.history_ids` is already capped by the
    loader, so both see the same rows. A serving path that truncated in one
    place and not the other would hand the ranker two columns describing two
    different users -- and only for users with more than MAX_HISTORY entries,
    which is the tail least likely to show up in a fixture.

    Most-recent-first, so the HEAD is kept: slicing the tail would hand the
    model the user's oldest reading and call it their history. Index 0 is the
    reserved OOV row that short lists pad with, and pooling it drags a short
    history toward a row that is not an article.
    """
    return [item for item in history[:MAX_HISTORY] if item > 0]


def _tower_width(tower: TwoTower, n_user_feats: int, device: torch.device) -> int:
    """The tower's output width, measured rather than configured.

    One forward pass on zeros. Two things come out of it: the number, which
    TwoTower does not expose as an attribute, and a startup smoke test -- a
    checkpoint whose shapes do not line up fails here, at boot, rather than on
    the first real request.
    """
    with torch.no_grad():
        embedded = tower.encode_user(
            torch.zeros((1, n_user_feats), dtype=torch.float32, device=device),
            torch.zeros((1, 1), dtype=torch.long, device=device),
            torch.zeros((1, 1), dtype=torch.float32, device=device),
        )
    return int(embedded.shape[-1])


@dataclass
class Loaded:
    """Everything the servicer serves from, loaded once at startup."""

    tower: TwoTower
    index: Any
    kind: str
    version: str
    #: The item content table, row 0 reserved. Held because two of the
    #: ranker's columns are computed from it and this is the only serving
    #: process that has it.
    content: torch.Tensor
    #: The INDEX's width, from the artifact.
    dim: int
    #: The TOWER's output width, measured by a forward pass at startup rather
    #: than read from a config or an attribute -- TwoTower does not carry one.
    #: The two being equal is the whole point: an index built from a different
    #: checkpoint than the tower loaded here searches a space the queries are
    #: not in, and every neighbour it returns looks perfectly reasonable.
    tower_dim: int
    n_user_feats: int
    n_items: int
    ef_search: int
    device: torch.device


def load(
    checkpoint: Path,
    artifacts: Path = DEFAULT_ARTIFACTS,
    pointer: Path = DEFAULT_POINTER,
    variant: str = CONTENT_VARIANTS[0],
    ef_search: int = DEFAULT_EF_SEARCH,
    settings: Settings | None = None,
    device: torch.device | None = None,
) -> Loaded:
    """Load the tower and the promoted index.

    Raises:
        RuntimeError: If nothing has been promoted. Refusing to start beats
            starting and answering every request with an empty candidate set,
            which the orchestrator would read as a cold user rather than as a
            missing index.
    """
    resolved = settings or load_settings()
    where = device or torch.device("cpu")

    label = current(pointer)
    if label is None:
        raise RuntimeError(
            f"no index promoted at {pointer}; run the hourly_index DAG or promote one by hand"
        )
    index = load_version(artifacts, label)

    items = load_item_tables(resolved, variant)
    state = torch.load(checkpoint, map_location=where, weights_only=True)["model"]
    if "user_norm.weight" not in state:
        raise RuntimeError(
            f"{checkpoint} has no user_norm.weight; it is not a two-tower checkpoint"
        )
    # LayerNorm(n_user_feats), so its width IS the feature count the tower was
    # fitted with. Read from the artifact rather than configured, because a
    # configured value that disagrees produces a plausible embedding for a user
    # who does not exist.
    n_user_feats = int(state["user_norm.weight"].shape[0])

    tower = load_tower(checkpoint, items, n_user_feats, where)
    tower.eval()
    tower_dim = _tower_width(tower, n_user_feats, where)

    kind = _index_kind(index)
    if kind == "hnsw":
        # The default, set once. A per-request override is handled in Retrieve
        # and only where the FAISS build supports it without mutation.
        index.hnsw.efSearch = ef_search

    return Loaded(
        tower=tower,
        index=index,
        kind=kind,
        version=label,
        content=items.content,
        dim=int(index.d),
        tower_dim=tower_dim,
        n_user_feats=n_user_feats,
        n_items=int(index.ntotal),
        ef_search=ef_search,
        device=where,
    )


class RetrievalServicer(retrieval_pb2_grpc.RetrievalServicer):
    """The gRPC surface. Validation, encode, search, translate."""

    def __init__(self, loaded: Loaded, cache: UserEmbeddingCache | None = None) -> None:
        self.loaded = loaded
        self.cache = cache
        self.search_params = _supports_search_params()
        # Guards the tower forward pass only. torch is not reentrant-safe for
        # a module being used from several threads, and the gRPC server runs a
        # pool. The FAISS search is deliberately OUTSIDE this lock: it releases
        # the GIL and parallelises, and serialising it would throw away the one
        # part of the request that scales.
        self._encode_lock = threading.Lock()

    # --- encoding ------------------------------------------------------------

    def encode(
        self, user_id: str, user_feats: list[float], history: list[int]
    ) -> tuple[npt.NDArray[np.float32], bool]:
        """The user's unit-norm vector, and whether it came from cache."""
        if self.cache is not None:
            cached = self.cache.get(user_id)
            if cached is not None:
                return cached, True

        vector = self._forward(user_feats, history)
        if self.cache is not None:
            self.cache.put(user_id, vector)
        return vector, False

    def _forward(self, user_feats: list[float], history: list[int]) -> npt.NDArray[np.float32]:
        feats = torch.tensor([user_feats], dtype=torch.float32, device=self.loaded.device)

        rows = recent(history)
        if rows:
            ids = torch.tensor([rows], dtype=torch.long, device=self.loaded.device)
            mask = torch.ones_like(ids, dtype=torch.float32)
        else:
            # A cold user is a legal request, not an error. One padded slot
            # rather than a zero-length tensor: the pooling divides by the mask
            # sum, and an all-zero mask is clamped rather than undefined.
            ids = torch.zeros((1, 1), dtype=torch.long, device=self.loaded.device)
            mask = torch.zeros((1, 1), dtype=torch.float32, device=self.loaded.device)

        with self._encode_lock, torch.no_grad():
            embedded = self.loaded.tower.encode_user(feats, ids, mask)
        return embedded.squeeze(0).cpu().numpy().astype(np.float32)

    # --- the ranker's content column -----------------------------------------

    def similarities(
        self, history: list[int], items: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float32]:
        """`content_similarity` for each candidate.

        A line-for-line mirror of models/ranking/dataset.py:build, and the
        asymmetry is deliberate on both sides: the POOLED history vector is
        L2-normalised and the candidate's content vector is NOT. The ranker was
        fitted on that, so normalising both here would be a tidier formula and
        a different feature -- the kind of divergence that shows up as a model
        mysteriously underperforming online.

        A cold user pools nothing and scores zero against everything, which is
        what the offline path produces too: an all-zero mask divides by a
        clamped 1.0 and normalises to zeros.
        """
        content = self.loaded.content
        rows = recent(history)
        if rows:
            pooled = content[torch.tensor(rows, dtype=torch.long)].mean(dim=0)
        else:
            pooled = torch.zeros(content.shape[1], dtype=content.dtype)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)

        candidates = content[torch.tensor(items, dtype=torch.long)]
        similarity = (candidates * pooled).sum(dim=-1)
        return similarity.numpy().astype(np.float32)

    # --- search --------------------------------------------------------------

    def search(
        self, query: npt.NDArray[np.float32], k: int, ef_search: int
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
        """Top-k item indices and scores, already shifted off FAISS positions."""
        matrix = np.ascontiguousarray(query[None, :], dtype="float32")

        params = None
        if ef_search and ef_search != self.loaded.ef_search and self.loaded.kind == "hnsw":
            import faiss

            params = faiss.SearchParametersHNSW()  # type: ignore[attr-defined]  # missing from faiss's bundled .pyi
            params.efSearch = ef_search

        if params is not None:
            scores, positions = self.loaded.index.search(matrix, k, params=params)
        else:
            scores, positions = self.loaded.index.search(matrix, k)

        # Position p holds item p + 1; FAISS returns -1 when it finds fewer
        # than k neighbours, which becomes 0 -- how every other module in this
        # project spells "no candidate".
        return to_item_ids(positions)[0], scores[0]

    # --- RPCs ----------------------------------------------------------------

    def Retrieve(  # noqa: N802 - the name is the proto's
        self, request: retrieval_pb2.RetrieveRequest, context: Any
    ) -> retrieval_pb2.RetrieveResponse:
        import grpc

        if not request.user_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "user_id is required")
        if request.k <= 0:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"k must be positive, got {request.k}")
        # Checked, not trusted. A short or permuted feature vector is a
        # correctly-typed request that produces a plausible embedding for a
        # user who does not exist, and the neighbours it retrieves look
        # entirely reasonable.
        if len(request.user_feats) != self.loaded.n_user_feats:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"{len(request.user_feats)} user features, the tower was fitted with "
                f"{self.loaded.n_user_feats}",
            )
        if request.ef_search and not self.search_params and self.loaded.kind == "hnsw":
            # Refused rather than serviced by mutating the shared index. See
            # scripts/probe_faiss_search_params.py: the mutation races under a
            # thread pool and silently searches at a parameter neither request
            # asked for.
            context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                "this FAISS build has no SearchParametersHNSW, so a per-request "
                "efSearch cannot be honoured without racing concurrent searches",
            )

        vector, cached = self.encode(
            request.user_id, list(request.user_feats), list(request.history)
        )
        items, scores = self.search(vector, request.k, request.ef_search)

        # Index 0 is the reserved OOV row, which is not an article and must not
        # occupy a slot. It appears here whenever FAISS returned fewer than k.
        #
        # Filtered BEFORE the content column is computed, so every array below
        # is aligned with the items actually returned. Computing first and
        # filtering after would be one more place for three parallel arrays to
        # drift out of step.
        keep = items > 0
        survivors = items[keep]
        return retrieval_pb2.RetrieveResponse(
            items=survivors.astype(np.int32).tolist(),
            scores=scores[keep].astype(np.float32).tolist(),
            content_similarity=self.similarities(list(request.history), survivors).tolist(),
            embedding_cached=cached,
            index_kind=self.loaded.kind,
            index_version=self.loaded.version,
        )

    def Health(  # noqa: N802 - the name is the proto's
        self, request: retrieval_pb2.RetrievalHealthRequest, context: Any
    ) -> retrieval_pb2.RetrievalHealthResponse:
        detail = ""
        ready = True
        if self.loaded.n_items <= 0:
            # An index that loaded but holds nothing answers every request with
            # an empty candidate set, which the orchestrator reads as a cold
            # user rather than as a broken index.
            ready = False
            detail = "index holds no vectors"
        elif self.loaded.dim != self.loaded.tower_dim:
            ready = False
            detail = (
                f"index is {self.loaded.dim}d and the tower emits "
                f"{self.loaded.tower_dim}d; they are from different builds"
            )

        return retrieval_pb2.RetrievalHealthResponse(
            ready=ready,
            detail=detail,
            index_kind=self.loaded.kind,
            index_version=self.loaded.version,
            index_ef_search=self.loaded.ef_search,
            item_count=self.loaded.n_items,
        )
