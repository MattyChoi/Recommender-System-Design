"""The retrieval servicer: validation, encoding, and the position-to-item shift.

Built on a `Loaded` assembled by hand rather than by `load()`. That function's
job is to read three artifacts off a disk that a test does not have; everything
that can be wrong PER REQUEST is in the servicer, and it is reachable without
them.

Two tests use a real FAISS index on purpose. The +1 shift between a FAISS
position and an item index is the kind of thing a fake agrees with by
construction, and being wrong about it returns real ids for the wrong articles.
"""

from __future__ import annotations

from typing import Any, cast

import faiss
import numpy as np
import pytest
import torch

from common.pb import retrieval_pb2
from data_pipeline.features.user_history import MAX_HISTORY
from models.retrieval.two_tower import TwoTower
from serving.retrieval.cache import UserEmbeddingCache
from serving.retrieval.service import (
    Loaded,
    RetrievalServicer,
    _index_kind,
    _tower_width,
    recent,
)
from serving.retrieval.tests.test_cache import FakeRedis

DIM = 4
N_USER_FEATS = 3


class AbortedError(Exception):
    """What a fake context raises, because the real one does.

    Load-bearing. `context.abort` in real gRPC raises, and the servicer relies
    on that to stop: there is no `return` after an abort. A fake that merely
    recorded the call would let execution continue into code production never
    reaches, and the tests would be describing a different program.
    """

    def __init__(self, code: Any, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


class FakeContext:
    def abort(self, code: Any, detail: str) -> None:
        raise AbortedError(code, detail)


class FakeTower:
    """Records what it was asked to encode and returns a fixed direction."""

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def encode_user(
        self, feats: torch.Tensor, ids: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        self.calls.append((feats.clone(), ids.clone(), mask.clone()))
        vector = torch.zeros((feats.shape[0], self.dim), dtype=torch.float32)
        vector[:, 0] = 1.0
        return vector


class FakeIndex:
    """Returns a fixed answer, including FAISS's -1 for "fewer than k found"."""

    def __init__(self, positions: list[int], scores: list[float]) -> None:
        self.positions = positions
        self.scores = scores
        self.calls: list[dict[str, Any]] = []

    def search(
        self, matrix: np.ndarray, k: int, params: Any = None
    ) -> tuple[np.ndarray, np.ndarray]:
        self.calls.append({"k": k, "params": params, "query": matrix.copy()})
        return (
            np.array([self.scores], dtype="float32"),
            np.array([self.positions], dtype="int64"),
        )


def content_table(rows: int = MAX_HISTORY + 16, width: int = DIM) -> torch.Tensor:
    """Distinct, non-unit content rows.

    Non-unit on purpose: `similarities` normalises the pooled history and NOT
    the candidate, and a table of unit vectors would make that asymmetry
    invisible -- the test would pass on an implementation that normalised both.
    """
    table = torch.zeros((rows, width), dtype=torch.float32)
    for row in range(rows):
        table[row, row % width] = float(row + 1)
    return table


def make_loaded(
    tower: Any = None,
    index: Any = None,
    kind: str = "hnsw",
    dim: int = DIM,
    tower_dim: int = DIM,
    n_items: int = 100,
    ef_search: int = 512,
    content: torch.Tensor | None = None,
) -> Loaded:
    return Loaded(
        tower=cast(TwoTower, tower or FakeTower()),
        index=index or FakeIndex([0, 1, 2], [0.9, 0.8, 0.7]),
        kind=kind,
        version="v=2026-09-22T12:00Z",
        content=content_table() if content is None else content,
        dim=dim,
        tower_dim=tower_dim,
        n_user_feats=N_USER_FEATS,
        n_items=n_items,
        ef_search=ef_search,
        device=torch.device("cpu"),
    )


def request(**overrides: Any) -> retrieval_pb2.RetrieveRequest:
    fields: dict[str, Any] = {
        "user_id": "U7",
        "k": 3,
        "user_feats": [0.1, 0.2, 0.3],
        "history": [5, 6],
    }
    fields.update(overrides)
    return retrieval_pb2.RetrieveRequest(**fields)


# --- Validation --------------------------------------------------------------


def test_an_empty_user_id_is_refused() -> None:
    """Empty is not a cold user, it is a caller bug.

    Served, it would encode a user with no history, cache the result under
    `uemb:` -- one key shared by every malformed request -- and return the same
    slate to all of them.
    """
    servicer = RetrievalServicer(make_loaded())

    with pytest.raises(AbortedError) as caught:
        servicer.Retrieve(request(user_id=""), FakeContext())

    assert "user_id is required" in caught.value.detail


def test_a_non_positive_k_is_refused() -> None:
    servicer = RetrievalServicer(make_loaded())

    with pytest.raises(AbortedError) as caught:
        servicer.Retrieve(request(k=0), FakeContext())

    assert "k must be positive" in caught.value.detail


def test_a_user_feature_vector_of_the_wrong_width_is_refused() -> None:
    """The one validation that cannot be skipped.

    A short or permuted feature vector is a correctly-typed request. The tower
    produces a plausible embedding for a user who does not exist and the index
    returns its neighbours, which look entirely reasonable. Nothing downstream
    can tell.
    """
    tower = FakeTower()
    servicer = RetrievalServicer(make_loaded(tower=tower))

    with pytest.raises(AbortedError) as caught:
        servicer.Retrieve(request(user_feats=[0.1, 0.2]), FakeContext())

    assert "the tower was fitted with 3" in caught.value.detail
    assert tower.calls == [], "nothing should be encoded once the width is known to be wrong"


def test_a_per_request_ef_search_is_refused_when_faiss_cannot_honour_it() -> None:
    """faiss 1.15.1 CAN honour it, so this path is forced rather than waited for.

    The alternative implementation mutates index.hnsw.efSearch on the shared
    index. gRPC runs a thread pool and FAISS releases the GIL, so two
    concurrent requests race and both search at whichever value won -- silently,
    at a parameter neither asked for.
    """
    servicer = RetrievalServicer(make_loaded())
    servicer.search_params = False

    with pytest.raises(AbortedError) as caught:
        servicer.Retrieve(request(ef_search=128), FakeContext())

    assert "racing" in caught.value.detail


# --- Encoding ----------------------------------------------------------------


def test_the_cache_is_consulted_before_the_tower() -> None:
    tower = FakeTower()
    cache = UserEmbeddingCache(FakeRedis(), dim=DIM)
    cache.put("U7", np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
    servicer = RetrievalServicer(make_loaded(tower=tower), cache)

    response = servicer.Retrieve(request(), FakeContext())

    assert response.embedding_cached is True
    assert tower.calls == [], "a cache hit must not cost a forward pass -- that is the point"


def test_a_miss_runs_the_tower_and_populates_the_cache() -> None:
    tower = FakeTower()
    client = FakeRedis()
    servicer = RetrievalServicer(make_loaded(tower=tower), UserEmbeddingCache(client, dim=DIM))

    response = servicer.Retrieve(request(), FakeContext())

    assert response.embedding_cached is False
    assert len(tower.calls) == 1
    # Written for the NEXT request, and for the Go fallback: ADR 0013's rung 2
    # reads exactly these keys when this service is unreachable.
    assert client.store != {}


def test_a_cold_user_is_served_not_refused() -> None:
    tower = FakeTower()
    servicer = RetrievalServicer(make_loaded(tower=tower))

    servicer.Retrieve(request(history=[]), FakeContext())

    _, ids, mask = tower.calls[0]
    # One padded slot rather than a zero-length tensor: the pooling divides by
    # the mask sum, and a zero-length dimension is undefined rather than clamped.
    assert ids.shape == (1, 1)
    assert float(mask.sum()) == 0.0


def test_history_is_truncated_to_the_most_recent() -> None:
    tower = FakeTower()
    servicer = RetrievalServicer(make_loaded(tower=tower))
    history = list(range(1, MAX_HISTORY + 51))

    servicer.Retrieve(request(history=history), FakeContext())

    _, ids, _ = tower.calls[0]
    assert ids.shape == (1, MAX_HISTORY)
    # The HEAD, because the proto says most-recent-first. Slicing the tail would
    # hand the model the user's oldest reading and call it their history.
    assert ids[0, 0].item() == 1
    assert ids[0, -1].item() == MAX_HISTORY


def test_padded_history_entries_are_dropped() -> None:
    tower = FakeTower()
    servicer = RetrievalServicer(make_loaded(tower=tower))

    servicer.Retrieve(request(history=[4, 0, 9, 0]), FakeContext())

    _, ids, mask = tower.calls[0]
    # Index 0 is the reserved OOV row, which short history lists pad with.
    # Pooled in, it would move every cold user's vector toward a row that is
    # not an article.
    assert ids.tolist() == [[4, 9]]
    assert float(mask.sum()) == 2.0


# --- The response ------------------------------------------------------------


def test_the_oov_row_is_never_returned() -> None:
    """FAISS answers -1 when it finds fewer than k, which to_item_ids maps to 0.

    Zero is the reserved row. Returned, it would put a placeholder in front of
    a user; kept in the list, it would consume a candidate slot.
    """
    index = FakeIndex(positions=[4, -1, -1], scores=[0.9, 0.0, 0.0])
    servicer = RetrievalServicer(make_loaded(index=index))

    response = servicer.Retrieve(request(), FakeContext())

    assert list(response.items) == [5]
    assert len(response.scores) == 1


def test_the_response_names_the_index_that_answered() -> None:
    servicer = RetrievalServicer(make_loaded(kind="flat"))

    response = servicer.Retrieve(request(), FakeContext())

    # A sidecar that quietly fell back to exact search is CORRECT and much
    # slower, and is invisible in every other signal.
    assert response.index_kind == "flat"
    assert response.index_version == "v=2026-09-22T12:00Z"


# --- content_similarity ------------------------------------------------------


def test_content_similarity_mirrors_the_offline_formula() -> None:
    """Recomputed here the way models/ranking/dataset.py:build does it.

    Written out longhand rather than calling the implementation, so the test
    fails if the formula changes rather than moving with it.
    """
    content = content_table()
    servicer = RetrievalServicer(make_loaded(content=content))
    # All four rows sit on axis 1 (index % DIM == 1), so the similarities are
    # non-zero. Candidates on a different axis would score 0 against this
    # history and the comparison would hold for any formula at all.
    history = [1, 5]
    items = np.array([9, 13], dtype=np.int64)

    got = servicer.similarities(history, items)

    pooled = content[history].mean(dim=0)
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
    want = (content[items.tolist()] * pooled).sum(dim=-1).numpy()

    assert got == pytest.approx(want, abs=1e-6)
    assert all(abs(value) > 1e-6 for value in got), "the fixture scores zero; it proves nothing"


def test_only_the_pooled_vector_is_normalised() -> None:
    """The asymmetry the ranker was fitted on.

    Normalising the candidate too is the tidier formula and a DIFFERENT
    feature. It would show up as a model that scores well offline and
    mysteriously underperforms online, with nothing in either system wrong.
    """
    content = content_table()
    servicer = RetrievalServicer(make_loaded(content=content))

    # Item 4 and item 8 share an axis (4 % DIM == 8 % DIM == 0) but have
    # different magnitudes, so a cosine would score them identically and a
    # raw dot product must not.
    got = servicer.similarities([4], np.array([4, 8], dtype=np.int64))

    assert got[0] != pytest.approx(got[1]), (
        "the candidate vector must not be normalised; both scored the same"
    )


def test_both_model_columns_pool_the_same_history() -> None:
    """The tower and the content column MUST agree on what the history is.

    Found by a fixture too small to index, which is luck: the divergence only
    appears past MAX_HISTORY entries, so it would have shipped and affected
    exactly the heaviest users -- the tail least likely to appear in any test.
    The query embedding would have been built from 50 items and
    `content_similarity` from all of them, and the ranker would have received
    two columns describing two different users with nothing reporting it.
    """
    tower = FakeTower()
    content = content_table()
    servicer = RetrievalServicer(make_loaded(tower=tower, content=content))
    long_history = list(range(1, MAX_HISTORY + 40))
    items = np.array([2], dtype=np.int64)

    servicer.Retrieve(request(history=long_history), FakeContext())
    got = servicer.similarities(long_history, items)

    _, ids, _ = tower.calls[0]
    pooled = [int(value) for value in ids[0]]
    assert pooled == recent(long_history), "the tower pooled something else"
    assert got == pytest.approx(servicer.similarities(pooled, items)), (
        "content_similarity pooled a different history than the tower did"
    )


def test_a_cold_user_scores_zero_similarity() -> None:
    # The offline path produces this too: an all-zero mask divides by a
    # clamped 1.0 and normalises to zeros.
    servicer = RetrievalServicer(make_loaded())

    got = servicer.similarities([], np.array([1, 2], dtype=np.int64))

    assert got == pytest.approx([0.0, 0.0])


def test_padded_history_entries_do_not_pollute_the_pooled_vector() -> None:
    # Row 0 is the reserved OOV row. Pooled in, it would pull every short
    # history toward a row that is not an article.
    servicer = RetrievalServicer(make_loaded())
    items = np.array([2], dtype=np.int64)

    assert servicer.similarities([3, 0, 0], items) == pytest.approx(
        servicer.similarities([3], items)
    )


def test_the_response_carries_one_similarity_per_returned_item() -> None:
    """Aligned with `items` after the OOV filter, not before.

    Three parallel arrays that are filtered at different moments is how a
    candidate ends up carrying the next candidate's features.
    """
    index = FakeIndex(positions=[4, -1, 2], scores=[0.9, 0.0, 0.5])
    servicer = RetrievalServicer(make_loaded(index=index))

    response = servicer.Retrieve(request(), FakeContext())

    assert len(response.items) == 2
    assert len(response.scores) == 2
    assert len(response.content_similarity) == 2


# --- Against a real FAISS index ----------------------------------------------


def _axis_rows(dim: int) -> np.ndarray:
    """One unit vector per axis, so every score is distinct and no tie-break
    rule can affect what comes back."""
    return np.eye(dim, dtype="float32")


def test_the_position_to_item_shift_holds_against_real_faiss() -> None:
    """Position p holds item p + 1, because row 0 never enters an index.

    A fake index agrees with whatever the code does. This one does not: FAISS
    decides the positions, and an off-by-one here returns real item ids
    pointing at the neighbouring article.
    """
    index = faiss.IndexFlatIP(DIM)
    index.add(_axis_rows(DIM))

    servicer = RetrievalServicer(make_loaded(index=index, n_items=DIM))
    # FakeTower emits [1, 0, 0, 0], so position 0 is the exact match -- and
    # position 0 must come back as item 1.
    response = servicer.Retrieve(request(k=1), FakeContext())

    assert list(response.items) == [1]
    assert response.scores[0] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "build,expected",
    [
        (lambda: faiss.IndexFlatIP(DIM), "flat"),
        (lambda: faiss.IndexHNSWFlat(DIM, 4, faiss.METRIC_INNER_PRODUCT), "hnsw"),
    ],
)
def test_the_index_kind_is_asked_of_the_object(build: Any, expected: str) -> None:
    """Not read from config and not inferred from the version label.

    A rebuild that wrote a flat index under a name saying hnsw is exactly what
    index_kind exists to surface; trusting the label reports the intention.
    """
    assert _index_kind(build()) == expected


# --- Health ------------------------------------------------------------------


def test_health_is_ready_when_the_tower_and_index_agree() -> None:
    servicer = RetrievalServicer(make_loaded())

    response = servicer.Health(retrieval_pb2.RetrievalHealthRequest(), FakeContext())

    assert response.ready is True
    assert response.detail == ""
    assert response.index_ef_search == 512
    assert response.item_count == 100


def test_health_refuses_a_tower_and_index_from_different_builds() -> None:
    """The failure with no other symptom.

    A 4d index searched with 8d queries would raise; a 4d index built from a
    DIFFERENT 4d checkpoint searches a space the queries are not in and returns
    perfectly reasonable-looking neighbours. Only the widths are checkable here,
    so the check is made where it can be made.
    """
    servicer = RetrievalServicer(make_loaded(dim=4, tower_dim=8))

    response = servicer.Health(retrieval_pb2.RetrievalHealthRequest(), FakeContext())

    assert response.ready is False
    assert "different builds" in response.detail


def test_health_refuses_an_empty_index() -> None:
    """An index that loaded but holds nothing answers every request with an
    empty candidate set, which the orchestrator reads as a cold user rather
    than as a broken index.

    Also pins that not-ready is an ANSWER: a health check returning a gRPC
    error is indistinguishable from one that could not reach the server, and
    those need different responses at 3am. FakeContext.abort raises, so a
    servicer that aborted here would fail this test rather than pass it.
    """
    servicer = RetrievalServicer(make_loaded(n_items=0))

    response = servicer.Health(retrieval_pb2.RetrievalHealthRequest(), FakeContext())

    assert response.ready is False
    assert "no vectors" in response.detail


# --- The startup measurement -------------------------------------------------


def test_the_tower_width_is_measured_rather_than_configured() -> None:
    """TwoTower takes out_dim and does not store it.

    Reading `tower.out_dim` was an AttributeError on every health call. One
    forward pass on zeros gets the number and smoke-tests the checkpoint's
    shapes at boot instead of on the first real request.
    """
    tower = FakeTower(dim=16)

    width = _tower_width(cast(TwoTower, tower), N_USER_FEATS, torch.device("cpu"))

    assert width == 16
    feats, ids, mask = tower.calls[0]
    assert feats.shape == (1, N_USER_FEATS)
    assert ids.shape == (1, 1) and float(mask.sum()) == 0.0
