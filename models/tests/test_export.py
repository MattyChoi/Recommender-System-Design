"""Training scores with ``torch_fit.predict``; production runs an ONNX graph
through onnxruntime. Those are two programs and they must agree.

Two things can diverge, and they are tested separately:

1. **The baked preprocessing.** The graph does its own reciprocal, log1p and
   standardisation, so the wrapper has to reproduce what the training transform
   did. The first test isolates this and needs NO onnxruntime.
2. **The column order.** Eleven columns in one matrix, two of them categorical.
   A permutation is a valid-shaped input that scores a different world.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from models.classes.ranking import Standardiser
from models.export.ranker_onnx import INPUT_NAME, OUTPUT_NAME, ServableRanker, export
from models.ranking.dataset import ABSENT, FEATURES, RankingRows
from models.ranking.mmoe import MMoE
from models.ranking.torch_fit import fit, predict

ROWS = 64
CPU = torch.device("cpu")


@pytest.fixture
def rows() -> RankingRows:
    """Every column in ``FEATURES``, so the wrapper's index logic is exercised
    against the real layout rather than a convenient subset."""
    per_request = 4
    n_requests = ROWS // per_request
    rng = np.random.default_rng(0)
    request = np.repeat(np.arange(n_requests), per_request).astype(np.int64)

    columns = {
        "retrieval_score": rng.normal(size=ROWS),
        "two_tower_rank": rng.integers(0, 50, size=ROWS),
        "trending_rank": np.where(rng.random(ROWS) < 0.3, ABSENT, rng.integers(0, 50, size=ROWS)),
        "n_sources": rng.integers(1, 4, size=ROWS),
        "prior_clicks": rng.integers(0, 500, size=ROWS),
        "train_clicks": rng.integers(0, 500, size=ROWS),
        "content_similarity": rng.random(ROWS),
        "history_length": rng.integers(1, 100, size=ROWS),
        "is_cold_item": rng.integers(0, 2, size=ROWS),
        "category_idx": rng.integers(1, 6, size=ROWS),
        "subcategory_idx": rng.integers(1, 10, size=ROWS),
    }
    features = np.stack([columns[name] for name in FEATURES], axis=1).astype(np.float32)

    labels = np.zeros(ROWS, dtype=np.int64)
    labels[::per_request] = 1
    return RankingRows(
        names=FEATURES,
        features=features,
        labels=labels,
        groups=np.full(n_requests, per_request, dtype=np.int64),
        items=np.arange(1, ROWS + 1, dtype=np.int64),
        request=request,
        user_ids=np.arange(n_requests, dtype=np.int64),
        observed=np.ones(ROWS, dtype=bool),
        found=np.ones(n_requests, dtype=bool),
    )


@pytest.fixture
def trained_pair(rows: RankingRows) -> tuple[MMoE, Standardiser]:
    model, scaler = fit(MMoE, rows, rows, CPU, epochs=1)
    assert isinstance(model, MMoE)
    return model, scaler


def make_session(graph: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(graph), providers=["CPUExecutionProvider"])


def test_the_wrapper_matches_eager_pytorch_on_raw_columns(
    rows: RankingRows, trained_pair: tuple[MMoE, Standardiser],
) -> None:
    """**The load-bearing test, and it needs no onnxruntime.**

    If the ServableRanker wrapper and the training transform disagree, the ONNX
    export is faithful to a wrapper that is already wrong -- and the export
    parity test below would pass while serving scored a different world.
    """
    model, scaler = trained_pair
    servable = ServableRanker(model, FEATURES, scaler.mean, scaler.scale)
    got = servable(torch.from_numpy(rows.features)).detach().numpy()
    expected = predict(model, scaler, rows, CPU)
    assert np.allclose(got.reshape(-1), expected, atol=1e-5)


def test_onnx_matches_eager_pytorch(
    tmp_path: Path, rows: RankingRows, trained_pair: tuple[MMoE, Standardiser],
) -> None:
    """The export itself. Looser tolerance than above -- 1e-4 rather than 1e-5 --
    because onnxruntime is entitled to fuse and reassociate operations."""
    model, scaler = trained_pair
    servable = ServableRanker(model, FEATURES, scaler.mean, scaler.scale)

    graph = tmp_path / "model.onnx"
    # export() returns the SIDECAR path, not the graph's -- it writes two files
    # and the one it hands back is the one a caller is most likely to forget.
    export(servable, graph)

    session = make_session(graph)
    got = np.asarray(session.run([OUTPUT_NAME], {INPUT_NAME: rows.features})[0])
    expected = predict(model, scaler, rows, CPU)
    assert np.allclose(got.reshape(-1), expected, atol=1e-4)


def test_the_column_sidecar_matches_the_graph(
    tmp_path: Path, trained_pair: tuple[MMoE, Standardiser],
) -> None:
    """The file the Go orchestrator reads must declare the order the graph was
    built for. This is the other half of readColumns in ch 5."""
    model, scaler = trained_pair
    servable = ServableRanker(model, FEATURES, scaler.mean, scaler.scale)

    sidecar = export(servable, tmp_path / "model.onnx")
    # The NAME is part of the contract too: ranking.Client derives it from the
    # graph path by suffix, so a rename here is a startup failure there.
    assert sidecar.name == "model.columns.json"
    assert json.loads(sidecar.read_text())["features"] == list(FEATURES)


def test_two_swapped_columns_change_the_scores(
    rows: RankingRows, trained_pair: tuple[MMoE, Standardiser],
) -> None:
    """Why the sidecar matters. Swapping two columns is a valid-shaped input that
    Triton accepts and the model scores, so nothing downstream can catch it --
    this asserts the scores actually MOVE, i.e. that the order is load-bearing
    rather than incidental."""
    model, scaler = trained_pair
    servable = ServableRanker(model, FEATURES, scaler.mean, scaler.scale)

    # Two columns of the SAME kind, deliberately. Both are counts, so both take
    # the same log1p in the wrapper and the swap survives preprocessing intact
    # -- what moves the score is the standardiser statistics and the weights,
    # which is exactly the divergence a shape check cannot see. Swapping a count
    # for a rank would also move the score, through the preprocessing, and would
    # prove something weaker.
    first = FEATURES.index("prior_clicks")
    second = FEATURES.index("train_clicks")

    straight = servable(torch.from_numpy(rows.features)).detach().numpy()
    permuted = rows.features.copy()
    permuted[:, [first, second]] = permuted[:, [second, first]]
    swapped = servable(torch.from_numpy(permuted)).detach().numpy()

    # Over the BATCH, not element-wise. A row whose two counts happen to be
    # equal scores identically either way -- cold items with 0 and 0 are common
    # -- so an element-wise assertion would be flaky for a reason that is not a
    # bug. The claim is that the order matters, and the batch is where it shows.
    assert not np.allclose(straight, swapped, atol=1e-6)


def test_the_categorical_columns_survive_the_float_round_trip() -> None:
    """Categoricals share a float32 matrix with the dense columns and are cast
    back to int64 inside the graph. A category index past 2**24 would not survive
    that round trip exactly."""