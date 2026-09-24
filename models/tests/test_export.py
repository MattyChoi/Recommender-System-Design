"""The exported ranker, and the four ways it could serve a different model.

**The artefact is not the model that was measured until something says so.**
Every number in `docs/ranking.md` came from eager PyTorch through
``torch_fit.predict``; production runs an ONNX graph through onnxruntime. Those
are two programs, and a claim that the deployed system scores 0.1403 NDCG@10 is
a claim about their equality.

Four ways they can differ while both look healthy:

1. **The preprocessing.** ``predict`` applies a reciprocal, a ``log1p`` and a
   fitted standardiser before the model sees anything. The export bakes those
   into the graph; if the baked version disagrees with the original, every score
   is wrong and nothing raises.
2. **The column order.** Eleven columns in one matrix, two of them categorical.
   A permuted order is a model scoring a different world.
3. **The batch axis.** Traced at one size and served at another is a shape error
   at best and a silently truncated batch at worst.
4. **A missing standardiser.** A checkpoint without ``mean``/``scale`` must
   fail loudly, because defaulting to an identity transform produces confident
   nonsense.

The first test runs without onnxruntime, on purpose: it isolates "the baked
preprocessing is right" from "the ONNX round trip is faithful", so a failure
names which half broke.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from models.export.onnx import INPUT_NAME, OUTPUT_NAME, ServableRanker, export, load_checkpoint
from models.ranking.dataset import FEATURES, RankingRows
from models.ranking.mmoe import MMoE
from models.ranking.torch_fit import (
    FeatureBlock,
    Standardiser,
    predict,
    split_columns,
    transform,
)

ROWS = 12
CARDINALITIES = (5, 9)


def make_rows(n: int = ROWS) -> RankingRows:
    """A table carrying every FEATURES column, with realistic magnitudes.

    The ranges matter: a rank sentinel at 10,000 beside a cosine near 0 is the
    spread the preprocessing exists to bound, so a fixture of small tidy numbers
    would let a broken transform pass.
    """
    rng = np.random.default_rng(0)
    columns = {
        "retrieval_score": rng.normal(size=n),
        "two_tower_rank": rng.choice([0, 7, 99, 10_000], size=n),
        "trending_rank": rng.choice([0, 3, 10_000], size=n),
        "n_sources": rng.integers(0, 4, size=n),
        "prior_clicks": rng.integers(0, 5_000, size=n),
        "train_clicks": rng.integers(0, 900, size=n),
        "content_similarity": rng.normal(size=n) * 0.1,
        "history_length": rng.integers(0, 50, size=n),
        "is_cold_item": rng.integers(0, 2, size=n),
        "category_idx": rng.integers(0, CARDINALITIES[0] + 1, size=n),
        "subcategory_idx": rng.integers(0, CARDINALITIES[1] + 1, size=n),
    }
    matrix = np.stack([columns[name] for name in FEATURES], axis=1).astype(np.float32)
    return RankingRows(
        names=FEATURES,
        features=matrix,
        labels=np.zeros(n, dtype=np.int64),
        groups=np.asarray([n], dtype=np.int64),
        items=np.arange(1, n + 1, dtype=np.int64),
        request=np.zeros(n, dtype=np.int64),
        user_ids=np.asarray([1], dtype=np.int64),
        observed=np.zeros(n, dtype=bool),
        found=np.asarray([False], dtype=bool),
    )


def trained_pair(rows: RankingRows) -> tuple[MMoE, Standardiser]:
    """An untrained-but-initialised model and a standardiser fitted on ``rows``.

    Training is not needed and would only add noise: the question is whether two
    executions of the SAME weights agree, and random weights exercise the
    arithmetic exactly as well as fitted ones.
    """
    torch.manual_seed(0)
    dense_columns, _ = split_columns(rows)
    scaler = Standardiser.fit(transform(rows)[:, dense_columns])
    model = MMoE(len(dense_columns), CARDINALITIES)
    model.eval()
    return model, scaler


def save_checkpoint(path: Path, model: MMoE, scaler: Standardiser) -> Path:
    torch.save(
        {
            "model": model.state_dict(),
            "mean": scaler.mean,
            "scale": scaler.scale,
            "features": FEATURES,
        },
        path,
    )
    return path


class TestTheBakedPreprocessing:
    def test_the_wrapper_reproduces_predict(self) -> None:
        """**The load-bearing test, and it needs no onnxruntime.** If the
        preprocessing baked into the exported graph disagrees with the
        preprocessing every offline number was computed through, the deployed
        model is a different model -- and this isolates that from the ONNX round
        trip, so a failure says which half broke."""
        rows = make_rows()
        model, scaler = trained_pair(rows)

        expected = predict(model, scaler, rows, torch.device("cpu"))

        servable = ServableRanker(model, FEATURES, scaler.mean, scaler.scale)
        with torch.no_grad():
            got = servable(torch.from_numpy(rows.features)).numpy()

        # [B, 1] against eager's [B]. The trailing axis is a serving
        # requirement -- Triton's batcher needs per-sample output dims -- so the
        # shape is asserted rather than quietly flattened away by broadcasting,
        # which would let a [B, B] result pass this comparison.
        assert got.shape == (len(rows.labels), 1)
        assert np.allclose(got.reshape(-1), expected, atol=1e-5)

    def test_the_raw_input_is_genuinely_untransformed(self) -> None:
        """The control. The wrapper takes RAW columns, so feeding it the
        already-transformed matrix must give a DIFFERENT answer -- otherwise the
        test above would pass on a wrapper that did no preprocessing at all."""
        rows = make_rows()
        model, scaler = trained_pair(rows)
        servable = ServableRanker(model, FEATURES, scaler.mean, scaler.scale)

        with torch.no_grad():
            raw = servable(torch.from_numpy(rows.features)).numpy()
            pre_transformed = servable(torch.from_numpy(transform(rows))).numpy()

        assert not np.allclose(raw, pre_transformed)

    def test_a_checkpoint_without_a_standardiser_is_refused(self, tmp_path: Path) -> None:
        """Loud, not defaulted. An identity transform here would score every row
        confidently and wrongly, and no shape would say anything was missing."""
        rows = make_rows()
        model, _ = trained_pair(rows)
        path = tmp_path / "bare.pt"
        torch.save({"model": model.state_dict(), "features": FEATURES}, path)

        with pytest.raises(KeyError, match=r"scale|mean"):
            load_checkpoint(path)

    def test_cardinalities_are_recovered_from_the_saved_tables(self, tmp_path: Path) -> None:
        """Read off the weights, not re-derived from data. A fresh scoring set
        can be missing a rare category, which would build a smaller embedding
        table than the one the weights were trained with."""
        rows = make_rows()
        model, scaler = trained_pair(rows)
        path = save_checkpoint(tmp_path / "ranker.pt", model, scaler)

        servable = load_checkpoint(path)

        # Two casts, and the second one is the interesting one. `MMoE.features`
        # is typed as the ABSTRACTION, `FeatureEmbedding`, which declares only
        # `width` and `forward` -- `embeddings` belongs to the dense backend,
        # `FeatureBlock`. So this assertion reaches through the interface into
        # one implementation's internals, and the cast is that being said out
        # loud rather than silently permitted. The TorchRec backend holds its
        # tables somewhere else entirely, which is exactly why the base class
        # does not promise them.
        model = cast("MMoE", servable.model)
        block = cast("FeatureBlock", model.features)
        for index, size in enumerate(CARDINALITIES):
            table = cast("torch.nn.Embedding", block.embeddings[index])
            assert table.weight.shape[0] == size + 1


class TestTheOnnxGraph:
    @staticmethod
    def session(path: Path):  # type: ignore[no-untyped-def]  # type is in an optional dep
        onnxruntime = pytest.importorskip(
            "onnxruntime", reason="pip install onnxruntime to run the export parity test"
        )
        return onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def test_onnx_matches_eager_pytorch(self, tmp_path: Path) -> None:
        """The claim the whole export rests on: the served graph and the
        measured model return the same scores for the same rows."""
        rows = make_rows()
        model, scaler = trained_pair(rows)
        path = save_checkpoint(tmp_path / "ranker.pt", model, scaler)
        servable = load_checkpoint(path)
        out = tmp_path / "model.onnx"
        export(servable, out)

        expected = predict(model, scaler, rows, torch.device("cpu"))
        got = self.session(out).run([OUTPUT_NAME], {INPUT_NAME: rows.features})[0]

        assert got.shape == (len(rows.labels), 1)
        assert np.allclose(got.reshape(-1), expected, atol=1e-4)

    def test_the_batch_axis_is_dynamic(self, tmp_path: Path) -> None:
        """Traced at one size, served at another. Part I's blend tops up short
        source lists, so a request can carry fewer candidates than the configured
        maximum and a fixed batch would reject it."""
        rows = make_rows(ROWS)
        model, scaler = trained_pair(rows)
        path = save_checkpoint(tmp_path / "ranker.pt", model, scaler)
        out = tmp_path / "model.onnx"
        export(load_checkpoint(path), out, batch=4)

        session = self.session(out)
        for size in (1, 3, 7, ROWS):
            smaller = make_rows(size)
            got = session.run([OUTPUT_NAME], {INPUT_NAME: smaller.features})[0]
            assert got.shape == (size, 1)

    def test_the_column_order_travels_with_the_graph(self, tmp_path: Path) -> None:
        """The one contract that could not be moved inside the graph, so it is
        written beside it."""
        rows = make_rows()
        model, scaler = trained_pair(rows)
        path = save_checkpoint(tmp_path / "ranker.pt", model, scaler)
        out = tmp_path / "model.onnx"

        sidecar = export(load_checkpoint(path), out)

        assert json.loads(sidecar.read_text())["features"] == list(FEATURES)

    def test_a_permuted_column_order_changes_the_scores(self, tmp_path: Path) -> None:
        """Why the sidecar matters. Swapping two columns is a valid-shaped input
        that scores a different world, with no error anywhere."""
        rows = make_rows()
        model, scaler = trained_pair(rows)
        path = save_checkpoint(tmp_path / "ranker.pt", model, scaler)
        out = tmp_path / "model.onnx"
        export(load_checkpoint(path), out)
        session = self.session(out)

        straight = session.run([OUTPUT_NAME], {INPUT_NAME: rows.features})[0]
        swapped = rows.features.copy()
        first = FEATURES.index("prior_clicks")
        second = FEATURES.index("train_clicks")
        swapped[:, [first, second]] = swapped[:, [second, first]]
        permuted = session.run([OUTPUT_NAME], {INPUT_NAME: swapped})[0]

        assert not np.allclose(straight, permuted)


def test_the_categorical_columns_survive_the_float_round_trip() -> None:
    """Categoricals share a float32 matrix with the dense columns and are cast
    to long inside the graph. float32 holds integers exactly to 2**24, and the
    largest index here is 121 -- but the cast truncates toward zero, so the
    property is worth pinning rather than assuming."""
    rows = make_rows()
    model, scaler = trained_pair(rows)
    servable = ServableRanker(model, FEATURES, scaler.mean, scaler.scale)

    prepared = torch.from_numpy(transform(rows))
    sparse_index = servable.get_buffer("sparse_index")
    recovered = prepared.index_select(1, sparse_index).long().numpy()
    _, sparse_columns = split_columns(rows)
    expected = transform(rows)[:, sparse_columns].astype(np.int64)

    assert np.array_equal(recovered, expected)
    # And the values are real category indices, not everything collapsed to 0 --
    # which is what a bad cast would produce and what the equality above would
    # still accept if `expected` were collapsed the same way.
    assert recovered.max() > 0
