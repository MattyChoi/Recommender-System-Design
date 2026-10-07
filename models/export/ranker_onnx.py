"""Export the serving ranker as ONNX, with its preprocessing inside the graph.

Run: ``uv run python -m models.export.ranker_onnx <checkpoint.pt> --out <model.onnx>``
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

from models.ranking.dataset import CATEGORICAL, FEATURES
from models.ranking.mmoe import MMoE
from models.ranking.torch_fit import COUNT_COLUMNS, RANK_COLUMNS

# The ONNX input and output names. Fixed, because Triton's config.pbtxt names
# them and a rename is a silent 400 at serving time.
INPUT_NAME = "features"
OUTPUT_NAME = "score"

OPSET = 18


class ServableRanker(nn.Module):
    """A trained ranker plus the preprocessing it was fitted with.

    Takes ``[B, n_features]`` RAW float32 columns in ``FEATURES`` order and
    returns ``[B, 1]`` logits.

    Args:
        model: The trained ranker, in eval mode.
        names: Column names in matrix order, from the checkpoint. Read from the
            checkpoint rather than from :data:`FEATURES` so a run that dropped a
            column exports the model it actually trained.
        mean: Standardiser means, over the DENSE columns only.
        scale: Standardiser scales, likewise.

    Note:
        The categorical columns arrive as floats -- they share one matrix with
        the dense ones -- and are cast to long inside. That cast is a truncation
        toward zero, which is exact for the integer-valued category indices this
        carries and would silently floor anything else. Stated because the input
        tensor's dtype no longer distinguishes the two kinds of column.
    """

    mean: torch.Tensor
    scale: torch.Tensor
    dense_index: torch.Tensor
    sparse_index: torch.Tensor

    def __init__(
        self,
        model: nn.Module,
        names: Sequence[str],
        mean: np.ndarray,
        scale: np.ndarray,
    ) -> None:
        super().__init__()
        self.model = model
        self.names = tuple(names)

        sparse = [self.names.index(name) for name in CATEGORICAL if name in self.names]
        dense = [index for index in range(len(self.names)) if index not in set(sparse)]

        # Registered as buffers, not Python lists: a buffer is exported into the
        # graph as an initialiser, so the indices and the statistics travel
        # inside the .onnx file rather than being re-derived by whoever loads it.
        self.register_buffer("dense_index", torch.tensor(dense, dtype=torch.long))
        self.register_buffer("sparse_index", torch.tensor(sparse, dtype=torch.long))
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))

        # Plain Python lists, NOT buffers. These drive a `for` loop in forward.
        # As constants they are folded into the exported ops, which is what they are.
        self.rank_positions = [self.names.index(n) for n in RANK_COLUMNS if n in self.names]
        self.count_positions = [self.names.index(n) for n in COUNT_COLUMNS if n in self.names]

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Column-wise scatter rather than in-place assignment: ONNX export
        # traces in-place index writes into something far uglier, and a copy of
        # an [N, 11] matrix is free next to the model.
        columns = [features[:, index] for index in range(len(self.names))]

        for index in self.rank_positions:
            # 1 / (1 + rank): rank 0 -> 1, rank 99 -> 0.01, the ABSENT sentinel
            # -> ~0. Bounded, monotone the right way, and it parks "absent"
            # beside "worst" instead of fifty standard deviations away.
            columns[index] = 1.0 / (1.0 + columns[index])
        for index in self.count_positions:
            columns[index] = torch.log1p(torch.clamp(columns[index], min=0.0))

        prepared = torch.stack(columns, dim=1)
        dense = (prepared.index_select(1, self.dense_index) - self.mean) / self.scale
        sparse = prepared.index_select(1, self.sparse_index).long()

        scores: torch.Tensor = self.model(dense, sparse)
        # `[batch, 1]`, not `[batch]`
        return scores.unsqueeze(-1)


def load_checkpoint(path: Path, device: torch.device | None = None) -> ServableRanker:
    """Rebuild the trained ranker and its standardiser from one file.

    Raises:
        KeyError: If the checkpoint predates the standardiser being saved
            alongside the weights. Loud rather than defaulting to an identity
            transform, which would score every row confidently and wrongly.
    """
    where = device or torch.device("cpu")
    blob = torch.load(path, map_location=where, weights_only=False)

    for required in ("model", "mean", "scale", "features"):
        if required not in blob:
            raise KeyError(
                f"{path} has no {required!r}. A ranker restored without the "
                "statistics it was fitted under scores confidently and wrongly, "
                "and no tensor shape says anything is missing."
            )

    names = tuple(blob["features"])
    sparse = [names.index(name) for name in CATEGORICAL if name in names]
    n_dense = len(names) - len(sparse)

    # Cardinalities are recovered from the saved embedding tables rather than
    # from the data: the table is what the weights were trained against, and
    # re-deriving it from a fresh scoring set could differ by a rare category.
    state = blob["model"]
    cardinalities = tuple(
        int(state[key].shape[0]) - 1
        for key in sorted(state, key=lambda k: int(k.split(".")[-2]))
        if key.startswith("features.embeddings.") and key.endswith(".weight")
    )

    model = MMoE(n_dense, cardinalities)
    model.load_state_dict(state)
    model.eval()

    servable = ServableRanker(model, names, blob["mean"], blob["scale"])
    servable.eval()
    return servable


def export(servable: ServableRanker, out: Path, batch: int = 4) -> Path:
    """Write the ONNX graph, with a dynamic batch axis.

    A fixed batch size would make the artefact usable only at the size it was
    traced with, and the candidate count per request is not constant --
    multi-source blending tops up short source lists, so a request can carry
    fewer than the configured maximum.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(batch, len(servable.names), dtype=torch.float32)

    # `dynamo=True` with `dynamic_shapes`, both stated explicitly.
    torch.onnx.export(
        servable,
        (example,),
        str(out),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_shapes={"features": {0: torch.export.Dim("batch")}},
        opset_version=OPSET,
        dynamo=True,
    )

    # The column order is the one contract that could not be moved inside the
    # graph, so it is written beside it. A serving client that builds columns in
    # a different order gets a model scoring a different world, silently.
    sidecar = out.with_suffix(".columns.json")
    sidecar.write_text(json.dumps({"features": list(servable.names)}, indent=2) + "\n")
    return sidecar


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="A .pt saved by the neural trainer.")
    parser.add_argument("--out", type=Path, default=Path("serving/triton/ranker/1/model.onnx"))
    args = parser.parse_args(argv)

    servable = load_checkpoint(args.checkpoint)
    sidecar = export(servable, args.out)

    print(f"  columns  {list(servable.names)}")
    print(f"  onnx  -> {args.out}")
    print(f"  order -> {sidecar}")
    print(
        "\n  The graph takes RAW features and does its own reciprocal, log1p and\n"
        "  standardisation, so the serving path cannot drift from the training\n"
        "  path by reimplementing them. What still has to agree is the column\n"
        "  ORDER above."
    )
    if list(servable.names) != list(FEATURES):
        print("\n  NOTE: this run dropped columns; the order above is NOT the full FEATURES list.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
