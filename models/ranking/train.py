"""Fit a neural ranker over retrieved candidates and score the funnel.

Two architectures share this entry point because they share everything else:
the same feature block, the same pointwise loss, the same training loop in
``torch_fit``, the same early-stopping rule. DCN v2 and MMoE differ in what
they do with the features and in nothing else, which is the only condition
under which a difference between them is a statement about architecture.

**The bar is the tree**, in ``models.ranking.baselines``. A neural ranker has
to clear it on a measured number before it earns a second serving stack, and on
this corpus it does not.

**Two things differ between these and the tree, not one.** Architecture, and the
objective: these are fitted pointwise with binary cross-entropy, while the tree
is fitted listwise on within-request comparisons. So a difference in NDCG cannot
be attributed to the architecture alone. The confound is stated rather than
quietly carried.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from functools import partial
from pathlib import Path

import torch

from models.ranking.dcn import DCNv2
from models.ranking.mmoe import MMoE, gate_usage
from models.ranking.pipeline import add_common_arguments, prepare, report
from models.ranking.torch_fit import Factory, predict, split_columns, transform
from models.ranking.torch_fit import fit as fit_torch

ARCHITECTURES = {"dcn": DCNv2, "mmoe": MMoE}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--model", choices=tuple(ARCHITECTURES), default="dcn")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Ray Train workers. Above 1 needs a non-uv driver; see rank-dist.",
    )
    parser.add_argument("--gpu-workers", action="store_true", help="Give each worker a GPU.")
    parser.add_argument(
        "--embeddings",
        choices=("torch", "torchrec"),
        default="torch",
        help="Categorical backend. torchrec is SCAFFOLDING at this scale -- these "
        "tables are 9 KB -- and exists so a million-level categorical is a config "
        "change rather than a rewrite. Needs the `sharded` extra.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")

    prepared = prepare(args, args.model)
    architecture = ARCHITECTURES[args.model]
    factory: Factory = architecture
    if args.embeddings == "torchrec":
        # Imported here so the default path never pays for torchrec, which is an
        # optional extra and pulls in fbgemm.
        from models.ranking.torchrec_block import TorchRecFeatureBlock

        factory = partial(architecture, block=TorchRecFeatureBlock)

    # Fitted on the requests retrieval solved, same as the tree: a group with no
    # positive teaches a pairwise objective nothing, and giving the pointwise
    # model rows the listwise one never saw would make the two incomparable on
    # the one axis they are meant to differ on.
    fitting, held = prepared.fitting.with_positives(), prepared.held.with_positives()

    trace: list[dict[str, float]] = []
    if args.workers > 1:
        from models.ranking.launcher import fit_distributed

        network, scaler, trace = fit_distributed(factory, fitting, held, args, prepared.device)
    else:
        network, scaler = fit_torch(
            factory,
            fitting,
            held,
            prepared.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            patience=args.patience,
            seed=args.seed,
            on_epoch=trace.append,
        )

    scores = predict(network, scaler, prepared.held, prepared.device)

    extra: dict[str, float] = {}
    if isinstance(network, MMoE):
        dense_columns, sparse_columns = split_columns(prepared.held)
        matrix = transform(prepared.held)
        usage = gate_usage(
            network,
            torch.from_numpy(scaler.apply(matrix[:, dense_columns])).to(prepared.device),
            torch.from_numpy(matrix[:, sparse_columns].astype("int64")).to(prepared.device),
        )
        print(
            f"\n  gate entropy OF THE MEAN {usage.entropy:.3f} of {usage.ceiling:.3f} "
            f"({usage.used:.1%} spread across requests)"
        )
        print(
            f"  MEAN per-row entropy     {usage.per_row:.3f} of {usage.ceiling:.3f} "
            f"({usage.specialised:.1%} specialised per request)"
        )
        print("  " + " ".join(f"{value:.3f}" for value in usage.weights) + "\n")
        extra = {"gate_entropy": usage.entropy, "gate_per_row_entropy": usage.per_row}

    def save(directory: Path) -> Path:
        artifact = directory / f"{prepared.run_name}.pt"
        # The standardiser rides along. A neural ranker restored without the
        # statistics it was fitted under scores confidently and wrongly, and
        # nothing about the tensor shapes says anything is missing.
        torch.save(
            {
                "model": network.state_dict(),
                "mean": scaler.mean,
                "scale": scaler.scale,
                "features": prepared.held.names,
                **prepared.marks,
            },
            artifact,
        )
        return artifact

    return report(args, prepared, args.model, scores, [], save, extra=extra, trace=trace)


if __name__ == "__main__":
    raise SystemExit(main())
