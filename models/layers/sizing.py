"""Embedding-table arithmetic, with no dependency on how the table is built.

Split out of ``sharded_embeddings`` because it has no business being behind an
optional extra: ``models/layers/hashing.py`` needs the same byte arithmetic and
must stay runnable on a machine that has never installed TorchRec. A constant
that two modules need is declared once or it eventually means two things.
"""

from __future__ import annotations

#: The manual's stated design target for G4. The corpus is 65,238; the gap is
#: the point, and G4's own text ("160K MIND articles") has the corpus wrong.
DESIGN_TARGET_ITEMS = 2_000_000

#: MEASURED by ``make shard-plan``: the catalogue size at which an 8 x 24 GB pod
#: stops placing a 64-dim fp32 table whole on one device. Matches
#: ``hbm x (1 - reservation) / (dim x 4)`` to 0.1%, which is the finding -- the
#: boundary is capacity arithmetic and the reservation fraction is a constant we
#: chose, not something TorchRec's cost model discovered.
CROSSOVER_ITEMS = 85_648_437

#: The same boundary on an 8 x 80 GB pod.
CROSSOVER_ITEMS_80GB = 286_015_625


def table_bytes(n_rows: int, dim: int, bytes_per_element: int = 4) -> int:
    """Raw parameter bytes for one embedding table.

    Deliberately excludes optimiser state. TorchRec's fused kernel defaults to
    rowwise Adagrad -- one float per ROW, so about 8 MB on a 2M-row table rather
    than the 512 MB that Adam's two moments per ELEMENT would cost. Quoting
    parameter bytes as "the memory" understates a table's footprint under a
    dense optimiser by 3x, and overstates the saving under a rowwise one; every
    report that uses this says which it is showing.

    Args:
        n_rows: Rows INCLUDING any reserved row.
        dim: Embedding width.
        bytes_per_element: 4 for fp32, 2 for fp16/bf16.

    Returns:
        Bytes.
    """
    return n_rows * dim * bytes_per_element
