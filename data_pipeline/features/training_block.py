"""The user-feature block training actually read, as a dense array.

Shared, not duplicated, because two things bin against it and they must bin
against the SAME distribution: ``scripts/writeups/skew_report.py`` compares it
with what serving logged, and ``data_pipeline.serving_artifacts.
dump_drift_baseline`` freezes its quantiles into the artifact the live drift
monitor bins on. Selected two different ways, the report's PSI and the monitor's
gauge would be computed against different baselines and disagree with no way to
see why.

Two selection rules carry that comparability, and both are easy to get wrong:

* **One row per impression.** ``training_examples`` holds a row per candidate,
  so an un-deduplicated read weights a request by its slate length -- and slate
  length is not a user feature.
* **Found users only.** A missing user row is filled with zeros on both sides,
  so pooling those rows lets the MISS RATE masquerade as a shift in the values.
  The miss rate is a coverage number and belongs beside the distribution, not
  inside it.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from common.features import USER_FEATURES, USER_FLAGS


def load_training(split: str) -> tuple[npt.NDArray[np.float64], int]:
    """Training's user block for found users, and the total request count.

    Args:
        split: A gold ``training_examples`` split, e.g. ``train``.

    Returns:
        ``(block, requests)`` -- ``[found requests, len(USER_FEATURES)]`` and the
        number of requests before the found-user filter, so a caller can report
        coverage rather than inferring it from the block's height.
    """
    from common.config import load_settings
    from common.spark import get_spark
    from common.utils import read_gold

    settings = load_settings()
    spark = get_spark(settings, app="training-block")
    frame = (
        read_gold(spark, settings, f"training_examples/{split}")
        .select("impression_id", *USER_FEATURES, *USER_FLAGS)
        .dropDuplicates(["impression_id"])
        .toPandas()
    )
    found = frame[USER_FLAGS[0]].fillna(False).astype(bool).to_numpy()
    block = frame.loc[found, list(USER_FEATURES)].to_numpy(dtype=np.float64)
    return block, len(frame)
