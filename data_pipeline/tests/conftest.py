"""Fixtures shared by every test module in this package.

Where the data comes from. The contract tests run against the real corpus when
it has been built, and against a small committed synthetic corpus otherwise --
which is what makes them meaningful in CI, where no dataset exists. The guide's
C4 is explicit that the contracts "belong in CI, on a fixture dataset small
enough to run in seconds"; tests that merely skip cleanly leave a green badge
on a repository whose labels could be inverted.

The fixture is built THROUGH the real pipeline rather than written as Parquet
by hand. Hand-built tables cannot catch a parse bug, and catching parse bugs is
most of what these contracts are for.
"""

from __future__ import annotations

from functools import reduce
from pathlib import Path

import pytest
from pyspark.sql import DataFrame, SparkSession

from common.config import Paths, Settings, load_settings
from data_pipeline.ingest.bronze import ingest
from data_pipeline.transform.id_maps import build_id_maps
from data_pipeline.transform.silver import build_silver

FIXTURE_RAW = Path(__file__).resolve().parent / "fixtures" / "raw"
REAL_DATA = Path("data")
_FIXTURE_SPLITS = ("train", "dev")


def _fixture_settings(root: Path) -> Settings:
    """Config pointing at the synthetic corpus, with everything else inherited.

    model_copy rather than Settings(paths=...) so the YAML and environment
    layers still apply -- only the paths and the storage backend move.

    The backend is forced local because this corpus IS local: the committed
    default is s3, and inheriting it would send a synthetic fixture to MinIO --
    failing in CI, where no container is running.. Pinned here rather than via
    an environment variable in CI so that a developer running pytest gets the same
    behaviour the pipeline does.
    """
    settings = load_settings().model_copy(
        update={
            "paths": Paths(
                raw=FIXTURE_RAW,
                bronze=root / "bronze",
                silver=root / "silver",
                gold=root / "gold",
            )
        },
        deep=True,
    )
    settings.storage.backend = "local"
    return settings


def _build_fixture(spark: SparkSession, root: Path) -> Path:
    """Run raw -> bronze -> id maps -> silver over the committed fixture."""
    settings = _fixture_settings(root)
    for split in _FIXTURE_SPLITS:
        ingest(spark, settings, split)

    def _union(table: str) -> DataFrame:
        frames = [
            spark.read.parquet(str(settings.paths.bronze / table / s)) for s in _FIXTURE_SPLITS
        ]
        return reduce(lambda a, b: a.unionByName(b), frames)

    # force=True because the guard in build_id_maps exists to protect trained
    # checkpoints, and a throwaway tmpdir has none.
    build_id_maps(_union("events"), _union("news"), settings, force=True)
    for split in _FIXTURE_SPLITS:
        build_silver(spark, settings, split)
    return root


@pytest.fixture(scope="session")
def data_root(spark: SparkSession, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Root of the medallion layers under test.

    Prefers the real corpus, so a local run still validates the actual data;
    falls back to building the synthetic fixture, so CI validates something.
    Which one ran is printed, because a contract suite that silently tested
    1,200 synthetic impressions when you believed it tested 8.6M real ones is
    worse than one that did not run.
    """
    if (REAL_DATA / "bronze" / "events" / "train" / "_SUCCESS").is_file():
        print("contracts: running against the real corpus under data/")
        return REAL_DATA
    print("contracts: no built corpus; building the synthetic fixture")
    return _build_fixture(spark, tmp_path_factory.mktemp("fixture"))
