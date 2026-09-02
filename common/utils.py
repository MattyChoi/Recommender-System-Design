from __future__ import annotations

from common.config import Settings

SPLITS = ("train", "dev", "test")
BRONZE_TABLES = ("events", "history", "news")
MIND_TS_FORMAT = "M/d/yyyy h:mm:ss a"


def _is_built(settings: Settings, layer: str, split: str) -> bool:
    """Whether the given data layer for ``split`` holds a committed write.

    Spark drops ``_SUCCESS`` into an output directory only after the write job
    commits, so a run that died midway leaves part-files behind but no marker
    and is correctly reported as unbuilt.
    """
    if layer not in ("bronze", "silver"):
        raise ValueError(f"unknown layer {layer!r}")

    if layer == "bronze":
        return all(
            (settings.paths.bronze / table / split / "_SUCCESS").is_file()
            for table in BRONZE_TABLES
        )
    elif layer == "silver":
        return (settings.paths.silver / "impressions" / split / "_SUCCESS").is_file()
    return False
