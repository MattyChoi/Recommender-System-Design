"""Smoke test so CI has something real to run from commit #1."""

import importlib


def test_packages_import() -> None:
    for name in ("data_pipeline", "models", "indexing", "evaluation"):
        assert importlib.import_module(name) is not None
