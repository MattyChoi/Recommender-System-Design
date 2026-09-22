"""Provenance, which is what makes a metric citable rather than merely produced.

The dirty flag is the one that earns its place. A SHA recorded from a modified
working tree names a commit that does not contain the code that ran, and it is
the usual way a run stops being reproducible while still looking traceable.

Nothing here needs MLflow, which is a compose service: :func:`track` is exercised
through its disabled arm.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from common.config import Settings, load_settings
from common.tracking import (
    config_hash,
    dataset_version,
    git_revision,
    mlflow_name,
    provenance,
    require_reachable,
    run_label,
    track,
)
from tests.test_config import REPO_ROOT


def _settings(gold: Path) -> Settings:
    local = load_settings(REPO_ROOT).model_copy(deep=True)
    local.paths.gold = gold
    local.storage.backend = "local"
    local.mlflow.enabled = False
    return local


class TestTheGitRevision:
    def test_a_clean_checkout_reports_clean(self, tmp_path: Path) -> None:
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        subprocess.run(
            ["git", "commit", "-q", "--allow-empty", "-m", "x"], cwd=tmp_path, check=True
        )

        got = git_revision(tmp_path)

        assert len(got["git_sha"]) == 40
        assert got["git_dirty"] == "false"

    def test_an_edited_tree_reports_dirty(self, tmp_path: Path) -> None:
        """The whole reason the flag exists: the SHA below is unchanged and no
        longer describes what would run."""
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        subprocess.run(
            ["git", "commit", "-q", "--allow-empty", "-m", "x"], cwd=tmp_path, check=True
        )
        (tmp_path / "changed.py").write_text("x = 1\n")

        assert git_revision(tmp_path)["git_dirty"] == "true"

    def test_no_repository_says_unknown(self, tmp_path: Path) -> None:
        """A tarball or a container without git. Saying 'unknown' beats logging
        a plausible-looking blank."""
        got = git_revision(tmp_path / "not-a-repo")

        assert got == {"git_sha": "unknown", "git_dirty": "unknown", "code_hash": "unknown"}


class TestTheCodeHash:
    """Written after two runs scoring 0.1040 and 0.1172 shared a name, a commit,
    a dirty flag and a checkpoint path -- the second overwrote the first."""

    @staticmethod
    def _repository(root: Path) -> None:
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", "commit", "-q", "--allow-empty", "-m", "x"], cwd=root, check=True)

    def test_a_clean_tree_names_its_commit(self, tmp_path: Path) -> None:
        self._repository(tmp_path)

        got = git_revision(tmp_path)

        assert got["code_hash"] == got["git_sha"][:12]

    def test_two_dirty_trees_at_one_commit_differ(self, tmp_path: Path) -> None:
        """**The bug, as a test.** Both states below share a SHA and a dirty
        flag, so a name built from either would collide. Only a digest of the
        working tree separates them."""
        self._repository(tmp_path)
        edited = tmp_path / "tracked.py"
        subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)

        edited.write_text("init_fn = None\n")
        first = git_revision(tmp_path)
        edited.write_text("init_fn = normal_\n")
        second = git_revision(tmp_path)

        assert first["git_sha"] == second["git_sha"]
        assert first["git_dirty"] == second["git_dirty"] == "true"
        assert first["code_hash"] != second["code_hash"]

    def test_an_untracked_file_counts(self, tmp_path: Path) -> None:
        """The edit that caused the collision was in a file git had never seen,
        so `git diff HEAD` alone would not have noticed it."""
        self._repository(tmp_path)
        before = git_revision(tmp_path)

        (tmp_path / "torchrec_block.py").write_text("x = 1\n")
        after = git_revision(tmp_path)

        assert before["code_hash"] != after["code_hash"]

    def test_editing_an_untracked_file_counts_too(self, tmp_path: Path) -> None:
        """The control. Hashing only the FILE NAMES would pass the test above
        and still miss every edit to a new module."""
        self._repository(tmp_path)
        new = tmp_path / "torchrec_block.py"

        new.write_text("init_fn = None\n")
        before = git_revision(tmp_path)
        new.write_text("init_fn = normal_\n")

        assert git_revision(tmp_path)["code_hash"] != before["code_hash"]

    def test_an_unchanged_dirty_tree_is_stable(self, tmp_path: Path) -> None:
        """A digest that moved on its own would make every rerun a new run and
        turn the overwrite bug into an infinite pile of checkpoints."""
        self._repository(tmp_path)
        (tmp_path / "changed.py").write_text("x = 1\n")

        assert git_revision(tmp_path)["code_hash"] == git_revision(tmp_path)["code_hash"]


class TestTheRunLabel:
    def test_it_carries_both_hashes(self) -> None:
        marks = {"config_hash": "cccccccccccccccc", "code_hash": "dddddddddddd"}

        assert run_label("lgbm-c100", marks) == "lgbm-c100-cccccccccccccccc-dddddddddddd"

    def test_a_code_change_alone_moves_the_name(self) -> None:
        """Which is exactly what the old name could not do."""
        config = {"config_hash": "cccccccccccccccc", "code_hash": "dddddddddddd"}
        edited = {**config, "code_hash": "eeeeeeeeeeee"}

        assert run_label("arm", config) != run_label("arm", edited)


class TestTheConfigHash:
    def test_the_same_inputs_give_the_same_hash(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)

        assert config_hash(settings, {"lr": 0.1}) == config_hash(settings, {"lr": 0.1})

    def test_an_argument_change_moves_it(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)

        assert config_hash(settings, {"lr": 0.1}) != config_hash(settings, {"lr": 0.2})

    def test_a_settings_change_moves_it(self, tmp_path: Path) -> None:
        """Two runs of the same command line under different RECSYS_ overrides
        are different configurations."""
        first = _settings(tmp_path)
        second = _settings(tmp_path)
        second.split.holdout_days = first.split.holdout_days + 1

        assert config_hash(first, {}) != config_hash(second, {})

    def test_argument_order_does_not_matter(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)

        assert config_hash(settings, {"a": 1, "b": 2}) == config_hash(settings, {"b": 2, "a": 1})


class TestTheDatasetVersion:
    def test_rebuilding_a_table_moves_it(self, tmp_path: Path) -> None:
        """The marker's mtime changes exactly when the table is rebuilt, which
        is the property that makes it usable as a version."""
        settings = _settings(tmp_path)
        marker = tmp_path / "training_examples" / "_SUCCESS"
        marker.parent.mkdir(parents=True)
        marker.touch()

        before = dataset_version(settings, ["training_examples"])
        os.utime(marker, (0, 0))

        assert dataset_version(settings, ["training_examples"]) != before

    def test_an_unbuilt_table_is_not_silently_skipped(self, tmp_path: Path) -> None:
        """Training against an unbuilt table is a different run from training
        against a built one, and the version has to say so."""
        settings = _settings(tmp_path)
        marker = tmp_path / "user_history" / "_SUCCESS"
        marker.parent.mkdir(parents=True)

        missing = dataset_version(settings, ["user_history"])
        marker.touch()

        assert dataset_version(settings, ["user_history"]) != missing

    def test_the_table_order_does_not_matter(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)

        assert dataset_version(settings, ["a", "b"]) == dataset_version(settings, ["b", "a"])


class TestTheMetricName:
    """MLflow's charset rejects `@`, which this project uses in every metric it
    reports. The translation lives at that one backend, not in the metric."""

    def test_the_at_notation_survives_as_words(self) -> None:
        assert mlflow_name("recall@100") == "recall_at_100"
        assert mlflow_name("ndcg@10_by_user") == "ndcg_at_10_by_user"

    def test_a_clean_name_is_untouched(self) -> None:
        assert mlflow_name("loss") == "loss"

    def test_nothing_mlflow_rejects_survives(self) -> None:
        """The rule, asserted rather than trusted: alphanumerics, `_-. :/` only."""
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-. :/")

        assert set(mlflow_name("recall@100")) <= allowed


class TestTrack:
    def test_disabled_tracking_still_yields_a_recorder(self, tmp_path: Path) -> None:
        """One code path in the caller, whether or not the server is up."""
        with track(_settings(tmp_path), "run", {"a": 1}) as run:
            run.log_metrics({"loss": 1.0}, step=0)
            run.log_artifact("nowhere")

    def test_an_unreachable_server_fails_before_any_work(self, tmp_path: Path) -> None:
        """The whole point is WHERE this raises. Left to `track`, a missing
        server costs a full Spark read first."""
        settings = _settings(tmp_path)
        settings.mlflow.enabled = True
        settings.mlflow.tracking_uri = "http://127.0.0.1:1"

        with pytest.raises(RuntimeError, match="make up"):
            require_reachable(settings, timeout=0.25)

    def test_disabled_tracking_is_never_probed(self, tmp_path: Path) -> None:
        """Turning tracking off is a decision, and it must not need a server to
        be absent in a particular way."""
        settings = _settings(tmp_path)
        settings.mlflow.tracking_uri = "http://127.0.0.1:1"

        require_reachable(settings, timeout=0.25)

    def test_provenance_carries_every_mark(self, tmp_path: Path) -> None:
        got = provenance(_settings(tmp_path), {"lr": 0.1}, ["training_examples"])

        assert set(got) == {
            "git_sha",
            "git_dirty",
            "code_hash",
            "config_hash",
            "dataset_version",
        }
