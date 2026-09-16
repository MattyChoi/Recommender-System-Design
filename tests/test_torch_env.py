"""Device decisions, and the rule that keeps Part G runnable on both machines.

This project is built on an Apple-silicon Mac and on a 4090 under WSL2, so every
device-dependent decision lives in ``common/torch_env.py`` and nowhere else. The
last test is the one that enforces that; the rest check the decisions themselves.

Nothing here needs a GPU. ``torch.device("cuda")`` is just a name, and every
function under test branches on ``device.type`` rather than touching hardware --
which is what makes the CUDA branches testable from the Mac and the MPS branches
testable from WSL2. A test suite that could only check the device it happens to
be running on would be no guard at all.
"""

from __future__ import annotations

import ast
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch

from common.torch_env import (
    autocast_for,
    dataloader_kwargs,
    distributed_backend,
    grad_scaler_for,
    select_device,
)

_REPO = Path(__file__).resolve().parents[1]
_ALL = [torch.device("cpu"), torch.device("cuda"), torch.device("mps")]


class TestDeviceDecisions:
    @pytest.mark.parametrize("device", _ALL, ids=lambda d: d.type)
    def test_the_scaler_is_disabled_everywhere(self, device: torch.device) -> None:
        """bf16 has fp32's exponent range and fp32 needs no scaling either, so
        there is no device on which loss scaling does anything but cost an
        inf/nan check. Pinned across all three because the bug this replaced was
        a condition that was true on exactly one of them."""
        assert grad_scaler_for(device).is_enabled() is False

    def test_autocast_is_on_only_for_cuda(self) -> None:
        """Returning a real no-op context rather than a disabled autocast is
        what keeps ``if device.type ==`` out of the training loop."""
        assert not isinstance(autocast_for(torch.device("cuda")), nullcontext)
        assert isinstance(autocast_for(torch.device("mps")), nullcontext)
        assert isinstance(autocast_for(torch.device("cpu")), nullcontext)

    def test_nccl_is_only_offered_to_cuda(self) -> None:
        """NCCL is CUDA-only. Offering it elsewhere fails at the first
        collective rather than at initialisation, which is much later and
        reads as a model bug."""
        assert distributed_backend(torch.device("cuda")) == "nccl"
        assert distributed_backend(torch.device("mps")) == "gloo"
        assert distributed_backend(torch.device("cpu")) == "gloo"

    def test_pin_memory_is_only_offered_to_cuda(self) -> None:
        """MPS shares memory with the host, so pinning buys nothing and warns."""
        assert dataloader_kwargs(torch.device("cuda"))["pin_memory"] is True
        assert dataloader_kwargs(torch.device("mps"))["pin_memory"] is False
        assert dataloader_kwargs(torch.device("cpu"))["pin_memory"] is False

    def test_prefer_overrides_whatever_is_present(self) -> None:
        """A test that must not depend on accelerator availability needs a way
        to say so, or it passes on one machine and fails on the other."""
        assert select_device("cpu").type == "cpu"

    def test_the_selected_device_is_one_we_have_decisions_for(self) -> None:
        assert select_device().type in {"cuda", "mps", "cpu"}


# --------------------------------------------------------------- the guard

_ROOTS = ("models", "evaluation", "indexing", "data_pipeline", "common")
_BANNED = {"cuda", "mps"}
_ESCAPE = "allow-device-literal"
# torch_env is the module whose whole job is to name devices. common/pb is
# generated protobuf -- excluded from ruff for the same reason it is excluded
# here: nothing in it was written by anyone.
_EXEMPT = ("common/torch_env.py", "common/pb")


def _docstrings(tree: ast.Module) -> set[int]:
    """Constant nodes that are docstrings, which may name a device freely.

    Prose explaining *why* CUDA is treated differently is the opposite of the
    problem; only executable strings are.
    """
    found: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            found.add(id(body[0].value))
    return found


def _is_exempt(path: Path) -> bool:
    rel = path.relative_to(_REPO).as_posix()
    return any(rel == entry or rel.startswith(f"{entry}/") for entry in _EXEMPT)


def _offences(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source)
    skip = _docstrings(tree)

    hits: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or id(node) in skip:
            continue
        if not isinstance(node.value, str) or node.value.lower() not in _BANNED:
            continue
        line = lines[node.lineno - 1]
        if _ESCAPE in line:
            continue
        hits.append(f"{path.relative_to(_REPO)}:{node.lineno}: {line.strip()}")
    return hits


def test_no_module_names_a_device_but_torch_env() -> None:
    """The portability rule, enforced rather than remembered.

    A comment asking people to route device choices through ``torch_env`` will
    not survive Part K. This will: the moment someone debugging on one machine
    writes ``torch.device("cuda")`` or ``autocast("cuda")`` into a model, the
    build goes red and names the line.

    Comments and docstrings are exempt -- this reads the AST, not the text --
    and a genuinely justified literal can be kept by putting
    ``# allow-device-literal`` on its line, which turns a silent exception into
    one that shows up in review.
    """
    offences = [
        hit
        for root in _ROOTS
        for path in sorted((_REPO / root).rglob("*.py"))
        if not _is_exempt(path)
        for hit in _offences(path)
    ]

    assert not offences, (
        "device literals outside common/torch_env.py -- route these through "
        "select_device()/autocast_for()/distributed_backend() so both machines "
        "still work:\n  " + "\n  ".join(offences)
    )
