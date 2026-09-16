"""Torch device and determinism, the way ``common/spark.py`` owns the session.

One place decides where tensors live, so no module reaches for
``torch.device("cuda")`` and no test quietly runs somewhere else than training
did.

**This project is built on two machines**: an Apple-silicon Mac (MPS) and a
Windows box with an RTX 4090 under WSL2 (CUDA). Retrieval sysmem is built for GPUs
--``torch.autocast("cuda", bfloat16)`` and ``GradScaler`` throughout -- and
transplanting that verbatim onto MPS does not fail loudly, it fails subtly:
``autocast("cuda")`` on MPS raises only once a tensor reaches it, and a
``GradScaler`` off CUDA is a silent no-op in some torch versions and an error
in others.

So every device-dependent choice is made HERE, from the device, and nothing
under ``models/`` names a device at all. That is enforced rather than
remembered: ``tests/test_torch_env.py`` fails the build on a ``"cuda"`` or
``"mps"`` literal anywhere outside this module.
"""

from __future__ import annotations

import os
import random
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

import numpy as np
import torch

# MPS falls back to CPU for kernels it lacks rather than raising. Without this
# an unimplemented op stops training; with it, training slows down and finishes.
# Set before the first MPS allocation or it has no effect.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def select_device(prefer: str | None = None) -> torch.device:
    """The device this project trains and evaluates on.

    Args:
        prefer: Force a device, e.g. ``"cpu"`` for a test that must not depend
            on accelerator availability.

    Returns:
        ``cuda`` if present, else ``mps``, else ``cpu``.
    """
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_for(device: torch.device) -> torch.amp.autocast_mode.autocast | nullcontext[None]:
    """Mixed precision where it helps, and nothing where it does not.

    bfloat16 autocast is roughly a 2x throughput win on CUDA. On MPS the gain is
    small and the numerics are less well trodden; on CPU bfloat16 is usually
    slower than fp32. Returning a real no-op context rather than a disabled
    autocast keeps the training loop free of ``if device.type ==`` branches.
    """
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def grad_scaler_for(device: torch.device) -> torch.amp.GradScaler:
    """A GradScaler that is disabled on every device, deliberately.

    Loss scaling exists to stop fp16 gradients underflowing. This project never
    trains in fp16: :func:`autocast_for` selects **bf16** on CUDA, which carries
    fp32's exponent range, and no autocast at all on MPS and CPU. There is
    nothing to scale anywhere.

    It still returns a scaler rather than ``None`` so the training loop can call
    ``scaler.scale(...)``/``step``/``update`` unconditionally and keep one code
    path instead of two. A disabled scaler makes all three pass-throughs.

    Note:
        This read ``enabled=device.type == "cuda"`` until 2026-09-15, which
        enabled scaling on precisely the device where autocast selects bf16 --
        the one place it is both unnecessary and not free, since
        ``scaler.step`` runs an inf/nan check and can skip an optimiser step.
        The condition was inverted relative to the paragraph above it. It never
        fired on the Mac, so it stayed invisible until this ran on CUDA.
    """
    return torch.amp.GradScaler(device.type, enabled=False)


def distributed_backend(device: torch.device) -> str:
    """The ``torch.distributed`` backend this device can actually use.

    NCCL is CUDA-only; MPS and CPU get gloo. Naming the backend rather than
    defaulting matters because the default is chosen from what is *compiled in*,
    not from what the tensors are on, so a Mac picks up a backend that then
    fails at the first collective rather than at initialisation.

    Note:
        At ``world_size == 1`` the all-gather in G5's loss is a no-op on either
        backend. The reason to initialise a process group on a single device at
        all is to exercise the gradient-flowing gather -- a plain
        ``dist.all_gather`` detaches, silently training on 1/N of the negatives,
        and a code path that is never entered cannot be tested.
    """
    return "nccl" if device.type == "cuda" else "gloo"


def dataloader_kwargs(device: torch.device) -> dict[str, Any]:
    """DataLoader options that depend on the device rather than on the task.

    ``pin_memory`` allocates page-locked host memory so the host-to-device copy
    can overlap compute. That is a CUDA concept: MPS shares memory with the
    host, so pinning buys nothing there and several torch versions warn about it
    on every epoch.

    Worker count and prefetch depth are deliberately NOT here -- they are task
    and dataset decisions, and putting them behind a device lookup would make
    them look like physics.
    """
    return {"pin_memory": device.type == "cuda"}


def set_seed(seed: int = 0) -> None:
    """Seed Python, NumPy and torch together.

    Seeding torch alone is the usual mistake: the DataLoader's shuffling and any
    negative sampling go through Python's and NumPy's generators, so a run
    seeded only in torch still differs between invocations in exactly the place
    an ablation needs it not to.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def deterministic(seed: int = 0) -> Iterator[None]:
    """Seeded and, where torch allows it, algorithmically deterministic.

    ``use_deterministic_algorithms`` makes some kernels raise rather than pick a
    nondeterministic implementation, which is what makes a determinism TEST able
    to fail. It is deliberately NOT on during training: several of the ops a
    two-tower needs have no deterministic kernel, and the cost is real.
    """
    set_seed(seed)
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=True)


def describe(device: torch.device) -> str:
    """One line for the training log and the MLflow run, so a metric can always
    be traced back to what produced it."""
    parts = [f"torch {torch.__version__}", f"device {device.type}"]
    if device.type == "cuda":
        parts.append(torch.cuda.get_device_name(0))
        parts.append(f"{torch.cuda.device_count()} visible")
    parts.append("autocast bf16" if device.type == "cuda" else "autocast off")
    return " · ".join(parts)
