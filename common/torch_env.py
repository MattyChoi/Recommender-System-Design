"""Torch device and determinism, the way ``common/spark.py`` owns the session.

One place decides where tensors live, so no module reaches for
``torch.device("cuda")`` and no test quietly runs somewhere else than training
did.

**This machine has no CUDA.** The guide's Part G is written for GPUs --
``torch.autocast("cuda", bfloat16)`` and ``GradScaler`` throughout -- and
transplanting that verbatim onto an Apple-silicon Mac does not fail loudly, it
fails subtly: ``GradScaler`` on a non-CUDA device is a silent no-op in some
torch versions and an error in others, and ``autocast("cuda")`` on MPS raises
only once a tensor reaches it. So both are chosen here from the device rather
than assumed, and the CPU/MPS path takes neither.
"""

from __future__ import annotations

import os
import random
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext

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
    """A GradScaler that is enabled only where it means anything.

    Loss scaling exists to stop fp16 gradients underflowing. bf16 has fp32's
    exponent range and does not need it, and neither does fp32 -- but the loop
    still calls ``scaler.scale(...)``/``step``/``update`` unconditionally, so a
    disabled scaler keeps one code path instead of two.
    """
    return torch.amp.GradScaler(device.type, enabled=device.type == "cuda")


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
