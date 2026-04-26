from __future__ import annotations

import torch


# IMPORTANT:
# For this app, "auto" is intentionally conservative.
# Most default experiments are small enough that CUDA/MPS startup and CPU<->GPU
# synchronization overhead can dominate the actual computation. In practice that
# made auto-selected GPUs much slower than CPU for ordinary UI runs.
# Users who want an accelerator can still choose cuda/mps explicitly.


def _mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


def list_device_preferences() -> list[str]:
    return ["auto", "cpu", "cuda", "mps"]


def list_dtype_preferences() -> list[str]:
    return ["auto", "float64", "float32"]


def list_execution_modes() -> list[str]:
    return ["accuracy", "balanced", "speed"]


def resolve_device_preference(device_preference: str | None) -> tuple[torch.device, str]:
    pref = str(device_preference or "auto")
    if pref == "auto":
        # Conservative auto: prefer CPU unless the user explicitly requests an
        # accelerator. This avoids very slow GPU/MPS startup/synchronization for
        # small local experiments, which are the common interactive case.
        return torch.device("cpu"), "cpu"
    if pref == "cpu":
        return torch.device("cpu"), "cpu"
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("Requested device='cuda' but CUDA is not available.")
        return torch.device("cuda"), "cuda"
    if pref == "mps":
        if not _mps_available():
            raise ValueError("Requested device='mps' but MPS is not available.")
        return torch.device("mps"), "mps"
    raise ValueError(f"Unsupported device_preference={pref!r}")


def resolve_dtype_preference(
    dtype_preference: str | None,
    execution_mode: str | None,
    resolved_device: torch.device,
) -> tuple[torch.dtype, str]:
    pref = str(dtype_preference or "auto")
    mode = str(execution_mode or "accuracy")
    if pref == "float64":
        return torch.float64, "float64"
    if pref == "float32":
        return torch.float32, "float32"
    if pref != "auto":
        raise ValueError(f"Unsupported dtype_preference={pref!r}")
    if mode == "accuracy":
        return torch.float64, "float64"
    if mode == "speed":
        return torch.float32, "float32"
    if mode != "balanced":
        raise ValueError(f"Unsupported execution_mode={mode!r}")
    if resolved_device.type in {"cuda", "mps"}:
        return torch.float32, "float32"
    return torch.float64, "float64"
