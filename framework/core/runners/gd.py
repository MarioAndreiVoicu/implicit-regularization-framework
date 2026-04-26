from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from framework.core.models import RunResult, RunSpec
from framework.core.runners.psd import run_psd_optimizer


def run_gd(
    spec: RunSpec,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> RunResult:
    if spec.optimizer != "gd":
        raise ValueError(f"Expected optimizer='gd', got {spec.optimizer!r}.")
    return run_psd_optimizer(spec, device=device, dtype=dtype, progress_callback=progress_callback)
