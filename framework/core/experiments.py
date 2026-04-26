from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from framework.core.device import resolve_device_preference, resolve_dtype_preference
from framework.core.models import RunResult, RunSpec
from framework.core.optimizers import list_optimizers as list_registered_optimizers
from framework.core.runners.gd import run_gd
from framework.core.runners.matrix_completion import run_matrix_completion_optimizer
from framework.core.runners.psd import run_psd_optimizer

Runner = Callable[..., RunResult]


_PROBLEM_REGISTRY: dict[str, dict[str, str]] = {
    "psd_sensing": {"label": "PSD sensing"},
    "matrix_completion": {"label": "Matrix completion"},
}

_OPTIMIZER_REGISTRY: dict[str, dict[str, str]] = {
    name: {"label": name.replace("_", " ").title()} for name in list_registered_optimizers()
}

_RUNNER_REGISTRY: dict[tuple[str, str], Runner] = {
    ("psd_sensing", "gd"): run_gd,
    ("psd_sensing", "adam"): run_psd_optimizer,
    ("psd_sensing", "sgd"): run_psd_optimizer,
    ("psd_sensing", "momentum"): run_psd_optimizer,
    ("psd_sensing", "coordinate_descent"): run_psd_optimizer,
    ("psd_sensing", "preconditioned_gd"): run_psd_optimizer,
    ("matrix_completion", "gd"): run_matrix_completion_optimizer,
    ("matrix_completion", "adam"): run_matrix_completion_optimizer,
    ("matrix_completion", "sgd"): run_matrix_completion_optimizer,
    ("matrix_completion", "momentum"): run_matrix_completion_optimizer,
    ("matrix_completion", "coordinate_descent"): run_matrix_completion_optimizer,
    ("matrix_completion", "preconditioned_gd"): run_matrix_completion_optimizer,
}


def list_problem_families() -> list[str]:
    return sorted(_PROBLEM_REGISTRY.keys())


def list_optimizers(problem_family: str | None = None) -> list[str]:
    if problem_family is None:
        return sorted(_OPTIMIZER_REGISTRY.keys())
    return sorted([opt for pf, opt in _RUNNER_REGISTRY.keys() if pf == problem_family])


def list_supported_runner_keys() -> list[tuple[str, str]]:
    return sorted(_RUNNER_REGISTRY.keys())


def get_runner(spec: RunSpec) -> Runner:
    key = (str(spec.problem_family), str(spec.optimizer))
    runner = _RUNNER_REGISTRY.get(key)
    if runner is None:
        supported = ", ".join(f"{problem}/{optimizer}" for problem, optimizer in list_supported_runner_keys())
        raise ValueError(
            "Unsupported experiment combination: "
            f"problem_family={spec.problem_family!r}, optimizer={spec.optimizer!r}. "
            f"Supported combinations: {supported}."
        )
    return runner


def run_experiment(
    spec: RunSpec,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> RunResult:
    spec.validate()
    runner = get_runner(spec)
    resolved_device = device
    resolved_device_name = None
    if resolved_device is None:
        resolved_device, resolved_device_name = resolve_device_preference(spec.device_preference)
    else:
        resolved_device_name = str(resolved_device)
    resolved_dtype = dtype
    if resolved_dtype is None:
        resolved_dtype, _ = resolve_dtype_preference(spec.dtype_preference, spec.execution_mode, resolved_device)
    result = runner(spec, device=resolved_device, dtype=resolved_dtype, progress_callback=progress_callback)
    result.summary.setdefault("resolved_device", resolved_device_name)
    result.summary.setdefault("resolved_dtype", str(resolved_dtype).replace("torch.", ""))
    result.summary.setdefault("execution_mode", spec.execution_mode)
    return result
