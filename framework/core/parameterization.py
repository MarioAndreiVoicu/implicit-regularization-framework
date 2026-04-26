from __future__ import annotations

from typing import Any

import torch

from framework.core.initialization import initialize_factor


def _mc_imbalance_ratio(level: float) -> float:
    return float(10.0 ** float(level))


_PARAMETERIZATION_REGISTRY: dict[str, dict[str, str]] = {
    "factorized_psd": {
        "label": "Factorized PSD",
        "theory_note": "Implicit-bias baseline X = U U^T.",
        "problem_family": "psd_sensing",
    },
    "symmetric_matrix": {
        "label": "Direct symmetric matrix",
        "theory_note": "Direct symmetric-matrix optimization to isolate factorization effects.",
        "problem_family": "psd_sensing",
    },
    "factorized_uv": {
        "label": "Factorized UV",
        "theory_note": "Rectangular low-rank factorization X = U V^T.",
        "problem_family": "matrix_completion",
    },
    "direct_matrix": {
        "label": "Direct matrix",
        "theory_note": "Direct rectangular matrix optimization to isolate factorization effects.",
        "problem_family": "matrix_completion",
    },
}


def list_parameterizations(problem_family: str | None = None) -> list[str]:
    if problem_family is None:
        return sorted(_PARAMETERIZATION_REGISTRY.keys())
    return sorted([k for k, meta in _PARAMETERIZATION_REGISTRY.items() if meta["problem_family"] == problem_family])


def initialize_psd_parameters(*, seed: int, n: int, factor_rank: int, init_scale: float, init_type: str, parameterization: str, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    if parameterization == "factorized_psd":
        u = initialize_factor(seed=seed, n=n, factor_rank=factor_rank, init_scale=init_scale, init_type=init_type, device=device, dtype=dtype)
        return {"U": u}
    if parameterization == "symmetric_matrix":
        torch.manual_seed(seed)
        s = init_scale * torch.randn((n, n), device=device, dtype=dtype)
        s = 0.5 * (s + s.transpose(0, 1))
        return {"S": s}
    raise ValueError(f"Unhandled parameterization={parameterization!r}.")


def initialize_matrix_completion_parameters(*, seed: int, n_rows: int, n_cols: int, factor_rank: int, init_scale: float, init_type: str, parameterization: str, mc_factor_balance: str, mc_factor_imbalance_level: float, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    if parameterization == "factorized_uv":
        u = initialize_factor(seed=seed, n=n_rows, factor_rank=factor_rank, init_scale=init_scale, init_type=init_type, device=device, dtype=dtype)
        v = initialize_factor(seed=seed + 1, n=n_cols, factor_rank=factor_rank, init_scale=init_scale, init_type=init_type, device=device, dtype=dtype)
        if mc_factor_balance == "unbalanced":
            ratio = _mc_imbalance_ratio(mc_factor_imbalance_level)
            scale_u = ratio ** 0.5
            scale_v = ratio ** -0.5
            u = u * scale_u
            v = v * scale_v
        elif mc_factor_balance != "balanced":
            raise ValueError(f"Unsupported mc_factor_balance={mc_factor_balance!r}.")
        return {"U": u, "V": v}
    if parameterization == "direct_matrix":
        torch.manual_seed(seed)
        x = init_scale * torch.randn((n_rows, n_cols), device=device, dtype=dtype)
        return {"X": x}
    raise ValueError(f"Unhandled parameterization={parameterization!r}.")


def psd_model_x(parameters: dict[str, torch.Tensor], parameterization: str) -> torch.Tensor:
    if parameterization == "factorized_psd":
        u = parameters["U"]
        return u @ u.transpose(0, 1)
    if parameterization == "symmetric_matrix":
        s = parameters["S"]
        return 0.5 * (s + s.transpose(0, 1))
    raise ValueError(f"Unhandled parameterization={parameterization!r}.")


def matrix_completion_model_x(parameters: dict[str, torch.Tensor], parameterization: str) -> torch.Tensor:
    if parameterization == "factorized_uv":
        return parameters["U"] @ parameters["V"].transpose(0, 1)
    if parameterization == "direct_matrix":
        return parameters["X"]
    raise ValueError(f"Unhandled parameterization={parameterization!r}.")


def project_parameters_inplace(parameters: dict[str, torch.Tensor], parameterization: str) -> None:
    if parameterization in {"factorized_psd", "factorized_uv", "direct_matrix"}:
        return
    if parameterization == "symmetric_matrix":
        s = parameters["S"]
        s.copy_(0.5 * (s + s.transpose(0, 1)))
        return
    raise ValueError(f"Unhandled parameterization={parameterization!r}.")


def parameter_artifacts(parameters: dict[str, torch.Tensor], parameterization: str) -> dict[str, Any]:
    if parameterization == "factorized_psd":
        return {"U_final": parameters["U"].detach().cpu().numpy()}
    if parameterization == "symmetric_matrix":
        return {"S_final": parameters["S"].detach().cpu().numpy()}
    if parameterization == "factorized_uv":
        return {
            "U_final": parameters["U"].detach().cpu().numpy(),
            "V_final": parameters["V"].detach().cpu().numpy(),
        }
    if parameterization == "direct_matrix":
        return {"X_final": parameters["X"].detach().cpu().numpy()}
    raise ValueError(f"Unhandled parameterization={parameterization!r}.")
