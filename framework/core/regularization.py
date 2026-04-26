from __future__ import annotations

import torch

from framework.core.parameterization import matrix_completion_model_x, psd_model_x


_EXPLICIT_REGISTRY: dict[str, dict[str, str]] = {
    "factor_l2": {
        "label": "Factor L2",
        "note": "0.5 * lambda * sum of squared factor norms baseline on factor parameters.",
    },
    "frobenius": {
        "label": "Frobenius",
        "note": "0.5 * lambda * ||X||_F^2 baseline on the model matrix X.",
    },
    "none": {
        "label": "None",
        "note": "No explicit regularization baseline.",
    },
}


def list_explicit_reg_types() -> list[str]:
    return sorted(_EXPLICIT_REGISTRY.keys())


def explicit_reg_types_for_parameterization(parameterization: str) -> list[str]:
    if parameterization in {"factorized_psd", "factorized_uv"}:
        return ["none", "factor_l2"]
    if parameterization in {"symmetric_matrix", "direct_matrix"}:
        return ["none", "frobenius"]
    return ["none"]


def explicit_reg_penalty(*, parameters: dict[str, torch.Tensor], parameterization: str, explicit_reg_type: str, explicit_reg_strength: float) -> torch.Tensor:
    first = next(iter(parameters.values()))
    if explicit_reg_type == "none":
        return torch.zeros((), device=first.device, dtype=first.dtype)
    if explicit_reg_type == "factor_l2":
        if parameterization not in {"factorized_psd", "factorized_uv"}:
            raise ValueError("explicit_reg_type='factor_l2' is supported only for factorized parameterizations.")
        penalty = torch.zeros((), device=first.device, dtype=first.dtype)
        for tensor in parameters.values():
            penalty = penalty + torch.sum(tensor * tensor)
        return 0.5 * explicit_reg_strength * penalty
    if explicit_reg_type == "frobenius":
        if parameterization == "symmetric_matrix":
            matrix = psd_model_x(parameters, parameterization)
        elif parameterization == "direct_matrix":
            matrix = matrix_completion_model_x(parameters, parameterization)
        else:
            raise ValueError("explicit_reg_type='frobenius' is supported only for direct matrix parameterizations.")
        penalty = torch.sum(matrix * matrix)
        return 0.5 * explicit_reg_strength * penalty
    supported = ", ".join(list_explicit_reg_types())
    raise ValueError(f"Unsupported explicit_reg_type={explicit_reg_type!r}. Supported: {supported}.")
