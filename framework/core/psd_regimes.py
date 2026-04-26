from __future__ import annotations

import torch


COMMUTING = "commuting"
NON_COMMUTING = "non_commuting"


def list_psd_regimes() -> list[str]:
    return [NON_COMMUTING, COMMUTING]


def normalize_symmetric_stack_fro(a: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    flat = a.reshape(a.shape[0], -1)
    norms = torch.linalg.norm(flat, dim=1, keepdim=True).clamp_min(eps)
    return a / norms.view(a.shape[0], 1, 1)


def normalized_target_commutators(a: torch.Tensor, x_star: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    comm = a @ x_star - x_star @ a
    numer = torch.linalg.norm(comm.reshape(comm.shape[0], -1), dim=1)
    a_norm = torch.linalg.norm(a.reshape(a.shape[0], -1), dim=1)
    x_norm = torch.linalg.norm(x_star)
    denom = (a_norm * x_norm).clamp_min(eps)
    return numer / denom


def summarize_target_commutators(a: torch.Tensor, x_star: torch.Tensor) -> dict[str, float]:
    vals = normalized_target_commutators(a, x_star)
    return {
        "mean_target_commutator": float(vals.mean().item()),
        "max_target_commutator": float(vals.max().item()),
    }
