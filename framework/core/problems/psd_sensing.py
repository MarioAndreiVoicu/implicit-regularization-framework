from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from framework.core.psd_regimes import (
    COMMUTING,
    NON_COMMUTING,
    list_psd_regimes,
    normalize_symmetric_stack_fro,
    summarize_target_commutators,
)


@dataclass
class Problem:
    X_star: torch.Tensor
    A: torch.Tensor
    y: torch.Tensor
    U_star: torch.Tensor
    mean_target_commutator: float
    max_target_commutator: float


def _make_symmetric_gaussian_a(
    m: int,
    n: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    g = torch.randn((m, n, n), device=device, dtype=dtype)
    return 0.5 * (g + g.transpose(-1, -2))



def _sample_orthogonal_basis(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn((n, n), device=device, dtype=dtype))
    return q



def _make_base_commuting_target_and_operators(
    m: int,
    n: int,
    r_star: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = _sample_orthogonal_basis(n=n, device=device, dtype=dtype)
    eigvals = torch.zeros(n, device=device, dtype=dtype)
    eigvals[:r_star] = torch.rand(r_star, device=device, dtype=dtype).abs() + 0.5
    x_star = q @ torch.diag(eigvals) @ q.transpose(0, 1)
    diag_coeffs = torch.randn((m, n), device=device, dtype=dtype)
    a_commuting = torch.einsum("ik,mk,jk->mij", q, diag_coeffs, q)
    u_star = q[:, :r_star] @ torch.diag(torch.sqrt(eigvals[:r_star]))
    return x_star, a_commuting, u_star



def make_psd_sensing_problem(
    seed: int,
    n: int,
    r_star: int,
    m: int,
    noise_std: float = 0.0,
    measurement_type: str = "gaussian_symmetric",
    psd_regime: str = NON_COMMUTING,
    noncommutativity_level: float = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> Problem:
    if device is None:
        device = torch.device("cpu")

    if measurement_type != "gaussian_symmetric":
        raise ValueError(
            f"Unsupported measurement_type={measurement_type}. "
            "Expected 'gaussian_symmetric'."
        )
    if psd_regime not in list_psd_regimes():
        raise ValueError(f"Unsupported psd_regime={psd_regime!r}.")
    if not (0.0 <= float(noncommutativity_level) <= 1.0):
        raise ValueError("noncommutativity_level must be in [0, 1].")

    torch.manual_seed(seed)
    np.random.seed(seed)

    x_star, a_commuting, u_star = _make_base_commuting_target_and_operators(
        m=m,
        n=n,
        r_star=r_star,
        device=device,
        dtype=dtype,
    )

    a_commuting = normalize_symmetric_stack_fro(a_commuting)

    if psd_regime == COMMUTING:
        level = 0.0
        a = a_commuting
    else:
        level = float(noncommutativity_level)
        a_random = normalize_symmetric_stack_fro(
            _make_symmetric_gaussian_a(m=m, n=n, device=device, dtype=dtype)
        )
        a = (1.0 - level) * a_commuting + level * a_random
        a = 0.5 * (a + a.transpose(-1, -2))
        a = normalize_symmetric_stack_fro(a)

    y = torch.einsum("mij,ij->m", a, x_star)

    if noise_std > 0:
        y = y + noise_std * torch.randn_like(y)

    comm_summary = summarize_target_commutators(a, x_star)
    if psd_regime == COMMUTING:
        comm_summary = {"mean_target_commutator": 0.0, "max_target_commutator": 0.0}
    return Problem(
        X_star=x_star,
        A=a,
        y=y,
        U_star=u_star,
        mean_target_commutator=comm_summary["mean_target_commutator"],
        max_target_commutator=comm_summary["max_target_commutator"],
    )
