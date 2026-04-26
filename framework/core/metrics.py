from __future__ import annotations

import math

import torch


@torch.no_grad()
def rel_frob_err(x: torch.Tensor, x_star: torch.Tensor) -> float:
    num = torch.linalg.norm(x - x_star, ord="fro")
    den = torch.linalg.norm(x_star, ord="fro") + 1e-12
    return float((num / den).item())


@torch.no_grad()
def frobenius_norm(x: torch.Tensor) -> float:
    return float(torch.linalg.norm(x, ord="fro").item())


@torch.no_grad()
def trace_value(x: torch.Tensor) -> float:
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        return float("nan")
    return float(torch.trace(x).item())


@torch.no_grad()
def effective_rank_from_singular_values(singular_values: torch.Tensor) -> float:
    s = torch.clamp(singular_values, min=0.0)
    total = torch.sum(s)
    if float(total.item()) <= 1e-12:
        return 0.0
    p = s / total
    mask = p > 0
    entropy = -torch.sum(p[mask] * torch.log(p[mask]))
    return float(torch.exp(entropy).item())


@torch.no_grad()
def stable_rank_from_singular_values(singular_values: torch.Tensor) -> float:
    s = torch.clamp(singular_values, min=0.0)
    if s.numel() == 0:
        return 0.0
    top = float(torch.max(s).item())
    if top <= 1e-12:
        return 0.0
    fro_sq = float(torch.sum(s**2).item())
    return fro_sq / (top * top)


@torch.no_grad()
def spectral_metrics_symmetric_matrix(x: torch.Tensor, num_eigs: int) -> dict[str, float | list[float]]:
    eigvals = torch.linalg.eigvalsh(x)
    eigvals_desc = torch.flip(eigvals, dims=[0])
    singular_values = torch.sort(torch.abs(eigvals), descending=True).values

    top_eigs = eigvals_desc[:num_eigs]
    top_svs = singular_values[:num_eigs]
    if top_eigs.numel() < num_eigs:
        top_eigs = torch.cat([top_eigs, torch.zeros(num_eigs - top_eigs.numel(), device=x.device, dtype=x.dtype)])
    if top_svs.numel() < num_eigs:
        top_svs = torch.cat([top_svs, torch.zeros(num_eigs - top_svs.numel(), device=x.device, dtype=x.dtype)])

    negative_eigs = eigvals[eigvals < 0]
    negative_eigen_mass = float(torch.sum(torch.abs(negative_eigs)).item()) if negative_eigs.numel() else 0.0

    return {
        "frobenius_norm_X": frobenius_norm(x),
        "trace_X": trace_value(x),
        "effective_rank": effective_rank_from_singular_values(singular_values),
        "stable_rank": stable_rank_from_singular_values(singular_values),
        "min_eigenvalue_X": float(torch.min(eigvals).item()),
        "negative_eigen_mass": negative_eigen_mass,
        "top_k_eigenvalues": [float(v.item()) for v in top_eigs],
        "top_k_singular_values": [float(v.item()) for v in top_svs],
    }


@torch.no_grad()
def spectral_metrics_matrix(x: torch.Tensor, num_svs: int) -> dict[str, float | list[float]]:
    singular_values = torch.linalg.svdvals(x)
    singular_values = torch.sort(singular_values, descending=True).values
    top_svs = singular_values[:num_svs]
    if top_svs.numel() < num_svs:
        top_svs = torch.cat([top_svs, torch.zeros(num_svs - top_svs.numel(), device=x.device, dtype=x.dtype)])
    nan_list = [float("nan")] * num_svs
    return {
        "frobenius_norm_X": frobenius_norm(x),
        "trace_X": trace_value(x),
        "effective_rank": effective_rank_from_singular_values(singular_values),
        "stable_rank": stable_rank_from_singular_values(singular_values),
        "min_eigenvalue_X": float("nan"),
        "negative_eigen_mass": float("nan"),
        "top_k_eigenvalues": nan_list,
        "top_k_singular_values": [float(v.item()) for v in top_svs],
    }


@torch.no_grad()
def _projection_distance_from_bases(q_hat: torch.Tensor, q_star: torch.Tensor) -> float:
    if q_hat.numel() == 0 or q_star.numel() == 0:
        return 0.0
    r = min(q_hat.shape[1], q_star.shape[1])
    if r == 0:
        return 0.0
    p_hat = q_hat @ q_hat.transpose(0, 1)
    p_star = q_star @ q_star.transpose(0, 1)
    return float((torch.linalg.norm(p_hat - p_star, ord="fro") / math.sqrt(2.0 * r)).item())


@torch.no_grad()
def _alignment_from_bases(q_hat: torch.Tensor, q_star: torch.Tensor) -> float:
    if q_hat.numel() == 0 or q_star.numel() == 0:
        return 0.0
    r = min(q_hat.shape[1], q_star.shape[1])
    if r == 0:
        return 0.0
    gram = q_hat.transpose(0, 1) @ q_star
    return float((torch.sum(gram**2) / r).item())


@torch.no_grad()
def top_eigenspace_basis_symmetric(x: torch.Tensor, rank: int) -> torch.Tensor:
    if rank <= 0:
        return torch.empty((x.shape[0], 0), device=x.device, dtype=x.dtype)
    x_sym = 0.5 * (x + x.transpose(0, 1))
    _, eigvecs = torch.linalg.eigh(x_sym)
    return eigvecs[:, -rank:]


@torch.no_grad()
def top_singular_bases(x: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    if rank <= 0:
        return (
            torch.empty((x.shape[0], 0), device=x.device, dtype=x.dtype),
            torch.empty((x.shape[1], 0), device=x.device, dtype=x.dtype),
        )
    u, _, vh = torch.linalg.svd(x, full_matrices=False)
    return u[:, :rank], vh.transpose(0, 1)[:, :rank]


@torch.no_grad()
def subspace_alignment_metrics_symmetric(x: torch.Tensor, x_star: torch.Tensor, rank: int) -> dict[str, float]:
    q_hat = top_eigenspace_basis_symmetric(x, rank)
    q_star = top_eigenspace_basis_symmetric(x_star, rank)
    return {
        "subspace_alignment": _alignment_from_bases(q_hat, q_star),
        "subspace_projection_distance": _projection_distance_from_bases(q_hat, q_star),
    }


@torch.no_grad()
def subspace_alignment_metrics_matrix(x: torch.Tensor, x_star: torch.Tensor, rank: int) -> dict[str, float]:
    u_hat, v_hat = top_singular_bases(x, rank)
    u_star, v_star = top_singular_bases(x_star, rank)
    left_alignment = _alignment_from_bases(u_hat, u_star)
    right_alignment = _alignment_from_bases(v_hat, v_star)
    left_distance = _projection_distance_from_bases(u_hat, u_star)
    right_distance = _projection_distance_from_bases(v_hat, v_star)
    return {
        "left_subspace_alignment": left_alignment,
        "right_subspace_alignment": right_alignment,
        "mean_subspace_alignment": 0.5 * (left_alignment + right_alignment),
        "left_subspace_projection_distance": left_distance,
        "right_subspace_projection_distance": right_distance,
        "mean_subspace_projection_distance": 0.5 * (left_distance + right_distance),
    }
