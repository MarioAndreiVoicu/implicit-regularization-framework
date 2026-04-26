from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from framework.core.datasets.custom_matrix_completion import make_custom_triplets_csv_split
from framework.core.datasets.movielens import make_movielens_100k_split


@dataclass
class MatrixCompletionDataBundle:
    data_source: str
    source_kind: str
    n_rows: int
    n_cols: int
    observed_row_idx: torch.Tensor
    observed_col_idx: torch.Tensor
    observed_y: torch.Tensor
    train_row_idx: torch.Tensor
    train_col_idx: torch.Tensor
    train_y: torch.Tensor
    val_row_idx: torch.Tensor
    val_col_idx: torch.Tensor
    val_y: torch.Tensor
    test_row_idx: torch.Tensor
    test_col_idx: torch.Tensor
    test_y: torch.Tensor
    observed_mask: torch.Tensor | None = None
    X_star: torch.Tensor | None = None
    U_star: torch.Tensor | None = None
    V_star: torch.Tensor | None = None
    rating_min: float | None = None
    rating_max: float | None = None
    target_row_coherence: float | None = None
    target_col_coherence: float | None = None
    target_condition_number: float | None = None
    target_effective_rank: float | None = None
    target_stable_rank: float | None = None
    target_singular_values: torch.Tensor | None = None
    coherence_level: float | None = None
    sampling_regime: str | None = None
    mc_spectrum_type: str | None = None
    mc_spectrum_decay: float | None = None
    dataset_version: str | None = None
    split_version: str | None = None
    preprocessing_version: str | None = None
    custom_dataset_path: str | None = None
    split_strategy: str | None = None
    has_ground_truth: bool = False

    @property
    def has_validation_split(self) -> bool:
        return bool(self.val_y.numel())

    @property
    def has_test_split(self) -> bool:
        return bool(self.test_y.numel())


@dataclass
class MatrixCompletionProblem:
    data_source: str
    source_kind: str
    n_rows: int
    n_cols: int
    train_row_idx: torch.Tensor
    train_col_idx: torch.Tensor
    train_y: torch.Tensor
    val_row_idx: torch.Tensor
    val_col_idx: torch.Tensor
    val_y: torch.Tensor
    test_row_idx: torch.Tensor
    test_col_idx: torch.Tensor
    test_y: torch.Tensor
    X_star: torch.Tensor | None = None
    U_star: torch.Tensor | None = None
    V_star: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    rating_min: float | None = None
    rating_max: float | None = None
    target_row_coherence: float | None = None
    target_col_coherence: float | None = None
    target_condition_number: float | None = None
    target_effective_rank: float | None = None
    target_stable_rank: float | None = None
    target_singular_values: torch.Tensor | None = None
    coherence_level: float | None = None
    sampling_regime: str | None = None
    mc_spectrum_type: str | None = None
    mc_spectrum_decay: float | None = None
    dataset_version: str | None = None
    split_version: str | None = None
    preprocessing_version: str | None = None
    custom_dataset_path: str | None = None
    split_strategy: str | None = None
    has_ground_truth: bool = False
    data_bundle: MatrixCompletionDataBundle | None = None


def _mixed_orthonormal_basis(n: int, r: int, level: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    base_random = torch.randn((n, r), device=device, dtype=dtype)
    q_random, _ = torch.linalg.qr(base_random, mode="reduced")
    perm = torch.randperm(n, device=device)[:r]
    q_spiky = torch.eye(n, device=device, dtype=dtype)[:, perm]
    mixed = (1.0 - level) * q_random + level * q_spiky
    q_mixed, _ = torch.linalg.qr(mixed, mode="reduced")
    return q_mixed[:, :r]


def _target_singular_values(*, r_star: int, mc_spectrum_type: str, mc_spectrum_decay: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    idx = torch.arange(1, r_star + 1, device=device, dtype=dtype)
    if mc_spectrum_type == "flat":
        s = torch.ones((r_star,), device=device, dtype=dtype)
    elif mc_spectrum_type == "power_decay":
        s = idx.pow(-mc_spectrum_decay)
    elif mc_spectrum_type == "exponential_decay":
        s = torch.exp(-mc_spectrum_decay * (idx - 1.0))
    else:
        raise ValueError(f"Unsupported mc_spectrum_type={mc_spectrum_type!r}.")
    return s / torch.mean(s)


def _spectral_target_diagnostics(singular_values: torch.Tensor) -> tuple[float, float, float]:
    p = singular_values / torch.sum(singular_values)
    entropy = -(p * torch.log(torch.clamp(p, min=1e-30))).sum()
    effective_rank = float(torch.exp(entropy).item())
    stable_rank = float((torch.sum(singular_values**2) / torch.clamp(torch.max(singular_values)**2, min=1e-30)).item())
    cond = float((torch.max(singular_values) / torch.clamp(torch.min(singular_values), min=1e-30)).item())
    return cond, effective_rank, stable_rank


def _synthetic_target(
    *,
    n_rows: int,
    n_cols: int,
    r_star: int,
    coherence_level: float,
    mc_spectrum_type: str,
    mc_spectrum_decay: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float, torch.Tensor, float, float, float]:
    u_basis = _mixed_orthonormal_basis(n_rows, r_star, coherence_level, device, dtype)
    v_basis = _mixed_orthonormal_basis(n_cols, r_star, coherence_level, device, dtype)
    singular_values = _target_singular_values(r_star=r_star, mc_spectrum_type=mc_spectrum_type, mc_spectrum_decay=mc_spectrum_decay, device=device, dtype=dtype)
    sqrt_s = torch.sqrt(singular_values).unsqueeze(0)
    u_star = u_basis * sqrt_s
    v_star = v_basis * sqrt_s
    x_star = u_star @ v_star.transpose(0, 1)

    row_lev = torch.sum(u_basis**2, dim=1)
    col_lev = torch.sum(v_basis**2, dim=1)
    row_coh = float((n_rows / r_star) * torch.max(row_lev).item())
    col_coh = float((n_cols / r_star) * torch.max(col_lev).item())
    cond, eff_rank, stable_rank = _spectral_target_diagnostics(singular_values)
    return x_star, u_star, v_star, row_coh, col_coh, singular_values, cond, eff_rank, stable_rank


def _sample_entry_indices(
    *,
    n_rows: int,
    n_cols: int,
    m: int,
    sampling_regime: str,
    u_star: torch.Tensor,
    v_star: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    total = n_rows * n_cols
    if m > total:
        raise ValueError(f"m={m} exceeds total entries {total}.")

    if sampling_regime == "uniform":
        perm = torch.randperm(total, device=device)[:m]
    elif sampling_regime == "leverage_biased":
        row_scores = torch.sum(u_star**2, dim=1)
        col_scores = torch.sum(v_star**2, dim=1)
        probs = row_scores[:, None] + col_scores[None, :]
        probs = probs.reshape(-1)
        probs = probs / torch.sum(probs)
        perm = torch.multinomial(probs, num_samples=m, replacement=False)
    else:
        raise ValueError(f"Unsupported sampling_regime={sampling_regime!r}.")

    row_idx = torch.div(perm, n_cols, rounding_mode="floor")
    col_idx = perm % n_cols
    return row_idx, col_idx


def _mask_from_indices(*, n_rows: int, n_cols: int, row_idx: torch.Tensor, col_idx: torch.Tensor, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((n_rows, n_cols), device=device, dtype=torch.bool)
    if row_idx.numel() and col_idx.numel():
        mask[row_idx, col_idx] = True
    return mask


def _concat_split_observations(*, device: torch.device, dtype: torch.dtype, splits: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    row_parts = [rows for rows, _, _ in splits if rows.numel()]
    col_parts = [cols for _, cols, _ in splits if cols.numel()]
    y_parts = [y for _, _, y in splits if y.numel()]
    if not row_parts:
        return (
            torch.empty((0,), device=device, dtype=torch.long),
            torch.empty((0,), device=device, dtype=torch.long),
            torch.empty((0,), device=device, dtype=dtype),
        )
    return torch.cat(row_parts), torch.cat(col_parts), torch.cat(y_parts)


def _bundle_from_movielens_split(
    *,
    seed: int,
    val_fraction: float,
    test_fraction: float,
    dataset_version: str | None,
    split_version: str | None,
    preprocessing_version: str | None,
    device: torch.device,
    dtype: torch.dtype,
) -> MatrixCompletionDataBundle:
    split = make_movielens_100k_split(
        seed=seed,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        device=device,
        dtype=dtype,
    )
    observed_row_idx, observed_col_idx, observed_y = _concat_split_observations(
        device=device,
        dtype=dtype,
        splits=[
            (split.train_row_idx, split.train_col_idx, split.train_y),
            (split.val_row_idx, split.val_col_idx, split.val_y),
            (split.test_row_idx, split.test_col_idx, split.test_y),
        ],
    )
    observed_mask = _mask_from_indices(
        n_rows=split.n_rows,
        n_cols=split.n_cols,
        row_idx=observed_row_idx,
        col_idx=observed_col_idx,
        device=device,
    )
    return MatrixCompletionDataBundle(
        data_source="movielens_100k",
        source_kind="real",
        n_rows=split.n_rows,
        n_cols=split.n_cols,
        observed_row_idx=observed_row_idx,
        observed_col_idx=observed_col_idx,
        observed_y=observed_y,
        train_row_idx=split.train_row_idx,
        train_col_idx=split.train_col_idx,
        train_y=split.train_y,
        val_row_idx=split.val_row_idx,
        val_col_idx=split.val_col_idx,
        val_y=split.val_y,
        test_row_idx=split.test_row_idx,
        test_col_idx=split.test_col_idx,
        test_y=split.test_y,
        observed_mask=observed_mask,
        rating_min=split.rating_min,
        rating_max=split.rating_max,
        sampling_regime="observed",
        dataset_version=dataset_version,
        split_version=split_version,
        preprocessing_version=preprocessing_version,
        split_strategy="heldout_observed_entries",
        has_ground_truth=False,
    )




def _bundle_from_custom_triplets_csv(
    *,
    path: str,
    seed: int,
    val_fraction: float,
    test_fraction: float,
    dataset_version: str | None,
    split_version: str | None,
    preprocessing_version: str | None,
    device: torch.device,
    dtype: torch.dtype,
) -> MatrixCompletionDataBundle:
    split = make_custom_triplets_csv_split(
        path=path,
        seed=seed,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        device=device,
        dtype=dtype,
    )
    observed_row_idx, observed_col_idx, observed_y = _concat_split_observations(
        device=device,
        dtype=dtype,
        splits=[
            (split.train_row_idx, split.train_col_idx, split.train_y),
            (split.val_row_idx, split.val_col_idx, split.val_y),
            (split.test_row_idx, split.test_col_idx, split.test_y),
        ],
    )
    observed_mask = _mask_from_indices(
        n_rows=split.n_rows,
        n_cols=split.n_cols,
        row_idx=observed_row_idx,
        col_idx=observed_col_idx,
        device=device,
    )
    return MatrixCompletionDataBundle(
        data_source="custom_triplets_csv",
        source_kind="real",
        n_rows=split.n_rows,
        n_cols=split.n_cols,
        observed_row_idx=observed_row_idx,
        observed_col_idx=observed_col_idx,
        observed_y=observed_y,
        train_row_idx=split.train_row_idx,
        train_col_idx=split.train_col_idx,
        train_y=split.train_y,
        val_row_idx=split.val_row_idx,
        val_col_idx=split.val_col_idx,
        val_y=split.val_y,
        test_row_idx=split.test_row_idx,
        test_col_idx=split.test_col_idx,
        test_y=split.test_y,
        observed_mask=observed_mask,
        rating_min=split.rating_min,
        rating_max=split.rating_max,
        sampling_regime="observed",
        dataset_version=dataset_version,
        split_version=split_version,
        preprocessing_version=preprocessing_version,
        custom_dataset_path=split.source_path,
        split_strategy="heldout_observed_entries",
        has_ground_truth=False,
    )


def _bundle_from_synthetic_target(
    *,
    seed: int,
    n_rows: int,
    n_cols: int,
    r_star: int,
    m: int,
    noise_std: float,
    measurement_type: str,
    coherence_level: float,
    sampling_regime: str,
    mc_spectrum_type: str,
    mc_spectrum_decay: float,
    dataset_version: str | None,
    split_version: str | None,
    preprocessing_version: str | None,
    device: torch.device,
    dtype: torch.dtype,
) -> MatrixCompletionDataBundle:
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_star, u_star, v_star, row_coh, col_coh, target_singular_values, target_condition_number, target_effective_rank, target_stable_rank = _synthetic_target(
        n_rows=n_rows,
        n_cols=n_cols,
        r_star=r_star,
        coherence_level=coherence_level,
        mc_spectrum_type=mc_spectrum_type,
        mc_spectrum_decay=mc_spectrum_decay,
        device=device,
        dtype=dtype,
    )

    effective_sampling = sampling_regime
    if measurement_type == "uniform_entries":
        effective_sampling = "uniform"
    elif measurement_type == "leverage_biased_entries":
        effective_sampling = "leverage_biased"

    row_idx, col_idx = _sample_entry_indices(
        n_rows=n_rows,
        n_cols=n_cols,
        m=m,
        sampling_regime=effective_sampling,
        u_star=u_star,
        v_star=v_star,
        device=device,
    )
    y = x_star[row_idx, col_idx]
    if noise_std > 0:
        y = y + noise_std * torch.randn_like(y)

    observed_mask = _mask_from_indices(
        n_rows=n_rows,
        n_cols=n_cols,
        row_idx=row_idx,
        col_idx=col_idx,
        device=device,
    )
    empty_idx = torch.empty((0,), device=device, dtype=torch.long)
    empty_y = torch.empty((0,), device=device, dtype=dtype)

    return MatrixCompletionDataBundle(
        data_source="synthetic",
        source_kind="synthetic",
        n_rows=n_rows,
        n_cols=n_cols,
        observed_row_idx=row_idx,
        observed_col_idx=col_idx,
        observed_y=y,
        train_row_idx=row_idx,
        train_col_idx=col_idx,
        train_y=y,
        val_row_idx=empty_idx,
        val_col_idx=empty_idx,
        val_y=empty_y,
        test_row_idx=empty_idx,
        test_col_idx=empty_idx,
        test_y=empty_y,
        observed_mask=observed_mask,
        X_star=x_star,
        U_star=u_star,
        V_star=v_star,
        target_row_coherence=row_coh,
        target_col_coherence=col_coh,
        target_condition_number=target_condition_number,
        target_effective_rank=target_effective_rank,
        target_stable_rank=target_stable_rank,
        target_singular_values=target_singular_values,
        coherence_level=coherence_level,
        sampling_regime=effective_sampling,
        mc_spectrum_type=mc_spectrum_type,
        mc_spectrum_decay=mc_spectrum_decay,
        dataset_version=dataset_version,
        split_version=split_version,
        preprocessing_version=preprocessing_version,
        split_strategy="observed_only",
        has_ground_truth=True,
    )




def _validate_source_aware_bundle(bundle: MatrixCompletionDataBundle) -> MatrixCompletionDataBundle:
    if bundle.source_kind not in {"synthetic", "real"}:
        raise ValueError(f"Unsupported matrix-completion source_kind={bundle.source_kind!r}.")
    if bundle.has_ground_truth and bundle.X_star is None:
        raise ValueError("Ground-truth-aware matrix-completion bundles require X_star.")
    if bundle.source_kind == "synthetic":
        if not bundle.has_ground_truth:
            raise ValueError("Synthetic matrix-completion bundles must expose ground truth.")
        if bundle.split_strategy is None:
            raise ValueError("Synthetic matrix-completion bundles must declare split_strategy.")
    if bundle.source_kind == "real" and bundle.has_ground_truth:
        raise ValueError("Real matrix-completion bundles cannot claim ground truth.")
    return bundle


def matrix_completion_supports_ground_truth_metrics(bundle_or_problem: MatrixCompletionDataBundle | MatrixCompletionProblem) -> bool:
    return bool(getattr(bundle_or_problem, "has_ground_truth", False))


def matrix_completion_has_heldout_observations(bundle_or_problem: MatrixCompletionDataBundle | MatrixCompletionProblem) -> bool:
    val_y = getattr(bundle_or_problem, "val_y")
    test_y = getattr(bundle_or_problem, "test_y")
    return bool(val_y.numel() or test_y.numel())


def make_matrix_completion_data_bundle(
    *,
    seed: int,
    n_rows: int,
    n_cols: int,
    r_star: int,
    m: int,
    noise_std: float = 0.0,
    measurement_type: str = "uniform_entries",
    data_source: str = "synthetic",
    dataset_version: str | None = None,
    split_version: str | None = None,
    preprocessing_version: str | None = None,
    custom_dataset_path: str | None = None,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    coherence_level: float = 0.0,
    sampling_regime: str = "uniform",
    mc_spectrum_type: str = "flat",
    mc_spectrum_decay: float = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> MatrixCompletionDataBundle:
    if device is None:
        device = torch.device("cpu")

    if data_source == "movielens_100k":
        if measurement_type != "observed_ratings":
            raise ValueError("MovieLens 100K requires measurement_type='observed_ratings'.")
        return _validate_source_aware_bundle(_bundle_from_movielens_split(
            seed=seed,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            dataset_version=dataset_version,
            split_version=split_version,
            preprocessing_version=preprocessing_version,
            device=device,
            dtype=dtype,
        ))

    if data_source == "custom_triplets_csv":
        if measurement_type != "observed_ratings":
            raise ValueError("custom_triplets_csv requires measurement_type='observed_ratings'.")
        if not str(custom_dataset_path or "").strip():
            raise ValueError("custom_triplets_csv requires custom_dataset_path")
        return _validate_source_aware_bundle(_bundle_from_custom_triplets_csv(
            path=str(custom_dataset_path),
            seed=seed,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            dataset_version=dataset_version,
            split_version=split_version,
            preprocessing_version=preprocessing_version,
            device=device,
            dtype=dtype,
        ))

    if measurement_type not in {"uniform_entries", "leverage_biased_entries"}:
        raise ValueError(
            f"Unsupported measurement_type={measurement_type!r}. Expected 'uniform_entries' or 'leverage_biased_entries'."
        )

    return _validate_source_aware_bundle(_bundle_from_synthetic_target(
        seed=seed,
        n_rows=n_rows,
        n_cols=n_cols,
        r_star=r_star,
        m=m,
        noise_std=noise_std,
        measurement_type=measurement_type,
        coherence_level=coherence_level,
        sampling_regime=sampling_regime,
        mc_spectrum_type=mc_spectrum_type,
        mc_spectrum_decay=mc_spectrum_decay,
        dataset_version=dataset_version,
        split_version=split_version,
        preprocessing_version=preprocessing_version,
        device=device,
        dtype=dtype,
    ))


def _problem_from_bundle(bundle: MatrixCompletionDataBundle) -> MatrixCompletionProblem:
    return MatrixCompletionProblem(
        data_source=bundle.data_source,
        source_kind=bundle.source_kind,
        n_rows=bundle.n_rows,
        n_cols=bundle.n_cols,
        train_row_idx=bundle.train_row_idx,
        train_col_idx=bundle.train_col_idx,
        train_y=bundle.train_y,
        val_row_idx=bundle.val_row_idx,
        val_col_idx=bundle.val_col_idx,
        val_y=bundle.val_y,
        test_row_idx=bundle.test_row_idx,
        test_col_idx=bundle.test_col_idx,
        test_y=bundle.test_y,
        X_star=bundle.X_star,
        U_star=bundle.U_star,
        V_star=bundle.V_star,
        mask=bundle.observed_mask,
        rating_min=bundle.rating_min,
        rating_max=bundle.rating_max,
        target_row_coherence=bundle.target_row_coherence,
        target_col_coherence=bundle.target_col_coherence,
        target_condition_number=bundle.target_condition_number,
        target_effective_rank=bundle.target_effective_rank,
        target_stable_rank=bundle.target_stable_rank,
        target_singular_values=bundle.target_singular_values,
        coherence_level=bundle.coherence_level,
        sampling_regime=bundle.sampling_regime,
        mc_spectrum_type=bundle.mc_spectrum_type,
        mc_spectrum_decay=bundle.mc_spectrum_decay,
        dataset_version=bundle.dataset_version,
        split_version=bundle.split_version,
        preprocessing_version=bundle.preprocessing_version,
        custom_dataset_path=bundle.custom_dataset_path,
        split_strategy=bundle.split_strategy,
        has_ground_truth=bundle.has_ground_truth,
        data_bundle=bundle,
    )


def make_matrix_completion_problem(
    *,
    seed: int,
    n_rows: int,
    n_cols: int,
    r_star: int,
    m: int,
    noise_std: float = 0.0,
    measurement_type: str = "uniform_entries",
    data_source: str = "synthetic",
    dataset_version: str | None = None,
    split_version: str | None = None,
    preprocessing_version: str | None = None,
    custom_dataset_path: str | None = None,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    coherence_level: float = 0.0,
    sampling_regime: str = "uniform",
    mc_spectrum_type: str = "flat",
    mc_spectrum_decay: float = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> MatrixCompletionProblem:
    bundle = make_matrix_completion_data_bundle(
        seed=seed,
        n_rows=n_rows,
        n_cols=n_cols,
        r_star=r_star,
        m=m,
        noise_std=noise_std,
        measurement_type=measurement_type,
        data_source=data_source,
        dataset_version=dataset_version,
        split_version=split_version,
        preprocessing_version=preprocessing_version,
        custom_dataset_path=custom_dataset_path,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        coherence_level=coherence_level,
        sampling_regime=sampling_regime,
        mc_spectrum_type=mc_spectrum_type,
        mc_spectrum_decay=mc_spectrum_decay,
        device=device,
        dtype=dtype,
    )
    return _problem_from_bundle(bundle)
