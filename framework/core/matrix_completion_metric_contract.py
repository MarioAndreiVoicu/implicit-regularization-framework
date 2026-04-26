from __future__ import annotations

from typing import Any


SYNTHETIC_ONLY_TRAJECTORY_METRICS = {
    "relative_reconstruction_error",
    "rel_recon_error",
    "target_row_coherence",
    "target_col_coherence",
    "target_condition_number",
    "target_effective_rank",
    "target_stable_rank",
    "left_subspace_alignment",
    "right_subspace_alignment",
    "mean_subspace_alignment",
    "left_subspace_projection_distance",
    "right_subspace_projection_distance",
    "mean_subspace_projection_distance",
}

HELDOUT_ONLY_TRAJECTORY_METRICS = {
    "val_loss",
    "test_loss",
    "val_rmse",
    "test_rmse",
}

SYNTHETIC_ONLY_SUMMARY_METRICS = {
    "final_relative_reconstruction_error",
    "final_rel_recon_error",
    "final_target_row_coherence",
    "final_target_col_coherence",
    "final_target_condition_number",
    "final_target_effective_rank",
    "final_target_stable_rank",
    "final_left_subspace_alignment",
    "final_right_subspace_alignment",
    "final_mean_subspace_alignment",
    "final_left_subspace_projection_distance",
    "final_right_subspace_projection_distance",
    "final_mean_subspace_projection_distance",
    "best_rel_recon_error",
    "best_rel_recon_error_step",
}

HELDOUT_ONLY_SUMMARY_METRICS = {
    "final_val_loss",
    "final_test_loss",
    "final_val_rmse",
    "final_test_rmse",
    "best_val_metric_name",
    "best_val_step",
    "best_val_metric",
    "best_val_loss",
    "best_val_rmse",
    "best_test_loss_at_best_val",
    "best_test_rmse_at_best_val",
}

SWEEP_METRIC_OPTIONS_BY_SOURCE_KIND: dict[str, dict[str, str]] = {
    "synthetic": {
        "final_rel_recon_error": "Final relative reconstruction error",
        "final_objective": "Final objective",
        "final_frobenius_norm_X": "Final Frobenius norm of X",
        "final_target_row_coherence": "Final target row coherence",
        "final_target_col_coherence": "Final target col coherence",
        "final_target_condition_number": "Final target condition number",
        "final_target_effective_rank": "Final target effective rank",
        "final_target_stable_rank": "Final target stable rank",
        "final_factor_u_norm": "Final factor U norm",
        "final_factor_v_norm": "Final factor V norm",
        "final_factor_norm_ratio": "Final factor norm ratio",
        "final_left_subspace_alignment": "Final left subspace alignment",
        "final_right_subspace_alignment": "Final right subspace alignment",
        "final_mean_subspace_alignment": "Final mean subspace alignment",
        "final_left_subspace_projection_distance": "Final left subspace projection distance",
        "final_right_subspace_projection_distance": "Final right subspace projection distance",
        "final_mean_subspace_projection_distance": "Final mean subspace projection distance",
    },
    "real": {
        "final_val_rmse": "Final validation RMSE",
        "final_test_rmse": "Final test RMSE",
        "best_val_rmse": "Best validation RMSE",
        "best_test_rmse_at_best_val": "Test RMSE at best validation step",
        "final_objective": "Final objective",
    },
}

FINAL_METRICS_BY_SOURCE_KIND: dict[str, list[str]] = {
    "synthetic": [
        "final_objective",
        "final_rel_recon_error",
        "final_frobenius_norm_X",
        "final_effective_rank",
        "final_stable_rank",
        "final_gradient_norm",
        "final_target_row_coherence",
        "final_target_col_coherence",
        "final_target_condition_number",
        "final_target_effective_rank",
        "final_target_stable_rank",
        "final_factor_u_norm",
        "final_factor_v_norm",
        "final_factor_norm_ratio",
        "final_factor_log_norm_ratio",
        "final_left_subspace_alignment",
        "final_right_subspace_alignment",
        "final_mean_subspace_alignment",
        "final_left_subspace_projection_distance",
        "final_right_subspace_projection_distance",
        "final_mean_subspace_projection_distance",
    ],
    "real": [
        "final_objective",
        "final_train_loss",
        "final_val_loss",
        "final_test_loss",
        "final_val_rmse",
        "final_test_rmse",
        "best_val_metric",
        "best_val_step",
        "best_val_loss",
        "best_val_rmse",
        "best_test_loss_at_best_val",
        "best_test_rmse_at_best_val",
        "final_gradient_norm",
    ],
}

TRAJECTORY_METRICS_BY_SOURCE_KIND: dict[str, list[str]] = {
    "synthetic": [
        "objective",
        "rel_recon_error",
        "frobenius_norm_X",
        "effective_rank",
        "stable_rank",
        "gradient_norm",
        "target_row_coherence",
        "target_col_coherence",
        "target_condition_number",
        "target_effective_rank",
        "target_stable_rank",
        "factor_u_norm",
        "factor_v_norm",
        "factor_norm_ratio",
        "factor_log_norm_ratio",
        "left_subspace_alignment",
        "right_subspace_alignment",
        "mean_subspace_alignment",
        "left_subspace_projection_distance",
        "right_subspace_projection_distance",
        "mean_subspace_projection_distance",
    ],
    "real": [
        "objective",
        "train_loss",
        "val_loss",
        "test_loss",
        "val_rmse",
        "test_rmse",
        "gradient_norm",
    ],
}


def matrix_completion_source_kind(*, source_kind: str | None = None, data_source: str | None = None) -> str:
    if source_kind in {"synthetic", "real"}:
        return str(source_kind)
    return "synthetic" if str(data_source or "synthetic") == "synthetic" else "real"


def matrix_completion_final_metrics(*, source_kind: str | None = None, data_source: str | None = None) -> list[str]:
    kind = matrix_completion_source_kind(source_kind=source_kind, data_source=data_source)
    return list(FINAL_METRICS_BY_SOURCE_KIND[kind])


def matrix_completion_trajectory_metrics(*, source_kind: str | None = None, data_source: str | None = None) -> list[str]:
    kind = matrix_completion_source_kind(source_kind=source_kind, data_source=data_source)
    return list(TRAJECTORY_METRICS_BY_SOURCE_KIND[kind])


def matrix_completion_sweep_metric_options(*, source_kind: str | None = None, data_source: str | None = None) -> dict[str, str]:
    kind = matrix_completion_source_kind(source_kind=source_kind, data_source=data_source)
    return dict(SWEEP_METRIC_OPTIONS_BY_SOURCE_KIND[kind])


def preferred_matrix_completion_fit_metric(*, source_kind: str | None = None, data_source: str | None = None) -> str:
    kind = matrix_completion_source_kind(source_kind=source_kind, data_source=data_source)
    return "rel_recon_error" if kind == "synthetic" else "val_rmse"


def matrix_completion_history_plot_metrics(*, source_kind: str | None = None, data_source: str | None = None) -> list[tuple[str, str]]:
    kind = matrix_completion_source_kind(source_kind=source_kind, data_source=data_source)
    if kind == "real":
        return [
            ("loss", "Loss vs step"),
            ("val_rmse", "Validation RMSE vs step"),
            ("test_rmse", "Test RMSE vs step"),
        ]
    return [
        ("loss", "Loss vs step"),
        ("rel_recon_error", "Relative reconstruction error vs step"),
        ("frobenius_norm_X", "||X||_F vs step"),
    ]


def _dynamic_source_only_keys(prefixes: set[str], num_eigs: int) -> set[str]:
    keys: set[str] = set()
    for i in range(1, max(0, int(num_eigs)) + 1):
        keys.update({f"target_sv_{i}", f"final_target_sv_{i}"})
    return keys


def filter_matrix_completion_metrics(metrics: dict[str, list[float]], *, source_kind: str, num_eigs: int) -> dict[str, list[float]]:
    kind = matrix_completion_source_kind(source_kind=source_kind)
    forbidden = set(HELDOUT_ONLY_TRAJECTORY_METRICS) if kind == "synthetic" else set(SYNTHETIC_ONLY_TRAJECTORY_METRICS)
    if kind == "real":
        forbidden.update({f"target_sv_{i}" for i in range(1, max(0, int(num_eigs)) + 1)})
    return {key: value for key, value in metrics.items() if key not in forbidden}


def filter_matrix_completion_summary(summary: dict[str, Any], *, source_kind: str, num_eigs: int) -> dict[str, Any]:
    kind = matrix_completion_source_kind(source_kind=source_kind)
    forbidden = set(HELDOUT_ONLY_SUMMARY_METRICS) if kind == "synthetic" else set(SYNTHETIC_ONLY_SUMMARY_METRICS)
    if kind == "real":
        forbidden.update({f"final_target_sv_{i}" for i in range(1, max(0, int(num_eigs)) + 1)})
    return {key: value for key, value in summary.items() if key not in forbidden}
