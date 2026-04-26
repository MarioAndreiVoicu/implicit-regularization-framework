from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch

from framework.core.metrics import rel_frob_err, spectral_metrics_matrix, subspace_alignment_metrics_matrix
from framework.core.models import RunResult, RunSpec, RunStatus
from framework.core.matrix_completion_metric_contract import filter_matrix_completion_metrics, filter_matrix_completion_summary
from framework.core.optimizers import apply_optimizer_step, current_learning_rate, init_optimizer_state, minibatch_indices, optimizer_kind, zero_grad
from framework.core.parameterization import (
    initialize_matrix_completion_parameters,
    matrix_completion_model_x,
    parameter_artifacts,
    project_parameters_inplace,
)
from framework.core.problems.matrix_completion import (
    make_matrix_completion_problem,
    matrix_completion_supports_ground_truth_metrics,
)
from framework.core.regularization import explicit_reg_penalty


def _observed_loss(x: torch.Tensor, row_idx: torch.Tensor, col_idx: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if y.numel() == 0:
        return torch.tensor(float("nan"), device=x.device, dtype=x.dtype)
    preds = x[row_idx, col_idx]
    resid = preds - y
    return 0.5 * torch.mean(resid**2)


def _observed_rmse(x: torch.Tensor, row_idx: torch.Tensor, col_idx: torch.Tensor, y: torch.Tensor) -> float:
    if y.numel() == 0:
        return float("nan")
    preds = x[row_idx, col_idx]
    resid = preds - y
    return float(torch.sqrt(torch.mean(resid**2)).item())


def _total_gradient_norm(parameters: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for tensor in parameters.values():
        if tensor.grad is None:
            return float("nan")
        grad_norm = float(torch.linalg.norm(tensor.grad).item())
        total += grad_norm * grad_norm
    return math.sqrt(total)


def _total_parameter_norm(parameters: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for tensor in parameters.values():
        param_norm = float(torch.linalg.norm(tensor).item())
        total += param_norm * param_norm
    return math.sqrt(total)


def _parameter_delta_norm(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for name in before.keys():
        delta = after[name] - before[name]
        delta_norm = float(torch.linalg.norm(delta).item())
        total += delta_norm * delta_norm
    return math.sqrt(total)


def _factor_balance_diagnostics(parameters: dict[str, torch.Tensor], parameterization: str) -> tuple[float, float, float, float]:
    if parameterization != "factorized_uv":
        return float("nan"), float("nan"), float("nan"), float("nan")
    u_norm = float(torch.linalg.norm(parameters["U"]).item())
    v_norm = float(torch.linalg.norm(parameters["V"]).item())
    if v_norm == 0.0:
        ratio = float('inf') if u_norm > 0.0 else 1.0
    else:
        ratio = u_norm / v_norm
    log_ratio = float(math.log(max(ratio, 1e-30))) if math.isfinite(ratio) and ratio > 0.0 else float('inf')
    return u_norm, v_norm, ratio, log_ratio


def _metrics_template(num_eigs: int) -> dict[str, list[float]]:
    metrics: dict[str, list[float]] = {
        "step": [],
        "train_loss": [],
        "val_loss": [],
        "test_loss": [],
        "val_rmse": [],
        "test_rmse": [],
        "observed_loss": [],
        "data_loss": [],
        "loss": [],
        "explicit_reg_term": [],
        "objective": [],
        "lr": [],
        "gradient_norm": [],
        "update_norm": [],
        "parameter_norm": [],
        "objective_delta": [],
        "relative_reconstruction_error": [],
        "rel_recon_error": [],
        "frobenius_norm_X": [],
        "effective_rank": [],
        "stable_rank": [],
        "trace_X": [],
        "min_eigenvalue_X": [],
        "negative_eigen_mass": [],
        "target_row_coherence": [],
        "target_col_coherence": [],
        "target_condition_number": [],
        "target_effective_rank": [],
        "target_stable_rank": [],
        "factor_u_norm": [],
        "factor_v_norm": [],
        "factor_norm_ratio": [],
        "factor_log_norm_ratio": [],
        "left_subspace_alignment": [],
        "right_subspace_alignment": [],
        "mean_subspace_alignment": [],
        "left_subspace_projection_distance": [],
        "right_subspace_projection_distance": [],
        "mean_subspace_projection_distance": [],
    }
    for i in range(num_eigs):
        metrics[f"eig_{i + 1}"] = []
        metrics[f"sv_{i + 1}"] = []
        metrics[f"target_sv_{i + 1}"] = []
    return metrics


def _append_metrics(
    metrics: dict[str, list[float]],
    *,
    step: int,
    train_loss_val: float,
    val_loss_val: float,
    test_loss_val: float,
    val_rmse_val: float,
    test_rmse_val: float,
    reg_term_val: float,
    objective_val: float,
    lr_val: float,
    gradient_norm_val: float,
    update_norm_val: float,
    parameter_norm_val: float,
    objective_delta_val: float,
    err_val: float,
    spectral: dict[str, float | list[float]],
    target_row_coherence_val: float,
    target_col_coherence_val: float,
    target_condition_number_val: float,
    target_effective_rank_val: float,
    target_stable_rank_val: float,
    target_singular_values: list[float],
    factor_u_norm_val: float,
    factor_v_norm_val: float,
    factor_norm_ratio_val: float,
    factor_log_norm_ratio_val: float,
    left_subspace_alignment_val: float,
    right_subspace_alignment_val: float,
    mean_subspace_alignment_val: float,
    left_subspace_projection_distance_val: float,
    right_subspace_projection_distance_val: float,
    mean_subspace_projection_distance_val: float,
) -> None:
    metrics["step"].append(float(step))
    metrics["train_loss"].append(train_loss_val)
    metrics["val_loss"].append(val_loss_val)
    metrics["test_loss"].append(test_loss_val)
    metrics["val_rmse"].append(val_rmse_val)
    metrics["test_rmse"].append(test_rmse_val)
    metrics["observed_loss"].append(train_loss_val)
    metrics["data_loss"].append(train_loss_val)
    metrics["loss"].append(train_loss_val)
    metrics["explicit_reg_term"].append(reg_term_val)
    metrics["objective"].append(objective_val)
    metrics["lr"].append(lr_val)
    metrics["gradient_norm"].append(gradient_norm_val)
    metrics["update_norm"].append(update_norm_val)
    metrics["parameter_norm"].append(parameter_norm_val)
    metrics["objective_delta"].append(objective_delta_val)
    metrics["relative_reconstruction_error"].append(err_val)
    metrics["rel_recon_error"].append(err_val)
    metrics["frobenius_norm_X"].append(float(spectral["frobenius_norm_X"]))
    metrics["effective_rank"].append(float(spectral["effective_rank"]))
    metrics["stable_rank"].append(float(spectral["stable_rank"]))
    metrics["trace_X"].append(float(spectral["trace_X"]))
    metrics["min_eigenvalue_X"].append(float(spectral["min_eigenvalue_X"]))
    metrics["negative_eigen_mass"].append(float(spectral["negative_eigen_mass"]))
    metrics["target_row_coherence"].append(target_row_coherence_val)
    metrics["target_col_coherence"].append(target_col_coherence_val)
    metrics["target_condition_number"].append(target_condition_number_val)
    metrics["target_effective_rank"].append(target_effective_rank_val)
    metrics["target_stable_rank"].append(target_stable_rank_val)
    metrics["factor_u_norm"].append(factor_u_norm_val)
    metrics["factor_v_norm"].append(factor_v_norm_val)
    metrics["factor_norm_ratio"].append(factor_norm_ratio_val)
    metrics["factor_log_norm_ratio"].append(factor_log_norm_ratio_val)
    metrics["left_subspace_alignment"].append(left_subspace_alignment_val)
    metrics["right_subspace_alignment"].append(right_subspace_alignment_val)
    metrics["mean_subspace_alignment"].append(mean_subspace_alignment_val)
    metrics["left_subspace_projection_distance"].append(left_subspace_projection_distance_val)
    metrics["right_subspace_projection_distance"].append(right_subspace_projection_distance_val)
    metrics["mean_subspace_projection_distance"].append(mean_subspace_projection_distance_val)
    for i, value in enumerate(spectral["top_k_eigenvalues"]):
        metrics[f"eig_{i + 1}"].append(float(value))
    for i, value in enumerate(spectral["top_k_singular_values"]):
        metrics[f"sv_{i + 1}"].append(float(value))
    for i, value in enumerate(target_singular_values):
        metrics[f"target_sv_{i + 1}"].append(float(value))




def _emit_progress_update(
    progress_callback: Callable[[dict[str, Any]], None] | None,
    *,
    step: int,
    max_steps: int,
    status: str,
    metrics: dict[str, list[float]],
    summary: dict[str, Any] | None = None,
) -> None:
    if progress_callback is None or not metrics.get("step"):
        return
    latest = {
        "step": int(metrics["step"][-1]),
        "train_loss": float(metrics["train_loss"][-1]),
        "val_loss": float(metrics["val_loss"][-1]),
        "test_loss": float(metrics["test_loss"][-1]),
        "relative_reconstruction_error": float(metrics["relative_reconstruction_error"][-1]),
        "gradient_norm": float(metrics["gradient_norm"][-1]),
        "lr": float(metrics["lr"][-1]),
    }
    history_keys = ["step", "train_loss", "val_loss", "test_loss", "relative_reconstruction_error", "gradient_norm", "lr"]
    history = {key: [float(x) for x in metrics.get(key, [])] for key in history_keys if metrics.get(key)}
    payload = {
        "status": str(status),
        "step": int(step),
        "max_steps": int(max_steps),
        "metrics": latest,
        "history": history,
    }
    if summary is not None:
        payload["summary"] = dict(summary)
    progress_callback(payload)

def _full_objective(*, parameters: dict[str, torch.Tensor], spec: RunSpec, problem) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = matrix_completion_model_x(parameters, spec.parameterization)
    train_loss = _observed_loss(x, problem.train_row_idx, problem.train_col_idx, problem.train_y)
    reg_term = explicit_reg_penalty(
        parameters=parameters,
        parameterization=spec.parameterization,
        explicit_reg_type=spec.explicit_reg_type,
        explicit_reg_strength=spec.explicit_reg_strength,
    )
    objective = train_loss + reg_term
    return x, train_loss, reg_term, objective




def _clone_parameters(parameters: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in parameters.items()}


def _early_stopping_score(metric_name: str, *, val_loss_val: float, val_rmse_val: float) -> float:
    if metric_name == "val_loss":
        return float(val_loss_val)
    if metric_name == "val_rmse":
        return float(val_rmse_val)
    raise ValueError(f"Unsupported early stopping metric: {metric_name!r}")

def _update_objective(*, parameters: dict[str, torch.Tensor], spec: RunSpec, problem, opt_kind: str, opt_state) -> torch.Tensor:
    x = matrix_completion_model_x(parameters, spec.parameterization)
    train_count = int(problem.train_y.numel())
    if opt_kind == "minibatch":
        if spec.batch_size is None or spec.batch_size > train_count:
            raise ValueError("batch_size cannot exceed the number of training observations")
        idx = minibatch_indices(m=train_count, batch_size=spec.batch_size, state=opt_state, device=problem.train_y.device)
        train_loss = _observed_loss(x, problem.train_row_idx[idx], problem.train_col_idx[idx], problem.train_y[idx])
    else:
        train_loss = _observed_loss(x, problem.train_row_idx, problem.train_col_idx, problem.train_y)
    reg_term = explicit_reg_penalty(
        parameters=parameters,
        parameterization=spec.parameterization,
        explicit_reg_type=spec.explicit_reg_type,
        explicit_reg_strength=spec.explicit_reg_strength,
    )
    return train_loss + reg_term


def run_matrix_completion_optimizer(
    spec: RunSpec,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> RunResult:
    spec.validate()
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float64
    if spec.problem_family != "matrix_completion":
        raise ValueError(f"run_matrix_completion_optimizer supports only problem_family='matrix_completion', got {spec.problem_family!r}.")

    n_cols = spec.n_cols if spec.n_cols is not None else spec.n
    problem = make_matrix_completion_problem(
        seed=spec.seed,
        n_rows=spec.n,
        n_cols=n_cols,
        r_star=spec.r_star,
        m=spec.m,
        noise_std=spec.noise_std,
        measurement_type=spec.measurement_type,
        data_source=spec.data_source,
        dataset_version=spec.dataset_version,
        split_version=spec.split_version,
        preprocessing_version=spec.preprocessing_version,
        custom_dataset_path=spec.custom_dataset_path,
        val_fraction=spec.val_fraction,
        test_fraction=spec.test_fraction,
        coherence_level=spec.coherence_level,
        sampling_regime=spec.sampling_regime,
        mc_spectrum_type=spec.mc_spectrum_type,
        mc_spectrum_decay=spec.mc_spectrum_decay,
        device=device,
        dtype=dtype,
    )

    # real dataset fixes the actual dimensions/observation count
    spec.n = int(problem.n_rows)
    spec.n_cols = int(problem.n_cols)
    spec.m = int(problem.train_y.numel() + problem.val_y.numel() + problem.test_y.numel())
    if problem.sampling_regime is not None:
        spec.sampling_regime = str(problem.sampling_regime)
    if problem.mc_spectrum_type is not None:
        spec.mc_spectrum_type = str(problem.mc_spectrum_type)
    if problem.mc_spectrum_decay is not None:
        spec.mc_spectrum_decay = float(problem.mc_spectrum_decay)

    parameters = initialize_matrix_completion_parameters(
        seed=spec.seed + 12345,
        n_rows=problem.n_rows,
        n_cols=problem.n_cols,
        factor_rank=spec.factor_rank,
        init_scale=spec.init_scale,
        init_type=spec.init_type,
        parameterization=spec.parameterization,
        mc_factor_balance=spec.mc_factor_balance,
        mc_factor_imbalance_level=spec.mc_factor_imbalance_level,
        device=device,
        dtype=dtype,
    )
    for tensor in parameters.values():
        tensor.requires_grad_(True)

    opt_kind = optimizer_kind(spec.optimizer)
    opt_state = init_optimizer_state(optimizer=spec.optimizer, parameters=parameters, seed=spec.seed + 54321)
    metrics = _metrics_template(spec.num_eigs)

    best_err = float("inf")
    best_step = 0
    best_objective = float("inf")
    status = RunStatus.COMPLETED
    diverged_step = None
    converged = False
    converged_step = None
    stop_reason = "max_steps"
    prev_objective_val: float | None = None
    last_update_norm = 0.0
    early_stopping_active = bool(spec.early_stopping_enabled and spec.problem_family == "matrix_completion" and spec.data_source in {"movielens_100k", "custom_triplets_csv"})
    best_val_metric_name = spec.early_stopping_metric if early_stopping_active else None
    best_val_metric = float("inf")
    best_val_step = None
    best_val_loss = float("nan")
    best_val_rmse = float("nan")
    best_test_loss_at_best_val = float("nan")
    best_test_rmse_at_best_val = float("nan")
    early_stopping_bad_steps = 0
    best_parameters = None

    for t in range(spec.steps + 1):
        zero_grad(parameters)
        x, train_loss, reg_term, objective = _full_objective(parameters=parameters, spec=spec, problem=problem)
        if not torch.isfinite(objective):
            status = RunStatus.DIVERGED
            diverged_step = t
            stop_reason = "nonfinite_objective"
            break

        parameter_norm_val = _total_parameter_norm(parameters)
        if (not math.isfinite(parameter_norm_val)) or (parameter_norm_val > spec.divergence_norm_threshold):
            status = RunStatus.DIVERGED
            diverged_step = t
            stop_reason = "parameter_norm_exceeded"
            break

        objective.backward()
        invalid_grad = any(tensor.grad is None or (not torch.isfinite(tensor.grad).all()) for tensor in parameters.values())
        if invalid_grad:
            status = RunStatus.DIVERGED
            diverged_step = t
            stop_reason = "nonfinite_gradient"
            break

        train_loss_val = float(train_loss.item())
        reg_term_val = float(reg_term.item())
        objective_val = float(objective.item())
        gradient_norm_val = _total_gradient_norm(parameters)
        objective_delta_val = 0.0 if prev_objective_val is None else abs(objective_val - prev_objective_val)
        has_ground_truth = matrix_completion_supports_ground_truth_metrics(problem)
        err_val = float("nan") if not has_ground_truth or problem.X_star is None else rel_frob_err(x, problem.X_star)
        u_norm_val, v_norm_val, factor_ratio_val, factor_log_ratio_val = _factor_balance_diagnostics(parameters, spec.parameterization)
        val_loss_val = float(_observed_loss(x, problem.val_row_idx, problem.val_col_idx, problem.val_y).item()) if problem.val_y.numel() else float("nan")
        test_loss_val = float(_observed_loss(x, problem.test_row_idx, problem.test_col_idx, problem.test_y).item()) if problem.test_y.numel() else float("nan")
        val_rmse_val = _observed_rmse(x, problem.val_row_idx, problem.val_col_idx, problem.val_y)
        test_rmse_val = _observed_rmse(x, problem.test_row_idx, problem.test_col_idx, problem.test_y)
        spectral = spectral_metrics_matrix(x, spec.num_eigs)
        alignment = subspace_alignment_metrics_matrix(x, problem.X_star, spec.r_star) if has_ground_truth and problem.X_star is not None else None

        should_stop_for_convergence = (
            spec.stopping_mode == "run_to_convergence"
            and prev_objective_val is not None
            and gradient_norm_val <= spec.grad_norm_tol
            and objective_delta_val <= spec.objective_tol
        )
        should_log = (t % spec.log_every == 0) or (t == spec.steps) or should_stop_for_convergence
        if should_log:
            _append_metrics(
                metrics,
                step=t,
                train_loss_val=train_loss_val,
                val_loss_val=val_loss_val,
                test_loss_val=test_loss_val,
                val_rmse_val=val_rmse_val,
                test_rmse_val=test_rmse_val,
                reg_term_val=reg_term_val,
                objective_val=objective_val,
                lr_val=current_learning_rate(spec=spec, step_index=t),
                gradient_norm_val=gradient_norm_val,
                update_norm_val=last_update_norm,
                parameter_norm_val=parameter_norm_val,
                objective_delta_val=objective_delta_val,
                err_val=err_val,
                spectral=spectral,
                target_row_coherence_val=float(problem.target_row_coherence) if problem.target_row_coherence is not None else float("nan"),
                target_col_coherence_val=float(problem.target_col_coherence) if problem.target_col_coherence is not None else float("nan"),
                target_condition_number_val=float(problem.target_condition_number) if problem.target_condition_number is not None else float("nan"),
                target_effective_rank_val=float(problem.target_effective_rank) if problem.target_effective_rank is not None else float("nan"),
                target_stable_rank_val=float(problem.target_stable_rank) if problem.target_stable_rank is not None else float("nan"),
                target_singular_values=(lambda vals: vals + [float("nan")] * (spec.num_eigs - len(vals)))([float(x) for x in (problem.target_singular_values[:spec.num_eigs].detach().cpu().tolist() if problem.target_singular_values is not None else [])]),
                factor_u_norm_val=u_norm_val,
                factor_v_norm_val=v_norm_val,
                factor_norm_ratio_val=factor_ratio_val,
                factor_log_norm_ratio_val=factor_log_ratio_val,
                left_subspace_alignment_val=float("nan") if alignment is None else alignment["left_subspace_alignment"],
                right_subspace_alignment_val=float("nan") if alignment is None else alignment["right_subspace_alignment"],
                mean_subspace_alignment_val=float("nan") if alignment is None else alignment["mean_subspace_alignment"],
                left_subspace_projection_distance_val=float("nan") if alignment is None else alignment["left_subspace_projection_distance"],
                right_subspace_projection_distance_val=float("nan") if alignment is None else alignment["right_subspace_projection_distance"],
                mean_subspace_projection_distance_val=float("nan") if alignment is None else alignment["mean_subspace_projection_distance"],
            )
            if math.isfinite(err_val) and err_val < best_err:
                best_err = err_val
                best_step = t
            if objective_val < best_objective:
                best_objective = objective_val
            _emit_progress_update(
                progress_callback,
                step=t,
                max_steps=spec.steps,
                status=RunStatus.RUNNING,
                metrics=metrics,
            )

        if early_stopping_active:
            current_val_metric = _early_stopping_score(
                spec.early_stopping_metric,
                val_loss_val=val_loss_val,
                val_rmse_val=val_rmse_val,
            )
            if math.isfinite(current_val_metric):
                improved = current_val_metric < (best_val_metric - spec.early_stopping_min_delta)
                if improved or best_val_step is None:
                    best_val_metric = current_val_metric
                    best_val_step = t
                    best_val_loss = val_loss_val
                    best_val_rmse = val_rmse_val
                    best_test_loss_at_best_val = test_loss_val
                    best_test_rmse_at_best_val = test_rmse_val
                    best_parameters = _clone_parameters(parameters)
                    early_stopping_bad_steps = 0
                else:
                    early_stopping_bad_steps += 1

        if should_stop_for_convergence:
            converged = True
            converged_step = t
            stop_reason = "converged"
            zero_grad(parameters)
            break

        if early_stopping_active and best_val_step is not None and early_stopping_bad_steps >= spec.early_stopping_patience:
            stop_reason = "early_stopping"
            zero_grad(parameters)
            break

        if t == spec.steps:
            stop_reason = "fixed_steps_complete" if spec.stopping_mode == "fixed_steps" else "max_steps"
            zero_grad(parameters)
            break

        zero_grad(parameters)
        update_objective = _update_objective(parameters=parameters, spec=spec, problem=problem, opt_kind=opt_kind, opt_state=opt_state)
        update_objective.backward()

        invalid_update_grad = any(tensor.grad is None or (not torch.isfinite(tensor.grad).all()) for tensor in parameters.values())
        if invalid_update_grad:
            status = RunStatus.DIVERGED
            diverged_step = t
            stop_reason = "nonfinite_gradient"
            break

        before_step = {name: tensor.detach().clone() for name, tensor in parameters.items()}
        with torch.no_grad():
            apply_optimizer_step(optimizer=spec.optimizer, spec=spec, parameters=parameters, state=opt_state)
            project_parameters_inplace(parameters, spec.parameterization)
        after_step = {name: tensor.detach() for name, tensor in parameters.items()}
        last_update_norm = _parameter_delta_norm(before_step, after_step)

        if any(not torch.isfinite(tensor).all() for tensor in parameters.values()):
            status = RunStatus.DIVERGED
            diverged_step = t + 1
            stop_reason = "nonfinite_parameters"
            zero_grad(parameters)
            break

        post_update_parameter_norm = _total_parameter_norm(parameters)
        if (not math.isfinite(post_update_parameter_norm)) or (post_update_parameter_norm > spec.divergence_norm_threshold):
            status = RunStatus.DIVERGED
            diverged_step = t + 1
            stop_reason = "parameter_norm_exceeded"
            zero_grad(parameters)
            break

        zero_grad(parameters)
        prev_objective_val = objective_val

    if not metrics["step"]:
        raise RuntimeError("No metrics were recorded before termination.")

    artifacts = parameter_artifacts(parameters, spec.parameterization)
    if best_parameters is not None:
        best_artifacts = parameter_artifacts(best_parameters, spec.parameterization)
        for key, value in best_artifacts.items():
            artifacts[key.replace("_final", "_best_val")] = value
    last_metrics = {k: v[-1] for k, v in metrics.items() if v}
    summary = {
        "data_source": spec.data_source,
        "source_kind": problem.source_kind,
        "split_strategy": problem.split_strategy,
        "has_ground_truth": bool(problem.has_ground_truth),
        "final_step": int(metrics["step"][-1]),
        "final_train_loss": float(last_metrics["train_loss"]),
        "final_val_loss": float(last_metrics["val_loss"]),
        "final_test_loss": float(last_metrics["test_loss"]),
        "final_val_rmse": float(last_metrics["val_rmse"]),
        "final_test_rmse": float(last_metrics["test_rmse"]),
        "best_val_metric_name": best_val_metric_name,
        "best_val_step": best_val_step,
        "best_val_metric": float(best_val_metric) if math.isfinite(best_val_metric) else float("nan"),
        "best_val_loss": float(best_val_loss),
        "best_val_rmse": float(best_val_rmse),
        "best_test_loss_at_best_val": float(best_test_loss_at_best_val),
        "best_test_rmse_at_best_val": float(best_test_rmse_at_best_val),
        "final_observed_loss": float(last_metrics["observed_loss"]),
        "final_data_loss": float(last_metrics["data_loss"]),
        "final_loss": float(last_metrics["loss"]),
        "final_explicit_reg_term": float(last_metrics["explicit_reg_term"]),
        "final_objective": float(last_metrics["objective"]),
        "final_lr": float(last_metrics["lr"]),
        "final_gradient_norm": float(last_metrics["gradient_norm"]),
        "final_update_norm": float(last_metrics["update_norm"]),
        "final_parameter_norm": float(last_metrics["parameter_norm"]),
        "final_objective_delta": float(last_metrics["objective_delta"]),
        "final_relative_reconstruction_error": float(last_metrics["relative_reconstruction_error"]),
        "final_rel_recon_error": float(last_metrics["rel_recon_error"]),
        "final_trace_X": float(last_metrics["trace_X"]),
        "final_frobenius_norm_X": float(last_metrics["frobenius_norm_X"]),
        "final_effective_rank": float(last_metrics["effective_rank"]),
        "final_stable_rank": float(last_metrics["stable_rank"]),
        "final_min_eigenvalue_X": float(last_metrics["min_eigenvalue_X"]),
        "final_negative_eigen_mass": float(last_metrics["negative_eigen_mass"]),
        "final_target_row_coherence": float(last_metrics["target_row_coherence"]),
        "final_target_col_coherence": float(last_metrics["target_col_coherence"]),
        "final_target_condition_number": float(last_metrics["target_condition_number"]),
        "final_target_effective_rank": float(last_metrics["target_effective_rank"]),
        "final_target_stable_rank": float(last_metrics["target_stable_rank"]),
        "final_factor_u_norm": float(last_metrics["factor_u_norm"]),
        "final_factor_v_norm": float(last_metrics["factor_v_norm"]),
        "final_factor_norm_ratio": float(last_metrics["factor_norm_ratio"]),
        "final_factor_log_norm_ratio": float(last_metrics["factor_log_norm_ratio"]),
        "final_left_subspace_alignment": float(last_metrics["left_subspace_alignment"]),
        "final_right_subspace_alignment": float(last_metrics["right_subspace_alignment"]),
        "final_mean_subspace_alignment": float(last_metrics["mean_subspace_alignment"]),
        "final_left_subspace_projection_distance": float(last_metrics["left_subspace_projection_distance"]),
        "final_right_subspace_projection_distance": float(last_metrics["right_subspace_projection_distance"]),
        "final_mean_subspace_projection_distance": float(last_metrics["mean_subspace_projection_distance"]),
        "best_rel_recon_error": float(best_err),
        "best_rel_recon_error_step": int(best_step),
        "best_objective": float(best_objective),
        "status": status,
        "resolved_device": str(device),
        "resolved_dtype": str(dtype).replace("torch.", ""),
        "execution_mode": spec.execution_mode,
        "is_diverged": bool(status == RunStatus.DIVERGED),
        "stopping_mode": spec.stopping_mode,
        "stop_reason": stop_reason,
        "early_stopping_enabled": bool(early_stopping_active),
        "early_stopping_metric": spec.early_stopping_metric,
        "early_stopping_patience": int(spec.early_stopping_patience),
        "early_stopping_min_delta": float(spec.early_stopping_min_delta),
        "early_stopped": bool(stop_reason == "early_stopping"),
        "converged": bool(converged),
        "converged_step": converged_step,
        "diverged_step": diverged_step,
        "steps_completed": int(metrics["step"][-1]),
    }
    for i in range(spec.num_eigs):
        summary[f"final_eig_{i + 1}"] = float(last_metrics[f"eig_{i + 1}"])
        summary[f"final_sv_{i + 1}"] = float(last_metrics[f"sv_{i + 1}"])
        summary[f"final_target_sv_{i + 1}"] = float(last_metrics[f"target_sv_{i + 1}"])

    filtered_metrics = filter_matrix_completion_metrics(metrics, source_kind=problem.source_kind, num_eigs=spec.num_eigs)
    filtered_summary = filter_matrix_completion_summary(summary, source_kind=problem.source_kind, num_eigs=spec.num_eigs)
    _emit_progress_update(
        progress_callback,
        step=int(filtered_summary["final_step"]),
        max_steps=spec.steps,
        status=status,
        metrics=filtered_metrics,
        summary=filtered_summary,
    )
    return RunResult(spec=spec, metrics=filtered_metrics, summary=filtered_summary, artifacts=artifacts)
