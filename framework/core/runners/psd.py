from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch

from framework.core.metrics import rel_frob_err, spectral_metrics_symmetric_matrix, subspace_alignment_metrics_symmetric
from framework.core.models import RunResult, RunSpec, RunStatus
from framework.core.optimizers import (
    apply_optimizer_step,
    current_learning_rate,
    init_optimizer_state,
    minibatch_indices,
    optimizer_kind,
    zero_grad,
)
from framework.core.parameterization import (
    initialize_psd_parameters,
    parameter_artifacts,
    project_parameters_inplace,
    psd_model_x,
)
from framework.core.problems.psd_sensing import make_psd_sensing_problem
from framework.core.regularization import explicit_reg_penalty


def _data_loss_psd(x: torch.Tensor, a: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    preds = torch.einsum("mij,ij->m", a, x)
    resid = preds - y
    return 0.5 * torch.mean(resid**2)


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


def _parameter_delta_norm(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> float:
    total = 0.0
    for name in before.keys():
        delta = after[name] - before[name]
        delta_norm = float(torch.linalg.norm(delta).item())
        total += delta_norm * delta_norm
    return math.sqrt(total)


def _metrics_template(num_eigs: int) -> dict[str, list[float]]:
    metrics: dict[str, list[float]] = {
        "step": [],
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
        "trace_X": [],
        "frobenius_norm_X": [],
        "effective_rank": [],
        "stable_rank": [],
        "min_eigenvalue_X": [],
        "negative_eigen_mass": [],
        "mean_target_commutator": [],
        "max_target_commutator": [],
        "subspace_alignment": [],
        "subspace_projection_distance": [],
    }
    for i in range(num_eigs):
        metrics[f"eig_{i + 1}"] = []
        metrics[f"sv_{i + 1}"] = []
    return metrics


def _append_metrics(
    metrics: dict[str, list[float]],
    *,
    step: int,
    data_loss_val: float,
    reg_term_val: float,
    objective_val: float,
    lr_val: float,
    gradient_norm_val: float,
    update_norm_val: float,
    parameter_norm_val: float,
    objective_delta_val: float,
    err_val: float,
    spectral: dict[str, float | list[float]],
    mean_target_commutator_val: float,
    max_target_commutator_val: float,
    subspace_alignment_val: float,
    subspace_projection_distance_val: float,
) -> None:
    metrics["step"].append(float(step))
    metrics["data_loss"].append(data_loss_val)
    metrics["loss"].append(data_loss_val)
    metrics["explicit_reg_term"].append(reg_term_val)
    metrics["objective"].append(objective_val)
    metrics["lr"].append(lr_val)
    metrics["gradient_norm"].append(gradient_norm_val)
    metrics["update_norm"].append(update_norm_val)
    metrics["parameter_norm"].append(parameter_norm_val)
    metrics["objective_delta"].append(objective_delta_val)
    metrics["relative_reconstruction_error"].append(err_val)
    metrics["rel_recon_error"].append(err_val)
    metrics["trace_X"].append(float(spectral["trace_X"]))
    metrics["frobenius_norm_X"].append(float(spectral["frobenius_norm_X"]))
    metrics["effective_rank"].append(float(spectral["effective_rank"]))
    metrics["stable_rank"].append(float(spectral["stable_rank"]))
    metrics["min_eigenvalue_X"].append(float(spectral["min_eigenvalue_X"]))
    metrics["negative_eigen_mass"].append(float(spectral["negative_eigen_mass"]))
    metrics["mean_target_commutator"].append(float(mean_target_commutator_val))
    metrics["max_target_commutator"].append(float(max_target_commutator_val))
    metrics["subspace_alignment"].append(float(subspace_alignment_val))
    metrics["subspace_projection_distance"].append(float(subspace_projection_distance_val))
    for i, value in enumerate(spectral["top_k_eigenvalues"]):
        metrics[f"eig_{i + 1}"].append(float(value))
    for i, value in enumerate(spectral["top_k_singular_values"]):
        metrics[f"sv_{i + 1}"].append(float(value))





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
        "objective": float(metrics["objective"][-1]),
        "data_loss": float(metrics["data_loss"][-1]),
        "relative_reconstruction_error": float(metrics["relative_reconstruction_error"][-1]),
        "gradient_norm": float(metrics["gradient_norm"][-1]),
        "lr": float(metrics["lr"][-1]),
    }
    history_keys = ["step", "objective", "data_loss", "relative_reconstruction_error", "gradient_norm", "lr"]
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

def _full_objective(
    *,
    parameters: dict[str, torch.Tensor],
    spec: RunSpec,
    A: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = psd_model_x(parameters, spec.parameterization)
    data_loss = _data_loss_psd(x, A, y)
    reg_term = explicit_reg_penalty(
        parameters=parameters,
        parameterization=spec.parameterization,
        explicit_reg_type=spec.explicit_reg_type,
        explicit_reg_strength=spec.explicit_reg_strength,
    )
    objective = data_loss + reg_term
    return x, data_loss, reg_term, objective



def _update_objective(
    *,
    parameters: dict[str, torch.Tensor],
    spec: RunSpec,
    problem: any,
    opt_kind: str,
    opt_state,
) -> torch.Tensor:
    if opt_kind == "minibatch":
        idx = minibatch_indices(
            m=spec.m,
            batch_size=spec.batch_size,
            state=opt_state,
            device=problem.A.device,
        )
        x = psd_model_x(parameters, spec.parameterization)
        data_loss = _data_loss_psd(x, problem.A[idx], problem.y[idx])
        reg_term = explicit_reg_penalty(
            parameters=parameters,
            parameterization=spec.parameterization,
            explicit_reg_type=spec.explicit_reg_type,
            explicit_reg_strength=spec.explicit_reg_strength,
        )
        return data_loss + reg_term

    _, data_loss, reg_term, _ = _full_objective(
        parameters=parameters,
        spec=spec,
        A=problem.A,
        y=problem.y,
    )
    return data_loss + reg_term



def run_psd_optimizer(
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

    if spec.problem_family != "psd_sensing":
        raise ValueError(f"run_psd_optimizer supports only problem_family='psd_sensing', got {spec.problem_family!r}.")

    problem = make_psd_sensing_problem(
        seed=spec.seed,
        n=spec.n,
        r_star=spec.r_star,
        m=spec.m,
        noise_std=spec.noise_std,
        measurement_type=spec.measurement_type,
        psd_regime=spec.psd_regime,
        noncommutativity_level=spec.noncommutativity_level,
        device=device,
        dtype=dtype,
    )

    parameters = initialize_psd_parameters(
        seed=spec.seed + 12345,
        n=spec.n,
        factor_rank=spec.factor_rank,
        init_scale=spec.init_scale,
        init_type=spec.init_type,
        parameterization=spec.parameterization,
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
    last_spectral: dict[str, float | list[float]] | None = None

    for t in range(spec.steps + 1):
        zero_grad(parameters)
        x, data_loss, reg_term, objective = _full_objective(
            parameters=parameters,
            spec=spec,
            A=problem.A,
            y=problem.y,
        )

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

        invalid_grad = any(
            tensor.grad is None or (not torch.isfinite(tensor.grad).all()) for tensor in parameters.values()
        )
        if invalid_grad:
            status = RunStatus.DIVERGED
            diverged_step = t
            stop_reason = "nonfinite_gradient"
            break

        data_loss_val = float(data_loss.item())
        reg_term_val = float(reg_term.item())
        objective_val = float(objective.item())
        gradient_norm_val = _total_gradient_norm(parameters)
        objective_delta_val = 0.0 if prev_objective_val is None else abs(objective_val - prev_objective_val)
        err_val = rel_frob_err(x, problem.X_star)
        spectral = spectral_metrics_symmetric_matrix(x, spec.num_eigs)
        alignment = subspace_alignment_metrics_symmetric(x, problem.X_star, spec.r_star)
        last_spectral = spectral

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
                data_loss_val=data_loss_val,
                reg_term_val=reg_term_val,
                objective_val=objective_val,
                lr_val=current_learning_rate(spec=spec, step_index=t),
                gradient_norm_val=gradient_norm_val,
                update_norm_val=last_update_norm,
                parameter_norm_val=parameter_norm_val,
                objective_delta_val=objective_delta_val,
                err_val=err_val,
                spectral=spectral,
                mean_target_commutator_val=problem.mean_target_commutator,
                max_target_commutator_val=problem.max_target_commutator,
                subspace_alignment_val=alignment["subspace_alignment"],
                subspace_projection_distance_val=alignment["subspace_projection_distance"],
            )
            if err_val < best_err:
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

        if should_stop_for_convergence:
            converged = True
            converged_step = t
            stop_reason = "converged"
            zero_grad(parameters)
            break

        if t == spec.steps:
            stop_reason = "fixed_steps_complete" if spec.stopping_mode == "fixed_steps" else "max_steps"
            zero_grad(parameters)
            break

        zero_grad(parameters)
        update_objective = _update_objective(
            parameters=parameters,
            spec=spec,
            problem=problem,
            opt_kind=opt_kind,
            opt_state=opt_state,
        )
        update_objective.backward()

        invalid_update_grad = any(
            tensor.grad is None or (not torch.isfinite(tensor.grad).all()) for tensor in parameters.values()
        )
        if invalid_update_grad:
            status = RunStatus.DIVERGED
            diverged_step = t
            stop_reason = "nonfinite_gradient"
            break

        before_step = {name: tensor.detach().clone() for name, tensor in parameters.items()}
        with torch.no_grad():
            apply_optimizer_step(
                optimizer=spec.optimizer,
                spec=spec,
                parameters=parameters,
                state=opt_state,
            )
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
    last_metrics = {k: v[-1] for k, v in metrics.items() if v}

    summary = {
        "final_step": int(metrics["step"][-1]),
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
        "final_mean_target_commutator": float(last_metrics["mean_target_commutator"]),
        "final_max_target_commutator": float(last_metrics["max_target_commutator"]),
        "final_subspace_alignment": float(last_metrics["subspace_alignment"]),
        "final_subspace_projection_distance": float(last_metrics["subspace_projection_distance"]),
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
        "converged": bool(converged),
        "converged_step": converged_step,
        "diverged_step": diverged_step,
        "steps_completed": int(metrics["step"][-1]),
    }
    for i in range(spec.num_eigs):
        summary[f"final_eig_{i + 1}"] = float(last_metrics[f"eig_{i + 1}"])
        summary[f"final_sv_{i + 1}"] = float(last_metrics[f"sv_{i + 1}"])

    _emit_progress_update(
        progress_callback,
        step=int(summary["final_step"]),
        max_steps=spec.steps,
        status=status,
        metrics=metrics,
        summary=summary,
    )
    result = RunResult(spec=spec, metrics=metrics, summary=summary, artifacts=artifacts)
    result.validate()
    return result
