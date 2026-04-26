from __future__ import annotations

from typing import Any

import math
import pandas as pd

from framework.core.models import RunStatus
from framework.core.matrix_completion_metric_contract import preferred_matrix_completion_fit_metric
from framework.services.seed_group_records import build_seed_group_analysis_items


ISSUE_PRIORITY = [
    "diverged",
    "invalid",
    "failed",
    "psd_violation",
    "stalled",
    "hit_cap",
    "ok",
]


def preferred_fit_metric(config: dict[str, Any], summary: dict[str, Any], metrics: pd.DataFrame | None = None) -> tuple[str, float | None, float | None]:
    problem_family = str(config.get("problem_family", "psd_sensing"))
    if problem_family == "matrix_completion":
        metric = preferred_matrix_completion_fit_metric(
            source_kind=summary.get("source_kind"),
            data_source=config.get("data_source"),
        )
        summary_key = "final_val_rmse" if metric == "val_rmse" else "final_rel_recon_error"
        return metric, _metric_start(metrics, metric), _summary_float(summary.get(summary_key))
    return "rel_recon_error", _metric_start(metrics, "rel_recon_error"), _summary_float(summary.get("final_rel_recon_error"))


def _summary_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _metric_start(metrics: pd.DataFrame | None, metric: str) -> float | None:
    if metrics is None or metric not in metrics.columns or metrics.empty:
        return None
    series = pd.to_numeric(metrics[metric], errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.iloc[0])


def _relative_drop(start: float | None, end: float | None) -> float | None:
    if start is None or end is None or not math.isfinite(start) or not math.isfinite(end):
        return None
    denom = max(abs(start), 1e-12)
    return float((start - end) / denom)


def analyze_run_instability(config: dict[str, Any], summary: dict[str, Any], metrics: pd.DataFrame | None = None) -> dict[str, Any]:
    status = str(summary.get("status", RunStatus.COMPLETED))
    stop_reason = str(summary.get("stop_reason", ""))
    converged = bool(summary.get("converged", False))
    grad_tol = float(config.get("grad_norm_tol", 1e-10) or 1e-10)
    final_gradient_norm = _summary_float(summary.get("final_gradient_norm"))
    final_update_norm = _summary_float(summary.get("final_update_norm"))
    final_objective_delta = _summary_float(summary.get("final_objective_delta"))
    final_parameter_norm = _summary_float(summary.get("final_parameter_norm"))
    final_min_eigenvalue = _summary_float(summary.get("final_min_eigenvalue_X"))
    final_negative_eigen_mass = _summary_float(summary.get("final_negative_eigen_mass"))
    fit_metric, fit_start, fit_end = preferred_fit_metric(config, summary, metrics)
    fit_drop = _relative_drop(fit_start, fit_end)
    objective_start = _metric_start(metrics, "objective")
    objective_end = _summary_float(summary.get("final_objective"))
    objective_drop = _relative_drop(objective_start, objective_end)

    labels: list[str] = []
    reasons: list[str] = []

    if status == RunStatus.DIVERGED or bool(summary.get("is_diverged", False)):
        labels.append("diverged")
        reasons.append(f"status={status}")
    elif status == RunStatus.INVALID:
        labels.append("invalid")
        reasons.append("run marked invalid")
    elif status == RunStatus.FAILED:
        labels.append("failed")
        reasons.append("run marked failed")

    if config.get("problem_family") == "psd_sensing" and config.get("parameterization") == "symmetric_matrix":
        if (final_negative_eigen_mass is not None and final_negative_eigen_mass > 1e-10) or (
            final_min_eigenvalue is not None and final_min_eigenvalue < -1e-8
        ):
            labels.append("psd_violation")
            reasons.append("direct symmetric parameterization left PSD regime")

    if stop_reason in {"max_steps", "fixed_steps_complete"} and not converged:
        labels.append("hit_cap")
        reasons.append("run hit step cap before convergence")

    tiny_gradient = final_gradient_norm is not None and final_gradient_norm <= max(grad_tol * 10.0, 1e-9)
    tiny_update = final_update_norm is not None and final_update_norm <= 1e-10
    tiny_objective_change = final_objective_delta is not None and final_objective_delta <= max(float(config.get("objective_tol", 1e-12) or 1e-12) * 10.0, 1e-10)
    poor_progress = (
        ((objective_drop is None) or (objective_drop < 1e-3))
        and ((fit_drop is None) or (fit_drop < 1e-3))
    )
    if status == RunStatus.COMPLETED and not converged and poor_progress and (tiny_gradient or tiny_update or tiny_objective_change):
        labels.append("stalled")
        reasons.append("objective/fit barely improved before stopping")

    if not labels:
        labels.append("ok")
        reasons.append("no obvious instability signal")

    primary_issue = next(label for label in ISSUE_PRIORITY if label in labels)
    return {
        "primary_issue": primary_issue,
        "issue_labels": labels,
        "issue_reason": "; ".join(dict.fromkeys(reasons)),
        "fit_metric": fit_metric,
        "fit_start": fit_start,
        "fit_end": fit_end,
        "fit_improvement_ratio": fit_drop,
        "objective_start": objective_start,
        "objective_end": objective_end,
        "objective_improvement_ratio": objective_drop,
        "final_gradient_norm": final_gradient_norm,
        "final_update_norm": final_update_norm,
        "final_parameter_norm": final_parameter_norm,
        "final_objective_delta": final_objective_delta,
        "final_min_eigenvalue_X": final_min_eigenvalue,
        "final_negative_eigen_mass": final_negative_eigen_mass,
        "stop_reason": stop_reason,
        "status": status,
        "converged": converged,
    }


def build_failure_table(loaded_runs: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in loaded_runs:
        config = item.get("config", {})
        summary = item.get("summary", {})
        metrics = item.get("metrics")
        analysis = analyze_run_instability(config, summary, metrics)
        rows.append(
            {
                "run_id": item.get("run_id"),
                "seed_group_id": (item.get("metadata") or {}).get("seed_group_id") or config.get("seed_group_id"),
                "group_id": (item.get("metadata") or {}).get("group_id") or config.get("group_id"),
                "problem_family": config.get("problem_family"),
                "data_source": config.get("data_source"),
                "optimizer": config.get("optimizer"),
                "parameterization": config.get("parameterization"),
                "lr": config.get("lr"),
                "seed": config.get("seed"),
                "status": analysis["status"],
                "stop_reason": analysis["stop_reason"],
                "primary_issue": analysis["primary_issue"],
                "issue_labels": ", ".join(analysis["issue_labels"]),
                "issue_reason": analysis["issue_reason"],
                "fit_metric": analysis["fit_metric"],
                "fit_start": analysis["fit_start"],
                "fit_end": analysis["fit_end"],
                "fit_improvement_ratio": analysis["fit_improvement_ratio"],
                "objective_start": analysis["objective_start"],
                "final_objective": summary.get("final_objective"),
                "objective_improvement_ratio": analysis["objective_improvement_ratio"],
                "final_gradient_norm": analysis["final_gradient_norm"],
                "final_update_norm": analysis["final_update_norm"],
                "final_parameter_norm": analysis["final_parameter_norm"],
                "final_objective_delta": analysis["final_objective_delta"],
                "final_rel_recon_error": summary.get("final_rel_recon_error"),
                "final_val_rmse": summary.get("final_val_rmse"),
                "final_test_rmse": summary.get("final_test_rmse"),
                "final_min_eigenvalue_X": analysis["final_min_eigenvalue_X"],
                "final_negative_eigen_mass": analysis["final_negative_eigen_mass"],
                "converged": analysis["converged"],
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def diagnostics_table(config: dict[str, Any], summary: dict[str, Any], metrics: pd.DataFrame | None = None) -> pd.DataFrame:
    analysis = analyze_run_instability(config, summary, metrics)
    rows = [
        {"field": "primary_issue", "value": analysis["primary_issue"]},
        {"field": "issue_labels", "value": ", ".join(analysis["issue_labels"])},
        {"field": "issue_reason", "value": analysis["issue_reason"]},
        {"field": "status", "value": analysis["status"]},
        {"field": "stop_reason", "value": analysis["stop_reason"]},
        {"field": "converged", "value": analysis["converged"]},
        {"field": "fit_metric", "value": analysis["fit_metric"]},
        {"field": "fit_start", "value": analysis["fit_start"]},
        {"field": "fit_end", "value": analysis["fit_end"]},
        {"field": "fit_improvement_ratio", "value": analysis["fit_improvement_ratio"]},
        {"field": "objective_start", "value": analysis["objective_start"]},
        {"field": "final_objective", "value": summary.get("final_objective")},
        {"field": "objective_improvement_ratio", "value": analysis["objective_improvement_ratio"]},
        {"field": "final_gradient_norm", "value": analysis["final_gradient_norm"]},
        {"field": "final_update_norm", "value": analysis["final_update_norm"]},
        {"field": "final_parameter_norm", "value": analysis["final_parameter_norm"]},
        {"field": "final_objective_delta", "value": analysis["final_objective_delta"]},
    ]
    if config.get("problem_family") == "psd_sensing":
        rows.extend(
            [
                {"field": "final_min_eigenvalue_X", "value": analysis["final_min_eigenvalue_X"]},
                {"field": "final_negative_eigen_mass", "value": analysis["final_negative_eigen_mass"]},
            ]
        )
    return pd.DataFrame(rows)


def instability_seed_group_table(loaded_runs: list[dict[str, Any]]) -> pd.DataFrame:
    groups = build_seed_group_analysis_items(loaded_runs)
    rows: list[dict[str, Any]] = []
    for group in groups:
        analyses = [analyze_run_instability(item["config"], item["summary"], item.get("metrics")) for item in group["items"]]
        counts: dict[str, int] = {}
        for analysis in analyses:
            counts[analysis["primary_issue"]] = counts.get(analysis["primary_issue"], 0) + 1
        record = group["record"]
        rows.append(
            {
                "seed_group_id": group["seed_group_id"],
                "run_count": group["run_count"],
                "seed_count": group["seed_count"],
                "issue_breakdown": ", ".join(f"{k}:{v}" for k, v in sorted(counts.items())),
                "has_problematic_seed": any(label != "ok" for label in counts),
                "problem_family": record.get("problem_family"),
                "optimizer": record.get("optimizer"),
                "parameterization": record.get("parameterization"),
                "group_id": record.get("group_id"),
            }
        )
    return pd.DataFrame(rows)
