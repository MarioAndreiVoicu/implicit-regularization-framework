from __future__ import annotations

from typing import Any

import pandas as pd

from framework.core.matrix_completion_metric_contract import matrix_completion_source_kind
from framework.services.local_runs import (
    find_matched_runs,
    load_runs_with_data,
    match_axis_options,
)
from framework.services.seed_group_records import build_seed_group_analysis_items, seed_group_final_metrics

FIT_METRICS_BY_KEY: dict[tuple[str, str], list[str]] = {
    ("psd_sensing", "synthetic"): ["final_rel_recon_error", "final_objective"],
    ("matrix_completion", "synthetic"): ["final_rel_recon_error", "final_objective"],
    ("matrix_completion", "real"): ["final_val_rmse", "final_test_rmse", "final_objective"],
}

COMPLEXITY_METRICS_BY_KEY: dict[tuple[str, str], list[str]] = {
    ("psd_sensing", "synthetic"): ["final_trace_X", "final_effective_rank", "final_stable_rank", "final_frobenius_norm_X"],
    ("matrix_completion", "synthetic"): ["final_frobenius_norm_X", "final_effective_rank", "final_stable_rank"],
    ("matrix_completion", "real"): ["final_frobenius_norm_X", "final_effective_rank", "final_stable_rank"],
}

METRIC_LABELS = {
    "final_rel_recon_error": "Final relative reconstruction error",
    "final_objective": "Final objective",
    "final_val_rmse": "Final validation RMSE",
    "final_test_rmse": "Final test RMSE",
    "final_trace_X": "Final trace_X",
    "final_frobenius_norm_X": "Final Frobenius norm of X",
    "final_effective_rank": "Final effective rank",
    "final_stable_rank": "Final stable rank",
}

COLOR_BY_OPTIONS = [
    "optimizer",
    "parameterization",
    "explicit_reg_type",
    "stopping_mode",
    "problem_family",
    "data_source",
]


def frontier_metric_options(configs: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    fit: list[str] = []
    complexity: list[str] = []
    for cfg in configs:
        family = str(cfg.get("problem_family", "psd_sensing"))
        source_key = str(cfg.get("data_source", "synthetic"))
        if family == "matrix_completion":
            source_key = matrix_completion_source_kind(source_kind=cfg.get("source_kind"), data_source=cfg.get("data_source"))
        key = (family, source_key)
        for metric in FIT_METRICS_BY_KEY.get(key, []):
            if metric not in fit:
                fit.append(metric)
        for metric in COMPLEXITY_METRICS_BY_KEY.get(key, []):
            if metric not in complexity:
                complexity.append(metric)
    return fit, complexity


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric)


def frontier_run_points(loaded_runs: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in loaded_runs:
        cfg = item["config"]
        summary = item["summary"]
        rows.append(
            {
                "point_id": item["run_id"],
                "point_type": "run",
                "run_id": item["run_id"],
                "seed_group_id": cfg.get("seed_group_id") or item.get("metadata", {}).get("seed_group_id"),
                "problem_family": cfg.get("problem_family", "psd_sensing"),
                "data_source": cfg.get("data_source", "synthetic"),
                "optimizer": cfg.get("optimizer"),
                "parameterization": cfg.get("parameterization"),
                "explicit_reg_type": cfg.get("explicit_reg_type"),
                "stopping_mode": cfg.get("stopping_mode"),
                "init_scale": cfg.get("init_scale"),
                "lr": cfg.get("lr"),
                "status": summary.get("status"),
                **{metric: summary.get(metric) for metric in set(sum(FIT_METRICS_BY_KEY.values(), [])) | set(sum(COMPLEXITY_METRICS_BY_KEY.values(), []))},
            }
        )
    return pd.DataFrame(rows)


def frontier_seed_group_points(loaded_runs: list[dict[str, Any]]) -> pd.DataFrame:
    groups = build_seed_group_analysis_items(loaded_runs)
    rows: list[dict[str, Any]] = []
    for group in groups:
        record = group["record"]
        agg = seed_group_final_metrics(group)
        metric_map = {row["metric"]: row["mean"] for row in agg.to_dict(orient="records")}
        rows.append(
            {
                "point_id": group["seed_group_id"],
                "point_type": "seed_group",
                "run_id": None,
                "seed_group_id": group["seed_group_id"],
                "problem_family": record.get("problem_family", "psd_sensing"),
                "data_source": record.get("data_source", "synthetic"),
                "optimizer": record.get("optimizer"),
                "parameterization": record.get("parameterization"),
                "explicit_reg_type": record.get("explicit_reg_type"),
                "stopping_mode": record.get("stopping_mode"),
                "init_scale": record.get("init_scale"),
                "lr": record.get("lr"),
                "status": record.get("status"),
                "seed_count": record.get("seed_count", len(group.get("items", []))),
                **metric_map,
            }
        )
    return pd.DataFrame(rows)


def frontier_matched_points(loaded_runs: list[dict[str, Any]], reference_run_id: str, varying_axis: str, strict_exact_match: bool = True) -> pd.DataFrame:
    reference, matches = find_matched_runs(loaded_runs, reference_run_id, varying_axis, strict_exact_match)
    if reference is None:
        return pd.DataFrame()
    subset = [reference] + matches
    df = frontier_run_points(subset)
    if df.empty:
        return df
    df["matched_reference_run"] = reference_run_id
    df["varying_axis"] = varying_axis
    return df


def pareto_frontier(df: pd.DataFrame, x_metric: str, y_metric: str) -> pd.DataFrame:
    if df.empty or x_metric not in df.columns or y_metric not in df.columns:
        return pd.DataFrame(columns=df.columns)
    small = df.dropna(subset=[x_metric, y_metric]).copy()
    if small.empty:
        return small
    small = small.sort_values([x_metric, y_metric], ascending=[True, True]).reset_index(drop=True)
    best_y = float("inf")
    keep_idx: list[int] = []
    for idx, row in small.iterrows():
        y = float(row[y_metric])
        if y <= best_y:
            keep_idx.append(idx)
            best_y = y
    return small.loc[keep_idx].copy()


__all__ = [
    "COLOR_BY_OPTIONS",
    "frontier_metric_options",
    "frontier_run_points",
    "frontier_seed_group_points",
    "frontier_matched_points",
    "match_axis_options",
    "metric_label",
    "pareto_frontier",
    "load_runs_with_data",
]
