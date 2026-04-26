from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any

import pandas as pd

from framework.services.local_runs import sweep_metric_options
from framework.services.sweep_repository import SweepRepository
from framework.core.models import SweepSpec
from framework.services.sweeps import SweepExecutionResult, load_saved_sweep_view


SWEEP_ANALYSIS_SUMMARY_COLUMNS = [
    "group_id",
    "kind",
    "fields",
    "preset",
    "seed_count",
    "run_count",
    "failure_count",
]


@dataclass(frozen=True)
class SweepAnalysisView:
    group_id: str
    kind: str | None
    fields: tuple[str, ...]
    metadata: dict[str, Any]
    summary: dict[str, Any]
    contract: dict[str, Any]
    confound_warnings: list[dict[str, str]]
    raw_df: pd.DataFrame
    aggregate_df: pd.DataFrame
    metric_options: dict[str, str]


def _safe_std(series: pd.Series) -> float:
    return float(pd.Series(series, dtype=float).std(ddof=0))


def _expected_seed_count(metadata: dict[str, Any], sweep_spec: SweepSpec) -> int:
    payload = metadata.get("expected_seed_count")
    if payload is not None:
        return int(payload)
    return int(len(sweep_spec.seeds))


def _expected_cell_count(sweep_spec: SweepSpec) -> int:
    return int(prod(len(axis.values) for axis in sweep_spec.axes))


def _cell_balance_frame(raw_df: pd.DataFrame, fields: tuple[str, ...], expected_seed_count: int) -> pd.DataFrame:
    if raw_df.empty or not fields:
        columns = list(fields) + ["seed_count", "run_count", "seed_balance_status"]
        return pd.DataFrame(columns=columns)
    grouped = (
        raw_df.groupby(list(fields), dropna=False, as_index=False)
        .agg(seed_count=("seed", "nunique"), run_count=("run", "count"))
        .reset_index(drop=True)
    )
    grouped["seed_balance_status"] = grouped["seed_count"].apply(
        lambda count: "balanced" if int(count) == int(expected_seed_count) else "incomplete"
    )
    return grouped


def aggregate_sweep_1d(raw_df: pd.DataFrame, field: str, metric_keys: list[str], *, expected_seed_count: int) -> pd.DataFrame:
    agg_spec: dict[str, tuple[str, Any]] = {"seed_count": ("seed", "nunique"), "run_count": ("run", "count")}
    present_metrics: list[str] = []
    for metric in metric_keys:
        if metric not in raw_df.columns:
            continue
        present_metrics.append(metric)
        agg_spec[f"{metric}_mean"] = (metric, "mean")
        agg_spec[f"{metric}_std"] = (metric, _safe_std)
        agg_spec[f"{metric}_min"] = (metric, "min")
        agg_spec[f"{metric}_max"] = (metric, "max")
        agg_spec[f"{metric}_median"] = (metric, "median")
    aggregate_df = raw_df.groupby(field, dropna=False, as_index=False).agg(**agg_spec).sort_values(field).reset_index(drop=True)
    aggregate_df["expected_seed_count"] = int(expected_seed_count)
    aggregate_df["seed_balance_status"] = aggregate_df["seed_count"].apply(
        lambda count: "balanced" if int(count) == int(expected_seed_count) else "incomplete"
    )
    aggregate_df["seed_coverage_ratio"] = aggregate_df["seed_count"].apply(
        lambda count: float(count) / float(expected_seed_count) if expected_seed_count > 0 else 0.0
    )
    for metric in present_metrics:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        min_col = f"{metric}_min"
        max_col = f"{metric}_max"
        aggregate_df[f"{metric}_sem"] = aggregate_df.apply(
            lambda row: float(row[std_col]) / (float(row["seed_count"]) ** 0.5) if float(row["seed_count"]) > 0 else 0.0,
            axis=1,
        )
        aggregate_df[f"{metric}_cv"] = aggregate_df.apply(
            lambda row: abs(float(row[std_col]) / float(row[mean_col])) if float(row[mean_col]) != 0.0 else None,
            axis=1,
        )
        aggregate_df[f"{metric}_range"] = aggregate_df.apply(
            lambda row: float(row[max_col]) - float(row[min_col]),
            axis=1,
        )
    return aggregate_df


def aggregate_sweep_2d(raw_df: pd.DataFrame, field_x: str, field_y: str, metric_keys: list[str], *, expected_seed_count: int) -> pd.DataFrame:
    agg_spec: dict[str, tuple[str, Any]] = {"seed_count": ("seed", "nunique"), "run_count": ("run", "count")}
    present_metrics: list[str] = []
    for metric in metric_keys:
        if metric not in raw_df.columns:
            continue
        present_metrics.append(metric)
        agg_spec[f"{metric}_mean"] = (metric, "mean")
        agg_spec[f"{metric}_std"] = (metric, _safe_std)
        agg_spec[f"{metric}_min"] = (metric, "min")
        agg_spec[f"{metric}_max"] = (metric, "max")
        agg_spec[f"{metric}_median"] = (metric, "median")
    aggregate_df = raw_df.groupby([field_x, field_y], dropna=False, as_index=False).agg(**agg_spec).reset_index(drop=True)
    aggregate_df["expected_seed_count"] = int(expected_seed_count)
    aggregate_df["seed_balance_status"] = aggregate_df["seed_count"].apply(
        lambda count: "balanced" if int(count) == int(expected_seed_count) else "incomplete"
    )
    aggregate_df["seed_coverage_ratio"] = aggregate_df["seed_count"].apply(
        lambda count: float(count) / float(expected_seed_count) if expected_seed_count > 0 else 0.0
    )
    for metric in present_metrics:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        min_col = f"{metric}_min"
        max_col = f"{metric}_max"
        aggregate_df[f"{metric}_sem"] = aggregate_df.apply(
            lambda row: float(row[std_col]) / (float(row["seed_count"]) ** 0.5) if float(row["seed_count"]) > 0 else 0.0,
            axis=1,
        )
        aggregate_df[f"{metric}_cv"] = aggregate_df.apply(
            lambda row: abs(float(row[std_col]) / float(row[mean_col])) if float(row[mean_col]) != 0.0 else None,
            axis=1,
        )
        aggregate_df[f"{metric}_range"] = aggregate_df.apply(
            lambda row: float(row[max_col]) - float(row[min_col]),
            axis=1,
        )
    return aggregate_df


def _build_metric_options(problem_family: str | None, data_source: str | None, raw_df: pd.DataFrame) -> dict[str, str]:
    options = sweep_metric_options(problem_family, data_source)
    if raw_df.empty:
        return options
    return {key: label for key, label in options.items() if key in raw_df.columns}


def _resolve_sweep_spec(metadata: dict[str, Any]) -> SweepSpec:
    return SweepSpec.from_dict(metadata.get("sweep_spec") or metadata)


def _build_sweep_summary(group_id: str, metadata: dict[str, Any], raw_df: pd.DataFrame) -> dict[str, Any]:
    sweep_spec = _resolve_sweep_spec(metadata)
    expected_seed_count = _expected_seed_count(metadata, sweep_spec)
    expected_cell_count = int(metadata.get("expected_cell_count") or _expected_cell_count(sweep_spec))
    expected_run_count = int(metadata.get("expected_run_count") or (expected_seed_count * expected_cell_count))
    cell_balance_df = _cell_balance_frame(raw_df, sweep_spec.fields, expected_seed_count)
    observed_cell_count = int(len(cell_balance_df.index))
    incomplete_cell_count = int((cell_balance_df.get("seed_balance_status") == "incomplete").sum()) if not cell_balance_df.empty else 0
    balanced_cell_count = int((cell_balance_df.get("seed_balance_status") == "balanced").sum()) if not cell_balance_df.empty else 0
    missing_cell_count = max(0, expected_cell_count - observed_cell_count)
    if expected_cell_count == 0:
        seed_balance_status = "empty"
    elif missing_cell_count == 0 and incomplete_cell_count == 0 and observed_cell_count == expected_cell_count:
        seed_balance_status = "balanced"
    else:
        seed_balance_status = "incomplete"
    return {
        "group_id": group_id,
        "kind": sweep_spec.kind,
        "fields": list(sweep_spec.fields),
        "varied_factors": list(metadata.get("varied_factors") or sweep_spec.fields),
        "fixed_factor_keys": list(metadata.get("fixed_factor_keys") or sorted(str(key) for key in sweep_spec.fixed_factors.keys())),
        "preset": sweep_spec.preset,
        "seed_count": int(pd.Series(raw_df.get("seed", pd.Series(dtype=float))).dropna().nunique()) if not raw_df.empty and "seed" in raw_df.columns else expected_seed_count,
        "expected_seed_count": expected_seed_count,
        "run_count": int(len(raw_df.index)),
        "observed_run_count": int(len(raw_df.index)),
        "expected_run_count": expected_run_count,
        "expected_cell_count": expected_cell_count,
        "observed_cell_count": observed_cell_count,
        "missing_cell_count": missing_cell_count,
        "balanced_cell_count": balanced_cell_count,
        "incomplete_cell_count": incomplete_cell_count,
        "seed_balance_status": seed_balance_status,
        "failure_count": int(metadata.get("failure_count", 0) or 0),
    }


def _build_sweep_contract(metadata: dict[str, Any], summary: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    fields = tuple(str(field) for field in summary.get("fields", []))
    protocol_manipulated_factor = str(metadata.get("protocol_manipulated_factor") or "").strip()
    protocol_held_fixed = {str(item) for item in metadata.get("protocol_held_fixed_factors", [])}
    warnings: list[dict[str, str]] = []

    if protocol_manipulated_factor and tuple([protocol_manipulated_factor]) != fields:
        warnings.append({
            "code": "protocol_manipulated_factor_mismatch",
            "severity": "error",
            "message": f"Protocol declares '{protocol_manipulated_factor}' as the manipulated factor, but the sweep varies {list(fields)}.",
        })

    violated_held_fixed = sorted(field for field in fields if field in protocol_held_fixed)
    if violated_held_fixed:
        warnings.append({
            "code": "protocol_held_fixed_violation",
            "severity": "error",
            "message": f"Sweep varies protocol held-fixed factor(s): {', '.join(violated_held_fixed)}.",
        })

    if int(summary.get("failure_count", 0) or 0) > 0:
        warnings.append({
            "code": "failed_runs_present",
            "severity": "warning",
            "message": f"Sweep has {int(summary['failure_count'])} failed run(s).",
        })

    if int(summary.get("missing_cell_count", 0) or 0) > 0:
        warnings.append({
            "code": "missing_sweep_cells",
            "severity": "warning",
            "message": f"Sweep is missing {int(summary['missing_cell_count'])} cell(s) relative to the declared grid.",
        })

    if int(summary.get("incomplete_cell_count", 0) or 0) > 0:
        warnings.append({
            "code": "unbalanced_seed_cells",
            "severity": "warning",
            "message": f"{int(summary['incomplete_cell_count'])} sweep cell(s) do not contain the full expected seed set.",
        })

    if any(item["severity"] == "error" for item in warnings):
        status = "protocol_conflict"
        label = "Protocol conflict"
    elif any(item["code"] in {"missing_sweep_cells", "unbalanced_seed_cells", "failed_runs_present"} for item in warnings):
        status = "partially_observed"
        label = "Partially observed"
    elif len(fields) > 1:
        status = "interaction_exploration"
        label = "Interaction sweep"
    else:
        status = "controlled_single_factor"
        label = "Controlled single-factor sweep"

    contract = {
        "status": status,
        "label": label,
        "varied_factors": list(fields),
        "fixed_factor_keys": list(summary.get("fixed_factor_keys", [])),
        "expected_seed_count": int(summary.get("expected_seed_count", 0) or 0),
        "seed_balance_status": str(summary.get("seed_balance_status") or ""),
    }
    return contract, warnings


def build_sweep_analysis_view(
    *,
    group_id: str,
    metadata: dict[str, Any],
    raw_df: pd.DataFrame,
    metric_options: dict[str, str],
) -> SweepAnalysisView:
    sweep_spec = _resolve_sweep_spec(metadata)
    kind = sweep_spec.kind
    fields = sweep_spec.fields
    metric_keys = list(metric_options.keys())
    expected_seed_count = _expected_seed_count(metadata, sweep_spec)
    if kind == "1D" and len(fields) >= 1 and not raw_df.empty:
        aggregate_df = aggregate_sweep_1d(raw_df, fields[0], metric_keys, expected_seed_count=expected_seed_count)
    elif kind == "2D" and len(fields) >= 2 and not raw_df.empty:
        aggregate_df = aggregate_sweep_2d(raw_df, fields[0], fields[1], metric_keys, expected_seed_count=expected_seed_count)
    else:
        aggregate_df = pd.DataFrame()
    summary = _build_sweep_summary(group_id, metadata, raw_df)
    contract, confound_warnings = _build_sweep_contract(metadata, summary)
    return SweepAnalysisView(
        group_id=group_id,
        kind=str(kind) if kind is not None else None,
        fields=fields,
        metadata=dict(metadata),
        summary=summary,
        contract=contract,
        confound_warnings=confound_warnings,
        raw_df=raw_df.copy(),
        aggregate_df=aggregate_df,
        metric_options=dict(metric_options),
    )


def build_sweep_analysis_from_execution_result(
    result: SweepExecutionResult,
    *,
    problem_family: str,
    data_source: str,
) -> SweepAnalysisView:
    metadata = dict(result.metadata)
    metadata["failure_count"] = result.failed_count
    metric_options = _build_metric_options(problem_family, data_source, result.raw_df)
    return build_sweep_analysis_view(
        group_id=result.group_id,
        metadata=metadata,
        raw_df=result.raw_df,
        metric_options=metric_options,
    )


def load_saved_sweep_analysis(group_id: str, repository: SweepRepository) -> SweepAnalysisView | None:
    saved_view = load_saved_sweep_view(group_id, repository)
    if saved_view is None:
        return None
    base_cfg = dict(saved_view.metadata.get("base_config") or saved_view.sweep_spec.fixed_factors)
    metric_options = _build_metric_options(
        base_cfg.get("problem_family"),
        base_cfg.get("data_source"),
        saved_view.raw_df,
    )
    return build_sweep_analysis_view(
        group_id=saved_view.group_id,
        metadata=saved_view.metadata,
        raw_df=saved_view.raw_df,
        metric_options=metric_options,
    )


__all__ = [
    "SWEEP_ANALYSIS_SUMMARY_COLUMNS",
    "SweepAnalysisView",
    "aggregate_sweep_1d",
    "aggregate_sweep_2d",
    "build_sweep_analysis_view",
    "build_sweep_analysis_from_execution_result",
    "load_saved_sweep_analysis",
]
