from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from framework.services.serialization import json_dumps

from framework.storage.local import read_run_header
from framework.core.matrix_completion_metric_contract import (
    matrix_completion_final_metrics,
    matrix_completion_sweep_metric_options,
    matrix_completion_trajectory_metrics,
)


DEFAULT_MATCH_FIELDS = [
    "problem_family",
    "data_source",
    "n",
    "n_cols",
    "r_star",
    "m",
    "factor_rank",
    "measurement_type",
    "psd_regime",
    "noncommutativity_level",
    "coherence_level",
    "sampling_regime",
    "mc_spectrum_type",
    "mc_spectrum_decay",
    "mc_factor_balance",
    "mc_factor_imbalance_level",
    "val_fraction",
    "test_fraction",
    "early_stopping_enabled",
    "early_stopping_metric",
    "early_stopping_patience",
    "early_stopping_min_delta",
    "early_stopping_enabled",
    "early_stopping_metric",
    "early_stopping_patience",
    "early_stopping_min_delta",
    "noise_std",
    "seed",
    "init_type",
    "init_scale",
    "optimizer",
    "batch_size",
    "momentum_beta",
    "adam_beta1",
    "adam_beta2",
    "adam_eps",
    "preconditioner_eps",
    "lr_schedule_type",
    "lr_decay_gamma",
    "lr_step_size",
    "lr_min",
    "parameterization",
    "explicit_reg_type",
    "explicit_reg_strength",
    "stopping_mode",
    "grad_norm_tol",
    "objective_tol",
    "divergence_norm_threshold",
    "lr",
    "steps",
]

MATCH_AXIS_FIELDS: dict[str, list[str]] = {
    "learning_rate": ["lr"],
    "learning_rate_schedule": ["lr_schedule_type", "lr_decay_gamma", "lr_step_size", "lr_min"],
    "training_budget": ["steps", "grad_norm_tol", "objective_tol"],
    "random_seed": ["seed"],
    "batch_size": ["batch_size"],
    "optimizer": ["optimizer", "momentum_beta", "adam_beta1", "adam_beta2", "adam_eps", "preconditioner_eps"],
    "parameterization": ["parameterization"],
    "explicit_regularization": ["explicit_reg_type", "explicit_reg_strength"],
    "init_scale": ["init_scale"],
    "stopping_mode": ["stopping_mode"],
    "psd_regime": ["psd_regime", "noncommutativity_level"],
    "noncommutativity_level": ["noncommutativity_level"],
    "coherence_level": ["coherence_level"],
    "sampling_regime": ["sampling_regime"],
    "mc_spectrum_type": ["mc_spectrum_type", "mc_spectrum_decay"],
    "mc_spectrum_decay": ["mc_spectrum_decay"],
    "mc_factor_balance": ["mc_factor_balance", "mc_factor_imbalance_level"],
    "mc_factor_imbalance_level": ["mc_factor_imbalance_level"],
}

PROBLEM_FINAL_METRICS = {
    ("psd_sensing", "synthetic"): [
        "final_objective",
        "final_rel_recon_error",
        "final_trace_X",
        "final_effective_rank",
        "final_stable_rank",
        "final_gradient_norm",
        "final_mean_target_commutator",
        "final_max_target_commutator",
        "final_subspace_alignment",
        "final_subspace_projection_distance",
    ],
    ("matrix_completion", "synthetic"): matrix_completion_final_metrics(data_source="synthetic"),
    ("matrix_completion", "movielens_100k"): matrix_completion_final_metrics(data_source="movielens_100k"),
    ("matrix_completion", "custom_triplets_csv"): matrix_completion_final_metrics(data_source="custom_triplets_csv"),
}

PROBLEM_TRAJECTORY_METRICS = {
    ("psd_sensing", "synthetic"): [
        "objective",
        "rel_recon_error",
        "trace_X",
        "effective_rank",
        "stable_rank",
        "gradient_norm",
        "mean_target_commutator",
        "max_target_commutator",
        "subspace_alignment",
        "subspace_projection_distance",
    ],
    ("matrix_completion", "synthetic"): matrix_completion_trajectory_metrics(data_source="synthetic"),
    ("matrix_completion", "movielens_100k"): matrix_completion_trajectory_metrics(data_source="movielens_100k"),
    ("matrix_completion", "custom_triplets_csv"): matrix_completion_trajectory_metrics(data_source="custom_triplets_csv"),
}


def metrics_for_config(config: dict[str, Any] | None, kind: str) -> list[str]:
    cfg = config or {}
    key = (str(cfg.get("problem_family", "psd_sensing")), str(cfg.get("data_source", "synthetic")))
    if kind == "final":
        return list(PROBLEM_FINAL_METRICS.get(key, PROBLEM_FINAL_METRICS[("psd_sensing", "synthetic")]))
    if kind == "trajectory":
        return list(PROBLEM_TRAJECTORY_METRICS.get(key, PROBLEM_TRAJECTORY_METRICS[("psd_sensing", "synthetic")]))
    raise ValueError(f"Unsupported metric kind: {kind}")


def metrics_for_problem_family(problem_family: str | None, kind: str, data_source: str | None = None) -> list[str]:
    return metrics_for_config({"problem_family": problem_family, "data_source": data_source or "synthetic"}, kind)


def sweep_metric_options(problem_family: str | None, data_source: str | None = None) -> dict[str, str]:
    family = str(problem_family or "psd_sensing")
    source = str(data_source or "synthetic")
    if family == "matrix_completion":
        return matrix_completion_sweep_metric_options(data_source=source)
    return {
        "final_rel_recon_error": "Final relative reconstruction error",
        "final_objective": "Final objective",
        "final_trace_X": "Final trace_X",
        "final_mean_target_commutator": "Final mean target commutator",
        "final_max_target_commutator": "Final max target commutator",
    }

FINAL_METRICS_DEFAULT = PROBLEM_FINAL_METRICS[("psd_sensing", "synthetic")]
TRAJECTORY_METRICS_DEFAULT = PROBLEM_TRAJECTORY_METRICS[("psd_sensing", "synthetic")]
SEED_AGG_FINAL_METRICS = FINAL_METRICS_DEFAULT
SEED_AGG_TRAJECTORY_METRICS = TRAJECTORY_METRICS_DEFAULT

SEED_GROUP_FIELDS = [
    "problem_family",
    "data_source",
    "n",
    "n_cols",
    "r_star",
    "m",
    "factor_rank",
    "measurement_type",
    "psd_regime",
    "noncommutativity_level",
    "coherence_level",
    "sampling_regime",
    "mc_spectrum_type",
    "mc_spectrum_decay",
    "mc_factor_balance",
    "mc_factor_imbalance_level",
    "val_fraction",
    "test_fraction",
    "noise_std",
    "init_type",
    "init_scale",
    "optimizer",
    "lr_schedule_type",
    "lr_decay_gamma",
    "lr_step_size",
    "lr_min",
    "batch_size",
    "momentum_beta",
    "adam_beta1",
    "adam_beta2",
    "adam_eps",
    "preconditioner_eps",
    "parameterization",
    "explicit_reg_type",
    "explicit_reg_strength",
    "stopping_mode",
    "grad_norm_tol",
    "objective_tol",
    "divergence_norm_threshold",
    "lr",
    "steps",
    "log_every",
]


NON_EXPERIMENTAL_CONFIG_FIELDS = {
    # Identifiers and UI bookkeeping are useful for provenance, but they are not
    # experimental factors and should not make a controlled comparison look
    # confounded. They remain visible in raw technical metadata.
    "seed_group_id",
    "group_id",
    "sweep_group_id",
    "run_name",
    "notes",
    "tags",
    "favorite",
}


def _experimental_config_keys(configs: list[dict]) -> list[str]:
    return sorted(
        key
        for key in set().union(*[cfg.keys() for cfg in configs])
        if key not in NON_EXPERIMENTAL_CONFIG_FIELDS
    )


def x_col(df: pd.DataFrame) -> str:
    if "step" in df.columns:
        return "step"
    if len(df.columns) > 0:
        return str(df.columns[0])
    return "step"


def list_run_dirs(runs_dir: Path) -> list[Path]:
    if not runs_dir.exists():
        return []
    return sorted([p for p in runs_dir.iterdir() if p.is_dir()], reverse=True)


def config_diff(configs: list[dict]) -> pd.DataFrame:
    if not configs:
        return pd.DataFrame()
    keys = _experimental_config_keys(configs)
    varying = []
    for k in keys:
        vals = [cfg.get(k, None) for cfg in configs]
        if len(set(map(str, vals))) > 1:
            varying.append(k)
    if not varying:
        return pd.DataFrame()
    rows = []
    for k in varying:
        row = {"field": k}
        for i, cfg in enumerate(configs):
            row[f"run_{i+1}"] = cfg.get(k, None)
        rows.append(row)
    return pd.DataFrame(rows)


def _normalize_value(value: Any) -> Any:
    if isinstance(value, float):
        return float(value)
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_value(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((str(k), _normalize_value(v)) for k, v in value.items()))
    return value


def _config_value(cfg: dict[str, Any], field: str) -> Any:
    return _normalize_value(cfg.get(field, None))


def match_axis_options() -> list[str]:
    return list(MATCH_AXIS_FIELDS.keys())


def match_key_fields(varying_axis: str) -> list[str]:
    varying_fields = set(MATCH_AXIS_FIELDS[varying_axis])
    return [field for field in DEFAULT_MATCH_FIELDS if field not in varying_fields]


def build_match_key(cfg: dict[str, Any], varying_axis: str) -> tuple[tuple[str, Any], ...]:
    return tuple((field, _config_value(cfg, field)) for field in match_key_fields(varying_axis))


def varying_axis_value(cfg: dict[str, Any], varying_axis: str) -> str:
    fields = MATCH_AXIS_FIELDS[varying_axis]
    if len(fields) == 1:
        return str(cfg.get(fields[0], None))
    return " | ".join(f"{field}={cfg.get(field, None)}" for field in fields)


def unexpected_mismatch_fields(reference_cfg: dict[str, Any], candidate_cfg: dict[str, Any], varying_axis: str) -> list[str]:
    mismatches: list[str] = []
    allowed = set(MATCH_AXIS_FIELDS[varying_axis])
    for field in DEFAULT_MATCH_FIELDS:
        if field in allowed:
            continue
        if _config_value(reference_cfg, field) != _config_value(candidate_cfg, field):
            mismatches.append(field)
    return mismatches


def find_matched_runs(
    loaded_runs: list[dict[str, Any]],
    reference_run_id: str,
    varying_axis: str,
    strict_exact_match: bool = True,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if varying_axis not in MATCH_AXIS_FIELDS:
        raise ValueError(f"Unsupported varying axis: {varying_axis}")
    reference = next((item for item in loaded_runs if item["run_id"] == reference_run_id), None)
    if reference is None:
        return None, []
    ref_cfg = reference["config"]
    ref_key = build_match_key(ref_cfg, varying_axis)
    ref_axis_value = varying_axis_value(ref_cfg, varying_axis)

    matches: list[dict[str, Any]] = []
    for item in loaded_runs:
        if item["run_id"] == reference_run_id:
            continue
        cfg = item["config"]
        axis_value = varying_axis_value(cfg, varying_axis)
        mismatches = unexpected_mismatch_fields(ref_cfg, cfg, varying_axis)
        same_axis_value = axis_value == ref_axis_value
        same_key = build_match_key(cfg, varying_axis) == ref_key
        if strict_exact_match:
            if (not same_key) or same_axis_value or mismatches:
                continue
        else:
            if same_axis_value:
                continue
            if not same_key and len(mismatches) > 0:
                continue
        enriched = dict(item)
        enriched["varying_axis"] = varying_axis
        enriched["varying_value"] = axis_value
        enriched["unexpected_mismatches"] = mismatches
        matches.append(enriched)
    matches.sort(key=lambda item: (str(item["varying_value"]), str(item["run_id"])))
    return reference, matches


def matched_candidates_table(reference: dict[str, Any], matches: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in matches:
        summary = item["summary"]
        rows.append(
            {
                "run_id": item["run_id"],
                "varying_value": item["varying_value"],
                "status": summary.get("status"),
                "stop_reason": summary.get("stop_reason"),
                "final_objective": summary.get("final_objective"),
                "final_rel_recon_error": summary.get("final_rel_recon_error"),
                "unexpected_mismatches": ", ".join(item.get("unexpected_mismatches", [])),
            }
        )
    return pd.DataFrame(rows)


def matched_config_diff(reference_cfg: dict[str, Any], candidate_cfg: dict[str, Any], varying_axis: str) -> pd.DataFrame:
    rows = []
    allowed = set(MATCH_AXIS_FIELDS[varying_axis])
    keys = _experimental_config_keys([reference_cfg, candidate_cfg])
    for field in keys:
        ref_val = reference_cfg.get(field, None)
        cand_val = candidate_cfg.get(field, None)
        if _normalize_value(ref_val) == _normalize_value(cand_val):
            continue
        rows.append({"field": field, "reference": ref_val, "candidate": cand_val, "allowed_change": field in allowed})
    return pd.DataFrame(rows)


def matched_final_metrics_table(reference_summary: dict[str, Any], candidate_summary: dict[str, Any], metrics: list[str] | None = None) -> pd.DataFrame:
    metrics = metrics or FINAL_METRICS_DEFAULT
    rows = []
    for metric in metrics:
        ref_val = reference_summary.get(metric)
        cand_val = candidate_summary.get(metric)
        delta = None
        pct_delta = None
        if isinstance(ref_val, (int, float)) and isinstance(cand_val, (int, float)):
            delta = float(cand_val) - float(ref_val)
            if float(ref_val) != 0.0:
                pct_delta = 100.0 * delta / float(ref_val)
        rows.append({"metric": metric, "reference": ref_val, "candidate": cand_val, "delta": delta, "pct_delta": pct_delta})
    return pd.DataFrame(rows)


def parse_seed_text(seed_text: str) -> list[int]:
    parts = [p for p in re.split(r"[\s,;]+", str(seed_text or "").strip()) if p]
    if not parts:
        raise ValueError("Provide at least one seed.")
    seeds: list[int] = []
    seen: set[int] = set()
    for part in parts:
        value = int(part)
        if value < 0:
            raise ValueError("Seeds must be >= 0.")
        if value in seen:
            continue
        seen.add(value)
        seeds.append(value)
    return seeds


def generate_seed_sequence(start_seed: int, num_seeds: int, step: int = 1) -> list[int]:
    if start_seed < 0:
        raise ValueError("start_seed must be >= 0.")
    if num_seeds < 1:
        raise ValueError("num_seeds must be >= 1.")
    if step < 1:
        raise ValueError("seed step must be >= 1.")
    return [int(start_seed) + i * int(step) for i in range(int(num_seeds))]


def resolve_seed_values(seed_mode: str, *, seed: int | None = None, start_seed: int | None = None, num_seeds: int | None = None, seed_step: int = 1, seeds_text: str = "") -> list[int]:
    if seed_mode == "single":
        if seed is None:
            raise ValueError("Single-seed mode requires seed.")
        return [int(seed)]
    if seed_mode == "generated":
        if start_seed is None or num_seeds is None:
            raise ValueError("Generated-seed mode requires start_seed and num_seeds.")
        return generate_seed_sequence(int(start_seed), int(num_seeds), int(seed_step))
    if seed_mode == "manual":
        return parse_seed_text(seeds_text)
    raise ValueError(f"Unsupported seed mode: {seed_mode}")


def _seed_group_key_payload(cfg: dict[str, Any]) -> dict[str, Any]:
    return {field: cfg.get(field) for field in SEED_GROUP_FIELDS}


def _short_hash(obj: dict[str, Any]) -> str:
    payload = json_dumps(obj, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def seed_group_identifier(cfg: dict[str, Any], metadata: dict[str, Any] | None = None) -> str:
    explicit = (metadata or {}).get("seed_group_id") or cfg.get("seed_group_id")
    if explicit:
        return str(explicit)
    return f"derived_{_short_hash(_seed_group_key_payload(cfg))}"


def load_runs_with_data(runs_dir: Path, load_run_fn) -> list[dict[str, Any]]:
    loaded_runs: list[dict[str, Any]] = []
    for rd in list_run_dirs(runs_dir):
        loaded = load_run_fn(str(rd))
        metrics = loaded["metrics"]
        if not isinstance(metrics, pd.DataFrame):
            metrics = pd.DataFrame(metrics)
        loaded_runs.append(
            {
                "run_id": rd.name,
                "config": loaded["config"],
                "summary": loaded["summary"],
                "metrics": metrics,
                "metadata": loaded.get("metadata", {}),
            }
        )
    return loaded_runs


def group_runs_by_seed(loaded_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in loaded_runs:
        groups[seed_group_identifier(item["config"], item.get("metadata"))].append(item)

    out: list[dict[str, Any]] = []
    for seed_group_id, items in groups.items():
        items = sorted(items, key=lambda item: (int(item["config"].get("seed", 0)), str(item["run_id"])))
        seeds = [int(item["config"].get("seed", item.get("metadata", {}).get("seed", 0))) for item in items]
        if len(set(seeds)) < 2:
            continue
        representative = items[0]
        out.append(
            {
                "seed_group_id": seed_group_id,
                "group_id": representative["config"].get("group_id"),
                "items": items,
                "seed_count": len(set(seeds)),
                "run_count": len(items),
                "seeds": sorted(set(seeds)),
                "config": representative["config"],
            }
        )
    out.sort(key=lambda group: str(group["seed_group_id"]))
    return out


def seed_groups_table(groups: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for group in groups:
        cfg = group["config"]
        rows.append(
            {
                "seed_group_id": group["seed_group_id"],
                "group_id": group["group_id"],
                "seed_count": group["seed_count"],
                "problem_family": cfg.get("problem_family"),
                "optimizer": cfg.get("optimizer"),
                "parameterization": cfg.get("parameterization"),
                "init_scale": cfg.get("init_scale"),
                "lr": cfg.get("lr"),
                "explicit_reg_type": cfg.get("explicit_reg_type"),
                "stopping_mode": cfg.get("stopping_mode"),
                "seeds": ", ".join(str(seed) for seed in group["seeds"]),
            }
        )
    return pd.DataFrame(rows)


def aggregate_seed_group_final_metrics(group: dict[str, Any], metrics: list[str] | None = None) -> pd.DataFrame:
    metrics = metrics or SEED_AGG_FINAL_METRICS
    rows = []
    for metric in metrics:
        values = []
        for item in group["items"]:
            value = item["summary"].get(metric)
            if isinstance(value, (int, float)):
                values.append(float(value))
        if not values:
            continue
        series = pd.Series(values, dtype=float)
        mean = float(series.mean())
        std = float(series.std(ddof=0))
        seed_count = int(series.shape[0])
        sem = float(std / (seed_count ** 0.5)) if seed_count > 0 else 0.0
        cv = float(abs(std / mean)) if mean != 0.0 else None
        min_value = float(series.min())
        max_value = float(series.max())
        rows.append(
            {
                "metric": metric,
                "mean": mean,
                "std": std,
                "sem": sem,
                "cv": cv,
                "median": float(series.median()),
                "min": min_value,
                "max": max_value,
                "range": float(max_value - min_value),
                "seed_count": seed_count,
            }
        )
    return pd.DataFrame(rows)


def aggregate_seed_group_trajectories(group: dict[str, Any], metrics: list[str] | None = None) -> pd.DataFrame:
    metrics = metrics or SEED_AGG_TRAJECTORY_METRICS
    rows: list[dict[str, Any]] = []
    for item in group["items"]:
        df = item["metrics"]
        if not isinstance(df, pd.DataFrame) or "step" not in df.columns:
            continue
        for metric in metrics:
            if metric not in df.columns:
                continue
            small = df[["step", metric]].copy()
            small["metric"] = metric
            small["run_id"] = item["run_id"]
            small = small.rename(columns={metric: "value"})
            rows.extend(small.to_dict(orient="records"))
    if not rows:
        return pd.DataFrame(columns=["metric", "step", "mean", "std", "min", "max", "seed_count"])
    big = pd.DataFrame(rows)
    grouped = big.groupby(["metric", "step"], as_index=False).agg(
        mean=("value", "mean"),
        std=("value", lambda s: float(pd.Series(s, dtype=float).std(ddof=0))),
        min=("value", "min"),
        max=("value", "max"),
        seed_count=("run_id", "nunique"),
    )
    return grouped.sort_values(["metric", "step"]).reset_index(drop=True)


def list_groups(runs_dir: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for rd in list_run_dirs(runs_dir):
        try:
            header = read_run_header(str(rd))
        except Exception:
            continue
        gid = header["config"].get("group_id") or header["metadata"].get("group_id")
        if gid:
            groups[str(gid)].append(rd)
    return dict(groups)
