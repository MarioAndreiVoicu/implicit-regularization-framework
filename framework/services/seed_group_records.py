from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from framework.services.analysis_records import load_analysis_items
from framework.services.local_runs import (
    aggregate_seed_group_final_metrics,
    aggregate_seed_group_trajectories,
    build_match_key,
    group_runs_by_seed,
    match_axis_options,
    matched_final_metrics_table,
    metrics_for_config,
    unexpected_mismatch_fields,
    varying_axis_value,
)


SEED_GROUP_RECORD_COLUMNS = [
    "seed_group_id",
    "group_id",
    "status",
    "run_count",
    "seed_count",
    "seeds",
    "problem_family",
    "data_source",
    "dataset_version",
    "split_version",
    "preprocessing_version",
    "optimizer",
    "parameterization",
    "explicit_reg_type",
    "stopping_mode",
    "measurement_type",
    "psd_regime",
    "sampling_regime",
    "init_scale",
    "lr",
    "steps",
]


def build_seed_group_analysis_record(group: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(group.get("config", {}))
    items = list(group.get("items", []))
    statuses = [str(item.get("summary", {}).get("status")) for item in items if item.get("summary") is not None]
    if not statuses:
        status = None
    elif len(set(statuses)) == 1:
        status = statuses[0]
    else:
        status = "mixed"
    return {
        "seed_group_id": group.get("seed_group_id"),
        "group_id": group.get("group_id"),
        "status": status,
        "run_count": int(group.get("run_count", len(items)) or 0),
        "seed_count": int(group.get("seed_count", len(group.get("seeds", []))) or 0),
        "seeds": list(group.get("seeds", [])),
        "problem_family": cfg.get("problem_family", "psd_sensing"),
        "data_source": cfg.get("data_source", "synthetic"),
        "dataset_version": cfg.get("dataset_version"),
        "split_version": cfg.get("split_version"),
        "preprocessing_version": cfg.get("preprocessing_version"),
        "optimizer": cfg.get("optimizer"),
        "parameterization": cfg.get("parameterization"),
        "explicit_reg_type": cfg.get("explicit_reg_type"),
        "stopping_mode": cfg.get("stopping_mode"),
        "measurement_type": cfg.get("measurement_type"),
        "psd_regime": cfg.get("psd_regime"),
        "sampling_regime": cfg.get("sampling_regime"),
        "init_scale": cfg.get("init_scale"),
        "lr": cfg.get("lr"),
        "steps": cfg.get("steps"),
    }



def build_seed_group_analysis_item(group: dict[str, Any]) -> dict[str, Any]:
    record = build_seed_group_analysis_record(group)
    return {
        "seed_group_id": record["seed_group_id"],
        "group_id": record["group_id"],
        "record": record,
        "config": group.get("config", {}),
        "items": list(group.get("items", [])),
        "run_count": record["run_count"],
        "seed_count": record["seed_count"],
        "seeds": list(record["seeds"]),
        "run_ids": [item.get("run_id") for item in group.get("items", [])],
        "metadata": _group_metadata(group),
    }



def build_seed_group_analysis_items(loaded_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = group_runs_by_seed(loaded_runs)
    return [build_seed_group_analysis_item(group) for group in groups]



def load_seed_group_analysis_items(
    read_repository,
    *,
    include_metrics: bool = True,
    run_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    loaded_runs = load_analysis_items(read_repository, include_metrics=include_metrics, run_ids=run_ids)
    return build_seed_group_analysis_items(loaded_runs)



def seed_group_records_table(groups: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for group in groups:
        record = dict(group.get("record") or build_seed_group_analysis_record(group))
        rows.append(
            {
                **record,
                "seeds": ", ".join(str(seed) for seed in record.get("seeds", [])),
            }
        )
    if not rows:
        return pd.DataFrame(columns=SEED_GROUP_RECORD_COLUMNS)
    df = pd.DataFrame(rows)
    ordered = [col for col in SEED_GROUP_RECORD_COLUMNS if col in df.columns]
    extras = [col for col in df.columns if col not in ordered]
    return df[ordered + extras].sort_values("seed_group_id").reset_index(drop=True)



def seed_group_final_metrics(group: dict[str, Any], metrics: list[str] | None = None) -> pd.DataFrame:
    raw_group = {
        "seed_group_id": group.get("seed_group_id"),
        "group_id": group.get("group_id"),
        "items": group.get("items", []),
        "seed_count": group.get("seed_count"),
        "run_count": group.get("run_count"),
        "seeds": group.get("seeds", []),
        "config": group.get("config", {}),
    }
    return aggregate_seed_group_final_metrics(raw_group, metrics)



def seed_group_trajectories(group: dict[str, Any], metrics: list[str] | None = None) -> pd.DataFrame:
    raw_group = {
        "seed_group_id": group.get("seed_group_id"),
        "group_id": group.get("group_id"),
        "items": group.get("items", []),
        "seed_count": group.get("seed_count"),
        "run_count": group.get("run_count"),
        "seeds": group.get("seeds", []),
        "config": group.get("config", {}),
    }
    return aggregate_seed_group_trajectories(raw_group, metrics)


def _group_metadata(group: dict[str, Any]) -> dict[str, Any]:
    if isinstance(group.get("metadata"), dict):
        return dict(group["metadata"])
    items = list(group.get("items", []))
    for item in items:
        metadata = item.get("metadata")
        if isinstance(metadata, dict):
            return dict(metadata)
    return {}


def seed_group_compare_config(group: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(group.get("config", {}))
    cfg.pop("seed_group_id", None)
    cfg.pop("group_id", None)
    cfg["seed"] = 0
    return cfg


def find_matched_seed_groups(
    groups: list[dict[str, Any]],
    reference_group_id: str,
    varying_axis: str,
    strict_exact_match: bool = True,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if varying_axis not in match_axis_options():
        raise ValueError(f"Unsupported varying axis: {varying_axis}")
    reference = next((group for group in groups if str(group.get("seed_group_id")) == str(reference_group_id)), None)
    if reference is None:
        return None, []

    ref_cfg = seed_group_compare_config(reference)
    ref_key = build_match_key(ref_cfg, varying_axis)
    ref_axis_value = varying_axis_value(ref_cfg, varying_axis)
    matches: list[dict[str, Any]] = []
    for group in groups:
        if str(group.get("seed_group_id")) == str(reference_group_id):
            continue
        cfg = seed_group_compare_config(group)
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
        enriched = dict(group)
        enriched["compare_config"] = cfg
        enriched["metadata"] = _group_metadata(group)
        enriched["varying_axis"] = varying_axis
        enriched["varying_value"] = axis_value
        enriched["unexpected_mismatches"] = mismatches
        matches.append(enriched)
    matches.sort(key=lambda group: (str(group.get("varying_value")), str(group.get("seed_group_id"))))
    reference_out = dict(reference)
    reference_out["compare_config"] = ref_cfg
    reference_out["metadata"] = _group_metadata(reference)
    return reference_out, matches


def seed_group_pairing_table(reference_group: dict[str, Any], candidate_group: dict[str, Any]) -> pd.DataFrame:
    reference_by_seed = {
        int(item.get("config", {}).get("seed")): item
        for item in reference_group.get("items", [])
        if item.get("config", {}).get("seed") is not None
    }
    candidate_by_seed = {
        int(item.get("config", {}).get("seed")): item
        for item in candidate_group.get("items", [])
        if item.get("config", {}).get("seed") is not None
    }
    rows: list[dict[str, Any]] = []
    for seed in sorted(set(reference_by_seed) | set(candidate_by_seed)):
        reference_item = reference_by_seed.get(seed)
        candidate_item = candidate_by_seed.get(seed)
        if reference_item is not None and candidate_item is not None:
            pairing_status = "paired"
        elif reference_item is None:
            pairing_status = "reference_missing"
        else:
            pairing_status = "candidate_missing"
        rows.append(
            {
                "seed": seed,
                "pairing_status": pairing_status,
                "reference_run_id": reference_item.get("run_id") if reference_item else None,
                "candidate_run_id": candidate_item.get("run_id") if candidate_item else None,
                "reference_status": reference_item.get("summary", {}).get("status") if reference_item else None,
                "candidate_status": candidate_item.get("summary", {}).get("status") if candidate_item else None,
            }
        )
    return pd.DataFrame(rows)


def paired_seed_final_metric_deltas(
    reference_group: dict[str, Any],
    candidate_group: dict[str, Any],
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    metrics = metrics or metrics_for_config(reference_group.get("config", {}), "final")
    reference_by_seed = {
        int(item.get("config", {}).get("seed")): item
        for item in reference_group.get("items", [])
        if item.get("config", {}).get("seed") is not None
    }
    candidate_by_seed = {
        int(item.get("config", {}).get("seed")): item
        for item in candidate_group.get("items", [])
        if item.get("config", {}).get("seed") is not None
    }
    rows: list[dict[str, Any]] = []
    for seed in sorted(set(reference_by_seed) & set(candidate_by_seed)):
        reference_item = reference_by_seed[seed]
        candidate_item = candidate_by_seed[seed]
        metric_df = matched_final_metrics_table(
            reference_item.get("summary", {}),
            candidate_item.get("summary", {}),
            metrics=metrics,
        ).copy()
        if metric_df.empty:
            continue
        metric_df.insert(0, "seed", seed)
        metric_df.insert(1, "reference_run_id", reference_item.get("run_id"))
        metric_df.insert(2, "candidate_run_id", candidate_item.get("run_id"))
        rows.extend(metric_df.to_dict(orient="records"))
    if not rows:
        return pd.DataFrame(columns=["seed", "reference_run_id", "candidate_run_id", "metric", "reference", "candidate", "delta", "pct_delta"])
    return pd.DataFrame(rows)


def paired_seed_delta_summary(delta_df: pd.DataFrame, pairing_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "metric",
        "pair_count",
        "reference_mean",
        "candidate_mean",
        "mean_delta",
        "median_delta",
        "mean_abs_delta",
        "std_delta",
        "min_delta",
        "max_delta",
        "positive_delta_fraction",
        "negative_delta_fraction",
        "zero_delta_fraction",
        "sign_consistency_ratio",
        "delta_direction",
        "reference_only_seed_count",
        "candidate_only_seed_count",
        "paired_seed_count",
    ]
    if delta_df.empty:
        return pd.DataFrame(columns=columns)
    reference_only = int((pairing_df.get("pairing_status") == "candidate_missing").sum()) if not pairing_df.empty else 0
    candidate_only = int((pairing_df.get("pairing_status") == "reference_missing").sum()) if not pairing_df.empty else 0
    paired_seed_count = int((pairing_df.get("pairing_status") == "paired").sum()) if not pairing_df.empty else 0
    rows: list[dict[str, Any]] = []
    for metric, small in delta_df.groupby("metric", sort=True):
        ref = pd.to_numeric(small["reference"], errors="coerce")
        cand = pd.to_numeric(small["candidate"], errors="coerce")
        delta = pd.to_numeric(small["delta"], errors="coerce")
        positive_fraction = float((delta > 0).mean()) if len(delta.index) else 0.0
        negative_fraction = float((delta < 0).mean()) if len(delta.index) else 0.0
        zero_fraction = float((delta == 0).mean()) if len(delta.index) else 0.0
        if positive_fraction == 1.0:
            delta_direction = "positive"
        elif negative_fraction == 1.0:
            delta_direction = "negative"
        elif zero_fraction == 1.0:
            delta_direction = "flat"
        else:
            delta_direction = "mixed"
        rows.append(
            {
                "metric": metric,
                "pair_count": int(delta.shape[0]),
                "reference_mean": float(ref.mean()),
                "candidate_mean": float(cand.mean()),
                "mean_delta": float(delta.mean()),
                "median_delta": float(delta.median()),
                "mean_abs_delta": float(delta.abs().mean()),
                "std_delta": float(delta.std(ddof=0)),
                "min_delta": float(delta.min()),
                "max_delta": float(delta.max()),
                "positive_delta_fraction": positive_fraction,
                "negative_delta_fraction": negative_fraction,
                "zero_delta_fraction": zero_fraction,
                "sign_consistency_ratio": float(max(positive_fraction, negative_fraction, zero_fraction)),
                "delta_direction": delta_direction,
                "reference_only_seed_count": reference_only,
                "candidate_only_seed_count": candidate_only,
                "paired_seed_count": paired_seed_count,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def paired_seed_pairing_summary(pairing_df: pd.DataFrame) -> dict[str, Any]:
    if pairing_df.empty:
        return {
            "paired_seed_count": 0,
            "reference_only_seed_count": 0,
            "candidate_only_seed_count": 0,
            "complete": False,
        }
    paired_seed_count = int((pairing_df["pairing_status"] == "paired").sum())
    reference_only_seed_count = int((pairing_df["pairing_status"] == "candidate_missing").sum())
    candidate_only_seed_count = int((pairing_df["pairing_status"] == "reference_missing").sum())
    return {
        "paired_seed_count": paired_seed_count,
        "reference_only_seed_count": reference_only_seed_count,
        "candidate_only_seed_count": candidate_only_seed_count,
        "complete": reference_only_seed_count == 0 and candidate_only_seed_count == 0 and paired_seed_count > 0,
    }


__all__ = [
    "SEED_GROUP_RECORD_COLUMNS",
    "build_seed_group_analysis_record",
    "build_seed_group_analysis_item",
    "build_seed_group_analysis_items",
    "load_seed_group_analysis_items",
    "seed_group_records_table",
    "seed_group_final_metrics",
    "seed_group_trajectories",
    "seed_group_compare_config",
    "find_matched_seed_groups",
    "seed_group_pairing_table",
    "paired_seed_final_metric_deltas",
    "paired_seed_delta_summary",
    "paired_seed_pairing_summary",
]
