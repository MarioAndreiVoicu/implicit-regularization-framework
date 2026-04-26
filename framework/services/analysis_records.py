from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from framework.services.local_runs import list_run_dirs
from framework.services.run_repository import ReadRunRepository
from framework.storage.local import read_run_header


ANALYSIS_RECORD_COLUMNS = [
    "run_id",
    "run_name",
    "notes",
    "status",
    "problem_family",
    "data_source",
    "source_kind",
    "dataset_version",
    "split_version",
    "preprocessing_version",
    "seed",
    "seed_group_id",
    "group_id",
    "optimizer",
    "parameterization",
    "explicit_reg_type",
    "stopping_mode",
    "measurement_type",
    "psd_regime",
    "sampling_regime",
    "n",
    "n_cols",
    "r_star",
    "m",
    "factor_rank",
    "init_scale",
    "lr",
    "steps",
    "val_fraction",
    "test_fraction",
    "schema_version",
    "framework_version",
    "git_commit",
    "python_version",
    "numpy_version",
    "pandas_version",
    "torch_version",
    "config_hash",
    "artifact_count",
    "metrics_rows",
    "saved_at",
    "final_objective",
    "final_rel_recon_error",
    "final_gradient_norm",
    "final_val_rmse",
    "final_test_rmse",
    "best_val_rmse",
    "best_test_rmse_at_best_val",
    "converged",
    "stop_reason",
    "resolved_device",
    "resolved_dtype",
    "integrity_status",
    "integrity_warning_count",
    "integrity_error_count",
    "protocol_id",
    "protocol_title",
    "protocol_manipulated_factor",
    "protocol_held_fixed_factors",
    "protocol_primary_metrics",
    "protocol_seed_policy",
]


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _iter_run_dirs(run_source: ReadRunRepository | Path) -> list[Path]:
    if hasattr(run_source, "list_run_dirs"):
        return list(run_source.list_run_dirs())
    return list_run_dirs(Path(run_source))


def _read_header(run_source: ReadRunRepository | Path, run_dir: Path) -> dict[str, Any]:
    if hasattr(run_source, "read_run_header"):
        return run_source.read_run_header(run_dir)
    return read_run_header(str(run_dir))


def build_run_analysis_record(
    *,
    run_id: str,
    config: dict[str, Any],
    summary: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    artifact_manifest: dict[str, Any] | None = None,
    integrity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = dict(metadata or {})
    artifact_manifest = dict(artifact_manifest or {})
    integrity = dict(integrity or {})
    record = {
        "run_id": str(run_id),
        "run_name": config.get("run_name"),
        "notes": config.get("notes"),
        "status": _coalesce(meta.get("status"), summary.get("status")),
        "problem_family": _coalesce(meta.get("problem_family"), config.get("problem_family"), "psd_sensing"),
        "data_source": _coalesce(meta.get("data_source"), config.get("data_source"), "synthetic"),
        "source_kind": _coalesce(meta.get("source_kind"), summary.get("source_kind"), "synthetic"),
        "dataset_version": _coalesce(meta.get("dataset_version"), config.get("dataset_version")),
        "split_version": _coalesce(meta.get("split_version"), config.get("split_version")),
        "preprocessing_version": _coalesce(meta.get("preprocessing_version"), config.get("preprocessing_version")),
        "seed": _coalesce(meta.get("seed"), config.get("seed")),
        "seed_group_id": _coalesce(meta.get("seed_group_id"), config.get("seed_group_id")),
        "group_id": _coalesce(meta.get("group_id"), config.get("group_id")),
        "optimizer": _coalesce(meta.get("optimizer"), config.get("optimizer")),
        "parameterization": _coalesce(meta.get("parameterization"), config.get("parameterization")),
        "explicit_reg_type": _coalesce(meta.get("explicit_reg_type"), config.get("explicit_reg_type")),
        "stopping_mode": _coalesce(meta.get("stopping_mode"), config.get("stopping_mode")),
        "measurement_type": _coalesce(meta.get("measurement_type"), config.get("measurement_type")),
        "psd_regime": _coalesce(meta.get("psd_regime"), config.get("psd_regime")),
        "sampling_regime": _coalesce(meta.get("sampling_regime"), config.get("sampling_regime")),
        "n": config.get("n"),
        "n_cols": config.get("n_cols"),
        "r_star": config.get("r_star"),
        "m": config.get("m"),
        "factor_rank": config.get("factor_rank"),
        "init_scale": config.get("init_scale"),
        "lr": config.get("lr"),
        "steps": config.get("steps"),
        "val_fraction": config.get("val_fraction"),
        "test_fraction": config.get("test_fraction"),
        "schema_version": _coalesce(meta.get("schema_version"), config.get("schema_version")),
        "framework_version": _coalesce(meta.get("framework_version"), config.get("framework_version")),
        "git_commit": _coalesce(meta.get("git_commit"), config.get("git_commit")),
        "python_version": meta.get("python_version"),
        "numpy_version": meta.get("numpy_version"),
        "pandas_version": meta.get("pandas_version"),
        "torch_version": meta.get("torch_version"),
        "config_hash": meta.get("config_hash"),
        "artifact_count": int(artifact_manifest.get("count", meta.get("artifact_count", 0)) or 0),
        "metrics_rows": meta.get("metrics_rows"),
        "saved_at": meta.get("saved_at"),
        "final_objective": summary.get("final_objective"),
        "final_rel_recon_error": summary.get("final_rel_recon_error"),
        "final_gradient_norm": summary.get("final_gradient_norm"),
        "final_val_rmse": summary.get("final_val_rmse"),
        "final_test_rmse": summary.get("final_test_rmse"),
        "best_val_rmse": _coalesce(summary.get("best_val_rmse"), summary.get("best_val_metric")),
        "best_test_rmse_at_best_val": summary.get("best_test_rmse_at_best_val"),
        "converged": summary.get("converged"),
        "stop_reason": summary.get("stop_reason"),
        "resolved_device": _coalesce(meta.get("resolved_device"), summary.get("resolved_device")),
        "resolved_dtype": _coalesce(meta.get("resolved_dtype"), summary.get("resolved_dtype")),
        "integrity_status": integrity.get("status", "valid"),
        "integrity_warning_count": int(integrity.get("warning_count", 0) or 0),
        "integrity_error_count": int(integrity.get("error_count", 0) or 0),
        "protocol_id": meta.get("protocol_id"),
        "protocol_title": meta.get("protocol_title"),
        "protocol_manipulated_factor": meta.get("protocol_manipulated_factor"),
        "protocol_held_fixed_factors": tuple(meta.get("protocol_held_fixed_factors", []) or ()),
        "protocol_primary_metrics": tuple(meta.get("protocol_primary_metrics", []) or ()),
        "protocol_seed_policy": meta.get("protocol_seed_policy"),
    }
    return record



def load_analysis_records(run_source: ReadRunRepository | Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_dir in _iter_run_dirs(run_source):
        try:
            header = _read_header(run_source, run_dir)
        except Exception:
            continue
        rows.append(
            build_run_analysis_record(
                run_id=run_dir.name,
                config=header["config"],
                summary=header["summary"],
                metadata=header.get("metadata"),
                artifact_manifest=header.get("artifact_manifest"),
                integrity=header.get("integrity"),
            )
        )
    if not rows:
        return pd.DataFrame(columns=ANALYSIS_RECORD_COLUMNS)
    df = pd.DataFrame(rows)
    ordered = [col for col in ANALYSIS_RECORD_COLUMNS if col in df.columns]
    extras = [col for col in df.columns if col not in ordered]
    return df[ordered + extras].sort_values("run_id").reset_index(drop=True)



def load_analysis_item(run_ref: Path | str, run_source: ReadRunRepository, include_metrics: bool = True) -> dict[str, Any]:
    run_dir = run_source.resolve_run_dir(run_ref)
    loaded = run_source.load_run(run_dir)
    metrics = loaded.get("metrics")
    if include_metrics:
        if not isinstance(metrics, pd.DataFrame):
            metrics = pd.DataFrame(metrics)
    else:
        metrics = pd.DataFrame()
    record = build_run_analysis_record(
        run_id=run_dir.name,
        config=loaded["config"],
        summary=loaded["summary"],
        metadata=loaded.get("metadata"),
        artifact_manifest=loaded.get("artifact_manifest"),
        integrity=loaded.get("integrity"),
    )
    return {
        "run_id": run_dir.name,
        "run_dir": run_dir,
        "record": record,
        "config": loaded["config"],
        "summary": loaded["summary"],
        "metadata": loaded.get("metadata", {}),
        "artifact_manifest": loaded.get("artifact_manifest", {}),
        "integrity": loaded.get("integrity", {}),
        "metrics": metrics,
    }



def load_analysis_items(
    run_source: ReadRunRepository,
    *,
    include_metrics: bool = True,
    run_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    wanted = {str(run_id) for run_id in run_ids} if run_ids is not None else None
    items: list[dict[str, Any]] = []
    for run_dir in run_source.list_run_dirs():
        if wanted is not None and run_dir.name not in wanted:
            continue
        items.append(load_analysis_item(run_dir, run_source, include_metrics=include_metrics))
    return items
