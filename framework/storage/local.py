from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from framework.core.models import RunResult
from framework.services.data_sources import get_data_source_default_provenance
from framework.core.runtime_info import RUNTIME_PROVENANCE_SCHEMA_VERSION, missing_runtime_snapshot
from framework.services.serialization import json_dumps
from framework.services.run_integrity import (
    validate_loaded_run_payload,
    validate_run_folder_structure,
)


CONFIG_FILENAME = "config.json"
METRICS_FILENAME = "metrics.csv"
SUMMARY_FILENAME = "summary.json"
METADATA_FILENAME = "metadata.json"
ARTIFACTS_FILENAME = "artifacts.npz"
ARTIFACT_MANIFEST_FILENAME = "artifacts_manifest.json"


RUN_REQUIRED_FILENAMES = (
    CONFIG_FILENAME,
    METRICS_FILENAME,
    SUMMARY_FILENAME,
    METADATA_FILENAME,
    ARTIFACTS_FILENAME,
    ARTIFACT_MANIFEST_FILENAME,
)


def _short_hash(obj: dict) -> str:
    payload = json_dumps(obj, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:10]


def _file_paths(folder: Path) -> dict[str, Path]:
    return {
        "config": folder / CONFIG_FILENAME,
        "metrics": folder / METRICS_FILENAME,
        "summary": folder / SUMMARY_FILENAME,
        "metadata": folder / METADATA_FILENAME,
        "artifacts": folder / ARTIFACTS_FILENAME,
        "artifact_manifest": folder / ARTIFACT_MANIFEST_FILENAME,
    }


def _artifact_manifest(artifacts: dict[str, Any]) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for key in sorted(artifacts.keys()):
        value = artifacts[key]
        arr = np.asarray(value)
        items.append({"name": str(key), "dtype": str(arr.dtype), "shape": list(arr.shape)})
    return {"count": len(items), "items": items}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_run_structure(folder_path: Path) -> None:
    structure = validate_run_folder_structure(folder_path)
    if structure.status == "invalid":
        messages = [issue.message for issue in structure.errors]
        raise ValueError("Invalid saved run structure: " + "; ".join(messages))


def _load_artifacts_dict(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as artifacts_npz:
        return {key: artifacts_npz[key] for key in artifacts_npz.files}


def _infer_legacy_parameterization(cfg: dict[str, Any]) -> str:
    return str(cfg.get("parameterization") or "factorized_psd")


def _infer_legacy_init_type(cfg: dict[str, Any]) -> str:
    if cfg.get("init_type"):
        return str(cfg["init_type"])
    return "near_zero_gaussian"


def _infer_legacy_explicit_reg_type(cfg: dict[str, Any]) -> str:
    return str(cfg.get("explicit_reg_type") or "none")


def _infer_legacy_stopping_mode(cfg: dict[str, Any]) -> str:
    return str(cfg.get("stopping_mode") or "run_to_convergence")


def _infer_legacy_data_provenance(cfg: dict[str, Any]) -> dict[str, Any]:
    problem_family = str(cfg.get("problem_family", "psd_sensing"))
    data_source = str(cfg.get("data_source", "synthetic"))
    defaults = get_data_source_default_provenance(problem_family, data_source)
    return {
        "dataset_version": str(cfg.get("dataset_version") or defaults["dataset_version"]),
        "split_version": str(cfg.get("split_version") or defaults["split_version"]),
        "preprocessing_version": str(cfg.get("preprocessing_version") or defaults["preprocessing_version"]),
    }


def _runtime_defaults(metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    meta = dict(metadata or {})
    runtime_snapshot = meta.get("runtime_snapshot")
    if not isinstance(runtime_snapshot, dict):
        runtime_snapshot = missing_runtime_snapshot()
    return {
        "provenance_schema_version": int(meta.get("provenance_schema_version", RUNTIME_PROVENANCE_SCHEMA_VERSION)),
        "runtime_snapshot": runtime_snapshot,
        "python_version": meta.get("python_version", runtime_snapshot.get("python_version")),
        "python_implementation": meta.get("python_implementation", runtime_snapshot.get("python_implementation")),
        "platform_system": meta.get("platform_system", runtime_snapshot.get("platform_system")),
        "platform_release": meta.get("platform_release", runtime_snapshot.get("platform_release")),
        "platform_machine": meta.get("platform_machine", runtime_snapshot.get("platform_machine")),
        "platform_version": meta.get("platform_version", runtime_snapshot.get("platform_version")),
        "executable": meta.get("executable", runtime_snapshot.get("executable")),
        "numpy_version": meta.get("numpy_version", runtime_snapshot.get("numpy_version")),
        "pandas_version": meta.get("pandas_version", runtime_snapshot.get("pandas_version")),
        "torch_version": meta.get("torch_version", runtime_snapshot.get("torch_version")),
    }


def _infer_legacy_metadata(folder_path: Path, cfg: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    return {
        **_runtime_defaults(),
        "schema_version": int(cfg.get("schema_version", 1)),
        "framework_version": str(cfg.get("framework_version", "0.1.0")),
        "git_commit": cfg.get("git_commit"),
        "saved_at": folder_path.name.split("_")[0] if "_" in folder_path.name else None,
        "run_id": folder_path.name,
        "seed": cfg.get("seed"),
        "problem_family": cfg.get("problem_family", "psd_sensing"),
        "data_source": cfg.get("data_source", "synthetic"),
        **_infer_legacy_data_provenance(cfg),
        "measurement_type": cfg.get("measurement_type"),
        "psd_regime": cfg.get("psd_regime", "non_commuting"),
        "noncommutativity_level": float(cfg.get("noncommutativity_level", 1.0)),
        "coherence_level": float(cfg.get("coherence_level", 0.0)),
        "sampling_regime": str(cfg.get("sampling_regime", "uniform")),
        "mc_spectrum_type": str(cfg.get("mc_spectrum_type", "flat")),
        "mc_spectrum_decay": float(cfg.get("mc_spectrum_decay", 1.0)),
        "mc_factor_balance": str(cfg.get("mc_factor_balance", "balanced")),
        "mc_factor_imbalance_level": float(cfg.get("mc_factor_imbalance_level", 0.0)),
        "optimizer": cfg.get("optimizer"),
        "lr_schedule_type": cfg.get("lr_schedule_type", "constant"),
        "lr_decay_gamma": float(cfg.get("lr_decay_gamma", 0.5)),
        "lr_step_size": int(cfg.get("lr_step_size", 100)),
        "lr_min": float(cfg.get("lr_min", 0.0)),
        "device_preference": cfg.get("device_preference", "auto"),
        "resolved_device": summary.get("resolved_device"),
        "parameterization": _infer_legacy_parameterization(cfg),
        "init_type": _infer_legacy_init_type(cfg),
        "explicit_reg_type": _infer_legacy_explicit_reg_type(cfg),
        "explicit_reg_strength": float(cfg.get("explicit_reg_strength", 0.0)),
        "stopping_mode": _infer_legacy_stopping_mode(cfg),
        "grad_norm_tol": float(cfg.get("grad_norm_tol", 1e-10)),
        "objective_tol": float(cfg.get("objective_tol", 1e-12)),
        "divergence_norm_threshold": float(cfg.get("divergence_norm_threshold", 1e6)),
        "group_id": cfg.get("group_id"),
        "seed_group_id": cfg.get("seed_group_id"),
        "status": summary.get("status"),
        "config_hash": _short_hash(cfg),
    }


def save_run(result: RunResult, runs_dir: str = "runs", metadata_overrides: dict[str, Any] | None = None) -> str:
    result.validate()

    runs_path = Path(runs_dir)
    runs_path.mkdir(parents=True, exist_ok=True)

    cfg = result.spec.to_dict()
    run_hash = _short_hash(cfg)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{ts}_{run_hash}"
    folder = runs_path / folder_name
    folder.mkdir(parents=True, exist_ok=False)

    paths = _file_paths(folder)
    paths["config"].write_text(json_dumps(cfg, indent=2), encoding="utf-8")
    pd.DataFrame(result.metrics).to_csv(paths["metrics"], index=False)
    paths["summary"].write_text(json_dumps(result.summary, indent=2), encoding="utf-8")

    metadata = result.metadata_dict(saved_at=ts, run_id=folder_name)
    metadata.update(dict(metadata_overrides or {}))
    metadata.update(_runtime_defaults(metadata))
    metadata["config_hash"] = run_hash
    metadata["metrics_rows"] = int(len(next(iter(result.metrics.values()))) if result.metrics else 0)
    metadata["artifact_keys"] = sorted(str(k) for k in result.artifacts.keys())
    paths["metadata"].write_text(json_dumps(metadata, indent=2), encoding="utf-8")

    manifest = _artifact_manifest(result.artifacts)
    paths["artifact_manifest"].write_text(json_dumps(manifest, indent=2), encoding="utf-8")
    np.savez_compressed(paths["artifacts"], **result.artifacts)
    return str(folder)


def read_run_header(folder: str) -> dict[str, Any]:
    folder_path = Path(folder)
    _ensure_run_structure(folder_path)
    paths = _file_paths(folder_path)

    cfg = _read_json(paths["config"])
    summary = _read_json(paths["summary"])

    if paths["metadata"].exists():
        metadata = _read_json(paths["metadata"])
        metadata.setdefault("data_source", cfg.get("data_source", "synthetic"))
        metadata.setdefault("dataset_version", _infer_legacy_data_provenance(cfg)["dataset_version"])
        metadata.setdefault("split_version", _infer_legacy_data_provenance(cfg)["split_version"])
        metadata.setdefault("preprocessing_version", _infer_legacy_data_provenance(cfg)["preprocessing_version"])
        metadata.setdefault("lr_schedule_type", cfg.get("lr_schedule_type", "constant"))
        metadata.setdefault("lr_decay_gamma", float(cfg.get("lr_decay_gamma", 0.5)))
        metadata.setdefault("lr_step_size", int(cfg.get("lr_step_size", 100)))
        metadata.setdefault("lr_min", float(cfg.get("lr_min", 0.0)))
        metadata.setdefault("parameterization", _infer_legacy_parameterization(cfg))
        metadata.setdefault("init_type", _infer_legacy_init_type(cfg))
        metadata.setdefault("explicit_reg_type", _infer_legacy_explicit_reg_type(cfg))
        metadata.setdefault("explicit_reg_strength", float(cfg.get("explicit_reg_strength", 0.0)))
        metadata.setdefault("stopping_mode", _infer_legacy_stopping_mode(cfg))
        metadata.setdefault("grad_norm_tol", float(cfg.get("grad_norm_tol", 1e-10)))
        metadata.setdefault("objective_tol", float(cfg.get("objective_tol", 1e-12)))
        metadata.setdefault("divergence_norm_threshold", float(cfg.get("divergence_norm_threshold", 1e6)))
        metadata.setdefault("psd_regime", cfg.get("psd_regime", "non_commuting"))
        metadata.setdefault("noncommutativity_level", float(cfg.get("noncommutativity_level", 1.0)))
        metadata.setdefault("coherence_level", float(cfg.get("coherence_level", 0.0)))
        metadata.setdefault("sampling_regime", str(cfg.get("sampling_regime", "uniform")))
        metadata.setdefault("mc_spectrum_type", str(cfg.get("mc_spectrum_type", "flat")))
        metadata.setdefault("mc_spectrum_decay", float(cfg.get("mc_spectrum_decay", 1.0)))
        metadata.setdefault("mc_factor_balance", str(cfg.get("mc_factor_balance", "balanced")))
        metadata.setdefault("mc_factor_imbalance_level", float(cfg.get("mc_factor_imbalance_level", 0.0)))
        metadata.setdefault("group_id", cfg.get("group_id"))
        metadata.setdefault("seed_group_id", cfg.get("seed_group_id"))
        metadata.update(_runtime_defaults(metadata))
    else:
        metadata = _infer_legacy_metadata(folder_path, cfg, summary)

    cfg.setdefault("dataset_version", _infer_legacy_data_provenance(cfg)["dataset_version"])
    cfg.setdefault("split_version", _infer_legacy_data_provenance(cfg)["split_version"])
    cfg.setdefault("preprocessing_version", _infer_legacy_data_provenance(cfg)["preprocessing_version"])
    cfg.setdefault("lr_schedule_type", "constant")
    cfg.setdefault("lr_decay_gamma", 0.5)
    cfg.setdefault("lr_step_size", 100)
    cfg.setdefault("lr_min", 0.0)
    cfg.setdefault("parameterization", _infer_legacy_parameterization(cfg))
    cfg.setdefault("init_type", _infer_legacy_init_type(cfg))
    cfg.setdefault("explicit_reg_type", _infer_legacy_explicit_reg_type(cfg))
    cfg.setdefault("explicit_reg_strength", 0.0)
    cfg.setdefault("stopping_mode", _infer_legacy_stopping_mode(cfg))
    cfg.setdefault("grad_norm_tol", 1e-10)
    cfg.setdefault("objective_tol", 1e-12)
    cfg.setdefault("divergence_norm_threshold", 1e6)
    cfg.setdefault("psd_regime", "non_commuting")
    cfg.setdefault("noncommutativity_level", 1.0)
    cfg.setdefault("coherence_level", 0.0)
    cfg.setdefault("sampling_regime", "uniform")
    cfg.setdefault("mc_spectrum_type", "flat")
    cfg.setdefault("mc_spectrum_decay", 1.0)
    cfg.setdefault("mc_factor_balance", "balanced")
    cfg.setdefault("mc_factor_imbalance_level", 0.0)
    cfg.setdefault("group_id", None)
    cfg.setdefault("device_preference", "auto")
    cfg.setdefault("seed_group_id", None)

    artifact_manifest = {"count": 0, "items": []}
    if paths["artifact_manifest"].exists():
        artifact_manifest = _read_json(paths["artifact_manifest"])
    elif paths["artifacts"].exists():
        artifact_manifest = _artifact_manifest(_load_artifacts_dict(paths["artifacts"]))

    header = {
        "config": cfg,
        "summary": summary,
        "metadata": metadata,
        "artifact_manifest": artifact_manifest,
    }
    header["integrity"] = validate_loaded_run_payload(folder_path, header, include_metrics=False, include_artifacts=False).to_dict()
    return header


def load_run(folder: str) -> dict[str, Any]:
    folder_path = Path(folder)
    _ensure_run_structure(folder_path)
    paths = _file_paths(folder_path)
    header = read_run_header(str(folder_path))
    metrics = pd.read_csv(paths["metrics"])
    artifacts = _load_artifacts_dict(paths["artifacts"])
    loaded = {**header, "metrics": metrics, "artifacts": artifacts}
    loaded["integrity"] = validate_loaded_run_payload(folder_path, loaded, include_metrics=True, include_artifacts=True).to_dict()
    return loaded


def is_run_dir_complete(folder: str | Path) -> bool:
    folder_path = Path(folder)
    return all((folder_path / name).exists() for name in RUN_REQUIRED_FILENAMES)
