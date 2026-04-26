from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from framework.core.models import RunSpec, VALID_RUN_STATUSES


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

RUN_CORE_FILENAMES = (
    CONFIG_FILENAME,
    METRICS_FILENAME,
    SUMMARY_FILENAME,
    ARTIFACTS_FILENAME,
)

RUN_OPTIONAL_FILENAMES = (
    METADATA_FILENAME,
    ARTIFACT_MANIFEST_FILENAME,
)


@dataclass(frozen=True)
class IntegrityIssue:
    level: str
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"level": self.level, "code": self.code, "message": self.message}


@dataclass
class RunIntegrityReport:
    status: str = "valid"
    warnings: list[IntegrityIssue] = field(default_factory=list)
    errors: list[IntegrityIssue] = field(default_factory=list)

    def add_warning(self, code: str, message: str) -> None:
        self.warnings.append(IntegrityIssue(level="warning", code=code, message=message))
        if self.status == "valid":
            self.status = "valid_with_warnings"

    def add_error(self, code: str, message: str) -> None:
        self.errors.append(IntegrityIssue(level="error", code=code, message=message))
        self.status = "invalid"

    def merge(self, other: "RunIntegrityReport") -> "RunIntegrityReport":
        merged = RunIntegrityReport()
        for issue in [*self.warnings, *other.warnings]:
            merged.add_warning(issue.code, issue.message)
        for issue in [*self.errors, *other.errors]:
            merged.add_error(issue.code, issue.message)
        if not merged.errors and not merged.warnings:
            merged.status = "valid"
        elif not merged.errors:
            merged.status = "valid_with_warnings"
        else:
            merged.status = "invalid"
        return merged

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "warnings": [issue.to_dict() for issue in self.warnings],
            "errors": [issue.to_dict() for issue in self.errors],
            "warning_count": len(self.warnings),
            "error_count": len(self.errors),
        }


def _short_hash(obj: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(obj), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:10]


def validate_run_folder_structure(folder: str | Path) -> RunIntegrityReport:
    folder_path = Path(folder)
    report = RunIntegrityReport()
    if not folder_path.exists():
        report.add_error("folder_missing", f"Run folder does not exist: {folder_path}")
        return report
    if not folder_path.is_dir():
        report.add_error("folder_not_dir", f"Run path is not a directory: {folder_path}")
        return report
    missing_core = [name for name in RUN_CORE_FILENAMES if not (folder_path / name).exists()]
    if missing_core:
        report.add_error("missing_files", f"Missing required run files: {', '.join(sorted(missing_core))}")
    missing_optional = [name for name in RUN_OPTIONAL_FILENAMES if not (folder_path / name).exists()]
    if missing_optional:
        report.add_warning("missing_optional_files", f"Missing optional run files: {', '.join(sorted(missing_optional))}")
    return report


def _validate_metadata_consistency(
    report: RunIntegrityReport,
    *,
    folder_path: Path,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> None:
    expected_hash = _short_hash(config)
    if metadata.get("config_hash") != expected_hash:
        report.add_warning(
            "config_hash_mismatch",
            f"metadata.config_hash={metadata.get('config_hash')!r} does not match config hash {expected_hash!r}",
        )
    run_id = metadata.get("run_id")
    if run_id and str(run_id) != folder_path.name:
        report.add_warning("run_id_mismatch", f"metadata.run_id={run_id!r} does not match folder name {folder_path.name!r}")
    schema_meta = metadata.get("schema_version")
    schema_cfg = config.get("schema_version")
    if schema_meta is None:
        report.add_warning("schema_version_missing", "metadata.schema_version is missing")
    elif schema_cfg is not None and int(schema_meta) != int(schema_cfg):
        report.add_warning("schema_version_mismatch", "metadata.schema_version does not match config.schema_version")
    for field in ("seed", "problem_family", "data_source", "dataset_version", "split_version", "preprocessing_version", "seed_group_id", "group_id"):
        if field in metadata and field in config and metadata.get(field) != config.get(field):
            report.add_warning("metadata_field_mismatch", f"metadata.{field} does not match config.{field}")
    status = summary.get("status")
    if status not in VALID_RUN_STATUSES:
        report.add_error("invalid_status", f"summary.status={status!r} is not a valid run status")
    elif metadata.get("status") not in {None, status}:
        report.add_warning("status_mismatch", "metadata.status does not match summary.status")


def _validate_runtime_provenance(report: RunIntegrityReport, *, metadata: Mapping[str, Any]) -> None:
    provenance_schema_version = metadata.get("provenance_schema_version")
    if provenance_schema_version is None:
        report.add_warning("provenance_schema_version_missing", "metadata.provenance_schema_version is missing")
    runtime_snapshot = metadata.get("runtime_snapshot")
    if not isinstance(runtime_snapshot, Mapping):
        report.add_warning("runtime_snapshot_missing", "metadata.runtime_snapshot is missing or invalid")
        return
    if runtime_snapshot.get("capture_status") == "missing_legacy":
        report.add_warning("runtime_snapshot_legacy_missing", "runtime provenance was not captured for this legacy run")
        return
    for field in ("python_version", "numpy_version", "pandas_version"):
        if metadata.get(field) != runtime_snapshot.get(field):
            report.add_warning("runtime_snapshot_field_mismatch", f"metadata.{field} does not match metadata.runtime_snapshot.{field}")


def _validate_artifacts(
    report: RunIntegrityReport,
    *,
    metadata: Mapping[str, Any],
    artifact_manifest: Mapping[str, Any],
    artifacts: Mapping[str, Any] | None,
) -> None:
    items = list(artifact_manifest.get("items", []))
    manifest_names = [str(item.get("name")) for item in items if item.get("name") is not None]
    manifest_count = int(artifact_manifest.get("count", len(manifest_names)) or 0)
    if manifest_count != len(items):
        report.add_warning("artifact_manifest_count_mismatch", "artifact_manifest.count does not match number of manifest items")
    metadata_keys = list(metadata.get("artifact_keys", [])) if isinstance(metadata.get("artifact_keys"), Sequence) and not isinstance(metadata.get("artifact_keys"), (str, bytes)) else []
    if metadata_keys and sorted(metadata_keys) != sorted(manifest_names):
        report.add_warning("artifact_keys_mismatch", "metadata.artifact_keys do not match artifact_manifest item names")
    if artifacts is not None:
        actual_keys = sorted(str(key) for key in artifacts.keys())
        if sorted(manifest_names) != actual_keys:
            report.add_warning("artifact_manifest_keys_mismatch", "artifact_manifest item names do not match saved artifacts")


def _validate_metrics(report: RunIntegrityReport, *, metadata: Mapping[str, Any], metrics: pd.DataFrame | None) -> None:
    if metrics is None:
        return
    if "step" not in metrics.columns:
        report.add_error("metrics_step_missing", "metrics are missing required 'step' column")
    expected_rows = metadata.get("metrics_rows")
    if expected_rows is not None and int(expected_rows) != int(len(metrics)):
        report.add_warning("metrics_rows_mismatch", "metadata.metrics_rows does not match actual metrics row count")


def validate_loaded_run_payload(
    folder: str | Path,
    loaded: Mapping[str, Any],
    *,
    include_metrics: bool = True,
    include_artifacts: bool = True,
) -> RunIntegrityReport:
    folder_path = Path(folder)
    report = RunIntegrityReport()
    config = dict(loaded.get("config") or {})
    summary = dict(loaded.get("summary") or {})
    metadata = dict(loaded.get("metadata") or {})
    artifact_manifest = dict(loaded.get("artifact_manifest") or {})
    metrics = loaded.get("metrics") if include_metrics else None
    artifacts = loaded.get("artifacts") if include_artifacts else None

    try:
        RunSpec(**config).validate()
    except Exception as exc:
        report.add_error("config_invalid", f"config does not validate as RunSpec: {exc}")

    _validate_metadata_consistency(report, folder_path=folder_path, config=config, summary=summary, metadata=metadata)
    _validate_runtime_provenance(report, metadata=metadata)

    if include_metrics:
        if metrics is None:
            report.add_error("metrics_missing", "loaded payload is missing metrics")
        elif not isinstance(metrics, pd.DataFrame):
            report.add_error("metrics_type_invalid", "loaded metrics are not a pandas DataFrame")
        else:
            _validate_metrics(report, metadata=metadata, metrics=metrics)

    _validate_artifacts(report, metadata=metadata, artifact_manifest=artifact_manifest, artifacts=artifacts if include_artifacts else None)
    return report
