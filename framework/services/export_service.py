from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
import io
import json
import zipfile

import pandas as pd

from framework.services.sweep_analysis import SweepAnalysisView
from framework.services.sweep_repository import SweepRepository


@dataclass(frozen=True)
class ExportBundle:
    filename: str
    content_type: str
    data: bytes




def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    ordered: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = Path(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(resolved)
    return ordered

class LocalExportService:
    def __init__(
        self,
        *,
        ensure_report_and_plots_fn: Callable[..., None],
        build_run_bundle_zip_bytes_fn: Callable[..., bytes],
        build_compare_bundle_zip_bytes_fn: Callable[..., bytes],
        build_sweep_bundle_zip_bytes_fn: Callable[..., bytes],
        load_run_fn: Callable[[str], dict],
    ) -> None:
        self._ensure_report_and_plots_fn = ensure_report_and_plots_fn
        self._build_run_bundle_zip_bytes_fn = build_run_bundle_zip_bytes_fn
        self._build_compare_bundle_zip_bytes_fn = build_compare_bundle_zip_bytes_fn
        self._build_sweep_bundle_zip_bytes_fn = build_sweep_bundle_zip_bytes_fn
        self._load_run_fn = load_run_fn

    def ensure_run_report(self, run_dir: Path) -> None:
        run_dir = Path(run_dir)
        loaded = self._load_run_fn(str(run_dir))
        metrics = loaded["metrics"]
        if not isinstance(metrics, pd.DataFrame):
            metrics = pd.DataFrame(metrics)
        self._ensure_report_and_plots_fn(run_dir, loaded["config"], loaded["summary"], metrics)

    def _load_run_payload(self, run_dir: Path) -> dict:
        return self._load_run_fn(str(Path(run_dir)))

    def _build_traceability_manifest(self, run_dirs: Iterable[Path], *, context: dict | None = None) -> dict:
        run_dirs = _unique_paths(run_dirs)
        run_entries: list[dict] = []
        protocol_ids: list[str] = []
        seed_group_ids: list[str] = []
        for run_dir in run_dirs:
            loaded = self._load_run_payload(Path(run_dir))
            metadata = dict(loaded.get("metadata") or {})
            protocol_id = metadata.get("protocol_id")
            seed_group_id = metadata.get("seed_group_id")
            if protocol_id is not None:
                protocol_ids.append(str(protocol_id))
            if seed_group_id is not None:
                seed_group_ids.append(str(seed_group_id))
            run_entries.append(
                {
                    "run_id": Path(run_dir).name,
                    "protocol_id": protocol_id,
                    "seed_group_id": seed_group_id,
                    "runtime_snapshot": metadata.get("runtime_snapshot"),
                    "provenance_schema_version": metadata.get("provenance_schema_version"),
                }
            )
        manifest = {
            "run_count": len(run_entries),
            "run_ids": [entry["run_id"] for entry in run_entries],
            "protocol_ids": sorted({pid for pid in protocol_ids if pid}),
            "seed_group_ids": sorted({gid for gid in seed_group_ids if gid}),
            "runs": run_entries,
        }
        if context:
            manifest["context"] = dict(context)
        return manifest

    def _append_bundle_entries(
        self,
        bundle_bytes: bytes,
        *,
        json_entries: dict[str, dict] | None = None,
        csv_entries: dict[str, pd.DataFrame] | None = None,
    ) -> bytes:
        if not json_entries and not csv_entries:
            return bundle_bytes
        src = io.BytesIO(bundle_bytes)
        out = io.BytesIO()
        with zipfile.ZipFile(src, "r") as zin, zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                zout.writestr(item, zin.read(item.filename))
            for arcname, payload in (json_entries or {}).items():
                zout.writestr(arcname, json.dumps(payload, indent=2))
            for arcname, df in (csv_entries or {}).items():
                frame = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
                zout.writestr(arcname, frame.to_csv(index=False))
        return out.getvalue()

    def build_run_bundle(
        self,
        *,
        run_dir: Path,
        run_id: str,
        include_artifacts: bool,
        include_report: bool,
    ) -> ExportBundle:
        run_dir = Path(run_dir)
        if include_report:
            self.ensure_run_report(run_dir)
        traceability_manifest = self._build_traceability_manifest([run_dir], context={"export_kind": "run"})
        data = self._build_run_bundle_zip_bytes_fn(
            selected_dir=run_dir,
            include_artifacts=include_artifacts,
            include_report=include_report,
        )
        data = self._append_bundle_entries(
            data,
            json_entries={"traceability_manifest.json": traceability_manifest},
        )
        return ExportBundle(
            filename=f"{run_id}.zip",
            content_type="application/zip",
            data=data,
        )

    def build_compare_bundle(
        self,
        *,
        run_dirs: Iterable[Path],
        include_run_reports: bool,
        include_artifacts: bool,
        filename: str = "compare_runs_bundle.zip",
        compare_summary: dict | None = None,
        compare_tables: dict[str, pd.DataFrame] | None = None,
        context: dict | None = None,
    ) -> ExportBundle:
        run_dirs = _unique_paths(Path(rd) for rd in run_dirs)
        if include_run_reports:
            for run_dir in run_dirs:
                self.ensure_run_report(run_dir)
        data = self._build_compare_bundle_zip_bytes_fn(
            run_dirs=run_dirs,
            include_run_reports=include_run_reports,
            include_artifacts=include_artifacts,
        )
        json_entries = {
            "traceability_manifest.json": self._build_traceability_manifest(
                run_dirs,
                context={"export_kind": "compare", **(context or {})},
            )
        }
        if compare_summary is not None:
            json_entries["compare/summary.json"] = compare_summary
        csv_entries = {f"compare/{name}.csv": df for name, df in (compare_tables or {}).items() if df is not None}
        data = self._append_bundle_entries(data, json_entries=json_entries, csv_entries=csv_entries)
        return ExportBundle(
            filename=filename,
            content_type="application/zip",
            data=data,
        )

    def build_sweep_bundle(
        self,
        *,
        analysis_view: SweepAnalysisView,
        repository: SweepRepository,
        include_run_reports: bool,
        include_artifacts: bool,
    ) -> ExportBundle:
        loaded_runs = repository.load_group_run_items(analysis_view.group_id)
        if include_run_reports:
            for run_dir, cfg, summary, metrics in loaded_runs:
                metrics_df = metrics if isinstance(metrics, pd.DataFrame) else pd.DataFrame(metrics)
                self._ensure_report_and_plots_fn(run_dir, cfg, summary, metrics_df)
        sweep_field = analysis_view.fields[0] if analysis_view.fields else "sweep_value"
        aggregate_df = analysis_view.aggregate_df if isinstance(analysis_view.aggregate_df, pd.DataFrame) else pd.DataFrame()
        data = self._build_sweep_bundle_zip_bytes_fn(
            group_id=analysis_view.group_id,
            loaded_runs=loaded_runs,
            sweep_field=sweep_field,
            sweep_summary_df=aggregate_df,
            include_run_reports=include_run_reports,
            include_artifacts=include_artifacts,
            sweep_meta=analysis_view.metadata,
        )
        run_dirs = _unique_paths(run_dir for run_dir, _, _, _ in loaded_runs)
        json_entries = {
            f"{analysis_view.group_id}/summary.json": analysis_view.summary,
            f"{analysis_view.group_id}/contract.json": analysis_view.contract,
            f"{analysis_view.group_id}/confound_warnings.json": {"warnings": analysis_view.confound_warnings},
            f"{analysis_view.group_id}/traceability_manifest.json": self._build_traceability_manifest(
                run_dirs,
                context={
                    "export_kind": "sweep",
                    "group_id": analysis_view.group_id,
                    "protocol_id": analysis_view.metadata.get("protocol_id"),
                    "sweep_contract_status": analysis_view.contract.get("status"),
                },
            ),
        }
        data = self._append_bundle_entries(data, json_entries=json_entries)
        return ExportBundle(
            filename=f"{analysis_view.group_id}.zip",
            content_type="application/zip",
            data=data,
        )
