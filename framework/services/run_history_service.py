from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from framework.services.analysis_records import load_analysis_item, load_analysis_records
from framework.services.export_service import ExportBundle, LocalExportService
from framework.core.matrix_completion_metric_contract import matrix_completion_history_plot_metrics
from framework.services.run_repository import ReadRunRepository


@dataclass(frozen=True)
class RunHistoryPlotSpec:
    metric: str
    title: str


@dataclass(frozen=True)
class RunHistoryView:
    run_id: str
    run_dir: Path
    records_df: pd.DataFrame
    selected_record: dict
    summary_df: pd.DataFrame
    config: dict
    summary: dict
    integrity: dict
    metrics_df: pd.DataFrame
    x_col: str
    plot_specs: list[RunHistoryPlotSpec]


class RunHistoryService:
    def __init__(
        self,
        *,
        read_repository: ReadRunRepository,
        x_col_fn: Callable[[pd.DataFrame], str],
        export_service: LocalExportService,
    ) -> None:
        self._read_repository = read_repository
        self._x_col_fn = x_col_fn
        self._export_service = export_service

    def list_records(self) -> pd.DataFrame:
        return load_analysis_records(self._read_repository)

    def load_run_ids(self) -> list[str]:
        records_df = self.list_records()
        if records_df.empty or 'run_id' not in records_df.columns:
            return []
        return [str(run_id) for run_id in records_df['run_id'].dropna().astype(str).tolist()]

    def build_view(self, run_id: str | None = None) -> RunHistoryView:
        records_df = load_analysis_records(self._read_repository)
        if records_df.empty:
            raise ValueError("No saved runs found.")
        selected_run_id = str(run_id or records_df.iloc[0]["run_id"])
        selected_row = records_df[records_df["run_id"] == selected_run_id]
        if selected_row.empty:
            raise ValueError(f"Run '{selected_run_id}' was not found.")
        selected_record = selected_row.iloc[0].to_dict()
        run_dir = self._read_repository.resolve_run_dir(selected_run_id)
        item = load_analysis_item(run_dir, self._read_repository, include_metrics=True)
        metrics_df = item["metrics"] if isinstance(item["metrics"], pd.DataFrame) else pd.DataFrame(item["metrics"])
        x_col = self._x_col_fn(metrics_df) if not metrics_df.empty else "step"
        summary_fields = [
            "run_id",
            "status",
            "integrity_status",
            "integrity_warning_count",
            "integrity_error_count",
            "problem_family",
            "data_source",
            "dataset_version",
            "split_version",
            "preprocessing_version",
            "seed",
            "seed_group_id",
            "group_id",
            "optimizer",
            "parameterization",
            "lr",
            "steps",
            "final_objective",
            "final_rel_recon_error",
            "final_val_rmse",
            "final_test_rmse",
            "resolved_device",
            "resolved_dtype",
            "saved_at",
            "config_hash",
        ]
        summary_row = {field: selected_record.get(field) for field in summary_fields if field in selected_record}
        return RunHistoryView(
            run_id=str(selected_run_id),
            run_dir=run_dir,
            records_df=records_df,
            selected_record=selected_record,
            summary_df=pd.DataFrame([summary_row]),
            config=item["config"],
            summary=item["summary"],
            integrity=item.get("integrity", {}),
            metrics_df=metrics_df,
            x_col=x_col,
            plot_specs=self._plot_specs(item["config"], item["summary"]),
        )

    def build_export_bundle(self, *, run_id: str, include_artifacts: bool, include_report: bool) -> ExportBundle:
        run_dir = self._read_repository.resolve_run_dir(run_id)
        return self._export_service.build_run_bundle(
            run_dir=run_dir,
            run_id=str(run_id),
            include_artifacts=include_artifacts,
            include_report=include_report,
        )

    @staticmethod
    def _plot_specs(config: dict, summary: dict) -> list[RunHistoryPlotSpec]:
        data_source = str(config.get("data_source", "synthetic"))
        problem_family = str(config.get("problem_family", "psd_sensing"))
        if problem_family == "matrix_completion":
            source_kind = str(summary.get("source_kind") or ("synthetic" if data_source == "synthetic" else "real"))
            return [RunHistoryPlotSpec(metric=metric, title=title) for metric, title in matrix_completion_history_plot_metrics(source_kind=source_kind)]
        specs = [RunHistoryPlotSpec(metric="loss", title="Loss vs step"), RunHistoryPlotSpec(metric="rel_recon_error", title="Relative reconstruction error vs step")]
        specs.append(RunHistoryPlotSpec(metric="trace_X", title="trace(UUᵀ) vs step (PSD complexity proxy)"))
        return specs
