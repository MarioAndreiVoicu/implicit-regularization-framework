from __future__ import annotations

from pathlib import Path
from hashlib import sha256
from typing import Any, Callable

import pandas as pd

from framework.core.models import RunSpec
from framework.services.compare_service import CompareService
from framework.services.custom_dataset_registry import LocalCustomDatasetRegistry
from framework.services.data_sources import (
    DEFAULT_CUSTOM_TRIPLETS_PREPROCESSING_VERSION,
    DEFAULT_CUSTOM_TRIPLETS_SPLIT_VERSION,
    DEFAULT_MOVIELENS_DATASET_VERSION,
    DEFAULT_MOVIELENS_PREPROCESSING_VERSION,
    DEFAULT_MOVIELENS_SPLIT_VERSION,
    movielens_100k_stats,
)
from framework.core.datasets.custom_matrix_completion import load_custom_triplets_csv_dataframe, custom_triplets_csv_stats
from framework.services.experiment_critic import CriticReport, audit_new_run_request, audit_sweep_request, audit_sweep_spec
from framework.services.export_service import LocalExportService
from framework.services.local_runs import aggregate_seed_group_final_metrics, sweep_metric_options, x_col
from framework.services.new_run import (
    NewRunExecutionPlan,
    NewRunExecutionResult,
    NewRunRequest,
    NewRunValidationResult,
    execute_new_run_plan,
    get_new_run_request_from_state,
    sync_default_new_run_request_state,
    validate_new_run_request,
)
from framework.services.run_history_service import RunHistoryService
from framework.services.run_repository import LocalReadRunRepository, LocalRunRepository, RunRepository
from framework.services.sweep_analysis import load_saved_sweep_analysis
from framework.services.sweep_repository import LocalSweepRepository
from framework.services.sweeps import (
    SavedSweepView,
    SweepExecutionPlan,
    SweepExecutionResult,
    SweepRequest,
    build_sweep_base_request,
    build_sweep_execution_plan,
    execute_sweep_plan,
    get_default_sweep_base_request,
    resolve_sweep_seeds,
    save_sweep_metadata,
    validate_sweep_request,
)
from framework.services.validation_service import ValidationSuiteResult, execute_validation_suite
from framework.services.custom_dataset_registry import CustomDatasetRecord


class NewRunPageService:
    def __init__(self, *, run_experiment_fn: Callable[[RunSpec], Any], run_repository: RunRepository, dataset_registry: LocalCustomDatasetRegistry | None = None) -> None:
        self._run_experiment_fn = run_experiment_fn
        self._run_repository = run_repository
        self._dataset_registry = dataset_registry

    def sync_default_state(self, state: dict, *, prefix: str, problem_family: str, data_source: str) -> NewRunRequest:
        return sync_default_new_run_request_state(state, prefix=prefix, problem_family=problem_family, data_source=data_source)

    def get_request_from_state(self, state: dict, *, prefix: str, problem_family: str, data_source: str) -> NewRunRequest:
        return get_new_run_request_from_state(state, prefix=prefix, problem_family=problem_family, data_source=data_source)


    def validate_request(self, request: NewRunRequest) -> NewRunValidationResult:
        return validate_new_run_request(request, dataset_registry=self._dataset_registry)

    def audit_request(self, request: NewRunRequest) -> CriticReport:
        return audit_new_run_request(request)

    def execute_plan(
        self,
        plan: NewRunExecutionPlan,
        *,
        on_status: Callable[[str], None] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> NewRunExecutionResult:
        return execute_new_run_plan(
            plan,
            run_experiment_fn=self._run_experiment_fn,
            run_repository=self._run_repository,
            on_status=on_status,
            on_progress=on_progress,
        )

    def aggregate_seed_group_final_metrics(self, group: dict[str, Any]) -> pd.DataFrame:
        return aggregate_seed_group_final_metrics(group)

    def list_custom_datasets(self) -> list[CustomDatasetRecord]:
        return self._dataset_registry.list_records() if self._dataset_registry is not None else []

    def get_custom_dataset(self, dataset_id: str) -> CustomDatasetRecord:
        if self._dataset_registry is None:
            raise KeyError(dataset_id)
        return self._dataset_registry.get(dataset_id)


class CustomDatasetsPageService:
    def __init__(self, *, dataset_registry: LocalCustomDatasetRegistry) -> None:
        self._dataset_registry = dataset_registry

    def list_records(self) -> list[CustomDatasetRecord]:
        return self._dataset_registry.list_records()

    def register_triplets_csv(self, path: str) -> CustomDatasetRecord:
        return self._dataset_registry.register_triplets_csv(path)

    def stage_uploaded_triplets_csv(self, filename: str, content: bytes) -> str:
        filename = str(filename or "uploaded_triplets.csv").strip() or "uploaded_triplets.csv"
        suffix = Path(filename).suffix.lower() or ".csv"
        stem = Path(filename).stem or "uploaded_triplets"
        safe_stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem).strip("._") or "uploaded_triplets"
        digest = sha256(content).hexdigest()[:12]
        uploads_dir = self._dataset_registry.registry_dir / "uploaded_sources"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        staged = uploads_dir / f"{safe_stem}_{digest}{suffix}"
        staged.write_bytes(content)
        return str(staged.resolve())

    def get_record(self, dataset_id: str) -> CustomDatasetRecord:
        return self._dataset_registry.get(dataset_id)

    def list_builtin_sources(self) -> list[dict[str, Any]]:
        """Return fixed built-in data sources for backwards-compatible callers."""
        return self.list_dataset_sources()

    def list_dataset_sources(self) -> list[dict[str, Any]]:
        return [
            {
                "label": "MovieLens 100K",
                "problem_family": "matrix_completion",
                "data_source": "movielens_100k",
                "kind": "dataset",
                "description": "Built-in real matrix-completion dataset with fixed dimensions and default split/provenance.",
                "dataset_version": DEFAULT_MOVIELENS_DATASET_VERSION,
                "split_version": DEFAULT_MOVIELENS_SPLIT_VERSION,
                "preprocessing_version": DEFAULT_MOVIELENS_PREPROCESSING_VERSION,
                "stats": movielens_100k_stats(),
            },
        ]

    def custom_registry_summary(self) -> dict[str, Any]:
        records = self.list_records()
        return {
            "data_source": "custom_triplets_csv",
            "registered_datasets": len(records),
            "dataset_version_template": "custom_triplets_csv:<registered-dataset-id>",
            "split_version": DEFAULT_CUSTOM_TRIPLETS_SPLIT_VERSION,
            "preprocessing_version": DEFAULT_CUSTOM_TRIPLETS_PREPROCESSING_VERSION,
        }

    def validate_triplets_csv(self, path: str, *, preview_rows: int = 8) -> dict[str, Any]:
        resolved = Path(path).expanduser().resolve()
        frame = load_custom_triplets_csv_dataframe(resolved)
        stats = custom_triplets_csv_stats(resolved)
        fingerprint = self._dataset_registry.compute_fingerprint(resolved)
        dataset_id = f"custom_mc_{fingerprint[:12]}"
        return {
            "resolved_path": str(resolved),
            "source_filename": resolved.name,
            "file_size_bytes": int(resolved.stat().st_size),
            "fingerprint": fingerprint,
            "dataset_id": dataset_id,
            "dataset_version": f"custom_triplets_csv:{dataset_id}",
            "split_version": DEFAULT_CUSTOM_TRIPLETS_SPLIT_VERSION,
            "preprocessing_version": DEFAULT_CUSTOM_TRIPLETS_PREPROCESSING_VERSION,
            "stats": dict(stats),
            "preview_rows": frame.head(preview_rows).to_dict(orient="records"),
            "preview_columns": list(frame.columns),
            "already_registered": any(record.fingerprint == fingerprint for record in self.list_records()),
        }

    def resolve_selected_dataset_id(self, selected_id: str | None = None) -> str | None:
        records = self.list_records()
        if not records:
            return None
        dataset_ids = [record.dataset_id for record in records]
        candidate = str(selected_id or "").strip()
        if candidate in dataset_ids:
            return candidate
        return dataset_ids[0]

    def inspect_record(self, dataset_id: str, *, preview_rows: int = 8) -> dict[str, Any]:
        record = self.get_record(dataset_id)
        source_path = Path(record.source_path).expanduser()
        source_exists = source_path.exists()
        preview: list[dict[str, Any]] = []
        preview_columns: list[str] = []
        validation_error = ""
        if source_exists:
            try:
                frame = load_custom_triplets_csv_dataframe(source_path)
                preview = frame.head(preview_rows).to_dict(orient="records")
                preview_columns = list(frame.columns)
            except Exception as exc:
                validation_error = str(exc)
        return {
            "record": record.to_dict(),
            "source_exists": source_exists,
            "file_size_bytes": int(source_path.stat().st_size) if source_exists else None,
            "preview_rows": preview,
            "preview_columns": preview_columns,
            "validation_error": validation_error,
        }


class ValidationPageService:
    _CHECK_META = {
        "Stable run reaches low error": {
            "category": "Stability",
            "pass_reason": "The stable configuration reached the target error bound.",
            "fail_reason": "The stable configuration did not reach the target error bound.",
        },
        "Reproducibility (same config/seed)": {
            "category": "Reproducibility",
            "pass_reason": "The repeated stable run matched within the reproducibility tolerance.",
            "fail_reason": "The repeated stable run drift exceeded the reproducibility tolerance or the run was missing.",
        },
        "Metric integrity": {
            "category": "Metric integrity",
            "pass_reason": "The saved validation runs contain the expected finite summary and metric values.",
            "fail_reason": "One or more required validation metrics were missing, empty, NaN, or infinite.",
        },
    }

    def __init__(self, *, run_experiment_fn: Callable[[RunSpec], Any], run_repository: RunRepository) -> None:
        self._run_experiment_fn = run_experiment_fn
        self._run_repository = run_repository

    def execute_suite(
        self,
        *,
        stable_lr: float,
        err_good_thresh: float,
        repro_tol: float,
        on_status: Callable[[str], None] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> ValidationSuiteResult:
        return execute_validation_suite(
            stable_lr=stable_lr,
            err_good_thresh=err_good_thresh,
            repro_tol=repro_tol,
            run_experiment_fn=self._run_experiment_fn,
            run_repository=self._run_repository,
            on_status=on_status,
            on_progress=on_progress,
        )

    def validation_contract(self) -> list[dict[str, str]]:
        return [
            {
                "category": "Stability",
                "title": "Stable run",
                "description": "Runs a conservative reference configuration and checks that it reaches a low final reconstruction error.",
            },
            {
                "category": "Reproducibility",
                "title": "Reproducibility",
                "description": "Repeats the same configuration with the same seed and checks that the final result stays within a tight tolerance.",
            },
            {
                "category": "Metric integrity",
                "title": "Metric integrity",
                "description": "Checks that saved validation runs contain the expected finite summary values and metric series.",
            },
        ]

    def build_suite_summary(self, result: ValidationSuiteResult) -> dict[str, Any]:
        total_checks = len(result.checks)
        passed_checks = sum(1 for check in result.checks if check.passed)
        failed_checks = total_checks - passed_checks
        categories: list[dict[str, Any]] = []
        for spec in self.validation_contract():
            cat_checks = [check for check in result.checks if self._check_meta(check.check)["category"] == spec["category"]]
            cat_passed = all(check.passed for check in cat_checks) if cat_checks else False
            categories.append({
                "category": spec["category"],
                "title": spec["title"],
                "status": "passed" if cat_passed else "failed",
                "check_count": len(cat_checks),
            })
        overall = "passed" if failed_checks == 0 and result.failed_count == 0 else "failed"
        if result.failed_count and result.succeeded_count:
            overall = "partial"
        return {
            "overall_status": overall,
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "failed_checks": failed_checks,
            "saved_runs": result.succeeded_count,
            "failed_runs": result.failed_count,
            "categories": categories,
        }

    def build_checks_frame(self, result: ValidationSuiteResult) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for check in result.checks:
            meta = self._check_meta(check.check)
            rows.append({
                "category": meta["category"],
                "check": check.check,
                "status": "passed" if check.passed else "failed",
                "value": check.value,
                "threshold": check.threshold,
                "run": check.run,
                "interpretation": meta["pass_reason"] if check.passed else meta["fail_reason"],
            })
        return pd.DataFrame(rows)

    def build_failures_frame(self, result: ValidationSuiteResult) -> pd.DataFrame:
        rows = [
            {
                "index": failure.index,
                "seed": failure.seed,
                "stage": failure.stage,
                "error_type": failure.error_type,
                "error_message": failure.error_message,
            }
            for failure in result.failures
        ]
        return pd.DataFrame(rows)

    def build_saved_runs_frame(self, result: ValidationSuiteResult) -> pd.DataFrame:
        role_map = {
            "validation_stable": "Stable",
            "validation_stable_repeat": "Stable repeat",
        }
        rows: list[dict[str, Any]] = []
        for item in result.items:
            run_name = str(item.config.get("run_name") or item.run_id)
            rows.append({
                "role": role_map.get(run_name, run_name),
                "run_id": item.run_id,
                "seed": item.seed,
                "status": item.summary.get("status"),
                "final_rel_recon_error": item.summary.get("final_rel_recon_error"),
                "final_objective": item.summary.get("final_objective"),
                "stop_reason": item.summary.get("stop_reason"),
                "resolved_device": item.summary.get("resolved_device"),
                "resolved_dtype": item.summary.get("resolved_dtype"),
                "dataset_version": item.config.get("dataset_version"),
                "split_version": item.config.get("split_version"),
                "preprocessing_version": item.config.get("preprocessing_version"),
            })
        return pd.DataFrame(rows)

    def recommended_compare_run_ids(self, result: ValidationSuiteResult, *, include_repeat: bool = True) -> list[str]:
        selected: list[str] = []
        stable_repeat_id = None
        for item in result.items:
            run_name = str(item.config.get("run_name") or item.run_id)
            if run_name == "validation_stable_repeat":
                stable_repeat_id = item.run_id
                break
        if result.stable_run_id is not None:
            selected.append(result.stable_run_id)
        if include_repeat and stable_repeat_id is not None:
            selected.append(stable_repeat_id)
        if len(selected) < 2:
            selected = list(result.saved_run_ids[:4])
        deduped: list[str] = []
        for run_id in selected:
            if run_id not in deduped:
                deduped.append(run_id)
        return deduped

    @classmethod
    def _check_meta(cls, name: str) -> dict[str, str]:
        return cls._CHECK_META.get(name, {
            "category": "Other",
            "pass_reason": "This validation check passed.",
            "fail_reason": "This validation check failed.",
        })

    @staticmethod
    def x_col(df: pd.DataFrame) -> str:
        return x_col(df)


class SweepsPageService:
    def __init__(
        self,
        *,
        sweeps_dir: Path,
        read_repository: LocalReadRunRepository,
        run_experiment_fn: Callable[[RunSpec], Any],
        run_repository: LocalRunRepository,
        export_service: LocalExportService,
        dataset_registry: LocalCustomDatasetRegistry | None = None,
        load_analysis_views_fn: Callable[[], dict],
        save_analysis_views_fn: Callable[[dict], None],
        apply_analysis_view_state_fn: Callable[[dict], None],
    ) -> None:
        self._run_experiment_fn = run_experiment_fn
        self._run_repository = run_repository
        self._export_service = export_service
        self._dataset_registry = dataset_registry
        self._load_analysis_views_fn = load_analysis_views_fn
        self._save_analysis_views_fn = save_analysis_views_fn
        self._apply_analysis_view_state_fn = apply_analysis_view_state_fn
        self._sweep_repository = LocalSweepRepository(
            sweeps_dir=sweeps_dir,
            read_repository=read_repository,
        )

    @property
    def export_service(self) -> LocalExportService:
        return self._export_service

    def load_analysis_views(self) -> dict:
        return self._load_analysis_views_fn()

    def save_analysis_views(self, meta: dict) -> None:
        self._save_analysis_views_fn(meta)

    def apply_analysis_view_state(self, state: dict) -> None:
        self._apply_analysis_view_state_fn(state)

    def list_group_ids(self) -> list[str]:
        return self._sweep_repository.list_group_ids()

    def load_saved_analysis(self, group_id: str):
        return load_saved_sweep_analysis(group_id, self._sweep_repository)

    def save_metadata(self, metadata: dict[str, Any]) -> None:
        save_sweep_metadata(self._sweep_repository, metadata)

    def build_sweep_bundle(self, *, analysis_view, include_run_reports: bool, include_artifacts: bool):
        return self._export_service.build_sweep_bundle(
            analysis_view=analysis_view,
            repository=self._sweep_repository,
            include_run_reports=include_run_reports,
            include_artifacts=include_artifacts,
        )

    def get_default_base_request(self, problem_family: str, data_source: str) -> SweepRequest:
        return get_default_sweep_base_request(problem_family, data_source)


    def resolve_seeds(self, request: SweepRequest) -> list[int]:
        return resolve_sweep_seeds(request)

    def build_base_request(self, **kwargs: Any) -> SweepRequest:
        return build_sweep_base_request(**kwargs, dataset_registry=self._dataset_registry)

    def build_execution_plan(self, request: SweepRequest) -> SweepExecutionPlan:
        return build_sweep_execution_plan(request, dataset_registry=self._dataset_registry)

    def validate_request(self, request: SweepRequest) -> list[str]:
        return validate_sweep_request(request, dataset_registry=self._dataset_registry)

    def audit_request(self, request: SweepRequest) -> CriticReport:
        report = audit_sweep_request(request)
        try:
            plan = build_sweep_execution_plan(request, dataset_registry=self._dataset_registry)
        except Exception:
            return report
        report.extend(audit_sweep_spec(plan.sweep_spec).warnings)
        return report

    def execute_plan(
        self,
        plan: SweepExecutionPlan,
        *,
        on_status: Callable[[str], None] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> SweepExecutionResult:
        return execute_sweep_plan(
            plan,
            run_experiment_fn=self._run_experiment_fn,
            run_repository=self._run_repository,
            on_status=on_status,
            on_progress=on_progress,
        )

    @staticmethod
    def sweep_metric_options(raw_df: pd.DataFrame, metadata: dict[str, Any]) -> dict[str, str]:
        return sweep_metric_options(raw_df, metadata)

    def list_custom_datasets(self) -> list[CustomDatasetRecord]:
        return self._dataset_registry.list_records() if self._dataset_registry is not None else []

    def get_custom_dataset(self, dataset_id: str) -> CustomDatasetRecord:
        if self._dataset_registry is None:
            raise KeyError(dataset_id)
        return self._dataset_registry.get(dataset_id)


class ComparePageService:
    def __init__(
        self,
        *,
        compare_service: CompareService,
        x_col_fn: Callable[[pd.DataFrame], str],
        load_analysis_views_fn: Callable[[], dict],
        save_analysis_views_fn: Callable[[dict], None],
        apply_analysis_view_state_fn: Callable[[dict], None],
    ) -> None:
        self.compare_service = compare_service
        self._x_col_fn = x_col_fn
        self._load_analysis_views_fn = load_analysis_views_fn
        self._save_analysis_views_fn = save_analysis_views_fn
        self._apply_analysis_view_state_fn = apply_analysis_view_state_fn

    def load_analysis_views(self) -> dict:
        return self._load_analysis_views_fn()

    def save_analysis_views(self, meta: dict) -> None:
        self._save_analysis_views_fn(meta)

    def apply_analysis_view_state(self, state: dict) -> None:
        self._apply_analysis_view_state_fn(state)

    def x_col(self, df: pd.DataFrame) -> str:
        return self._x_col_fn(df)



class ResultsExportService:
    def __init__(
        self,
        *,
        run_history_service: RunHistoryService,
        compare_service: ComparePageService,
        sweeps_service: SweepsPageService,
        run_manager_service,
    ) -> None:
        self._run_history_service = run_history_service
        self._compare_service = compare_service
        self._sweeps_service = sweeps_service
        self._run_manager_service = run_manager_service

    def load_run_ids(self) -> list[str]:
        return self._run_history_service.load_run_ids()

    def load_sweep_group_ids(self) -> list[str]:
        return self._sweeps_service.list_group_ids()

    def load_saved_sweep_analysis(self, group_id: str):
        return self._sweeps_service.load_saved_analysis(group_id)

    def matched_compare_context(self):
        return self._compare_service.compare_service.matched_compare_context()

    def matched_compare_result(self, *, reference_run_id: str, varying_axis: str, strict_exact_match: bool, candidate_run_id: str | None = None):
        return self._compare_service.compare_service.matched_compare_result(
            reference_run_id=reference_run_id,
            varying_axis=varying_axis,
            strict_exact_match=strict_exact_match,
            candidate_run_id=candidate_run_id,
        )

    def seed_group_ids(self) -> list[str]:
        result = self._compare_service.compare_service.seed_aggregation_result()
        if result is None or result.groups_df.empty:
            return []
        return result.groups_df["seed_group_id"].astype(str).tolist()

    def paired_seed_group_compare_result(
        self,
        *,
        reference_group_id: str,
        varying_axis: str,
        strict_exact_match: bool,
        candidate_group_id: str | None = None,
    ):
        return self._compare_service.compare_service.paired_seed_group_compare_result(
            reference_group_id=reference_group_id,
            varying_axis=varying_axis,
            strict_exact_match=strict_exact_match,
            candidate_group_id=candidate_group_id,
        )

    def build_run_bundle(self, *, run_id: str, include_artifacts: bool, include_report: bool):
        return self._run_history_service.build_export_bundle(
            run_id=run_id,
            include_artifacts=include_artifacts,
            include_report=include_report,
        )

    def build_selected_runs_bundle(self, *, run_ids: list[str], include_run_reports: bool, include_artifacts: bool) -> tuple[str, bytes]:
        return (
            self._run_manager_service.selected_file_name(),
            self._run_manager_service.build_bundle_zip(
                run_ids=run_ids,
                include_run_reports=include_run_reports,
                include_artifacts=include_artifacts,
            ),
        )

    def build_free_compare_bundle(self, *, run_ids: list[str], include_run_reports: bool, include_artifacts: bool):
        return self._compare_service.compare_service.build_free_compare_bundle(
            selected_run_ids=run_ids,
            include_run_reports=include_run_reports,
            include_artifacts=include_artifacts,
        )

    def build_matched_compare_bundle(
        self,
        *,
        reference_run_id: str,
        varying_axis: str,
        strict_exact_match: bool,
        candidate_run_id: str | None,
        include_run_reports: bool,
        include_artifacts: bool,
    ):
        result = self.matched_compare_result(
            reference_run_id=reference_run_id,
            varying_axis=varying_axis,
            strict_exact_match=strict_exact_match,
            candidate_run_id=candidate_run_id,
        )
        if result is None or not result.candidate:
            raise ValueError("Matched comparison could not be built for the selected runs.")
        return self._compare_service.compare_service.build_matched_compare_bundle(
            result=result,
            include_run_reports=include_run_reports,
            include_artifacts=include_artifacts,
        )

    def build_paired_seed_group_compare_bundle(
        self,
        *,
        reference_group_id: str,
        varying_axis: str,
        strict_exact_match: bool,
        candidate_group_id: str | None,
        include_run_reports: bool,
        include_artifacts: bool,
    ):
        result = self.paired_seed_group_compare_result(
            reference_group_id=reference_group_id,
            varying_axis=varying_axis,
            strict_exact_match=strict_exact_match,
            candidate_group_id=candidate_group_id,
        )
        if result is None or not result.candidate_group:
            raise ValueError("Paired seed-group comparison could not be built for the selected groups.")
        return self._compare_service.compare_service.build_paired_seed_group_compare_bundle(
            result=result,
            include_run_reports=include_run_reports,
            include_artifacts=include_artifacts,
        )

    def build_sweep_bundle(self, *, group_id: str, include_run_reports: bool, include_artifacts: bool):
        analysis_view = self._sweeps_service.load_saved_analysis(group_id)
        if analysis_view is None:
            raise ValueError(f"Sweep '{group_id}' could not be loaded.")
        return self._sweeps_service.build_sweep_bundle(
            analysis_view=analysis_view,
            include_run_reports=include_run_reports,
            include_artifacts=include_artifacts,
        )


__all__ = [
    'ComparePageService',
    'CustomDatasetsPageService',
    'NewRunPageService',
    'RunHistoryService',
    'SweepsPageService',
    'ValidationPageService',
    'ResultsExportService',
    'DocsPageService',
]


class DocsPageService:
    def __init__(self, *, docs_dir: Path) -> None:
        self._docs_dir = docs_dir

    def list_sections(self) -> list[dict[str, str]]:
        sections: list[dict[str, str]] = []
        for path in sorted(self._docs_dir.glob('*.md')):
            title = self._extract_title(path.read_text(encoding="utf-8"))
            sections.append({
                'slug': path.stem,
                'title': title,
                'path': str(path),
            })
        return sections

    def load_section(self, slug: str) -> dict[str, str]:
        path = self._docs_dir / f'{slug}.md'
        content = path.read_text(encoding="utf-8")
        return {
            'slug': slug,
            'title': self._extract_title(content),
            'content': content,
        }

    @staticmethod
    def _extract_title(content: str) -> str:
        for line in content.splitlines():
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()
        return 'Untitled'
