from __future__ import annotations

from pathlib import Path

from framework.core.experiments import run_experiment
from framework.storage.local import load_run, read_run_header, save_run as _save_run
from framework.services.exports_local import (
    build_compare_bundle_zip_bytes,
    build_run_bundle_zip_bytes,
    build_runs_collection_zip_bytes,
    build_sweep_bundle_zip_bytes,
    ensure_report_and_plots,
)
from framework.services.analysis_views import apply_analysis_view_state, load_analysis_views, save_analysis_views
from framework.services.compare_service import CompareService
from framework.services.custom_dataset_registry import LocalCustomDatasetRegistry
from framework.services.export_service import LocalExportService
from framework.services.local_runs import config_diff, x_col
from framework.services.run_history_service import RunHistoryService
from framework.services.run_repository import LocalReadRunRepository, LocalRunRepository
from framework.services.run_manager_service import LocalRunManagerService
from framework.services.run_manager_local import get_runs_df_for_manager, scan_runs_table_base
from framework.ui.deps import build_ui_deps
from framework.ui.page_services import (
    ComparePageService,
    CustomDatasetsPageService,
    NewRunPageService,
    ResultsExportService,
    SweepsPageService,
    ValidationPageService,
    DocsPageService,
)
from framework.ui.router import render_page
from framework.ui.shell import render_sidebar_navigation


def run_streamlit_app(runs_dir: Path) -> None:
    page = render_sidebar_navigation()
    sweeps_dir = Path("sweeps")
    sweeps_dir.mkdir(exist_ok=True)
    datasets_registry_dir = Path("datasets_registry")
    datasets_registry_dir.mkdir(exist_ok=True)

    dataset_registry = LocalCustomDatasetRegistry(datasets_registry_dir)
    run_repository = LocalRunRepository(runs_dir=runs_dir, save_run_fn=_save_run)
    read_repository = LocalReadRunRepository(runs_dir=runs_dir, load_run_fn=load_run, read_run_header_fn=read_run_header)
    export_service = LocalExportService(
        ensure_report_and_plots_fn=ensure_report_and_plots,
        build_run_bundle_zip_bytes_fn=build_run_bundle_zip_bytes,
        build_compare_bundle_zip_bytes_fn=build_compare_bundle_zip_bytes,
        build_sweep_bundle_zip_bytes_fn=build_sweep_bundle_zip_bytes,
        load_run_fn=load_run,
    )
    compare_service_core = CompareService(
        read_repository=read_repository,
        config_diff_fn=config_diff,
        export_service=export_service,
    )

    new_run_service = NewRunPageService(
        run_experiment_fn=run_experiment,
        run_repository=run_repository,
        dataset_registry=dataset_registry,
    )
    sweeps_service = SweepsPageService(
        sweeps_dir=sweeps_dir,
        read_repository=read_repository,
        run_experiment_fn=run_experiment,
        run_repository=run_repository,
        export_service=export_service,
        dataset_registry=dataset_registry,
        load_analysis_views_fn=load_analysis_views,
        save_analysis_views_fn=save_analysis_views,
        apply_analysis_view_state_fn=apply_analysis_view_state,
    )
    run_history_service = RunHistoryService(
        read_repository=read_repository,
        x_col_fn=x_col,
        export_service=export_service,
    )
    compare_service = ComparePageService(
        compare_service=compare_service_core,
        x_col_fn=x_col,
        load_analysis_views_fn=load_analysis_views,
        save_analysis_views_fn=save_analysis_views,
        apply_analysis_view_state_fn=apply_analysis_view_state,
    )
    run_manager_service = LocalRunManagerService(
        runs_dir=runs_dir,
        get_runs_df_for_manager_fn=lambda: get_runs_df_for_manager(runs_dir),
        clear_runs_cache_fn=scan_runs_table_base.cache_clear,
        build_runs_collection_zip_bytes_fn=build_runs_collection_zip_bytes,
    )

    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)

    ui_deps = build_ui_deps(
        new_run_service=new_run_service,
        run_history_service=run_history_service,
        compare_service=compare_service,
        validation_service=ValidationPageService(
            run_experiment_fn=run_experiment,
            run_repository=run_repository,
        ),
        sweeps_service=sweeps_service,
        run_manager_service=run_manager_service,
        datasets_service=CustomDatasetsPageService(dataset_registry=dataset_registry),
        results_export_service=ResultsExportService(
            run_history_service=run_history_service,
            compare_service=compare_service,
            sweeps_service=sweeps_service,
            run_manager_service=run_manager_service,
        ),
        docs_service=DocsPageService(docs_dir=docs_dir),
    )

    render_page(page, ui_deps)
