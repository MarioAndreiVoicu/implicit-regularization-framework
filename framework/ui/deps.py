from __future__ import annotations

from typing import Any


def build_ui_deps(
    *,
    new_run_service,
    run_history_service,
    compare_service,
    validation_service,
    sweeps_service,
    run_manager_service,
    datasets_service,
    results_export_service,
    docs_service,
) -> dict[str, Any]:
    return {
        'new_run_service': new_run_service,
        'run_history_service': run_history_service,
        'compare_service': compare_service,
        'validation_service': validation_service,
        'sweeps_service': sweeps_service,
        'run_manager_service': run_manager_service,
        'datasets_service': datasets_service,
        'results_export_service': results_export_service,
        'docs_service': docs_service,
    }
