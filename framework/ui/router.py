from __future__ import annotations

from framework.ui.pages.execution import render_execution_page
from framework.ui.pages.results import render_results_page
from framework.ui.pages.docs_theory import render_docs_theory_page
from framework.ui.pages.compare import render_compare_page
from framework.ui.pages.datasets import render_datasets_page
from framework.ui.pages.validation import render_validation_page


def render_page(page: str, deps: dict) -> None:
    if page == "Experiments":
        render_execution_page(new_run_service=deps["new_run_service"], sweeps_service=deps["sweeps_service"], page_title="Experiments")
        return

    if page == "Results":
        render_results_page(
            run_manager_service=deps["run_manager_service"],
            results_export_service=deps["results_export_service"],
        )
        return

    if page == "Compare":
        render_compare_page(service=deps["compare_service"])
        return

    if page == "Datasets":
        render_datasets_page(service=deps["datasets_service"])
        return

    if page == "Validation":
        render_validation_page(service=deps["validation_service"])
        return

    if page == "Docs / Theory":
        render_docs_theory_page(service=deps["docs_service"])
        return

    raise ValueError(f"Unknown page: {page}")
