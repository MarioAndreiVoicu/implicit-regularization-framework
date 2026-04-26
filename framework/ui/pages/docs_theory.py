from __future__ import annotations

import streamlit as st

from framework.ui.page_services import DocsPageService


_DOC_SECTIONS = [
    ("Start Here", "00_start_here"),
    ("Experiment Design", "01_experiment_design"),
    ("Core Theory", "02_core_theory"),
    ("Workflow", "03_workflow"),
    ("Reading Results", "04_reading_results"),
    ("Fields & Metrics", "05_fields_metrics_reference"),
    ("Reproducibility", "06_reproducibility_provenance"),
    ("Limitations", "07_limitations"),
]


def _render_doc(service: DocsPageService, slug: str) -> None:
    try:
        payload = service.load_section(slug)
    except FileNotFoundError:
        st.warning(f"Documentation section not found: {slug}")
        return
    st.markdown(payload["content"])


def render_docs_theory_page(*, service: DocsPageService) -> None:
    st.title("Docs / Theory")
    st.caption("A practical guide to designing, running, reading, and reproducing implicit-regularization experiments.")

    tabs = st.tabs([title for title, _slug in _DOC_SECTIONS])
    for tab, (_title, slug) in zip(tabs, _DOC_SECTIONS):
        with tab:
            _render_doc(service, slug)
