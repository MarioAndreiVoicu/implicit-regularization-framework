from __future__ import annotations

import streamlit as st

from framework.ui.page_services import NewRunPageService, SweepsPageService
from framework.ui.pages.new_run import render_new_run_page
from framework.ui.pages.sweeps import render_sweeps_page


def render_execution_page(*, new_run_service: NewRunPageService, sweeps_service: SweepsPageService, page_title: str = "Experiments") -> None:
    st.title(page_title)
    st.caption("Define and launch experiments from one place.")

    tab_new_run, tab_sweeps = st.tabs(["New Run", "Sweeps"])
    with tab_new_run:
        render_new_run_page(service=new_run_service)
    with tab_sweeps:
        render_sweeps_page(service=sweeps_service)
