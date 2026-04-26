from __future__ import annotations

import io
import pandas as pd
import streamlit as st

from framework.ui.page_services import ValidationPageService

_RESULT_KEY = "validation_last_result"


def _checks_csv(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def _status_badge(status: str) -> str:
    if status == "passed":
        return "✅ Passed"
    if status == "partial":
        return "⚠️ Partial"
    return "❌ Failed"


def render_validation_page(service: ValidationPageService) -> None:
    st.title("Validation")
    st.caption("Run a small self-check to confirm the framework is behaving as expected.")

    st.markdown("This validation suite runs three small framework self-checks.")
    info_cols = st.columns(3)
    with info_cols[0]:
        st.markdown("#### 1. Stable run")
        st.write("Runs a conservative reference configuration and checks that it reaches a low final reconstruction error.")
    with info_cols[1]:
        st.markdown("#### 2. Reproducibility")
        st.write("Repeats the same configuration with the same seed and checks that the final result stays within a tight tolerance.")
    with info_cols[2]:
        st.markdown("#### 3. Metric integrity")
        st.write("Checks that saved validation runs contain the expected finite summary values and metric series.")

    with st.expander("Validation settings", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            stable_lr = float(st.number_input("stable_lr", min_value=0.0, value=1e-3, format="%.6f", key="validation_stable_lr"))
            err_good_thresh = float(st.number_input("stable error threshold", min_value=0.0, value=0.20, format="%.6f", key="validation_err_good"))
        with c2:
            repro_tol = float(st.number_input("reproducibility tolerance", min_value=0.0, value=1e-8, format="%.2e", key="validation_repro_tol"))
            st.caption("Metric integrity has no threshold setting: required metrics must be present and finite.")

    status_slot = st.empty()
    progress_slot = st.empty()
    if st.button("Run validation", type="primary", key="validation_run_button"):
        progress = progress_slot.progress(0)

        def on_status(message: str) -> None:
            status_slot.info(message)

        def on_progress(done: int, total: int) -> None:
            progress.progress(done / max(total, 1))

        with st.spinner("Running validation suite..."):
            result = service.execute_suite(
                stable_lr=stable_lr,
                err_good_thresh=err_good_thresh,
                repro_tol=repro_tol,
                on_status=on_status,
                on_progress=on_progress,
            )
        st.session_state[_RESULT_KEY] = result
        status_slot.empty()
        progress_slot.empty()

    result = st.session_state.get(_RESULT_KEY)
    if result is None:
        st.info("No validation result yet. Click Run validation to run the framework self-check.")
        return

    summary = service.build_suite_summary(result)
    st.divider()
    st.subheader("Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", _status_badge(str(summary["overall_status"])))
    c2.metric("Checks passed", f"{summary['passed_checks']}/{summary['total_checks']}")
    c3.metric("Saved runs", int(summary["saved_runs"]))
    c4.metric("Failed runs", int(summary["failed_runs"]))

    checks_frame = service.build_checks_frame(result)
    st.markdown("#### Checks")
    st.dataframe(checks_frame, use_container_width=True, hide_index=True)

    with st.expander("Saved validation runs", expanded=False):
        saved_runs = service.build_saved_runs_frame(result)
        if saved_runs.empty:
            st.info("No validation runs were saved.")
        else:
            st.dataframe(saved_runs, use_container_width=True, hide_index=True)

    failures = service.build_failures_frame(result)
    if not failures.empty:
        with st.expander("Failure details", expanded=True):
            st.dataframe(failures, use_container_width=True, hide_index=True)

    with st.expander("Download", expanded=False):
        st.download_button(
            "Download validation checks CSV",
            data=_checks_csv(checks_frame),
            file_name="validation_checks.csv",
            mime="text/csv",
            key="validation_download_checks",
        )
