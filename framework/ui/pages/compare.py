from __future__ import annotations

import json
from typing import Any, Callable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from framework.services.ui_state import remove_conflicting_widget_defaults

_ST = st

def _state_safe_kwargs(key: str | None, kwargs: dict, *, default_arg_names: tuple[str, ...]) -> dict:
    return remove_conflicting_widget_defaults(st.session_state, key, kwargs, default_arg_names=default_arg_names)


def _selectbox(label, options=None, **kwargs):
    key = kwargs.get("key")
    return _ST.selectbox(label, options, **_state_safe_kwargs(key, kwargs, default_arg_names=("index",)))


def _checkbox(label, **kwargs):
    key = kwargs.get("key")
    return _ST.checkbox(label, **_state_safe_kwargs(key, kwargs, default_arg_names=("value",)))


def _radio(label, options=None, **kwargs):
    key = kwargs.get("key")
    return _ST.radio(label, options, **_state_safe_kwargs(key, kwargs, default_arg_names=("index",)))


def _multiselect(label, options=None, **kwargs):
    key = kwargs.get("key")
    return _ST.multiselect(label, options, **_state_safe_kwargs(key, kwargs, default_arg_names=("default",)))


def _text_input(label, **kwargs):
    key = kwargs.get("key")
    return _ST.text_input(label, **_state_safe_kwargs(key, kwargs, default_arg_names=("value",)))

from framework.services.frontier import COLOR_BY_OPTIONS, metric_label, pareto_frontier
from framework.services.analysis_views import consume_pending_analysis_view_state
from framework.services.comparison_audit import comparison_audit_details_table
from framework.services.compare_service import CompareService
from framework.ui.page_services import ComparePageService
from framework.services.local_runs import metrics_for_config

COMPARE_VIEW_KEYS = [
    "compare_mode",
    "compare_run_type",
    "free_compare_selected_runs",
    "matched_reference_run",
    "matched_varying_axis",
    "matched_strict_exact_match",
    "matched_candidate_run",
    "seed_group_select",
    "paired_seed_reference_group",
    "paired_seed_varying_axis",
    "paired_seed_strict_exact_match",
    "paired_seed_candidate_group",
    "frontier_source_mode",
    "frontier_reference_run",
    "frontier_varying_axis",
    "frontier_strict_exact_match",
    "frontier_fit_metric",
    "frontier_complexity_metric",
    "frontier_color_by",
    "frontier_show_frontier",
    "failure_source",
    "failure_reference_run",
    "failure_varying_axis",
    "failure_strict_exact_match",
    "failure_only_problematic",
    "failure_issue_filter",
    "failure_group_filter",
    "failure_run_select",
]


def _ensure_dataframe(df_or_dict) -> pd.DataFrame:
    if isinstance(df_or_dict, pd.DataFrame):
        return df_or_dict
    return pd.DataFrame(df_or_dict)


def _current_compare_view_state() -> dict:
    return {key: st.session_state[key] for key in COMPARE_VIEW_KEYS if key in st.session_state}


def _render_saved_compare_views(service: ComparePageService) -> None:
    meta = service.load_analysis_views()
    namespace = (meta or {}).get("compare_runs", {})
    views = namespace.get("views", {}) if isinstance(namespace, dict) else {}
    order = [name for name in namespace.get("order", []) if name in views] if isinstance(namespace, dict) else []
    current_state = _current_compare_view_state()

    with st.expander("Saved analysis views", expanded=False):
        c1, c2 = st.columns([1.4, 1.0])
        with c1:
            selected_name = _selectbox(
                "Saved compare view",
                options=["(none)"] + order,
                index=0,
                key="compare_saved_view_pick",
            )
        with c2:
            save_name = _text_input("Save current as", value="", key="compare_save_view_name").strip()

        a1, a2, a3 = st.columns(3)
        with a1:
            if st.button("Apply saved view", key="compare_apply_saved_view"):
                if selected_name != "(none)":
                    service.apply_analysis_view_state(views.get(selected_name, {}))
                    st.rerun()
        with a2:
            if st.button("Save current view", key="compare_save_current_view"):
                if not save_name:
                    st.error("Give the view a name.")
                else:
                    namespace = meta.setdefault("compare_runs", {"views": {}, "order": []})
                    namespace.setdefault("views", {})[save_name] = current_state
                    if save_name not in namespace.setdefault("order", []):
                        namespace["order"].append(save_name)
                    service.save_analysis_views(meta)
                    st.success(f"Saved view '{save_name}'.")
        with a3:
            if st.button("Delete saved view", key="compare_delete_saved_view"):
                if selected_name != "(none)":
                    namespace = meta.setdefault("compare_runs", {"views": {}, "order": []})
                    namespace.setdefault("views", {}).pop(selected_name, None)
                    namespace["order"] = [name for name in namespace.get("order", []) if name != selected_name]
                    service.save_analysis_views(meta)
                    st.success(f"Deleted view '{selected_name}'.")
                    st.rerun()

        st.download_button(
            "Download current view spec (.json)",
            data=json.dumps(current_state, indent=2),
            file_name="compare_view.json",
            mime="application/json",
            key="compare_download_current_view",
        )


def _render_export_panel(payload: dict | None) -> None:
    if not payload:
        return
    df = payload.get("df")
    if not isinstance(df, pd.DataFrame) or df.empty:
        return
    filename = payload.get("file_name", "analysis_view.csv")
    label = payload.get("label", "current table")
    st.markdown("### Export")
    st.download_button(
        f"Download {label} (.csv)",
        data=df.to_csv(index=False),
        file_name=filename,
        mime="text/csv",
        key=f"export_{filename}",
    )


def _render_metric_overlays(loaded: list[tuple[str, dict, dict, pd.DataFrame]], x_col_fn: Callable, metrics: list[str]) -> None:
    combined = []
    for name, _, _, df in loaded:
        x = x_col_fn(df)
        df2 = df.copy()
        df2["run"] = name
        if x != "step":
            df2 = df2.rename(columns={x: "step"})
        combined.append(df2)
    if not combined:
        st.info("No runs to plot.")
        return
    big = pd.concat(combined, ignore_index=True)
    for y in metrics:
        if y in big.columns:
            fig = px.line(big, x="step", y=y, color="run", title=f"{y} vs step")
            st.plotly_chart(fig, use_container_width=True)


def _render_seed_group_trajectory_plots(agg_df: pd.DataFrame, metrics: list[str]) -> None:
    for metric in metrics:
        metric_df = agg_df[agg_df["metric"] == metric].sort_values("step")
        if metric_df.empty:
            continue
        mean = metric_df["mean"].astype(float)
        std = metric_df["std"].fillna(0.0).astype(float)
        steps = metric_df["step"]
        upper = mean + std
        lower = mean - std
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=steps, y=upper, mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=steps, y=lower, mode="lines", line=dict(width=0), fill="tonexty", name=f"{metric} ± std"))
        fig.add_trace(go.Scatter(x=steps, y=mean, mode="lines", name=f"{metric} mean"))
        fig.update_layout(title=f"{metric} aggregated over seeds", xaxis_title="step", yaxis_title=metric)
        st.plotly_chart(fig, use_container_width=True)


def _ordered_columns(df: pd.DataFrame, preferred: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    cols = [name for name in preferred if name in df.columns]
    cols += [name for name in df.columns if name not in cols]
    return df[cols]


def _render_audit_block(audit) -> None:
    if audit is None:
        return
    if audit.severity == "success":
        st.success(audit.summary)
    elif audit.severity == "warning":
        st.warning(audit.summary)
    else:
        st.error(audit.summary)

    mismatch_count = len(audit.held_fixed_violations) + len(audit.core_mismatches) + len(audit.secondary_confounds)
    protocol_conflict_count = len(audit.protocol_conflicts)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Validity", audit.validity_label)
    col2.metric("Seed alignment", audit.seed_alignment)
    col3.metric("Confounds", mismatch_count)
    col4.metric("Protocol conflicts", protocol_conflict_count)

    with st.expander("Audit details", expanded=False):
        st.dataframe(comparison_audit_details_table(audit), use_container_width=True)


def _render_pairing_summary(summary: dict) -> None:
    if not summary:
        return
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Paired seeds", int(summary.get("paired_seed_count", 0)))
    col2.metric("Reference only", int(summary.get("reference_only_seed_count", 0)))
    col3.metric("Candidate only", int(summary.get("candidate_only_seed_count", 0)))
    col4.metric("Complete pairing", "yes" if bool(summary.get("complete", False)) else "no")


def _render_free_compare(
    compare_service: CompareService,
    records_df: pd.DataFrame,
    x_col_fn: Callable,
) -> dict | None:
    if records_df.empty or len(records_df) < 2:
        st.info("Need at least 2 runs to compare.")
        return None
    labels = records_df["run_id"].astype(str).tolist()
    st.caption("Use free compare for exploratory overlays. Use matched compare when you want a controlled single-factor check.")
    selected = _multiselect("Select 2–4 runs", options=labels, default=st.session_state.get("free_compare_selected_runs", []), key="free_compare_selected_runs")
    if len(selected) < 2:
        st.warning("Select at least 2 runs.")
        return None
    if len(selected) > 4:
        st.warning("Select at most 4 runs.")
        return None
    result = compare_service.free_compare_result(selected)
    tabs = st.tabs(["Summary", "Config diff", "Trajectories"])
    with tabs[0]:
        st.dataframe(result.summary_df, use_container_width=True)
    with tabs[1]:
        st.dataframe(result.diff_df if not result.diff_df.empty else pd.DataFrame([{"message": "Configs are identical."}]), use_container_width=True)
    with tabs[2]:
        _render_metric_overlays(result.overlay_items, x_col_fn=x_col_fn, metrics=result.overlay_metrics)
    return {"df": result.summary_df, "file_name": "free_compare_summary.csv", "label": "free-compare summary"}


def _render_matched_compare(compare_service: CompareService, x_col_fn: Callable) -> dict | None:
    context = compare_service.matched_compare_context()
    if len(context.labels) < 2:
        st.info("Need at least 2 runs to compare.")
        return None

    control_col, axis_col, strict_col = st.columns([1.6, 1.1, 0.9])
    with control_col:
        reference_run_id = _selectbox("Reference run", options=context.labels, key="matched_reference_run")
    with axis_col:
        varying_axis = _selectbox("Varying axis", options=context.varying_axis_options, key="matched_varying_axis")
    with strict_col:
        strict_exact_match = _checkbox("Strict exact match", value=True, key="matched_strict_exact_match")

    preview = compare_service.matched_compare_result(
        reference_run_id=reference_run_id,
        varying_axis=varying_axis,
        strict_exact_match=strict_exact_match,
        candidate_run_id=None,
    )
    if preview is None:
        st.warning("Reference run could not be loaded.")
        return None

    if preview.candidates_df.empty:
        st.info("No matched runs found.")
        return None

    st.markdown("### Candidate runs")
    st.dataframe(
        _ordered_columns(
            preview.candidates_df,
            [
                "run_id", "varying_value", "audit_label", "audit_severity", "audit_seed_alignment",
                "audit_confound_count", "audit_summary", "status", "final_objective", "final_rel_recon_error",
            ],
        ),
        use_container_width=True,
    )

    candidate_ids = [item["run_id"] for item in preview.matches]
    default_candidate = st.session_state.get("matched_candidate_run") or candidate_ids[0]
    if default_candidate not in candidate_ids:
        default_candidate = candidate_ids[0]
    selected_index = candidate_ids.index(default_candidate)
    candidate_run_id = _selectbox("Candidate run", options=candidate_ids, index=selected_index, key="matched_candidate_run")

    result = compare_service.matched_compare_result(
        reference_run_id=reference_run_id,
        varying_axis=varying_axis,
        strict_exact_match=strict_exact_match,
        candidate_run_id=candidate_run_id,
    )
    if result is None:
        return None

    left, right = st.columns(2)
    left.markdown(f"**Reference:** `{result.reference['run_id']}`")
    right.markdown(f"**Candidate:** `{result.candidate['run_id']}`")
    st.caption(f"Controlled axis: {varying_axis}")
    _render_audit_block(result.audit)

    tabs = st.tabs(["Overview", "Config diff", "Trajectories"])
    with tabs[0]:
        st.dataframe(result.final_metrics_df, use_container_width=True)
    with tabs[1]:
        st.dataframe(result.config_diff_df, use_container_width=True)
    with tabs[2]:
        _render_metric_overlays(
            [
                (result.reference["run_id"], result.reference["config"], result.reference["summary"], result.reference["metrics"]),
                (result.candidate["run_id"], result.candidate["config"], result.candidate["summary"], result.candidate["metrics"]),
            ],
            x_col_fn=x_col_fn,
            metrics=result.trajectory_metrics,
        )
    return {"df": result.final_metrics_df, "file_name": "matched_compare_final_metrics.csv", "label": "matched-comparison final metrics"}


def _render_seed_aggregation(compare_service: CompareService) -> dict | None:
    base = compare_service.seed_aggregation_result()
    if base is None:
        st.info("No multi-seed groups found yet.")
        return None
    st.markdown("### Seed groups")
    st.dataframe(base.groups_df, use_container_width=True)
    selected_group_id = _selectbox("Seed group", options=list(base.groups_df["seed_group_id"]), key="seed_group_select")
    result = compare_service.seed_aggregation_result(selected_group_id)
    if result is None:
        return None
    tabs = st.tabs(["Final metrics", "Group runs", "Trajectories"])
    with tabs[0]:
        st.dataframe(result.agg_final_df, use_container_width=True)
    with tabs[1]:
        st.dataframe(result.run_rows_df, use_container_width=True)
    with tabs[2]:
        _render_seed_group_trajectory_plots(result.trajectory_df, result.trajectory_metrics)
    return {"df": result.agg_final_df, "file_name": "seed_aggregation_final_metrics.csv", "label": "seed-aggregation metrics"}


def _render_paired_seed_compare(compare_service: CompareService) -> dict | None:
    base = compare_service.seed_aggregation_result()
    if base is None or base.groups_df.empty:
        st.info("No multi-seed groups found yet.")
        return None
    group_ids = base.groups_df["seed_group_id"].astype(str).tolist()
    axis_options = compare_service.matched_compare_context().varying_axis_options

    control_col, axis_col, strict_col = st.columns([1.6, 1.1, 0.9])
    with control_col:
        reference_group_id = _selectbox("Reference seed group", options=group_ids, key="paired_seed_reference_group")
    with axis_col:
        varying_axis = _selectbox("Varying axis", options=axis_options, key="paired_seed_varying_axis")
    with strict_col:
        strict_exact_match = _checkbox("Strict exact match", value=True, key="paired_seed_strict_exact_match")

    preview = compare_service.paired_seed_group_compare_result(
        reference_group_id=reference_group_id,
        varying_axis=varying_axis,
        strict_exact_match=strict_exact_match,
        candidate_group_id=None,
    )
    if preview is None:
        st.warning("Reference seed group could not be loaded.")
        return None

    if preview.candidates_df.empty:
        st.info("No matched seed groups found.")
        return None

    st.markdown("### Candidate seed groups")
    st.dataframe(
        _ordered_columns(
            preview.candidates_df,
            [
                "seed_group_id", "varying_value", "paired_seed_count", "reference_only_seed_count",
                "candidate_only_seed_count", "pairing_complete", "audit_label", "audit_severity", "audit_summary",
            ],
        ),
        use_container_width=True,
    )

    candidate_group_ids = [item["seed_group_id"] for item in preview.matches]
    default_candidate = st.session_state.get("paired_seed_candidate_group") or candidate_group_ids[0]
    if default_candidate not in candidate_group_ids:
        default_candidate = candidate_group_ids[0]
    selected_index = candidate_group_ids.index(default_candidate)
    candidate_group_id = _selectbox("Candidate seed group", options=candidate_group_ids, index=selected_index, key="paired_seed_candidate_group")

    result = compare_service.paired_seed_group_compare_result(
        reference_group_id=reference_group_id,
        varying_axis=varying_axis,
        strict_exact_match=strict_exact_match,
        candidate_group_id=candidate_group_id,
    )
    if result is None:
        return None

    left, right = st.columns(2)
    left.markdown(f"**Reference group:** `{result.reference_group.get('seed_group_id', '')}`")
    right.markdown(f"**Candidate group:** `{result.candidate_group.get('seed_group_id', '')}`")
    st.caption(f"Controlled axis: {varying_axis}")
    _render_audit_block(result.audit)
    _render_pairing_summary(result.pairing_summary)

    tabs = st.tabs(["Delta summary", "Per-seed pairing", "Per-seed deltas"])
    with tabs[0]:
        st.dataframe(result.delta_summary_df, use_container_width=True)
    with tabs[1]:
        st.dataframe(result.pairing_df, use_container_width=True)
    with tabs[2]:
        st.dataframe(result.per_seed_delta_df, use_container_width=True)

    return {"df": result.delta_summary_df, "file_name": "paired_seed_compare_delta_summary.csv", "label": "paired-seed delta summary"}


def _failure_overlay_metrics(config: dict) -> list[str]:
    metrics = ["objective", "gradient_norm", "update_norm", "parameter_norm", "objective_delta"]
    problem_family = str(config.get("problem_family", "psd_sensing"))
    data_source = str(config.get("data_source", "synthetic"))
    if problem_family == "matrix_completion" and data_source == "movielens_100k":
        metrics += ["val_rmse", "test_rmse"]
    else:
        metrics += ["rel_recon_error"]
    if problem_family == "psd_sensing":
        metrics += ["min_eigenvalue_X", "negative_eigen_mass"]
    return metrics


def _render_failure_instability(compare_service: CompareService, x_col_fn: Callable) -> dict | None:
    context = compare_service.failure_context()
    if not context.labels:
        st.info("No runs found.")
        return None

    source = st.radio("Source", ["All runs", "Matched set"], horizontal=True, key="failure_source")
    reference_run_id = None
    varying_axis = context.varying_axis_options[0]
    strict_exact_match = True
    if source == "Matched set":
        reference_run_id = _selectbox("Reference run", options=context.labels, key="failure_reference_run")
        varying_axis = _selectbox("Varying axis", options=context.varying_axis_options, key="failure_varying_axis")
        strict_exact_match = _checkbox("Strict exact match", value=True, key="failure_strict_exact_match")

    initial = compare_service.failure_result(
        source=source,
        reference_run_id=reference_run_id,
        varying_axis=varying_axis,
        strict_exact_match=strict_exact_match,
        only_problematic=False,
        selected_issues=[],
        selected_group="",
        run_id=None,
    )
    if initial is None or initial.filtered_df.empty:
        st.info("No runs available for failure analysis.")
        return None
    failures_df = initial.filtered_df

    col1, col2, col3 = st.columns(3)
    with col1:
        only_problematic = _checkbox("Only problematic runs", value=True, key="failure_only_problematic")
    with col2:
        issue_options = sorted(failures_df["primary_issue"].dropna().unique().tolist())
        selected_issues = _multiselect("Issue filter", options=issue_options, default=issue_options, key="failure_issue_filter")
    with col3:
        group_options = [""] + sorted([str(x) for x in failures_df["group_id"].dropna().unique().tolist()])
        selected_group = _selectbox("Sweep/group filter", options=group_options, key="failure_group_filter")

    result = compare_service.failure_result(
        source=source,
        reference_run_id=reference_run_id,
        varying_axis=varying_axis,
        strict_exact_match=strict_exact_match,
        only_problematic=only_problematic,
        selected_issues=selected_issues,
        selected_group=selected_group,
        run_id=st.session_state.get("failure_run_select"),
    )
    if result is None:
        return None
    st.markdown("### Problematic runs")
    st.dataframe(result.filtered_df, use_container_width=True)
    if result.filtered_df.empty:
        st.info("No runs match the current filters.")
        return {"df": result.filtered_df, "file_name": "failure_instability_filtered.csv", "label": "failure-analysis table"}

    _selectbox("Run", options=result.filtered_df["run_id"].tolist(), key="failure_run_select")
    tabs = st.tabs(["Diagnostics", "Trajectories", "Seed instability"])
    with tabs[0]:
        st.dataframe(result.diagnostics_df, use_container_width=True)
    with tabs[1]:
        if result.selected_run is not None:
            _render_metric_overlays(result.overlay_items, x_col_fn=x_col_fn, metrics=_failure_overlay_metrics(result.selected_run["config"]))
    with tabs[2]:
        if result.seed_instability_df.empty:
            st.info("No multi-seed groups in this selection.")
        else:
            st.dataframe(result.seed_instability_df, use_container_width=True)
    return {"df": result.filtered_df, "file_name": "failure_instability_filtered.csv", "label": "failure-analysis table"}


def _render_frontier_plot(points_df: pd.DataFrame, fit_metric: str, complexity_metric: str, color_by: str, show_frontier: bool) -> pd.DataFrame:
    valid = points_df.dropna(subset=[fit_metric, complexity_metric]).copy()
    if valid.empty:
        st.info("No valid points for the selected metrics.")
        return valid
    hover = [c for c in ["point_id", "run_id", "seed_group_id", "optimizer", "parameterization", "explicit_reg_type", "stopping_mode", "problem_family", "data_source", "status"] if c in valid.columns]
    fig = px.scatter(
        valid,
        x=complexity_metric,
        y=fit_metric,
        color=color_by if color_by in valid.columns else None,
        hover_data=hover,
        title=f"{metric_label(fit_metric)} vs {metric_label(complexity_metric)}",
    )
    if show_frontier:
        frontier = pareto_frontier(valid, complexity_metric, fit_metric)
        if not frontier.empty:
            fig.add_trace(
                go.Scatter(
                    x=frontier[complexity_metric],
                    y=frontier[fit_metric],
                    mode="lines+markers",
                    name="frontier",
                    line=dict(width=3),
                )
            )
    fig.update_layout(xaxis_title=metric_label(complexity_metric), yaxis_title=metric_label(fit_metric))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(valid, use_container_width=True)
    return valid


def _render_complexity_vs_fit(compare_service: CompareService) -> dict | None:
    context = compare_service.frontier_context()
    if not context.labels:
        st.info("No runs available.")
        return None

    source_mode = st.radio(
        "Point source",
        options=["All runs", "Matched set", "Seed groups"],
        horizontal=True,
        key="frontier_source_mode",
    )

    reference_run_id = None
    varying_axis = context.varying_axis_options[0]
    strict_exact_match = True
    if source_mode == "Matched set":
        reference_run_id = _selectbox("Reference run", options=context.labels, key="frontier_reference_run")
        varying_axis = _selectbox("Varying axis", options=context.varying_axis_options, key="frontier_varying_axis")
        strict_exact_match = _checkbox("Strict exact match", value=True, key="frontier_strict_exact_match")

    points_result = compare_service.frontier_points_result(
        source_mode=source_mode,
        reference_run_id=reference_run_id,
        varying_axis=varying_axis,
        strict_exact_match=strict_exact_match,
    )
    if points_result is None or points_result.points_df.empty:
        st.info("No points available for this selection.")
        return None

    col1, col2, col3 = st.columns(3)
    with col1:
        fit_metric = _selectbox("Fit metric", options=points_result.fit_options, format_func=metric_label, key="frontier_fit_metric")
    with col2:
        complexity_metric = _selectbox("Complexity metric", options=points_result.complexity_options, format_func=metric_label, key="frontier_complexity_metric")
    with col3:
        color_by = _selectbox("Color by", options=COLOR_BY_OPTIONS, key="frontier_color_by")

    show_frontier = _checkbox("Show frontier", value=True, key="frontier_show_frontier")
    valid = _render_frontier_plot(points_result.points_df, fit_metric, complexity_metric, color_by, show_frontier)
    return {"df": valid, "file_name": "complexity_vs_fit_points.csv", "label": "frontier points"}



def render_compare_page(service: ComparePageService) -> None:
    consume_pending_analysis_view_state()
    compare_service = service.compare_service
    st.title("Compare")
    st.caption("Compare runs. Sweep inspection lives in Results → Sweeps.")

    st.markdown("### Comparison type")
    run_type = _radio(
        "Run comparison type",
        options=["Basic run comparison", "Controlled pair comparison", "Paired-seed comparison"],
        horizontal=True,
        key="compare_run_type",
    )

    records_df = compare_service.load_records()
    payload = None

    st.divider()
    st.markdown("### Select runs and inspect comparison")
    if run_type == "Basic run comparison":
        payload = _render_free_compare(compare_service, records_df, service.x_col)
    elif run_type == "Controlled pair comparison":
        payload = _render_matched_compare(compare_service, service.x_col)
    else:
        payload = _render_paired_seed_compare(compare_service)

    with st.expander("Additional run analyses", expanded=False):
        advanced_type = _selectbox(
            "Analysis",
            options=["Seed aggregation", "Complexity vs fit", "Failure / instability"],
            key="compare_advanced_run_type",
        )
        if advanced_type == "Seed aggregation":
            advanced_payload = _render_seed_aggregation(compare_service)
        elif advanced_type == "Complexity vs fit":
            advanced_payload = _render_complexity_vs_fit(compare_service)
        else:
            advanced_payload = _render_failure_instability(compare_service, service.x_col)
        if advanced_payload is not None:
            payload = advanced_payload

    if payload:
        st.divider()
        _render_export_panel(payload)
