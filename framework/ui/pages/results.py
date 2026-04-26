from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from framework.services.run_manager_service import LocalRunManagerService
from framework.ui.page_services import ResultsExportService

_RUN_FILTER_PREFIX = "results_runs"
_SWEEP_PREFIX = "results_sweeps"

_RUN_TABLE_COLUMNS = [
    "run_id",
    "artifact_status",
    "favorite",
    "problem_family",
    "data_source",
    "optimizer",
    "parameterization",
    "seed",
    "group_id",
    "final_rel_recon_error",
    "status",
]

_RUN_INSPECT_COLUMNS = [
    "run_id",
    "run_name",
    "notes",
    "artifact_status",
    "tags",
    "favorite",
    "group_id",
    "seed_group_id",
    "problem_family",
    "data_source",
    "optimizer",
    "parameterization",
    "init_type",
    "init_scale",
    "lr",
    "steps",
    "final_rel_recon_error",
    "final_objective",
    "stop_reason",
    "integrity_status",
    "config_hash",
]

_SWEEP_TABLE_COLUMNS = [
    "group_id",
    "kind",
    "fields",
    "preset",
    "run_count",
    "seed_count",
    "seed_balance_status",
    "failure_count",
]

_RUN_PLOT_METRICS = [
    "objective",
    "rel_recon_error",
    "reconstruction_error",
    "train_rmse",
    "val_rmse",
    "test_rmse",
    "grad_norm",
    "gradient_norm",
    "update_norm",
    "parameter_norm",
    "trace_X",
    "frobenius_norm_X",
    "subspace_alignment",
    "subspace_projection_distance",
    "mean_subspace_alignment",
    "target_commutator",
    "mean_target_commutator",
    "max_target_commutator",
]

_SWEEP_PLOT_METRICS = [
    "final_rel_recon_error",
    "mean_final_rel_recon_error",
    "median_final_rel_recon_error",
    "min_final_rel_recon_error",
    "max_final_rel_recon_error",
    "final_objective",
    "mean_final_objective",
    "median_final_objective",
    "min_final_objective",
    "max_final_objective",
    "final_train_rmse",
    "mean_final_train_rmse",
    "final_val_rmse",
    "mean_final_val_rmse",
    "final_test_rmse",
    "mean_final_test_rmse",
    "final_trace_X",
    "mean_final_trace_X",
    "final_frobenius_norm_X",
    "mean_final_frobenius_norm_X",
    "final_subspace_alignment",
    "mean_final_subspace_alignment",
    "final_subspace_projection_distance",
    "mean_final_subspace_projection_distance",
    "final_mean_subspace_alignment",
    "mean_final_mean_subspace_alignment",
    "final_mean_target_commutator",
    "mean_final_mean_target_commutator",
    "final_max_target_commutator",
    "mean_final_max_target_commutator",
]

_METRIC_LABELS = {
    "objective": "Objective",
    "final_objective": "Final objective",
    "mean_final_objective": "Mean final objective",
    "rel_recon_error": "Relative reconstruction error",
    "final_rel_recon_error": "Final relative reconstruction error",
    "mean_final_rel_recon_error": "Mean final relative reconstruction error",
    "train_rmse": "Train RMSE",
    "val_rmse": "Validation RMSE",
    "test_rmse": "Test RMSE",
    "final_train_rmse": "Final train RMSE",
    "final_val_rmse": "Final validation RMSE",
    "final_test_rmse": "Final test RMSE",
    "mean_final_train_rmse": "Mean final train RMSE",
    "mean_final_val_rmse": "Mean final validation RMSE",
    "mean_final_test_rmse": "Mean final test RMSE",
    "grad_norm": "Gradient norm",
    "gradient_norm": "Gradient norm",
    "update_norm": "Update norm",
    "parameter_norm": "Parameter norm",
    "trace_X": "Trace(X)",
    "final_trace_X": "Final trace(X)",
    "frobenius_norm_X": "Frobenius norm(X)",
    "final_frobenius_norm_X": "Final Frobenius norm(X)",
    "subspace_alignment": "Subspace alignment",
    "final_subspace_alignment": "Final subspace alignment",
    "subspace_projection_distance": "Subspace projection distance",
    "final_subspace_projection_distance": "Final subspace projection distance",
    "mean_subspace_alignment": "Mean subspace alignment",
    "final_mean_subspace_alignment": "Final mean subspace alignment",
    "mean_target_commutator": "Mean target commutator",
    "final_mean_target_commutator": "Final mean target commutator",
    "max_target_commutator": "Max target commutator",
    "final_max_target_commutator": "Final max target commutator",
}


def _apply_session_updates(updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        st.session_state[key] = value


def _existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def _curated_numeric_columns(df: pd.DataFrame, candidates: list[str], *, excluded: set[str] | None = None) -> list[str]:
    excluded = excluded or set()
    cols: list[str] = []
    for col in candidates:
        if col in excluded or col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().any():
            cols.append(col)
    return cols


def _metric_label(metric: str) -> str:
    return _METRIC_LABELS.get(str(metric), str(metric).replace("_", " ").strip().title())


def _ensure_valid_metric_selection(key: str, options: list[str], default: list[str] | str | None = None):
    if not options:
        return [] if isinstance(default, list) else None
    current = st.session_state.get(key)
    if isinstance(default, list):
        if isinstance(current, list):
            valid = [item for item in current if item in options]
        else:
            valid = []
        if not valid:
            valid = [item for item in default if item in options]
        if not valid:
            valid = options[: min(4, len(options))]
        st.session_state[key] = valid
        return valid
    if current not in options:
        st.session_state[key] = default if isinstance(default, str) and default in options else options[0]
    return st.session_state[key]


def _compact_runs_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = _existing_columns(df, _RUN_TABLE_COLUMNS)
    out = df[cols].copy() if cols else df.copy()
    if "final_rel_recon_error" in out.columns:
        out["final_rel_recon_error"] = pd.to_numeric(out["final_rel_recon_error"], errors="coerce")
    return out




def _load_run_metrics(run_manager_service: LocalRunManagerService, run_id: str) -> pd.DataFrame:
    metrics_path = run_manager_service.runs_dir / str(run_id) / "metrics.csv"
    if not metrics_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(metrics_path)
    except Exception:
        return pd.DataFrame()


def _render_run_plots(run_manager_service: LocalRunManagerService, run_id: str) -> None:
    metrics_df = _load_run_metrics(run_manager_service, run_id)
    if metrics_df.empty:
        st.info("No metric trajectory file found for this run.")
        return

    metric_options = _curated_numeric_columns(metrics_df, _RUN_PLOT_METRICS, excluded={"step"})
    if not metric_options:
        st.info("No supported metric trajectories available for plotting.")
        return

    preferred_defaults = [
        metric for metric in ["objective", "rel_recon_error", "val_rmse", "test_rmse", "gradient_norm", "grad_norm"]
        if metric in metric_options
    ]
    if not preferred_defaults:
        preferred_defaults = metric_options[: min(3, len(metric_options))]
    key = f"{_RUN_FILTER_PREFIX}_plot_metrics"
    default_selection = _ensure_valid_metric_selection(key, metric_options, default=preferred_defaults[: min(4, len(preferred_defaults))])
    selected_metrics = st.multiselect(
        "Plot metrics",
        metric_options,
        default=default_selection,
        format_func=_metric_label,
        key=key,
    )
    if not selected_metrics:
        return

    plot_df = metrics_df[[*(["step"] if "step" in metrics_df.columns else []), *selected_metrics]].copy()
    if "step" in plot_df.columns:
        plot_df = plot_df.set_index("step")
    st.line_chart(plot_df[selected_metrics], use_container_width=True)


def _filter_runs(df: pd.DataFrame, *, search: str, artifact_status: str, family: str, favorite_only: bool) -> pd.DataFrame:
    filtered = df.copy()
    search = str(search or "").strip().lower()
    if search:
        mask = pd.Series(False, index=filtered.index)
        for col in ["run_id", "run_name", "notes", "tags", "group_id", "optimizer", "parameterization", "problem_family", "data_source"]:
            if col in filtered.columns:
                mask = mask | filtered[col].astype(str).str.lower().str.contains(search, na=False)
        filtered = filtered[mask]
    if artifact_status and artifact_status != "(all)" and "artifact_status" in filtered.columns:
        if artifact_status == "(unreviewed)":
            filtered = filtered[filtered["artifact_status"].fillna("").astype(str).str.strip() == ""]
        else:
            filtered = filtered[filtered["artifact_status"].fillna("").astype(str) == artifact_status]
    if family and family != "(all)" and "problem_family" in filtered.columns:
        filtered = filtered[filtered["problem_family"].fillna("").astype(str) == family]
    if favorite_only and "favorite" in filtered.columns:
        filtered = filtered[filtered["favorite"] == True]
    return filtered.reset_index(drop=True)


def _filter_sweeps(df: pd.DataFrame, *, search: str, kind: str) -> pd.DataFrame:
    filtered = df.copy()
    search = str(search or "").strip().lower()
    if search:
        mask = pd.Series(False, index=filtered.index)
        for col in ["group_id", "kind", "fields", "preset", "seed_balance_status", "load_error"]:
            if col in filtered.columns:
                mask = mask | filtered[col].astype(str).str.lower().str.contains(search, na=False)
        filtered = filtered[mask]
    if kind and kind != "(all)" and "kind" in filtered.columns:
        filtered = filtered[filtered["kind"].fillna("").astype(str) == kind]
    return filtered.reset_index(drop=True)


def _run_family_options(df: pd.DataFrame) -> list[str]:
    if df.empty or "problem_family" not in df.columns:
        return ["(all)"]
    values = sorted(x for x in df["problem_family"].fillna("").astype(str).unique().tolist() if x)
    return ["(all)"] + values


def _sweep_kind_options(df: pd.DataFrame) -> list[str]:
    if df.empty or "kind" not in df.columns:
        return ["(all)"]
    values = sorted(x for x in df["kind"].fillna("").astype(str).unique().tolist() if x)
    return ["(all)"] + values


def _render_run_inspector(row: pd.Series, *, run_manager_service: LocalRunManagerService) -> None:
    st.markdown("### Inspect selected run")
    compact = {col: row.get(col) for col in _RUN_INSPECT_COLUMNS if col in row.index and pd.notna(row.get(col))}
    top_cols = st.columns(4)
    top_cols[0].metric("Status", str(compact.get("status", "—")))
    top_cols[1].metric("Artifact", str(compact.get("artifact_status") or "unreviewed"))
    top_cols[2].metric("Family", str(compact.get("problem_family", "—")))
    err = compact.get("final_rel_recon_error")
    top_cols[3].metric("Final rel. error", "—" if err is None else f"{float(err):.3g}" if isinstance(err, (int, float)) else str(err))

    with st.expander("Run details", expanded=True):
        st.json(compact, expanded=True)

    st.markdown("#### Metric trajectories")
    _render_run_plots(run_manager_service, str(compact.get("run_id") or row.get("run_id")))


def _render_runs_actions(*, selected_ids: list[str], run_manager_service: LocalRunManagerService, results_export_service: ResultsExportService) -> None:
    action_cols = st.columns([1.0, 1.0, 1.0, 1.15])
    with action_cols[0]:
        if st.button("Open in Compare", disabled=len(selected_ids) < 2, key=f"{_RUN_FILTER_PREFIX}_compare"):
            _apply_session_updates(run_manager_service.build_free_compare_handoff(selected_ids))
            st.rerun()
    with action_cols[1]:
        status_options = ["", "exploratory", "keep", "thesis_candidate", "final", "invalid"]
        new_status = st.selectbox("Set status", status_options, format_func=lambda x: "unreviewed" if x == "" else x, key=f"{_RUN_FILTER_PREFIX}_set_status")
        if st.button("Apply status", disabled=not selected_ids, key=f"{_RUN_FILTER_PREFIX}_apply_status"):
            run_manager_service.set_artifact_status(selected_ids, new_status)
            st.rerun()
    with action_cols[2]:
        tag_text = st.text_input("Set tags", key=f"{_RUN_FILTER_PREFIX}_tags", placeholder="comma,separated")
        if st.button("Apply tags", disabled=not selected_ids, key=f"{_RUN_FILTER_PREFIX}_apply_tags"):
            run_manager_service.set_tags(selected_ids, tag_text)
            st.rerun()
    with action_cols[3]:
        include_reports = st.checkbox("Reports", value=False, key=f"{_RUN_FILTER_PREFIX}_export_reports")
        include_artifacts = st.checkbox("Artifacts", value=False, key=f"{_RUN_FILTER_PREFIX}_export_artifacts")
        if selected_ids:
            filename, data = results_export_service.build_selected_runs_bundle(
                run_ids=selected_ids,
                include_run_reports=include_reports,
                include_artifacts=include_artifacts,
            )
            st.download_button("Download selected", data=data, file_name=filename, mime="application/zip", key=f"{_RUN_FILTER_PREFIX}_download_selected")


def _render_runs_tab(*, run_manager_service: LocalRunManagerService, results_export_service: ResultsExportService) -> None:
    run_manager_service.consume_pending_view_state()
    browse = run_manager_service.build_browse_context()
    df_all = browse.runs_df
    if df_all.empty:
        st.info("No runs found yet.")
        return

    st.markdown("### Search & results")
    search = st.text_input("Search runs", key="results_runs_search", placeholder="Search by id, name, tag, field, optimizer, metric...")
    with st.expander("Advanced filters", expanded=False):
        filter_cols = st.columns([1.0, 1.0, 0.75])
        artifact_status = filter_cols[0].selectbox("Artifact status", browse.artifact_status_options, key=f"{_RUN_FILTER_PREFIX}_artifact_status")
        family = filter_cols[1].selectbox("Family", _run_family_options(df_all), key=f"{_RUN_FILTER_PREFIX}_family")
        favorite_only = filter_cols[2].checkbox("Favorites only", key=f"{_RUN_FILTER_PREFIX}_favorites")

    filtered = _filter_runs(df_all, search=search, artifact_status=artifact_status, family=family, favorite_only=favorite_only)
    st.caption(f"{len(filtered)} run(s) shown")
    if filtered.empty:
        st.info("No runs match the current search/filters.")
        return

    run_ids = filtered["run_id"].astype(str).tolist() if "run_id" in filtered.columns else []
    default_inspect = st.session_state.get(f"{_RUN_FILTER_PREFIX}_inspect_run", run_ids[0])
    if default_inspect not in run_ids:
        default_inspect = run_ids[0]
    with st.expander("Show matching runs table", expanded=False):
        st.dataframe(_compact_runs_df(filtered), use_container_width=True, hide_index=True)

    inspect_id = st.selectbox("Selected run", run_ids, index=run_ids.index(default_inspect), key=f"{_RUN_FILTER_PREFIX}_inspect_run")

    inspect_row = filtered[filtered["run_id"].astype(str) == str(inspect_id)].iloc[0]
    st.divider()
    _render_run_inspector(inspect_row, run_manager_service=run_manager_service)

    st.divider()
    st.markdown("### Selection / curation / export")
    with st.expander("Selection, curation, and export actions", expanded=False):
        selected_ids = st.multiselect("Select runs", run_ids, default=[inspect_id] if inspect_id in run_ids else [], key=f"{_RUN_FILTER_PREFIX}_selected_runs")
        _render_runs_actions(selected_ids=selected_ids, run_manager_service=run_manager_service, results_export_service=results_export_service)


def _sweep_summary_row(group_id: str, analysis_view, *, load_error: str | None = None) -> dict[str, Any]:
    if analysis_view is None:
        return {"group_id": group_id, "kind": None, "fields": None, "preset": None, "run_count": None, "seed_count": None, "seed_balance_status": None, "failure_count": None, "load_error": load_error}
    summary = dict(analysis_view.summary or {})
    return {
        "group_id": group_id,
        "kind": analysis_view.kind,
        "fields": ", ".join(str(x) for x in analysis_view.fields),
        "preset": summary.get("preset") or analysis_view.metadata.get("preset"),
        "run_count": summary.get("run_count"),
        "seed_count": summary.get("seed_count"),
        "seed_balance_status": summary.get("seed_balance_status"),
        "failure_count": summary.get("failure_count"),
        "load_error": load_error,
    }


def _render_sweep_actions(*, selected_group_id: str, analysis_view, results_export_service: ResultsExportService) -> None:
    run_ids: list[str] = []
    if not analysis_view.raw_df.empty and "run" in analysis_view.raw_df.columns:
        run_ids = [str(x) for x in analysis_view.raw_df["run"].dropna().astype(str).unique().tolist()]
    compare_ids = st.multiselect("Runs for Compare", run_ids, default=run_ids[: min(2, len(run_ids))], key=f"{_SWEEP_PREFIX}_compare_runs")
    if st.button("Open runs in Compare", disabled=len(compare_ids) < 2, key=f"{_SWEEP_PREFIX}_open_compare"):
        st.session_state["nav_page"] = "Compare"
        st.session_state["compare_mode"] = "Free Compare"
        st.session_state["free_compare_selected_runs"] = compare_ids[:4]
        st.rerun()
    include_reports = st.checkbox("Include per-run reports", value=False, key=f"{_SWEEP_PREFIX}_reports")
    include_artifacts = st.checkbox("Include artifacts", value=False, key=f"{_SWEEP_PREFIX}_artifacts")
    if st.button("Build sweep export", key=f"{_SWEEP_PREFIX}_build_export"):
        bundle = results_export_service.build_sweep_bundle(
            group_id=selected_group_id,
            include_run_reports=include_reports,
            include_artifacts=include_artifacts,
        )
        st.download_button("Download sweep bundle", data=bundle.data, file_name=bundle.filename, mime=bundle.content_type, key=f"{_SWEEP_PREFIX}_download_export")



def _sweep_display_df(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    if "seed_balance_status" in display.columns:
        display = display.rename(columns={"seed_balance_status": "seed_balance"})
    return display


def _preferred_sweep_metric_columns(df: pd.DataFrame, fields: list[str]) -> list[str]:
    excluded = {"run", "seed", "status", "source_kind", "seed_group_id", *[str(field) for field in fields]}
    return _curated_numeric_columns(df, _SWEEP_PLOT_METRICS, excluded=excluded)


def _render_sweep_plots(analysis_view) -> None:
    fields = [str(field) for field in analysis_view.fields]
    plot_df = analysis_view.aggregate_df.copy()
    if plot_df.empty:
        plot_df = analysis_view.raw_df.copy()
    if plot_df.empty or not fields:
        st.info("No plottable sweep data available.")
        return

    metric_options = _preferred_sweep_metric_columns(plot_df, fields)
    if not metric_options:
        st.info("No numeric sweep metrics available for plotting.")
        return

    key = f"{_SWEEP_PREFIX}_plot_metric"
    _ensure_valid_metric_selection(key, metric_options, default=metric_options[0])
    metric = st.selectbox("Sweep plot metric", metric_options, format_func=_metric_label, key=key)
    if len(fields) == 1 and fields[0] in plot_df.columns:
        x_field = fields[0]
        line_df = plot_df[[x_field, metric]].dropna().copy()
        if line_df.empty:
            st.info("No valid rows for the selected sweep plot.")
            return
        line_df = line_df.sort_values(x_field).set_index(x_field)
        st.line_chart(line_df[metric], use_container_width=True)
        return

    if len(fields) >= 2 and all(field in plot_df.columns for field in fields[:2]):
        x_field, y_field = fields[:2]
        heat_df = plot_df[[x_field, y_field, metric]].dropna().copy()
        if heat_df.empty:
            st.info("No valid rows for the selected heatmap.")
            return
        pivot = heat_df.pivot_table(index=y_field, columns=x_field, values=metric, aggfunc="mean")
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 4.5))
            image = ax.imshow(pivot.values, aspect="auto", origin="lower")
            ax.set_xlabel(x_field)
            ax.set_ylabel(y_field)
            ax.set_title(metric)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([str(x) for x in pivot.columns], rotation=45, ha="right")
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([str(y) for y in pivot.index])
            fig.colorbar(image, ax=ax)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
        except Exception:
            st.dataframe(pivot, use_container_width=True)
        return

    st.info("Sweep plot is available for 1D or 2D sweeps with numeric metrics.")



def _render_sweeps_tab(*, results_export_service: ResultsExportService) -> None:
    group_ids = results_export_service.load_sweep_group_ids()
    if not group_ids:
        st.info("No saved sweeps found yet.")
        return

    st.markdown("### Search & results")
    search = st.text_input("Search sweeps", key="results_sweeps_search", placeholder="Search by group id, fields, preset, status...")

    analysis_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for group_id in group_ids:
        load_error = None
        try:
            analysis_view = results_export_service.load_saved_sweep_analysis(group_id)
        except Exception as exc:
            analysis_view = None
            load_error = str(exc)
        analysis_cache[group_id] = analysis_view
        rows.append(_sweep_summary_row(group_id, analysis_view, load_error=load_error))

    table_df = pd.DataFrame(rows)
    with st.expander("Advanced filters", expanded=False):
        kind = st.selectbox("Sweep kind", _sweep_kind_options(table_df), key=f"{_SWEEP_PREFIX}_kind")

    filtered = _filter_sweeps(table_df, search=search, kind=kind)
    st.caption(f"{len(filtered)} sweep(s) shown")
    if filtered.empty:
        st.info("No sweeps match the current search/filters.")
        return

    filtered_group_ids = filtered["group_id"].astype(str).tolist()
    default_group_id = st.session_state.get(f"{_SWEEP_PREFIX}_selected_group", filtered_group_ids[0])
    if default_group_id not in filtered_group_ids:
        default_group_id = filtered_group_ids[0]
    with st.expander("Show matching sweeps table", expanded=False):
        st.dataframe(_sweep_display_df(filtered[_existing_columns(filtered, _SWEEP_TABLE_COLUMNS + ["load_error"])]), use_container_width=True, hide_index=True)

    selected_group_id = st.selectbox("Selected sweep", filtered_group_ids, index=filtered_group_ids.index(default_group_id), key=f"{_SWEEP_PREFIX}_selected_group")
    analysis_view = analysis_cache.get(selected_group_id) or results_export_service.load_saved_sweep_analysis(selected_group_id)
    if analysis_view is None:
        selected_row = filtered[filtered["group_id"].astype(str) == str(selected_group_id)]
        error_text = selected_row.iloc[0].get("load_error", "Selected sweep could not be loaded.") if not selected_row.empty else "Selected sweep could not be loaded."
        st.warning(f"Selected sweep could not be loaded: {error_text}")
        with st.expander("Matching sweeps", expanded=False):
            st.dataframe(_sweep_display_df(filtered[_existing_columns(filtered, _SWEEP_TABLE_COLUMNS + ["load_error"])]), use_container_width=True, hide_index=True)
        return

    summary = dict(analysis_view.summary or {})
    cols = st.columns(4)
    cols[0].metric("Kind", str(analysis_view.kind or "—"))
    cols[1].metric("Runs", int(summary.get("run_count", 0) or 0))
    cols[2].metric("Seeds", int(summary.get("seed_count", 0) or 0))
    cols[3].metric("Seed balance", str(summary.get("seed_balance_status") or "—"))

    st.divider()
    st.markdown("### Inspect selected sweep")
    detail_payload = {"group_id": analysis_view.group_id, "kind": analysis_view.kind, "fields": list(analysis_view.fields), "summary": summary}
    with st.expander("Sweep details", expanded=True):
        st.json(detail_payload, expanded=True)

    st.markdown("#### Sweep plots")
    _render_sweep_plots(analysis_view)

    st.divider()
    st.markdown("### Selection / curation / export")
    with st.expander("Actions / export", expanded=False):
        _render_sweep_actions(selected_group_id=selected_group_id, analysis_view=analysis_view, results_export_service=results_export_service)

    if analysis_view.confound_warnings:
        with st.expander("Warnings", expanded=False):
            st.dataframe(pd.DataFrame(analysis_view.confound_warnings), use_container_width=True, hide_index=True)

    with st.expander("Aggregate preview", expanded=True):
        if analysis_view.aggregate_df.empty:
            st.info("No aggregate rows available.")
        else:
            st.dataframe(analysis_view.aggregate_df.head(100), use_container_width=True, hide_index=True)

    if not analysis_view.raw_df.empty:
        with st.expander("Runs in sweep", expanded=False):
            preview_cols = _existing_columns(analysis_view.raw_df, ["run", "seed", *list(analysis_view.fields), "status", "final_rel_recon_error", "final_objective"])
            st.dataframe(analysis_view.raw_df[preview_cols].head(200), use_container_width=True)


def render_results_page(*, run_manager_service: LocalRunManagerService, results_export_service: ResultsExportService) -> None:
    st.title("Results")
    st.caption("Search, inspect, curate, and export completed artifacts.")

    runs_tab, sweeps_tab = st.tabs(["Runs", "Sweeps"])
    with runs_tab:
        _render_runs_tab(run_manager_service=run_manager_service, results_export_service=results_export_service)
    with sweeps_tab:
        _render_sweeps_tab(results_export_service=results_export_service)
