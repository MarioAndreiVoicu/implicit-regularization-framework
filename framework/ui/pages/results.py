from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from framework.services.run_manager_service import LocalRunManagerService
from framework.services.local_runs import metrics_for_config
from framework.ui.page_services import ResultsExportService
from framework.services.serialization import make_json_safe
from framework.ui.cards import render_value_columns
from framework.ui.formatting import format_display_value
from framework.ui.shell import apply_navigation_handoff

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

_RUN_PRIMARY_FIT_METRICS = [
    "rel_recon_error",
    "relative_reconstruction_error",
    "val_rmse",
    "test_rmse",
    "train_rmse",
    "train_loss",
    "val_loss",
    "test_loss",
]

_RUN_OPTIMIZATION_METRICS = [
    "objective",
    "loss",
    "data_loss",
    "observed_loss",
    "explicit_reg_term",
    "objective_delta",
    "gradient_norm",
    "grad_norm",
    "update_norm",
    "lr",
]

_RUN_COMPLEXITY_METRICS = [
    "trace_X",
    "frobenius_norm_X",
    "effective_rank",
    "stable_rank",
    "parameter_norm",
    "factor_u_norm",
    "factor_v_norm",
    "factor_norm_ratio",
    "factor_log_norm_ratio",
]

_RUN_PROBLEM_DIAGNOSTIC_METRICS = [
    "reconstruction_error",
    "target_commutator",
    "mean_target_commutator",
    "max_target_commutator",
    "subspace_alignment",
    "subspace_projection_distance",
    "left_subspace_alignment",
    "right_subspace_alignment",
    "mean_subspace_alignment",
    "left_subspace_projection_distance",
    "right_subspace_projection_distance",
    "mean_subspace_projection_distance",
    "target_row_coherence",
    "target_col_coherence",
    "target_condition_number",
    "target_effective_rank",
    "target_stable_rank",
]

_RUN_PLOT_METRIC_GROUPS = [
    ("Main fit", _RUN_PRIMARY_FIT_METRICS),
    ("Optimization diagnostics", _RUN_OPTIMIZATION_METRICS),
    ("Implicit regularization / complexity", _RUN_COMPLEXITY_METRICS),
    ("Problem-specific diagnostics", _RUN_PROBLEM_DIAGNOSTIC_METRICS),
]

_RUN_PLOT_METRICS = [
    metric
    for _, metrics in _RUN_PLOT_METRIC_GROUPS
    for metric in metrics
]
_SWEEP_BASE_PLOT_METRICS = [
    "final_rel_recon_error",
    "final_objective",
    "final_train_rmse",
    "final_val_rmse",
    "final_test_rmse",
    "final_train_loss",
    "final_val_loss",
    "final_test_loss",
    "final_trace_X",
    "final_frobenius_norm_X",
    "final_effective_rank",
    "final_stable_rank",
    "final_gradient_norm",
    "final_subspace_alignment",
    "final_subspace_projection_distance",
    "final_mean_subspace_alignment",
    "final_mean_subspace_projection_distance",
    "final_mean_target_commutator",
    "final_max_target_commutator",
]
_SWEEP_AGGREGATE_SUFFIXES = ["_mean", "_median", "_min", "_max"]
_SWEEP_NON_METRIC_COLUMNS = {
    "run",
    "run_id",
    "seed",
    "status",
    "source_kind",
    "seed_group_id",
    "group_id",
    "seed_count",
    "run_count",
    "expected_seed_count",
    "seed_balance_status",
    "seed_coverage_ratio",
    "failure_count",
    "load_error",
}

_METRIC_LABELS = {
    "objective": "Objective",
    "loss": "Loss",
    "data_loss": "Data loss",
    "observed_loss": "Observed loss",
    "explicit_reg_term": "Explicit regularization term",
    "objective_delta": "Objective delta",
    "final_objective": "Final objective",
    "mean_final_objective": "Mean final objective",
    "rel_recon_error": "Relative reconstruction error",
    "relative_reconstruction_error": "Relative reconstruction error",
    "final_rel_recon_error": "Final relative reconstruction error",
    "mean_final_rel_recon_error": "Mean final relative reconstruction error",
    "train_rmse": "Train RMSE",
    "val_rmse": "Validation RMSE",
    "test_rmse": "Test RMSE",
    "final_train_rmse": "Final train RMSE",
    "final_val_rmse": "Final validation RMSE",
    "final_test_rmse": "Final test RMSE",
    "final_train_loss": "Final train loss",
    "final_val_loss": "Final validation loss",
    "final_test_loss": "Final test loss",
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
    "final_effective_rank": "Final effective rank",
    "final_stable_rank": "Final stable rank",
    "final_gradient_norm": "Final gradient norm",
    "subspace_alignment": "Subspace alignment",
    "final_subspace_alignment": "Final subspace alignment",
    "subspace_projection_distance": "Subspace projection distance",
    "final_subspace_projection_distance": "Final subspace projection distance",
    "mean_subspace_alignment": "Mean subspace alignment",
    "final_mean_subspace_alignment": "Final mean subspace alignment",
    "final_mean_subspace_projection_distance": "Final mean subspace projection distance",
    "mean_target_commutator": "Mean target commutator",
    "final_mean_target_commutator": "Final mean target commutator",
    "max_target_commutator": "Max target commutator",
    "final_max_target_commutator": "Final max target commutator",
    "reconstruction_error": "Reconstruction error",
    "train_loss": "Train loss",
    "val_loss": "Validation loss",
    "test_loss": "Test loss",
    "lr": "Learning rate",
    "effective_rank": "Effective rank",
    "stable_rank": "Stable rank",
    "factor_u_norm": "Factor U norm",
    "factor_v_norm": "Factor V norm",
    "factor_norm_ratio": "Factor norm ratio",
    "factor_log_norm_ratio": "Log factor norm ratio",
    "left_subspace_alignment": "Left subspace alignment",
    "right_subspace_alignment": "Right subspace alignment",
    "left_subspace_projection_distance": "Left subspace projection distance",
    "right_subspace_projection_distance": "Right subspace projection distance",
    "mean_subspace_projection_distance": "Mean subspace projection distance",
    "target_row_coherence": "Target row coherence",
    "target_col_coherence": "Target col coherence",
    "target_condition_number": "Target condition number",
    "target_effective_rank": "Target effective rank",
    "target_stable_rank": "Target stable rank",
}


def _apply_session_updates(updates: dict[str, Any]) -> None:
    apply_navigation_handoff(updates)


def _existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def _literal_search_mask(df: pd.DataFrame, columns: list[str], search: str) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    needle = str(search or "").strip().lower()
    if not needle:
        return mask
    for col in columns:
        if col in df.columns:
            haystack = df[col].fillna("").astype(str).str.lower()
            mask = mask | haystack.str.contains(needle, na=False, regex=False)
    return mask


def _truthy_filter_values(series: pd.Series) -> pd.Series:
    return series.apply(lambda value: bool(value) if isinstance(value, bool) else str(value).strip().lower() in {"true", "1", "yes", "y"})


def _ensure_valid_multiselect_state(key: str, options: list[str], default: list[str] | None = None) -> list[str]:
    option_set = {str(option) for option in options}
    if key in st.session_state:
        current = st.session_state.get(key)
        if isinstance(current, (list, tuple, set)):
            valid = [str(item) for item in current if str(item) in option_set]
        elif current is None:
            valid = []
        else:
            valid = [str(current)] if str(current) in option_set else []
    else:
        valid = [str(item) for item in (default or []) if str(item) in option_set]
    st.session_state[key] = valid
    return valid


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


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:] if text else text


def _metric_label(metric: str) -> str:
    metric = str(metric)
    suffix_labels = {
        "_mean": "Mean",
        "_median": "Median",
        "_std": "Std. dev. of",
        "_min": "Minimum",
        "_max": "Maximum",
        "_sem": "SEM of",
        "_cv": "CV of",
        "_range": "Range of",
    }
    for suffix, prefix in suffix_labels.items():
        if metric.endswith(suffix):
            base_label = _metric_label(metric[: -len(suffix)])
            return f"{prefix} {_lower_first(base_label)}"
    return _METRIC_LABELS.get(metric, metric.replace("_", " ").strip().title())


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




def _unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        value = str(value)
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _metric_group_label(metric: str) -> str:
    metric = str(metric)
    for group, metrics in _RUN_PLOT_METRIC_GROUPS:
        if metric in metrics:
            return group
    return "Other"


def _run_plot_metric_candidates(run_metadata: dict[str, Any] | None = None) -> list[str]:
    metadata = run_metadata or {}
    try:
        configured = metrics_for_config(
            {
                "problem_family": metadata.get("problem_family"),
                "data_source": metadata.get("data_source"),
            },
            "trajectory",
        )
    except Exception:
        configured = []
    grouped = [metric for _, metrics in _RUN_PLOT_METRIC_GROUPS for metric in metrics]
    return _unique_preserve_order([*grouped, *configured])


def _run_plot_metric_options(metrics_df: pd.DataFrame, run_metadata: dict[str, Any] | None = None) -> list[str]:
    return _curated_numeric_columns(
        metrics_df,
        _run_plot_metric_candidates(run_metadata),
        excluded={"step"},
    )


def _default_run_plot_metric(metric_options: list[str], run_metadata: dict[str, Any] | None = None) -> str | None:
    if not metric_options:
        return None
    metadata = run_metadata or {}
    family = str(metadata.get("problem_family") or "psd_sensing")
    source = str(metadata.get("data_source") or "synthetic")
    if family == "matrix_completion" and source != "synthetic":
        preferred = ["val_rmse", "test_rmse", "val_loss", "test_loss", "train_loss", "objective", "loss"]
    else:
        preferred = ["rel_recon_error", "relative_reconstruction_error", "objective", "loss"]
    for metric in preferred:
        if metric in metric_options:
            return metric
    for metric in _RUN_PRIMARY_FIT_METRICS:
        if metric in metric_options:
            return metric
    return metric_options[0]


def _filter_valid_metric_list(values: Any, options: list[str], *, excluded: set[str] | None = None) -> list[str]:
    excluded = excluded or set()
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if str(value) in options and str(value) not in excluded]


def _run_metric_select_label(metric: str) -> str:
    return f"{_metric_label(metric)} · {_metric_group_label(metric)}"


def _compact_runs_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = _existing_columns(df, _RUN_TABLE_COLUMNS)
    out = df[cols].copy() if cols else df.copy()
    if "final_rel_recon_error" in out.columns:
        out["final_rel_recon_error"] = pd.to_numeric(out["final_rel_recon_error"], errors="coerce")
    return out


def _compact_run_label(row: pd.Series) -> str:
    run_id = str(row.get("run_id", "")).strip()
    pieces = []
    for col in ["problem_family", "data_source", "optimizer", "parameterization", "seed"]:
        value = row.get(col)
        if pd.notna(value) and str(value).strip():
            label = "seed" if col == "seed" else str(value)
            pieces.append(label if col != "seed" else f"seed {format_display_value(value)}")
    suffix = " | ".join(pieces)
    return f"{run_id} — {suffix}" if suffix else run_id


def _render_key_value_table(data: dict[str, Any], *, title: str | None = None) -> None:
    cleaned = make_json_safe(data)
    if not isinstance(cleaned, dict) or not cleaned:
        return
    rows = []
    for key, value in cleaned.items():
        if value in (None, ""):
            continue
        rows.append({"Field": str(key).replace("_", " ").title(), "Value": format_display_value(value)})
    if not rows:
        return
    if title:
        st.markdown(title)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)




def _load_run_metrics(run_manager_service: LocalRunManagerService, run_id: str) -> pd.DataFrame:
    metrics_path = run_manager_service.runs_dir / str(run_id) / "metrics.csv"
    if not metrics_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(metrics_path)
    except Exception:
        return pd.DataFrame()


def _render_single_metric_plot(metrics_df: pd.DataFrame, metric: str) -> None:
    columns = [metric]
    if "step" in metrics_df.columns:
        columns.insert(0, "step")
    plot_df = metrics_df[columns].copy()
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df.dropna(subset=[metric])
    if plot_df.empty:
        st.info(f"No valid values for {_metric_label(metric)}.")
        return
    if "step" in plot_df.columns:
        plot_df["step"] = pd.to_numeric(plot_df["step"], errors="coerce")
        plot_df = plot_df.dropna(subset=["step"]).sort_values("step").set_index("step")
    display_df = plot_df[[metric]].rename(columns={metric: _metric_label(metric)})
    st.line_chart(display_df, use_container_width=True)


def _render_run_plots(run_manager_service: LocalRunManagerService, run_id: str, run_metadata: dict[str, Any] | None = None) -> None:
    metrics_df = _load_run_metrics(run_manager_service, run_id)
    if metrics_df.empty:
        st.info("No metric trajectory file found for this run.")
        return

    metric_options = _run_plot_metric_options(metrics_df, run_metadata)
    if not metric_options:
        st.info("No supported metric trajectories available for plotting.")
        return

    default_metric = _default_run_plot_metric(metric_options, run_metadata)
    primary_key = f"{_RUN_FILTER_PREFIX}_primary_plot_metric"
    additional_key = f"{_RUN_FILTER_PREFIX}_additional_plot_metrics"
    run_key = f"{_RUN_FILTER_PREFIX}_plot_run_id"
    if st.session_state.get(run_key) != run_id:
        st.session_state[run_key] = run_id
        st.session_state[primary_key] = default_metric
        st.session_state[additional_key] = []
    _ensure_valid_metric_selection(primary_key, metric_options, default=default_metric)

    primary_metric = st.selectbox(
        "Metric",
        metric_options,
        format_func=_run_metric_select_label,
        key=primary_key,
    )
    if primary_metric:
        _render_single_metric_plot(metrics_df, primary_metric)

    advanced_options = [metric for metric in metric_options if metric != primary_metric]
    with st.expander("More trajectory metrics", expanded=False):
        st.caption("Optional diagnostics are plotted separately so unrelated units are not overlaid on one axis.")
        st.session_state[additional_key] = _filter_valid_metric_list(
            st.session_state.get(additional_key),
            advanced_options,
            excluded={str(primary_metric)},
        )
        additional_metrics = st.multiselect(
            "Additional metrics",
            advanced_options,
            format_func=_run_metric_select_label,
            key=additional_key,
        )
        for metric in additional_metrics:
            st.markdown(f"**{_metric_label(metric)}**")
            _render_single_metric_plot(metrics_df, metric)


def _filter_runs(df: pd.DataFrame, *, search: str, artifact_status: str, family: str, favorite_only: bool) -> pd.DataFrame:
    filtered = df.copy()
    search = str(search or "").strip().lower()
    if search:
        filtered = filtered[
            _literal_search_mask(
                filtered,
                ["run_id", "run_name", "notes", "tags", "group_id", "optimizer", "parameterization", "problem_family", "data_source"],
                search,
            )
        ]
    if artifact_status and artifact_status != "(all)" and "artifact_status" in filtered.columns:
        if artifact_status == "(unreviewed)":
            filtered = filtered[filtered["artifact_status"].fillna("").astype(str).str.strip() == ""]
        else:
            filtered = filtered[filtered["artifact_status"].fillna("").astype(str) == artifact_status]
    if family and family != "(all)" and "problem_family" in filtered.columns:
        filtered = filtered[filtered["problem_family"].fillna("").astype(str) == family]
    if favorite_only and "favorite" in filtered.columns:
        filtered = filtered[_truthy_filter_values(filtered["favorite"])]
    return filtered.reset_index(drop=True)


def _filter_sweeps(df: pd.DataFrame, *, search: str, kind: str) -> pd.DataFrame:
    filtered = df.copy()
    search = str(search or "").strip().lower()
    if search:
        filtered = filtered[
            _literal_search_mask(
                filtered,
                ["group_id", "kind", "fields", "preset", "seed_balance_status", "load_error"],
                search,
            )
        ]
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
    st.markdown("### Selected run")
    compact = {col: row.get(col) for col in _RUN_INSPECT_COLUMNS if col in row.index and pd.notna(row.get(col))}
    err = compact.get("final_rel_recon_error")
    render_value_columns(
        [
            ("Status", str(compact.get("status", "—"))),
            ("Artifact", str(compact.get("artifact_status") or "unreviewed")),
            ("Family", str(compact.get("problem_family", "—"))),
            ("Final rel. error", "—" if err is None else f"{float(err):.3g}" if isinstance(err, (int, float)) else str(err)),
        ]
    )

    key_fields = [
        "run_name",
        "problem_family",
        "data_source",
        "optimizer",
        "parameterization",
        "init_type",
        "init_scale",
        "lr",
        "steps",
        "seed",
        "group_id",
        "seed_group_id",
        "stop_reason",
    ]
    config_preview = {field: compact.get(field) for field in key_fields if compact.get(field) not in (None, "")}
    if config_preview:
        with st.expander("Key configuration", expanded=True):
            _render_key_value_table(config_preview)

    st.markdown("#### Trajectories")
    _render_run_plots(run_manager_service, str(compact.get("run_id") or row.get("run_id")), compact)

    with st.expander("Technical details", expanded=False):
        st.json(make_json_safe(compact), expanded=True)


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

    search_cols = st.columns([1.4, 0.9, 0.9, 0.55])
    search = search_cols[0].text_input("Search runs", key="results_runs_search", placeholder="ID, tag, optimizer, dataset...")
    artifact_status = search_cols[1].selectbox("Artifact status", browse.artifact_status_options, key=f"{_RUN_FILTER_PREFIX}_artifact_status")
    family = search_cols[2].selectbox("Family", _run_family_options(df_all), key=f"{_RUN_FILTER_PREFIX}_family")
    favorite_only = search_cols[3].checkbox("Favorites", key=f"{_RUN_FILTER_PREFIX}_favorites")

    filtered = _filter_runs(df_all, search=search, artifact_status=artifact_status, family=family, favorite_only=favorite_only)
    st.caption(f"{len(filtered)} run(s) shown")
    if filtered.empty:
        st.info("No runs match the current search/filters.")
        return

    run_ids = filtered["run_id"].dropna().astype(str).tolist() if "run_id" in filtered.columns else []
    if not run_ids:
        st.warning("Matching run records do not contain valid run IDs.")
        return
    default_inspect = st.session_state.get(f"{_RUN_FILTER_PREFIX}_inspect_run", run_ids[0])
    if default_inspect not in run_ids:
        default_inspect = run_ids[0]
    st.dataframe(_compact_runs_df(filtered), use_container_width=True, hide_index=True)

    label_by_id = {
        str(row.get("run_id")): _compact_run_label(row)
        for _, row in filtered.iterrows()
        if pd.notna(row.get("run_id"))
    }
    inspect_id = st.selectbox(
        "Inspect run",
        run_ids,
        index=run_ids.index(default_inspect),
        key=f"{_RUN_FILTER_PREFIX}_inspect_run",
        format_func=lambda run_id: label_by_id.get(str(run_id), str(run_id)),
    )

    inspect_row = filtered[filtered["run_id"].astype(str) == str(inspect_id)].iloc[0]
    st.divider()
    _render_run_inspector(inspect_row, run_manager_service=run_manager_service)

    with st.expander("Actions", expanded=False):
        selected_key = f"{_RUN_FILTER_PREFIX}_selected_runs"
        _ensure_valid_multiselect_state(selected_key, run_ids, default=[inspect_id] if inspect_id in run_ids else [])
        selected_ids = st.multiselect("Select runs", run_ids, key=selected_key)
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
    group_key = str(selected_group_id).replace(" ", "_")
    compare_key = f"{_SWEEP_PREFIX}_compare_runs_{group_key}"
    _ensure_valid_multiselect_state(compare_key, run_ids, default=run_ids[: min(2, len(run_ids))])
    compare_ids = st.multiselect("Runs for Compare", run_ids, key=compare_key)
    if st.button("Open runs in Compare", disabled=len(compare_ids) < 2, key=f"{_SWEEP_PREFIX}_open_compare_{group_key}"):
        apply_navigation_handoff({
            "nav_page": "Compare",
            "compare_mode": "Free Compare",
            "free_compare_selected_runs": compare_ids[:4],
        })
        st.rerun()
    include_reports = st.checkbox("Include per-run reports", value=False, key=f"{_SWEEP_PREFIX}_reports_{group_key}")
    include_artifacts = st.checkbox("Include artifacts", value=False, key=f"{_SWEEP_PREFIX}_artifacts_{group_key}")
    export_key = f"{_SWEEP_PREFIX}_export_bundle_{group_key}"
    export_settings_key = f"{_SWEEP_PREFIX}_export_settings_{group_key}"
    export_settings = (bool(include_reports), bool(include_artifacts))
    if st.session_state.get(export_settings_key) != export_settings:
        st.session_state[export_settings_key] = export_settings
        st.session_state.pop(export_key, None)
    if st.button("Build sweep export", key=f"{_SWEEP_PREFIX}_build_export_{group_key}"):
        st.session_state[export_key] = results_export_service.build_sweep_bundle(
            group_id=selected_group_id,
            include_run_reports=include_reports,
            include_artifacts=include_artifacts,
        )
    bundle = st.session_state.get(export_key)
    if bundle is not None:
        st.download_button("Download sweep bundle", data=bundle.data, file_name=bundle.filename, mime=bundle.content_type, key=f"{_SWEEP_PREFIX}_download_export_{group_key}")


def _compact_sweep_label(row: pd.Series) -> str:
    group_id = str(row.get("group_id", "")).strip()
    pieces = []
    for col in ["kind", "fields", "preset", "run_count"]:
        value = row.get(col)
        if pd.notna(value) and str(value).strip():
            if col == "run_count":
                pieces.append(f"{format_display_value(value)} runs")
            else:
                pieces.append(str(value))
    suffix = " | ".join(pieces)
    return f"{group_id} — {suffix}" if suffix else group_id



def _sweep_display_df(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    if "seed_balance_status" in display.columns:
        display = display.rename(columns={"seed_balance_status": "seed_balance"})
    return display


def _sweep_metric_candidates(metric_options: dict[str, str] | None = None) -> list[str]:
    base_metrics = _unique_preserve_order([*(metric_options or {}).keys(), *_SWEEP_BASE_PLOT_METRICS])
    candidates: list[str] = []
    for metric in base_metrics:
        candidates.extend(f"{metric}{suffix}" for suffix in _SWEEP_AGGREGATE_SUFFIXES)
        candidates.append(metric)
    return _unique_preserve_order(candidates)


def _preferred_sweep_metric_columns(
    df: pd.DataFrame,
    fields: list[str],
    metric_options: dict[str, str] | None = None,
) -> list[str]:
    excluded = set(_SWEEP_NON_METRIC_COLUMNS) | {str(field) for field in fields}
    preferred = _curated_numeric_columns(df, _sweep_metric_candidates(metric_options), excluded=excluded)
    fallback: list[str] = []
    for col in df.columns:
        if str(col) in excluded or str(col) in preferred:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().any():
            fallback.append(str(col))
    return _unique_preserve_order([*preferred, *fallback])



def _declared_sweep_axis_values(analysis_view: Any, field: str, observed_values: list[Any]) -> list[Any]:
    metadata = dict(getattr(analysis_view, "metadata", {}) or {})
    values_payload = metadata.get("values")
    if not isinstance(values_payload, dict):
        sweep_spec = metadata.get("sweep_spec")
        if isinstance(sweep_spec, dict):
            values_payload = sweep_spec.get("values")
    declared = list(values_payload.get(field, [])) if isinstance(values_payload, dict) else []
    ordered: list[Any] = []
    for value in declared:
        if value in observed_values and value not in ordered:
            ordered.append(value)
    for value in observed_values:
        if value not in ordered:
            ordered.append(value)
    return ordered


def _sweep_coupling(analysis_view: Any) -> str:
    metadata = dict(getattr(analysis_view, "metadata", {}) or {})
    sweep_spec = metadata.get("sweep_spec")
    if isinstance(sweep_spec, dict) and sweep_spec.get("coupling"):
        return str(sweep_spec.get("coupling"))
    return str(metadata.get("coupling") or "grid")

def _render_sweep_plots(analysis_view) -> None:
    fields = [str(field) for field in analysis_view.fields]
    plot_df = analysis_view.aggregate_df.copy()
    if plot_df.empty:
        plot_df = analysis_view.raw_df.copy()
    if plot_df.empty or not fields:
        st.info("No plottable sweep data available.")
        return

    metric_options = _preferred_sweep_metric_columns(
        plot_df,
        fields,
        metric_options=dict(getattr(analysis_view, "metric_options", {}) or {}),
    )
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
        display_series = line_df[metric].rename(_metric_label(metric))
        st.line_chart(display_series, use_container_width=True)
        return

    if len(fields) >= 2 and all(field in plot_df.columns for field in fields[:2]):
        if _sweep_coupling(analysis_view) != "grid":
            st.info("Heatmaps are shown for full-grid 2D sweeps. This saved sweep uses coupled pairs, so use the aggregate table instead.")
            return
        x_field, y_field = fields[:2]
        heat_df = plot_df[[x_field, y_field, metric]].dropna().copy()
        if heat_df.empty:
            st.info("No valid rows for the selected heatmap.")
            return
        pivot = heat_df.pivot_table(index=y_field, columns=x_field, values=metric, aggfunc="mean")
        x_order = _declared_sweep_axis_values(analysis_view, x_field, list(pivot.columns))
        y_order = _declared_sweep_axis_values(analysis_view, y_field, list(pivot.index))
        if x_order:
            pivot = pivot.reindex(columns=x_order)
        if y_order:
            pivot = pivot.reindex(index=y_order)
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 4.5))
            image = ax.imshow(pivot.values, aspect="auto", origin="lower")
            ax.set_xlabel(x_field)
            ax.set_ylabel(y_field)
            ax.set_title(_metric_label(metric))
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

    search_cols = st.columns([1.6, 0.8])
    search = search_cols[0].text_input("Search sweeps", key="results_sweeps_search", placeholder="Group id, fields, preset, status...")

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
    kind = search_cols[1].selectbox("Sweep kind", _sweep_kind_options(table_df), key=f"{_SWEEP_PREFIX}_kind")

    filtered = _filter_sweeps(table_df, search=search, kind=kind)
    st.caption(f"{len(filtered)} sweep(s) shown")
    if filtered.empty:
        st.info("No sweeps match the current search/filters.")
        return

    filtered_group_ids = filtered["group_id"].astype(str).tolist()
    default_group_id = st.session_state.get(f"{_SWEEP_PREFIX}_selected_group", filtered_group_ids[0])
    if default_group_id not in filtered_group_ids:
        default_group_id = filtered_group_ids[0]
    st.dataframe(_sweep_display_df(filtered[_existing_columns(filtered, _SWEEP_TABLE_COLUMNS + ["load_error"])]), use_container_width=True, hide_index=True)

    label_by_group_id = {
        str(row.get("group_id")): _compact_sweep_label(row)
        for _, row in filtered.iterrows()
        if pd.notna(row.get("group_id"))
    }
    selected_group_id = st.selectbox(
        "Inspect sweep",
        filtered_group_ids,
        index=filtered_group_ids.index(default_group_id),
        key=f"{_SWEEP_PREFIX}_selected_group",
        format_func=lambda group_id: label_by_group_id.get(str(group_id), str(group_id)),
    )
    analysis_view = analysis_cache.get(selected_group_id) or results_export_service.load_saved_sweep_analysis(selected_group_id)
    if analysis_view is None:
        selected_row = filtered[filtered["group_id"].astype(str) == str(selected_group_id)]
        error_text = selected_row.iloc[0].get("load_error", "Selected sweep could not be loaded.") if not selected_row.empty else "Selected sweep could not be loaded."
        st.warning(f"Selected sweep could not be loaded: {error_text}")
        with st.expander("Matching sweeps", expanded=False):
            st.dataframe(_sweep_display_df(filtered[_existing_columns(filtered, _SWEEP_TABLE_COLUMNS + ["load_error"])]), use_container_width=True, hide_index=True)
        return

    summary = dict(analysis_view.summary or {})
    render_value_columns(
        [
            ("Kind", str(analysis_view.kind or "—")),
            ("Runs", int(summary.get("run_count", 0) or 0)),
            ("Seeds", int(summary.get("seed_count", 0) or 0)),
            ("Seed balance", str(summary.get("seed_balance_status") or "—")),
        ]
    )

    st.markdown("### Selected sweep")
    detail_payload = {"group_id": analysis_view.group_id, "kind": analysis_view.kind, "fields": list(analysis_view.fields), "summary": summary}
    _render_key_value_table(
        {
            "group_id": analysis_view.group_id,
            "fields": ", ".join(str(field) for field in analysis_view.fields),
            "preset": summary.get("preset"),
            "failure_count": summary.get("failure_count"),
        },
        title="#### Sweep setup",
    )

    st.markdown("#### Plots")
    _render_sweep_plots(analysis_view)

    if analysis_view.confound_warnings:
        with st.expander("Warnings", expanded=False):
            st.dataframe(pd.DataFrame(analysis_view.confound_warnings), use_container_width=True, hide_index=True)

    with st.expander("Aggregate results", expanded=True):
        if analysis_view.aggregate_df.empty:
            st.info("No aggregate rows available.")
        else:
            st.dataframe(analysis_view.aggregate_df.head(100), use_container_width=True, hide_index=True)

    with st.expander("Actions", expanded=False):
        _render_sweep_actions(selected_group_id=selected_group_id, analysis_view=analysis_view, results_export_service=results_export_service)

    with st.expander("Technical details", expanded=False):
        st.json(make_json_safe(detail_payload), expanded=True)

    if not analysis_view.raw_df.empty:
        with st.expander("Runs in sweep", expanded=False):
            preview_cols = _existing_columns(analysis_view.raw_df, ["run", "seed", *list(analysis_view.fields), "status", "final_rel_recon_error", "final_objective"])
            st.dataframe(analysis_view.raw_df[preview_cols].head(200), use_container_width=True)


def render_results_page(*, run_manager_service: LocalRunManagerService, results_export_service: ResultsExportService) -> None:
    st.title("Results")
    st.caption("Browse completed runs and saved sweeps, inspect the main metrics, and use optional actions only when needed.")

    runs_tab, sweeps_tab = st.tabs(["Runs", "Sweeps"])
    with runs_tab:
        _render_runs_tab(run_manager_service=run_manager_service, results_export_service=results_export_service)
    with sweeps_tab:
        _render_sweeps_tab(results_export_service=results_export_service)
