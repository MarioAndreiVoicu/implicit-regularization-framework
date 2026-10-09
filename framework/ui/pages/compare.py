from __future__ import annotations

from typing import Any, Callable

import pandas as pd
import plotly.express as px
import streamlit as st

from framework.services.analysis_views import consume_pending_analysis_view_state
from framework.services.comparison_audit import comparison_audit_details_table
from framework.services.compare_service import CompareService
from framework.services.local_runs import metrics_for_config
from framework.services.ui_state import remove_conflicting_widget_defaults
from framework.ui.cards import render_value_columns
from framework.ui.compare_axes import (
    AUTO_AUDIT_AXIS as _AUTO_AUDIT_AXIS,
    NO_ADVANCED_AUDIT_AXIS as _NO_ADVANCED_AUDIT_AXIS,
    axis_label as _axis_label,
    split_audit_axis_options as _split_audit_axis_options,
)
from framework.ui.page_services import ComparePageService

_ST = st

_RUN_TABLE_COLUMNS = [
    "run_id",
    "created_at",
    "status",
    "problem_family",
    "data_source",
    "optimizer",
    "parameterization",
    "explicit_reg_type",
    "seed",
    "seed_group_id",
    "sweep_group_id",
    "final_objective",
    "final_rel_recon_error",
    "final_val_rmse",
    "final_test_rmse",
]

_DIFF_COLUMNS = ["field", "run_1", "run_2", "run_3", "run_4"]

_COMPARE_PRIMARY_FIT_METRICS = [
    "rel_recon_error",
    "relative_reconstruction_error",
    "val_rmse",
    "test_rmse",
    "train_rmse",
    "train_loss",
    "val_loss",
    "test_loss",
]

_COMPARE_OPTIMIZATION_METRICS = [
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

_COMPARE_COMPLEXITY_METRICS = [
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

_COMPARE_PROBLEM_DIAGNOSTIC_METRICS = [
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

_COMPARE_METRIC_GROUPS = [
    ("Main fit", _COMPARE_PRIMARY_FIT_METRICS),
    ("Optimization diagnostics", _COMPARE_OPTIMIZATION_METRICS),
    ("Implicit regularization / complexity", _COMPARE_COMPLEXITY_METRICS),
    ("Problem-specific diagnostics", _COMPARE_PROBLEM_DIAGNOSTIC_METRICS),
]

_COMPARE_METRIC_LABELS = {
    "objective": "Objective",
    "loss": "Loss",
    "data_loss": "Data loss",
    "observed_loss": "Observed loss",
    "explicit_reg_term": "Explicit regularization term",
    "objective_delta": "Objective delta",
    "rel_recon_error": "Relative reconstruction error",
    "relative_reconstruction_error": "Relative reconstruction error",
    "reconstruction_error": "Reconstruction error",
    "train_loss": "Training loss",
    "val_loss": "Validation loss",
    "test_loss": "Test loss",
    "train_rmse": "Training RMSE",
    "val_rmse": "Validation RMSE",
    "test_rmse": "Test RMSE",
    "gradient_norm": "Gradient norm",
    "grad_norm": "Gradient norm",
    "update_norm": "Update norm",
    "lr": "Learning rate",
    "trace_X": "Trace of X",
    "frobenius_norm_X": "Frobenius norm of X",
    "effective_rank": "Effective rank",
    "stable_rank": "Stable rank",
    "parameter_norm": "Parameter norm",
    "factor_u_norm": "Factor U norm",
    "factor_v_norm": "Factor V norm",
    "factor_norm_ratio": "Factor norm ratio",
    "factor_log_norm_ratio": "Log factor norm ratio",
    "target_commutator": "Target commutator",
    "mean_target_commutator": "Mean target commutator",
    "max_target_commutator": "Max target commutator",
    "subspace_alignment": "Subspace alignment",
    "subspace_projection_distance": "Subspace projection distance",
    "left_subspace_alignment": "Left subspace alignment",
    "right_subspace_alignment": "Right subspace alignment",
    "mean_subspace_alignment": "Mean subspace alignment",
    "left_subspace_projection_distance": "Left subspace projection distance",
    "right_subspace_projection_distance": "Right subspace projection distance",
    "mean_subspace_projection_distance": "Mean subspace projection distance",
    "target_row_coherence": "Target row coherence",
    "target_col_coherence": "Target column coherence",
    "target_condition_number": "Target condition number",
    "target_effective_rank": "Target effective rank",
    "target_stable_rank": "Target stable rank",
}

_COMPARE_CURATED_METRICS = [
    metric
    for _, metrics in _COMPARE_METRIC_GROUPS
    for metric in metrics
]

_COMPARE_TRAJECTORY_PREFIX = "compare_trajectory"

def _state_safe_kwargs(key: str | None, kwargs: dict, *, default_arg_names: tuple[str, ...]) -> dict:
    return remove_conflicting_widget_defaults(st.session_state, key, kwargs, default_arg_names=default_arg_names)


def _selectbox(label, options=None, **kwargs):
    key = kwargs.get("key")
    return _ST.selectbox(label, options, **_state_safe_kwargs(key, kwargs, default_arg_names=("index",)))


def _multiselect(label, options=None, **kwargs):
    key = kwargs.get("key")
    return _ST.multiselect(label, options, **_state_safe_kwargs(key, kwargs, default_arg_names=("default",)))



def _available_run_table(records_df: pd.DataFrame) -> pd.DataFrame:
    if records_df.empty:
        return records_df
    cols = [col for col in _RUN_TABLE_COLUMNS if col in records_df.columns]
    if not cols:
        return records_df
    return records_df[cols]


def _ordered_columns(df: pd.DataFrame, preferred: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    cols = [name for name in preferred if name in df.columns]
    cols += [name for name in df.columns if name not in cols]
    return df[cols]


def _render_export_panel(payload: dict | None) -> None:
    if not payload:
        return
    df = payload.get("df")
    if not isinstance(df, pd.DataFrame) or df.empty:
        return
    filename = payload.get("file_name", "compare_summary.csv")
    label = payload.get("label", "comparison summary")
    st.markdown("### Export")
    st.download_button(
        f"Download {label} (.csv)",
        data=df.to_csv(index=False),
        file_name=filename,
        mime="text/csv",
        key=f"export_{filename}",
    )


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        value = str(value)
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _metric_label(metric: str) -> str:
    return _COMPARE_METRIC_LABELS.get(str(metric), str(metric).replace("_", " ").strip().title())


def _metric_group_label(metric: str) -> str:
    metric = str(metric)
    for group, metrics in _COMPARE_METRIC_GROUPS:
        if metric in metrics:
            return group
    return "Other"


def _compare_metric_select_label(metric: str) -> str:
    return f"{_metric_label(metric)} · {_metric_group_label(metric)}"


def _numeric_columns(df: pd.DataFrame, *, excluded: set[str] | None = None) -> set[str]:
    excluded = excluded or set()
    if not isinstance(df, pd.DataFrame) or df.empty:
        return set()
    cols: set[str] = set()
    for col in df.columns:
        if col in excluded:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().any():
            cols.add(str(col))
    return cols


def _configured_metric_candidates(configs: list[dict[str, Any]]) -> list[str]:
    candidates: list[str] = []
    for config in configs:
        try:
            candidates.extend(metrics_for_config(config, "trajectory"))
        except Exception:
            continue
    return _unique_preserve_order(candidates)


def _compare_metric_options(loaded: list[tuple[str, dict, dict, pd.DataFrame]], candidate_metrics: list[str] | None = None) -> list[str]:
    dataframes = [df for _, _, _, df in loaded if isinstance(df, pd.DataFrame)]
    if len(dataframes) != len(loaded) or any(df.empty for df in dataframes):
        return []
    common: set[str] | None = None
    for df in dataframes:
        numeric = _numeric_columns(df, excluded={"step", "epoch", "iteration"})
        common = numeric if common is None else common & numeric
    if not common:
        return []
    configs = [config for _, config, _, _ in loaded]
    ordered_candidates = _unique_preserve_order([
        *(_COMPARE_CURATED_METRICS),
        *(_configured_metric_candidates(configs)),
        *(candidate_metrics or []),
        *sorted(common),
    ])
    return [metric for metric in ordered_candidates if metric in common]


def _primary_preference_sequence(configs: list[dict[str, Any]]) -> list[str]:
    families = {str((config or {}).get("problem_family") or "psd_sensing") for config in configs}
    sources = {str((config or {}).get("data_source") or "synthetic") for config in configs}
    if len(families) == 1 and "matrix_completion" in families:
        if len(sources) == 1 and "synthetic" in sources:
            return ["rel_recon_error", "relative_reconstruction_error", "val_rmse", "test_rmse", "objective", "loss"]
        if len(sources) == 1:
            return ["val_rmse", "test_rmse", "train_rmse", "val_loss", "test_loss", "train_loss", "objective", "loss"]
        return ["val_rmse", "test_rmse", "rel_recon_error", "relative_reconstruction_error", "objective", "loss"]
    if len(families) == 1 and "psd_sensing" in families:
        return ["rel_recon_error", "relative_reconstruction_error", "objective", "gradient_norm", "loss"]
    return ["val_rmse", "test_rmse", "rel_recon_error", "relative_reconstruction_error", "loss", "objective"]


def _default_compare_metric(metric_options: list[str], loaded: list[tuple[str, dict, dict, pd.DataFrame]]) -> tuple[str | None, str | None]:
    if not metric_options:
        return None, None
    configs = [config for _, config, _, _ in loaded]
    preference = _primary_preference_sequence(configs)
    preferred = preference[0] if preference else None
    for metric in preference:
        if metric in metric_options:
            return metric, preferred
    return metric_options[0], preferred


def _ensure_metric_selection(key: str, options: list[str], default: list[str] | str | None = None):
    if not options:
        return [] if isinstance(default, list) else None
    current = st.session_state.get(key)
    if isinstance(default, list):
        valid = [item for item in current if item in options] if isinstance(current, list) else []
        if not valid:
            valid = [item for item in default if item in options]
        st.session_state[key] = valid
        return valid
    if current not in options:
        st.session_state[key] = default if isinstance(default, str) and default in options else options[0]
    return st.session_state[key]


def _filter_valid_metric_list(values: Any, options: list[str], *, excluded: set[str] | None = None) -> list[str]:
    excluded = excluded or set()
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if str(value) in options and str(value) not in excluded]


def _trajectory_plot_frame(
    loaded: list[tuple[str, dict, dict, pd.DataFrame]],
    x_col_fn: Callable,
    metric: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for name, _, _, df in loaded:
        if not isinstance(df, pd.DataFrame) or df.empty or metric not in df.columns:
            continue
        x_col = x_col_fn(df)
        plot_df = pd.DataFrame({
            "step": df[x_col] if x_col in df.columns else range(len(df)),
            "value": pd.to_numeric(df[metric], errors="coerce"),
            "run": str(name),
        })
        plot_df["step"] = pd.to_numeric(plot_df["step"], errors="coerce")
        plot_df = plot_df.dropna(subset=["step", "value"]).sort_values("step")
        if not plot_df.empty:
            frames.append(plot_df)
    if not frames:
        return pd.DataFrame(columns=["step", "value", "run"])
    return pd.concat(frames, ignore_index=True)


def _render_compare_metric_plot(
    loaded: list[tuple[str, dict, dict, pd.DataFrame]],
    x_col_fn: Callable,
    metric: str,
) -> None:
    plot_df = _trajectory_plot_frame(loaded, x_col_fn, metric)
    if plot_df.empty:
        st.info(f"No valid trajectory values for {_metric_label(metric)}.")
        return
    label = _metric_label(metric)
    fig = px.line(
        plot_df,
        x="step",
        y="value",
        color="run",
        title=f"{label} vs step",
        labels={"step": "Step", "value": label, "run": "Run"},
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_metric_overlays(
    loaded: list[tuple[str, dict, dict, pd.DataFrame]],
    x_col_fn: Callable,
    metrics: list[str],
) -> None:
    metric_options = _compare_metric_options(loaded, candidate_metrics=metrics)
    if not metric_options:
        st.info("No shared metric trajectories found for the selected runs.")
        return

    default_metric, preferred_metric = _default_compare_metric(metric_options, loaded)
    selected_ids = tuple(name for name, _, _, _ in loaded)
    primary_key = f"{_COMPARE_TRAJECTORY_PREFIX}_primary_metric"
    additional_key = f"{_COMPARE_TRAJECTORY_PREFIX}_additional_metrics"
    selected_key = f"{_COMPARE_TRAJECTORY_PREFIX}_selected_runs"
    if st.session_state.get(selected_key) != selected_ids:
        st.session_state[selected_key] = selected_ids
        st.session_state[primary_key] = default_metric
        st.session_state[additional_key] = []
    _ensure_metric_selection(primary_key, metric_options, default=default_metric)

    primary_metric = _selectbox(
        "Metric to compare",
        options=metric_options,
        format_func=_compare_metric_select_label,
        key=primary_key,
    )
    if preferred_metric and preferred_metric not in metric_options and primary_metric:
        st.caption(f"{_metric_label(preferred_metric)} is not available for every selected run. Showing {_metric_label(primary_metric)} instead.")
    if primary_metric:
        _render_compare_metric_plot(loaded, x_col_fn, primary_metric)

    advanced_options = [metric for metric in metric_options if metric != primary_metric]
    with st.expander("More trajectory metrics", expanded=False):
        st.caption("Additional metrics are plotted separately so unrelated units are not overlaid on one axis.")
        st.session_state[additional_key] = _filter_valid_metric_list(
            st.session_state.get(additional_key),
            advanced_options,
            excluded={str(primary_metric)},
        )
        additional_metrics = _multiselect(
            "Additional metrics",
            options=advanced_options,
            format_func=_compare_metric_select_label,
            key=additional_key,
        )
        for metric in additional_metrics:
            st.markdown(f"**{_metric_label(metric)}**")
            _render_compare_metric_plot(loaded, x_col_fn, metric)


def _audit_score(audit) -> tuple[int, int, int]:
    status_score = {
        "valid_single_factor": 3,
        "weakly_controlled": 2,
        "protocol_conflict": 1,
        "invalid_context": 0,
    }.get(getattr(audit, "status", ""), 0)
    confound_count = (
        len(getattr(audit, "secondary_confounds", ()))
        + len(getattr(audit, "core_mismatches", ()))
        + len(getattr(audit, "held_fixed_violations", ()))
        + len(getattr(audit, "protocol_conflicts", ()))
    )
    return (status_score, len(getattr(audit, "allowed_differences", ())), -confound_count)


def _default_audit_axis(compare_service: CompareService, selected_run_ids: list[str], options: list[str]) -> str:
    if len(selected_run_ids) != 2 or not options:
        return options[0] if options else ""
    scored = []
    for axis in options:
        audit = compare_service.selected_pair_audit(
            reference_run_id=selected_run_ids[0],
            candidate_run_id=selected_run_ids[1],
            varying_axis=axis,
        )
        if audit is not None:
            scored.append((_audit_score(audit), axis))
    if not scored:
        return options[0]
    return max(scored, key=lambda item: item[0])[1]


def _render_audit_block(audit) -> None:
    if audit is None:
        return
    if audit.severity == "success":
        st.success(audit.summary)
    elif audit.severity == "warning":
        st.warning(audit.summary)
    else:
        st.error(audit.summary)

    confound_count = len(audit.held_fixed_violations) + len(audit.core_mismatches) + len(audit.secondary_confounds)
    protocol_conflict_count = len(audit.protocol_conflicts)
    render_value_columns(
        [
            ("Validity", audit.validity_label),
            ("Audit axis", _axis_label(audit.varying_axis)),
            ("Seed alignment", audit.seed_alignment),
            ("Confounds", confound_count + protocol_conflict_count),
        ]
    )

    with st.expander("Audit details", expanded=False):
        st.dataframe(comparison_audit_details_table(audit), use_container_width=True)


def _render_controlled_check(compare_service: CompareService, selected_run_ids: list[str]) -> None:
    st.markdown("### Controlled comparison check")
    if len(selected_run_ids) != 2:
        st.info("Select exactly two runs to audit whether the comparison is controlled.")
        return

    axis_options = compare_service.matched_compare_context().varying_axis_options
    if not axis_options:
        st.info("No audit axes are configured.")
        return

    common_axes, advanced_axes = _split_audit_axis_options(axis_options)
    if not common_axes and not advanced_axes:
        st.info("No audit axes are configured.")
        return

    default_axis = _default_audit_axis(compare_service, selected_run_ids, axis_options)
    selected_pair_key = tuple(selected_run_ids)
    common_options = [_AUTO_AUDIT_AXIS] + common_axes
    advanced_options = [_NO_ADVANCED_AUDIT_AXIS] + advanced_axes
    expected_key = "compare_audit_expected_axis"
    advanced_key = "compare_audit_advanced_axis"
    if (
        st.session_state.get("_compare_audit_pair") != selected_pair_key
        or st.session_state.get(expected_key) not in common_options
        or st.session_state.get(advanced_key) not in advanced_options
    ):
        st.session_state["_compare_audit_pair"] = selected_pair_key
        st.session_state[expected_key] = _AUTO_AUDIT_AXIS
        st.session_state[advanced_key] = _NO_ADVANCED_AUDIT_AXIS

    expected_axis = _selectbox(
        "Expected change",
        options=common_options,
        index=0,
        key=expected_key,
        format_func=_axis_label,
    )

    selected_advanced_axis = st.session_state.get(advanced_key, _NO_ADVANCED_AUDIT_AXIS)
    if selected_advanced_axis not in advanced_options:
        selected_advanced_axis = _NO_ADVANCED_AUDIT_AXIS

    with st.expander("Advanced comparison axes", expanded=False):
        if advanced_axes:
            selected_advanced_axis = _selectbox(
                "Advanced axis",
                options=advanced_options,
                index=advanced_options.index(selected_advanced_axis),
                key=advanced_key,
                format_func=_axis_label,
            )
            st.caption("Use this only when the intended change is more specific than the common options above.")
        else:
            st.info("No advanced axes are available for these runs.")
            selected_advanced_axis = _NO_ADVANCED_AUDIT_AXIS

    if selected_advanced_axis != _NO_ADVANCED_AUDIT_AXIS:
        varying_axis = selected_advanced_axis
        st.caption(f"Using advanced axis: {_axis_label(varying_axis)}.")
    elif expected_axis == _AUTO_AUDIT_AXIS:
        varying_axis = default_axis
        st.caption(f"Auto-detected expected change: {_axis_label(varying_axis)}.")
    else:
        varying_axis = expected_axis

    audit = compare_service.selected_pair_audit(
        reference_run_id=selected_run_ids[0],
        candidate_run_id=selected_run_ids[1],
        varying_axis=varying_axis,
    )
    _render_audit_block(audit)


def _render_selected_run_compare(
    compare_service: CompareService,
    records_df: pd.DataFrame,
    x_col_fn: Callable,
) -> dict | None:
    if records_df.empty or len(records_df) < 2:
        st.info("Need at least two runs to compare.")
        return None

    labels = records_df["run_id"].astype(str).tolist()
    default_selected = [
        run_id for run_id in st.session_state.get("free_compare_selected_runs", [])
        if run_id in labels
    ][:4]
    selected = _multiselect(
        "Select 2–4 runs",
        options=labels,
        default=default_selected,
        key="free_compare_selected_runs",
    )

    if len(selected) < 2:
        st.warning("Select at least two runs.")
        return None
    if len(selected) > 4:
        st.warning("Select at most four runs.")
        return None

    result = compare_service.free_compare_result(selected)
    if result.summary_df.empty:
        st.warning("The selected runs could not be loaded.")
        return None

    _render_controlled_check(compare_service, result.selected_run_ids)

    tabs = st.tabs(["Summary", "Config differences", "Trajectories"])
    with tabs[0]:
        st.dataframe(result.summary_df, use_container_width=True)
    with tabs[1]:
        diff_df = result.diff_df if not result.diff_df.empty else pd.DataFrame([{"message": "Configs are identical."}])
        st.dataframe(_ordered_columns(diff_df, _DIFF_COLUMNS), use_container_width=True)
    with tabs[2]:
        _render_metric_overlays(result.overlay_items, x_col_fn=x_col_fn, metrics=result.overlay_metrics)

    return {"df": result.summary_df, "file_name": "compare_summary.csv", "label": "comparison summary"}


def render_compare_page(service: ComparePageService) -> None:
    consume_pending_analysis_view_state()
    compare_service = service.compare_service
    records_df = compare_service.load_records()

    st.title("Compare")
    st.caption("Select a small set of runs, inspect metric differences, and check whether a two-run comparison is controlled.")

    with st.expander("Available runs", expanded=False):
        st.dataframe(_available_run_table(records_df), use_container_width=True)

    st.divider()
    payload = _render_selected_run_compare(compare_service, records_df, service.x_col)

    if payload:
        st.divider()
        _render_export_panel(payload)
