from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import streamlit as st

from framework.services.ui_state import remove_conflicting_widget_defaults
from framework.services.serialization import json_dumps
from framework.ui.formatting import format_display_sequence

from framework.core.device import (
    list_device_preferences,
    list_dtype_preferences,
)
from framework.core.experiments import list_optimizers, list_problem_families
from framework.core.parameterization import list_parameterizations
from framework.core.regularization import explicit_reg_types_for_parameterization
from framework.core.psd_regimes import list_psd_regimes
from framework.core.schedules import list_lr_schedule_types
from framework.core.stopping import list_stopping_modes
from framework.services.analysis_views import consume_pending_analysis_view_state
from framework.services.data_sources import get_data_source_stats, is_real_data_source, list_data_sources
from framework.services.sweep_analysis import (
    aggregate_sweep_1d,
    aggregate_sweep_2d,
    build_sweep_analysis_from_execution_result,
)
from framework.services.sweeps import SweepRequest, generate_numeric_sweep_values, parse_sweep_values_text
from framework.services.new_run import normalize_choice_state_value, normalize_optional_numeric_state_value, normalize_text_state_value
from framework.ui.page_services import SweepsPageService

def _state_safe_kwargs(key: str | None, kwargs: dict, *, default_arg_names: tuple[str, ...]) -> dict:
    return remove_conflicting_widget_defaults(st.session_state, key, kwargs, default_arg_names=default_arg_names)


def _selectbox(label, options, **kwargs):
    key = kwargs.get("key")
    return st.selectbox(label, options, **_state_safe_kwargs(key, kwargs, default_arg_names=("index",)))


def _radio(label, options, **kwargs):
    key = kwargs.get("key")
    return st.radio(label, options, **_state_safe_kwargs(key, kwargs, default_arg_names=("index",)))


def _number_input(label, **kwargs):
    key = kwargs.get("key")
    return st.number_input(label, **_state_safe_kwargs(key, kwargs, default_arg_names=("value",)))


def _checkbox(label, **kwargs):
    key = kwargs.get("key")
    return st.checkbox(label, **_state_safe_kwargs(key, kwargs, default_arg_names=("value",)))


def _text_input(label, **kwargs):
    key = kwargs.get("key")
    return st.text_input(label, **_state_safe_kwargs(key, kwargs, default_arg_names=("value",)))


def _sweep_audit_messages(report) -> list[str]:
    messages: list[str] = []
    seen: set[str] = set()
    for warning in getattr(report, "warnings", []):
        if getattr(warning, "code", "") in {"low_seed_count", "interaction_sweep", "multi_factor_sweep"}:
            continue
        message = str(getattr(warning, "message", ""))
        if message and message not in seen:
            messages.append(message)
            seen.add(message)
    return messages


MAX_RUNS_1D_DEFAULT = 24
MAX_RUNS_2D_DEFAULT = 36
SWEEP_MAX_RUNS_1D_KEY = "sweep_cell_cap_1d_v2"
SWEEP_MAX_RUNS_2D_KEY = "sweep_cell_cap_2d_v2"


def _normalize_choice_state(key: str, options: list[str] | tuple[str, ...], *, default: str) -> str:
    return normalize_choice_state_value(st.session_state, key, options=options, default=default)


def _normalize_numeric_state(
    key: str,
    *,
    default: int | float,
    cast_type: type[int] | type[float],
    min_value: int | float | None = None,
    max_value: int | float | None = None,
) -> int | float:
    return normalize_optional_numeric_state_value(
        st.session_state,
        key,
        default=default,
        cast_type=cast_type,
        min_value=min_value,
        max_value=max_value,
    )


def _default_preset_values(problem_family: str, field: str) -> list[object]:
    if field == "lr":
        return [1e-4, 3e-4, 1e-3, 3e-3, 1e-2] if problem_family == "psd_sensing" else [2e-2, 5e-2, 1e-1, 2e-1, 4e-1]
    if field == "init_scale":
        return [1e-4, 1e-3, 1e-2, 1e-1] if problem_family == "psd_sensing" else [1e-2, 3e-2, 1e-1, 3e-1]
    if field == "noncommutativity_level":
        return [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    if field == "mc_factor_imbalance_level":
        return [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    if field == "coherence_level":
        return [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    if field == "mc_spectrum_decay":
        return [0.0, 0.25, 0.5, 1.0, 2.0]
    if field == "explicit_reg_strength":
        return [0.0, 1e-4, 1e-3, 1e-2]
    if field == "factor_rank":
        return [2, 4, 6, 8, 10, 12]
    return []


def _sweep_field_catalog(problem_family: str, data_source: str, *, allow_2d: bool) -> dict[str, dict[str, object]]:
    catalog: dict[str, dict[str, object]] = {
        "lr": {"label": "learning rate", "value_type": "float", "supports_range": True, "supports_log": True},
        "init_scale": {"label": "init scale", "value_type": "float", "supports_range": True, "supports_log": True},
        "optimizer": {"label": "optimizer", "value_type": "choice", "choices": list_optimizers(problem_family)},
        "parameterization": {"label": "parameterization", "value_type": "choice", "choices": list_parameterizations(problem_family)},
        "lr_schedule_type": {"label": "lr schedule", "value_type": "choice", "choices": list_lr_schedule_types()},
        "explicit_reg_strength": {"label": "explicit reg strength", "value_type": "float", "supports_range": True, "supports_log": True},
        "factor_rank": {"label": "factor rank", "value_type": "int", "supports_range": True, "supports_log": False},
    }
    if problem_family == "psd_sensing":
        catalog["psd_regime"] = {"label": "PSD regime", "value_type": "choice", "choices": list_psd_regimes()}
        catalog["noncommutativity_level"] = {"label": "noncommutativity level", "value_type": "float", "supports_range": True, "supports_log": False}
    if problem_family == "matrix_completion":
        catalog["mc_factor_balance"] = {"label": "factor balance", "value_type": "choice", "choices": ["balanced", "unbalanced"]}
        catalog["mc_factor_imbalance_level"] = {"label": "factor imbalance level", "value_type": "float", "supports_range": True, "supports_log": False}
        if data_source == "synthetic":
            catalog["coherence_level"] = {"label": "coherence level", "value_type": "float", "supports_range": True, "supports_log": False}
            catalog["sampling_regime"] = {"label": "sampling regime", "value_type": "choice", "choices": ["uniform", "leverage_biased"]}
            catalog["mc_spectrum_type"] = {"label": "spectrum type", "value_type": "choice", "choices": ["flat", "power_decay", "exponential_decay"]}
            catalog["mc_spectrum_decay"] = {"label": "spectrum decay", "value_type": "float", "supports_range": True, "supports_log": False}
    if not allow_2d:
        return catalog
    return catalog


def _render_axis_values_editor(
    *,
    problem_family: str,
    field: str,
    meta: dict[str, object],
    key_prefix: str,
) -> tuple[list[object], str | None]:
    value_type = str(meta.get("value_type") or "float")
    default_values = _default_preset_values(problem_family, field)
    if value_type == "choice":
        choices = [str(item) for item in meta.get("choices", [])]
        default_choice_values = [item for item in default_values if item in choices] or choices[: min(4, len(choices))]
        selected = st.multiselect(
            f"{meta['label']} values",
            choices,
            default=default_choice_values,
            key=f"{key_prefix}_{field}_choices",
        )
        return list(selected), None if selected else "Select at least one value."

    input_mode_options = ["Manual list", "Generated range"]
    input_mode_key = f"{key_prefix}_{field}_input_mode"
    _normalize_choice_state(input_mode_key, input_mode_options, default="Manual list")
    input_mode = _radio(
        f"{meta['label']} values",
        input_mode_options,
        horizontal=True,
        key=input_mode_key,
    )
    try:
        if input_mode == "Manual list":
            default_text = ", ".join(str(item) for item in default_values) if default_values else ("1, 2, 3" if value_type == "int" else "0.1, 0.2, 0.5")
            raw = _text_input(
                f"{meta['label']} list",
                value=default_text,
                key=f"{key_prefix}_{field}_manual",
                help="Comma-separated values.",
            )
            values = parse_sweep_values_text(raw, value_type=value_type)
        else:
            supports_log = bool(meta.get("supports_log", False))
            spacing_options = ["linear", "log"] if supports_log else ["linear"]
            default_spacing = "log" if supports_log else "linear"
            cols = st.columns(4)
            with cols[0]:
                start = _number_input(f"{meta['label']} start", value=float(default_values[0] if default_values else (1.0 if value_type == "int" else 0.1)), key=f"{key_prefix}_{field}_start")
            with cols[1]:
                stop_default = float(default_values[-1] if default_values else (10.0 if value_type == "int" else 1.0))
                stop = _number_input(f"{meta['label']} stop", value=stop_default, key=f"{key_prefix}_{field}_stop")
            with cols[2]:
                count_default = max(2, len(default_values)) if default_values else 4
                count_key = f"{key_prefix}_{field}_count"
                _normalize_numeric_state(count_key, default=count_default, cast_type=int, min_value=1)
                count = int(_number_input(f"{meta['label']} count", min_value=1, value=count_default, step=1, key=count_key))
            with cols[3]:
                spacing = _selectbox(f"{meta['label']} spacing", spacing_options, index=spacing_options.index(default_spacing), key=f"{key_prefix}_{field}_spacing")
            values = generate_numeric_sweep_values(start=float(start), stop=float(stop), num=count, spacing=str(spacing), as_int=(value_type == "int"))
        preview = format_display_sequence(values, limit=8)
        st.caption(f"Values: {preview}")
        return list(values), None
    except Exception as exc:
        return [], str(exc)


def _preview_coupled_cells(field_x: str, x_vals: list[object], field_y: str, y_vals: list[object]) -> None:
    if not x_vals or not y_vals or len(x_vals) != len(y_vals):
        return
    preview_df = pd.DataFrame({field_x: x_vals, field_y: y_vals}).head(8)
    st.caption("Coupled configurations preview")
    st.dataframe(preview_df, use_container_width=True, hide_index=True)


def _format_sweep_values(values: list[object], *, limit: int = 12) -> str:
    return format_display_sequence(values, limit=limit)


def _render_sweep_values_preview(label: str, field_values: dict[str, list[object]], *, coupling: str = "grid") -> None:
    rows = [
        {"field": field, "values": _format_sweep_values(values), "count": len(values)}
        for field, values in field_values.items()
    ]
    st.caption(label)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if len(field_values) == 2 and coupling == "paired":
        fields = list(field_values.keys())
        _preview_coupled_cells(fields[0], field_values[fields[0]], fields[1], field_values[fields[1]])


def _render_sweep_axis_setup(*, problem_family: str, data_source: str, sweep_kind: str) -> dict[str, object]:
    st.divider()
    st.markdown("### Sweep axes")
    st.caption("Choose which values vary across the base experiment.")

    if sweep_kind == "1D sweep":
        builder_options = ["Preset", "Custom"]
        _normalize_choice_state("sweep_1d_builder", builder_options, default="Preset")
        builder_mode = _radio("Sweep builder", builder_options, horizontal=True, key="sweep_1d_builder")
        custom_error = None
        coupling = "grid"
        if builder_mode == "Preset":
            preset_options = ["Sweep learning rate", "Sweep init scale", "Sweep optimizer", "Sweep parameterization", "Sweep explicit reg strength", "Sweep lr schedule"]
            if problem_family == "psd_sensing":
                preset_options.extend(["Sweep noncommutativity level"])
            if problem_family == "matrix_completion":
                preset_options.extend(["Sweep factor balance", "Sweep factor imbalance level"])
            if problem_family == "matrix_completion" and data_source == "synthetic":
                preset_options.extend(["Sweep coherence level", "Sweep sampling regime", "Sweep spectrum type", "Sweep spectrum decay"])
            preset = _selectbox("Preset", preset_options, index=0)
            if preset == "Sweep learning rate":
                sweep_field, values = ("lr", [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]) if problem_family == "psd_sensing" else ("lr", [2e-2, 5e-2, 1e-1, 2e-1, 4e-1])
            elif preset == "Sweep init scale":
                sweep_field, values = ("init_scale", [1e-4, 1e-3, 1e-2, 1e-1]) if problem_family == "psd_sensing" else ("init_scale", [1e-2, 3e-2, 1e-1, 3e-1])
            elif preset == "Sweep optimizer":
                sweep_field, values = "optimizer", list_optimizers(problem_family)
            elif preset == "Sweep parameterization":
                sweep_field, values = "parameterization", list_parameterizations(problem_family)
            elif preset == "Sweep lr schedule":
                sweep_field, values = "lr_schedule_type", list_lr_schedule_types()
            elif preset == "Sweep noncommutativity level":
                sweep_field, values = "noncommutativity_level", [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
            elif preset == "Sweep factor balance":
                sweep_field, values = "mc_factor_balance", ["balanced", "unbalanced"]
            elif preset == "Sweep factor imbalance level":
                sweep_field, values = "mc_factor_imbalance_level", [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
            elif preset == "Sweep coherence level":
                sweep_field, values = "coherence_level", [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
            elif preset == "Sweep sampling regime":
                sweep_field, values = "sampling_regime", ["uniform", "leverage_biased"]
            elif preset == "Sweep spectrum type":
                sweep_field, values = "mc_spectrum_type", ["flat", "power_decay", "exponential_decay"]
            elif preset == "Sweep spectrum decay":
                sweep_field, values = "mc_spectrum_decay", [0.0, 0.25, 0.5, 1.0, 2.0]
            else:
                sweep_field, values = "explicit_reg_strength", [0.0, 1e-4, 1e-3, 1e-2]
            _render_sweep_values_preview("Preset values", {sweep_field: list(values)})
        else:
            catalog = _sweep_field_catalog(problem_family, data_source, allow_2d=False)
            field_options = list(catalog.keys())
            _normalize_choice_state("sweep_1d_field", field_options, default=field_options[0])
            sweep_field = _selectbox(
                "Sweep field",
                field_options,
                format_func=lambda key: str(catalog[key]["label"]),
                key="sweep_1d_field",
            )
            values, custom_error = _render_axis_values_editor(
                problem_family=problem_family,
                field=sweep_field,
                meta=catalog[sweep_field],
                key_prefix="sweep_1d_custom",
            )
            preset = f"Custom 1D: {sweep_field}"

        return {
            "kind": "1D",
            "preset": preset,
            "fields": (sweep_field,),
            "values_map": {sweep_field: list(values)},
            "sweep_field": sweep_field,
            "values": list(values),
            "custom_error": custom_error,
            "coupling": coupling,
        }

    builder_options = ["Preset", "Custom"]
    _normalize_choice_state("sweep_2d_builder", builder_options, default="Preset")
    builder_mode = _radio("Grid builder", builder_options, horizontal=True, key="sweep_2d_builder")
    coupling = "grid"
    custom_error = None
    if builder_mode == "Preset":
        grid_preset = _selectbox("Grid preset", ["Grid: (lr, init_scale)", "Grid: (lr, factor_rank)"], index=0)
        if grid_preset == "Grid: (lr, init_scale)":
            field_x, x_vals = ("lr", [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]) if problem_family == "psd_sensing" else ("lr", [2e-2, 5e-2, 1e-1, 2e-1, 4e-1])
            field_y, y_vals = ("init_scale", [1e-4, 1e-3, 1e-2, 1e-1]) if problem_family == "psd_sensing" else ("init_scale", [1e-2, 3e-2, 1e-1, 3e-1])
        else:
            field_x, x_vals = ("lr", [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]) if problem_family == "psd_sensing" else ("lr", [2e-2, 5e-2, 1e-1, 2e-1, 4e-1])
            field_y, y_vals = "factor_rank", [2, 4, 6, 8, 10, 12]
        preset = grid_preset
        _render_sweep_values_preview("Preset grid values", {field_x: list(x_vals), field_y: list(y_vals)}, coupling=coupling)
    else:
        catalog = _sweep_field_catalog(problem_family, data_source, allow_2d=True)
        coupling = "grid"
        st.caption("2D sweeps use a full grid so the result can be shown as a heatmap.")
        x_options = list(catalog.keys())
        _normalize_choice_state("sweep_2d_x_field", x_options, default=x_options[0])
        field_x = _selectbox(
            "X field",
            x_options,
            format_func=lambda key: f"X · {catalog[key]['label']}",
            key="sweep_2d_x_field",
        )
        y_candidates = [key for key in catalog.keys() if key != field_x]
        _normalize_choice_state("sweep_2d_y_field", y_candidates, default=y_candidates[0])
        field_y = _selectbox(
            "Y field",
            y_candidates,
            format_func=lambda key: f"Y · {catalog[key]['label']}",
            key="sweep_2d_y_field",
        )
        x_vals, x_error = _render_axis_values_editor(
            problem_family=problem_family,
            field=field_x,
            meta=catalog[field_x],
            key_prefix="sweep_2d_x",
        )
        y_vals, y_error = _render_axis_values_editor(
            problem_family=problem_family,
            field=field_y,
            meta=catalog[field_y],
            key_prefix="sweep_2d_y",
        )
        if x_error:
            custom_error = x_error
        elif y_error:
            custom_error = y_error
        preset = f"Custom 2D grid: {field_x} vs {field_y}"

    return {
        "kind": "2D",
        "preset": preset,
        "fields": (field_x, field_y),
        "values_map": {field_x: list(x_vals), field_y: list(y_vals)},
        "field_x": field_x,
        "field_y": field_y,
        "x_vals": list(x_vals),
        "y_vals": list(y_vals),
        "custom_error": custom_error,
        "coupling": coupling,
    }


def _safe_std(series: pd.Series) -> float:
    return float(pd.Series(series, dtype=float).std(ddof=0))


def _aggregate_1d(df: pd.DataFrame, sweep_field: str, metric_keys: list[str]) -> pd.DataFrame:
    expected_seed_count = 1
    if "seed" in df.columns and not df.empty:
        expected_seed_count = max(int(pd.Series(df["seed"]).dropna().nunique()), 1)
    return aggregate_sweep_1d(df, sweep_field, metric_keys, expected_seed_count=expected_seed_count)


def _aggregate_2d(df: pd.DataFrame, field_x: str, field_y: str, metric_keys: list[str]) -> pd.DataFrame:
    expected_seed_count = 1
    if "seed" in df.columns and not df.empty:
        expected_seed_count = max(int(pd.Series(df["seed"]).dropna().nunique()), 1)
    return aggregate_sweep_2d(df, field_x, field_y, metric_keys, expected_seed_count=expected_seed_count)


def _plot_1d_aggregate(agg: pd.DataFrame, sweep_field: str, metric: str, show_variability: bool, metric_options: dict[str, str]) -> go.Figure:
    mean_col = f"{metric}_mean"
    min_col = f"{metric}_min"
    max_col = f"{metric}_max"
    fig = go.Figure()
    if show_variability:
        fig.add_trace(go.Scatter(x=agg[sweep_field], y=agg[max_col], mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=agg[sweep_field], y=agg[min_col], mode="lines", line=dict(width=0), fill="tonexty", name="min/max band"))
    fig.add_trace(go.Scatter(x=agg[sweep_field], y=agg[mean_col], mode="lines+markers", name="mean"))
    fig.update_layout(title=f"{metric_options[metric]} vs {sweep_field}", xaxis_title=sweep_field, yaxis_title=metric_options[metric])
    if sweep_field in {"lr", "init_scale", "explicit_reg_strength"}:
        fig.update_xaxes(type="log")
    return fig




def _render_2d_phase_diagram(raw_df: pd.DataFrame, field_x: str, field_y: str, metric_options: dict[str, str], key_prefix: str) -> None:
    if not metric_options:
        st.info("No numeric sweep metrics available for the 2D heatmap.")
        return
    status_values = sorted(str(s) for s in raw_df["status"].dropna().unique().tolist()) if "status" in raw_df.columns else []
    cols = st.columns([1.2, 1.0, 1.2])
    with cols[0]:
        metric = _selectbox(
            "Metric",
            list(metric_options.keys()),
            format_func=lambda x: metric_options[x],
            key=f"{key_prefix}_metric",
        )
    with cols[1]:
        aggregation = _selectbox(
            "Configuration value",
            ["mean", "min", "max", "std"],
            index=0,
            key=f"{key_prefix}_aggregation",
        )
    with cols[2]:
        selected_statuses = st.multiselect(
            "Statuses",
            options=status_values,
            default=status_values,
            key=f"{key_prefix}_statuses",
        ) if status_values else []

    filtered = raw_df.copy()
    if status_values:
        filtered = filtered[filtered["status"].astype(str).isin(selected_statuses)]
    if filtered.empty:
        st.warning("No runs remain after status filtering.")
        return

    agg = _aggregate_2d(filtered, field_x, field_y, list(metric_options.keys()))
    st.dataframe(agg, use_container_width=True)

    value_col = f"{metric}_{aggregation}"
    pivot = agg.pivot(index=field_y, columns=field_x, values=value_col).sort_index().sort_index(axis=1)
    fig = px.imshow(
        pivot,
        aspect="auto",
        text_auto=True,
        labels={"x": field_x, "y": field_y, "color": f"{metric_options[metric]} ({aggregation})"},
        title=f"{metric_options[metric]} ({aggregation}) phase diagram",
    )
    fig.update_xaxes(type="log" if field_x in {"lr", "init_scale", "explicit_reg_strength"} else "linear")
    fig.update_yaxes(type="log" if field_y in {"lr", "init_scale", "explicit_reg_strength"} else "linear")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Configuration drill-down")
    d1, d2 = st.columns(2)
    with d1:
        x_values = list(pivot.columns.tolist())
        selected_x = _selectbox(field_x, x_values, key=f"{key_prefix}_selected_x")
    with d2:
        y_values = list(pivot.index.tolist())
        selected_y = _selectbox(field_y, y_values, key=f"{key_prefix}_selected_y")
    cell_df = filtered[(filtered[field_x] == selected_x) & (filtered[field_y] == selected_y)].copy()
    if cell_df.empty:
        st.info("No runs in selected configuration.")
        return
    st.dataframe(cell_df.sort_values([field_y, field_x, "seed", "run"]).reset_index(drop=True), use_container_width=True)

    summary_rows = []
    for agg_name in ["mean", "std", "min", "max"]:
        row = {"aggregate": agg_name}
        for metric_key in metric_options.keys():
            series = pd.to_numeric(cell_df[metric_key], errors="coerce").dropna()
            if series.empty:
                row[metric_key] = np.nan
            elif agg_name == "mean":
                row[metric_key] = float(series.mean())
            elif agg_name == "std":
                row[metric_key] = float(series.std(ddof=0))
            elif agg_name == "min":
                row[metric_key] = float(series.min())
            else:
                row[metric_key] = float(series.max())
        summary_rows.append(row)
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)


SWEEP_VIEW_KEYS = [
    "sweeps_mode",
    "sweeps_view_gid",
    "sweep_view_metric",
    "sweep_view_band",
    "sweep_view_phase_metric",
    "sweep_view_phase_aggregation",
    "sweep_view_phase_statuses",
    "sweep_view_phase_selected_x",
    "sweep_view_phase_selected_y",
]


def _current_sweep_view_state() -> dict:
    return {key: st.session_state[key] for key in SWEEP_VIEW_KEYS if key in st.session_state}


def _render_saved_sweep_views(service: SweepsPageService) -> None:
    meta = service.load_analysis_views()
    namespace = (meta or {}).get("sweeps", {})
    views = namespace.get("views", {}) if isinstance(namespace, dict) else {}
    order = [name for name in namespace.get("order", []) if name in views] if isinstance(namespace, dict) else []
    current_state = _current_sweep_view_state()

    with st.expander("Saved analysis views", expanded=False):
        c1, c2 = st.columns([1.4, 1.0])
        with c1:
            view_options = ["(none)"] + order
            if st.session_state.get("sweep_saved_view_pick") not in view_options:
                st.session_state["sweep_saved_view_pick"] = "(none)"
            selected_name = _selectbox("Saved sweep view", view_options, index=0, key="sweep_saved_view_pick")
        with c2:
            normalize_text_state_value(st.session_state, "sweep_save_view_name", default="")
            save_name = _text_input("Save current as", value="", key="sweep_save_view_name").strip()

        a1, a2, a3 = st.columns(3)
        with a1:
            if st.button("Apply saved view", key="sweep_apply_saved_view"):
                if selected_name != "(none)":
                    service.apply_analysis_view_state(views.get(selected_name, {}))
                    st.rerun()
        with a2:
            if st.button("Save current view", key="sweep_save_current_view"):
                if not save_name:
                    st.error("Give the view a name.")
                else:
                    namespace = meta.setdefault("sweeps", {"views": {}, "order": []})
                    namespace.setdefault("views", {})[save_name] = current_state
                    if save_name not in namespace.setdefault("order", []):
                        namespace["order"].append(save_name)
                    service.save_analysis_views(meta)
                    st.success(f"Saved view '{save_name}'.")
        with a3:
            if st.button("Delete saved view", key="sweep_delete_saved_view"):
                if selected_name != "(none)":
                    namespace = meta.setdefault("sweeps", {"views": {}, "order": []})
                    namespace.setdefault("views", {}).pop(selected_name, None)
                    namespace["order"] = [name for name in namespace.get("order", []) if name != selected_name]
                    service.save_analysis_views(meta)
                    st.success(f"Deleted view '{selected_name}'.")
                    st.rerun()

        st.download_button(
            "Download current view spec (.json)",
            data=json_dumps(current_state),
            file_name="sweep_view.json",
            mime="application/json",
            key="sweep_download_current_view",
        )


def _render_export_buttons(raw_df: pd.DataFrame, agg_df: pd.DataFrame | None, raw_name: str, agg_name: str | None) -> None:
    st.markdown("### Export")
    if isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
        st.download_button(
            "Download current raw table (.csv)",
            data=raw_df.to_csv(index=False),
            file_name=raw_name,
            mime="text/csv",
            key=f"dl_{raw_name}",
        )
    if isinstance(agg_df, pd.DataFrame) and not agg_df.empty and agg_name:
        st.download_button(
            "Download current aggregated table (.csv)",
            data=agg_df.to_csv(index=False),
            file_name=agg_name,
            mime="text/csv",
            key=f"dl_{agg_name}",
        )


def _render_sweep_bundle_download(
    *,
    analysis_view,
    service: SweepsPageService,
    key_prefix: str,
) -> None:
    with st.expander("Sweep bundle", expanded=False):
        include_reports = _checkbox("Include per-run reports", value=False, key=f"{key_prefix}_bundle_reports")
        include_artifacts = _checkbox("Include artifacts", value=False, key=f"{key_prefix}_bundle_artifacts")
        if st.button("Build sweep bundle", key=f"{key_prefix}_build_bundle"):
            bundle = service.build_sweep_bundle(
                analysis_view=analysis_view,
                include_run_reports=include_reports,
                include_artifacts=include_artifacts,
            )
            st.download_button(
                "Download sweep bundle",
                data=bundle.data,
                file_name=bundle.filename,
                mime=bundle.content_type,
                key=f"{key_prefix}_download_bundle",
            )


_LATEST_SWEEP_ANALYSIS_KEY = "sweeps_latest_execution_analysis"


def _store_latest_sweep_analysis(analysis_view) -> None:
    st.session_state[_LATEST_SWEEP_ANALYSIS_KEY] = analysis_view


def _latest_sweep_analysis():
    return st.session_state.get(_LATEST_SWEEP_ANALYSIS_KEY)


def _render_latest_sweep_analysis(*, service: SweepsPageService, expected_kind: str) -> None:
    analysis = _latest_sweep_analysis()
    if analysis is None or str(getattr(analysis, "kind", "")) != expected_kind:
        return
    raw_df = analysis.raw_df
    if not isinstance(raw_df, pd.DataFrame) or raw_df.empty:
        return
    metric_options = dict(analysis.metric_options or {})
    fields = tuple(getattr(analysis, "fields", ()) or ())
    st.markdown("### Latest sweep result")
    st.success(f"Saved sweep. group_id = {analysis.group_id}")
    st.dataframe(raw_df, use_container_width=True)

    if expected_kind == "1D" and len(fields) >= 1:
        sweep_field = str(fields[0])
        agg = analysis.aggregate_df
        st.dataframe(agg, use_container_width=True)
        if metric_options:
            metric = _selectbox(
                "Metric",
                list(metric_options.keys()),
                format_func=lambda x: metric_options[x],
                key=f"metric_latest_{analysis.group_id}",
            )
            show_variability = _checkbox("Show min/max band", value=False, key=f"band_latest_{analysis.group_id}")
            st.plotly_chart(_plot_1d_aggregate(agg, sweep_field, metric, show_variability, metric_options), use_container_width=True)
        _render_export_buttons(raw_df, agg, f"{analysis.group_id}_raw.csv", f"{analysis.group_id}_agg_1d.csv")
    elif expected_kind == "2D" and len(fields) >= 2:
        _render_2d_phase_diagram(
            raw_df,
            str(fields[0]),
            str(fields[1]),
            metric_options,
            key_prefix=f"phase_latest_{analysis.group_id}",
        )
        _render_export_buttons(raw_df, analysis.aggregate_df, f"{analysis.group_id}_raw.csv", f"{analysis.group_id}_agg_2d.csv")

    _render_sweep_bundle_download(analysis_view=analysis, service=service, key_prefix=f"latest_{analysis.group_id}")


def render_sweeps_page(
    service: SweepsPageService,
    *,
    show_title: bool = True,
    submit_1d_label: str = "Run 1D sweep",
    submit_2d_label: str = "Run 2D grid",
) -> None:
    consume_pending_analysis_view_state()
    if show_title:
        st.title("Sweeps")
    max_runs_1d = MAX_RUNS_1D_DEFAULT
    max_runs_2d = MAX_RUNS_2D_DEFAULT

    st.subheader("Run a new sweep")
    st.caption("Define a sweep by choosing a base experiment and the axis values to vary.")

    st.divider()
    st.markdown("### Sweep setup")
    sweep_kind_options = ["1D sweep", "2D grid"]
    _normalize_choice_state("sweep_kind", sweep_kind_options, default="1D sweep")
    sweep_kind = _radio("Sweep type", sweep_kind_options, horizontal=True, key="sweep_kind")
    _normalize_numeric_state(SWEEP_MAX_RUNS_1D_KEY, default=MAX_RUNS_1D_DEFAULT, cast_type=int, min_value=2)
    _normalize_numeric_state(SWEEP_MAX_RUNS_2D_KEY, default=MAX_RUNS_2D_DEFAULT, cast_type=int, min_value=2)
    with st.expander("Sweep limits", expanded=False):
        if sweep_kind == "1D sweep":
            max_runs_1d = int(_number_input("Max runs allowed", min_value=2, value=MAX_RUNS_1D_DEFAULT, step=1, key=SWEEP_MAX_RUNS_1D_KEY))
        else:
            max_runs_2d = int(_number_input("Max runs allowed", min_value=2, value=MAX_RUNS_2D_DEFAULT, step=1, key=SWEEP_MAX_RUNS_2D_KEY))

    st.divider()
    st.markdown("### Base experiment")
    st.markdown("#### Data setup")
    problem_family = _selectbox("problem_family", list_problem_families(), index=1)
    data_source = "synthetic"
    if problem_family == "matrix_completion":
        data_source_options = list_data_sources(problem_family)
        data_source = _selectbox("data_source", data_source_options, index=0)
    psd_regime = _selectbox("psd_regime", list_psd_regimes(), index=0) if problem_family == "psd_sensing" else "non_commuting"
    default_request = service.get_default_base_request(problem_family, data_source)

    custom_dataset_path = ""
    custom_dataset_id = ""
    custom_stats = None
    custom_stats_error = ""
    if problem_family == "matrix_completion" and data_source == "custom_triplets_csv":
        custom_records = service.list_custom_datasets()
        dataset_options = [""] + [record.dataset_id for record in custom_records]
        selected_dataset_id = str(st.session_state.get("sw_custom_dataset_id", str(default_request.custom_dataset_id or "")))
        if selected_dataset_id not in dataset_options:
            selected_dataset_id = ""
        st.session_state["sw_custom_dataset_id"] = selected_dataset_id
        custom_dataset_id = _selectbox(
            "custom_dataset_id",
            options=dataset_options,
            index=dataset_options.index(selected_dataset_id),
            format_func=lambda value: "(select registered dataset)" if value == "" else value,
            key="sw_custom_dataset_id",
        )
        if custom_dataset_id:
            try:
                record = service.get_custom_dataset(custom_dataset_id)
                custom_stats = dict(record.stats)
                st.caption(f"Registered dataset · file={record.source_filename} · fingerprint={record.fingerprint[:12]}")
            except Exception as exc:
                custom_stats_error = str(exc)
        else:
            custom_stats_error = "Select a registered dataset from the Data Sources page first."

    noncommutativity_level = 1.0
    coherence_level = float(default_request.coherence_level)
    sampling_regime = str(default_request.sampling_regime)
    mc_spectrum_type = str(default_request.mc_spectrum_type)
    mc_spectrum_decay = float(default_request.mc_spectrum_decay)
    mc_factor_balance = str(default_request.mc_factor_balance)
    mc_factor_imbalance_level = float(default_request.mc_factor_imbalance_level)
    early_stopping_enabled = bool(default_request.early_stopping_enabled)
    early_stopping_metric = str(default_request.early_stopping_metric)
    early_stopping_patience = int(default_request.early_stopping_patience)
    early_stopping_min_delta = float(default_request.early_stopping_min_delta)

    with st.expander("Structure / sampling controls", expanded=False):
        if problem_family == "psd_sensing":
            if psd_regime == "commuting":
                noncommutativity_level = float(_number_input("noncommutativity_level", min_value=0.0, max_value=0.0, value=0.0, step=0.05, disabled=True))
            else:
                noncommutativity_level = float(st.slider("noncommutativity_level", min_value=0.0, max_value=1.0, value=float(default_request.noncommutativity_level), step=0.05))
        elif data_source == "synthetic":
            coherence_level = float(st.slider("coherence_level", min_value=0.0, max_value=1.0, value=float(default_request.coherence_level), step=0.05))
            sampling_regime_options = ["uniform", "leverage_biased"]
            sampling_regime = _selectbox("sampling_regime", sampling_regime_options, index=sampling_regime_options.index(str(default_request.sampling_regime)) if str(default_request.sampling_regime) in sampling_regime_options else 0)
            spectrum_options = ["flat", "power_decay", "exponential_decay"]
            mc_spectrum_type = _selectbox("mc_spectrum_type", spectrum_options, index=spectrum_options.index(str(default_request.mc_spectrum_type)) if str(default_request.mc_spectrum_type) in spectrum_options else 0)
            if mc_spectrum_type == "flat":
                mc_spectrum_decay = float(_number_input("mc_spectrum_decay", min_value=0.0, max_value=0.0, value=0.0, step=0.1, disabled=True))
            else:
                mc_spectrum_decay = float(_number_input("mc_spectrum_decay", min_value=0.0, value=max(0.0, float(default_request.mc_spectrum_decay)), step=0.1))

        if problem_family == "matrix_completion":
            factor_balance_options = ["balanced", "unbalanced"]
            mc_factor_balance = _selectbox("mc_factor_balance", factor_balance_options, index=factor_balance_options.index(str(default_request.mc_factor_balance)) if str(default_request.mc_factor_balance) in factor_balance_options else 0)
            if mc_factor_balance == "balanced":
                mc_factor_imbalance_level = float(_number_input("mc_factor_imbalance_level", min_value=0.0, max_value=0.0, value=0.0, step=0.05, disabled=True))
            else:
                mc_factor_imbalance_level = float(st.slider("mc_factor_imbalance_level", min_value=0.0, max_value=1.0, value=max(0.0, float(default_request.mc_factor_imbalance_level or 0.5)), step=0.05))

    if problem_family == "matrix_completion" and data_source == "movielens_100k":
        n = int(default_request.n)
        n_cols = int(default_request.n_cols) if default_request.n_cols is not None else None
        m = int(default_request.m)
        stats = get_data_source_stats(problem_family, data_source) or {"n_rows": n, "n_cols": n_cols, "m": m}
        st.caption(f"MovieLens 100K · users={stats['n_rows']} · items={stats['n_cols']} · ratings={stats['m']}")
        val_fraction = float(_number_input("val_fraction", min_value=0.0, max_value=0.49, value=float(default_request.val_fraction), step=0.05, format="%.2f"))
        test_fraction = float(_number_input("test_fraction", min_value=0.0, max_value=0.49, value=float(default_request.test_fraction), step=0.05, format="%.2f"))
    elif problem_family == "matrix_completion" and data_source == "custom_triplets_csv":
        if custom_stats is not None:
            st.caption(
                f"Custom triplets CSV · rows={custom_stats['n_rows']} · cols={custom_stats['n_cols']} · observations={custom_stats['m']}"
            )
            n = int(custom_stats["n_rows"])
            n_cols = int(custom_stats["n_cols"])
            m = int(custom_stats["m"])
        else:
            if custom_stats_error:
                st.error(custom_stats_error)
            st.caption("Select a registered dataset to infer matrix dimensions and observation count.")
            n = int(default_request.n)
            n_cols = int(default_request.n_cols) if default_request.n_cols is not None else int(default_request.n)
            m = int(default_request.m)
        _number_input("n", value=int(n), disabled=True)
        _number_input("n_cols", value=int(n_cols), disabled=True)
        _number_input("m observed entries", value=int(m), disabled=True)
        val_fraction = float(_number_input("val_fraction", min_value=0.0, max_value=0.49, value=float(default_request.val_fraction), step=0.05, format="%.2f"))
        test_fraction = float(_number_input("test_fraction", min_value=0.0, max_value=0.49, value=float(default_request.test_fraction), step=0.05, format="%.2f"))
    else:
        val_fraction = float(default_request.val_fraction)
        test_fraction = float(default_request.test_fraction)
        n = int(_number_input("n", min_value=2, value=int(default_request.n), step=1))
        n_cols = None if default_request.n_cols is None else int(_number_input("n_cols", min_value=2, value=int(default_request.n_cols), step=1))
        m = int(_number_input("m measurements" if problem_family == "psd_sensing" else "m observed entries", min_value=1, value=int(default_request.m), step=50))

    if problem_family == "matrix_completion" and is_real_data_source(problem_family, data_source):
        base_noise = 0.0
    else:
        base_noise = float(_number_input("noise_std", min_value=0.0, value=float(default_request.noise_std), format="%.6f"))

    axis_state = _render_sweep_axis_setup(problem_family=problem_family, data_source=data_source, sweep_kind=sweep_kind)

    st.divider()
    st.markdown("### Model / parameterization")
    r_star = int(_number_input("r_star", min_value=1, value=int(default_request.r_star), step=1))
    factor_rank = int(_number_input("factor_rank", min_value=1, value=int(default_request.factor_rank), step=1))
    parameterization_options = list_parameterizations(problem_family)
    base_parameterization = _selectbox("base parameterization", parameterization_options, index=parameterization_options.index(str(default_request.parameterization)) if str(default_request.parameterization) in parameterization_options else 0)
    base_init = float(_number_input("base init_scale", min_value=0.0, value=float(default_request.init_scale), format="%.6f"))

    base_explicit_reg_type = str(default_request.explicit_reg_type)
    base_explicit_reg_strength = float(default_request.explicit_reg_strength)
    with st.expander("Advanced model / regularization", expanded=False):
        explicit_reg_options = explicit_reg_types_for_parameterization(base_parameterization)
        default_reg_type = str(default_request.explicit_reg_type) if str(default_request.explicit_reg_type) in explicit_reg_options else ("none" if "none" in explicit_reg_options else explicit_reg_options[0])
        base_explicit_reg_type = _selectbox("base explicit_reg_type", explicit_reg_options, index=explicit_reg_options.index(default_reg_type))
        base_explicit_reg_strength = float(_number_input("base explicit_reg_strength", min_value=0.0, value=float(default_request.explicit_reg_strength), format="%.6f", disabled=(base_explicit_reg_type == "none")))
        if base_explicit_reg_type == "none":
            base_explicit_reg_strength = 0.0

    st.divider()
    st.markdown("### Optimization")
    opt_left, opt_right = st.columns(2)
    with opt_left:
        optimizer_options = list_optimizers(problem_family)
        base_optimizer = _selectbox("base optimizer", optimizer_options, index=optimizer_options.index(str(default_request.optimizer)) if str(default_request.optimizer) in optimizer_options else 0)
    with opt_right:
        base_lr = float(_number_input("base lr", min_value=0.0, value=float(default_request.lr), format="%.6f"))

    base_batch_size = None
    if base_optimizer in {"sgd", "momentum"}:
        train_estimate = int(m * (1.0 - val_fraction - test_fraction))
        default_batch_size = int(default_request.batch_size or min(64, max(1, train_estimate)))
        base_batch_size = int(_number_input("base batch_size", min_value=1, value=min(default_batch_size, max(1, train_estimate)), max_value=max(1, train_estimate), step=1))

    base_lr_schedule_type = str(default_request.lr_schedule_type)
    base_lr_decay_gamma = float(default_request.lr_decay_gamma)
    base_lr_step_size = int(default_request.lr_step_size)
    base_lr_min = float(default_request.lr_min)
    with st.expander("LR schedule", expanded=False):
        lr_schedule_options = list_lr_schedule_types()
        base_lr_schedule_type = _selectbox("base lr_schedule_type", lr_schedule_options, index=lr_schedule_options.index(str(default_request.lr_schedule_type)) if str(default_request.lr_schedule_type) in lr_schedule_options else 0)
        if base_lr_schedule_type == "step_decay":
            c1, c2 = st.columns(2)
            with c1:
                base_lr_decay_gamma = float(_number_input("base lr_decay_gamma", min_value=1e-6, max_value=1.0, value=float(default_request.lr_decay_gamma), format="%.6f"))
            with c2:
                base_lr_step_size = int(_number_input("base lr_step_size", min_value=1, value=int(default_request.lr_step_size), step=1))
        elif base_lr_schedule_type == "cosine":
            base_lr_min = float(_number_input("base lr_min", min_value=0.0, value=float(default_request.lr_min), format="%.6f"))

    st.divider()
    st.markdown("### Runtime / logging")
    runtime_left, runtime_right, runtime_third = st.columns(3)
    with runtime_left:
        log_every = int(_number_input("log_every", min_value=1, value=int(default_request.log_every), step=1))
    with runtime_right:
        device_options = [option for option in list_device_preferences() if option != "auto"]
        default_device = str(default_request.device_preference) if str(default_request.device_preference) in device_options else "cpu"
        base_device_preference = _selectbox("device", device_options, index=device_options.index(default_device))
    with runtime_third:
        dtype_options = [option for option in list_dtype_preferences() if option != "auto"]
        default_dtype = str(default_request.dtype_preference) if str(default_request.dtype_preference) in dtype_options else "float64"
        base_dtype_preference = _selectbox("dtype", dtype_options, index=dtype_options.index(default_dtype))
    base_execution_mode = str(default_request.execution_mode)

    st.divider()
    st.markdown("### Stopping")
    stopping_mode_options = list_stopping_modes()
    default_stopping_mode = str(default_request.stopping_mode) if str(default_request.stopping_mode) in stopping_mode_options else "fixed_steps"
    stop_left, stop_right = st.columns(2)
    with stop_left:
        base_stopping_mode = _selectbox("base stopping_mode", stopping_mode_options, index=stopping_mode_options.index(default_stopping_mode))
    with stop_right:
        if base_stopping_mode == "run_to_convergence":
            steps = int(_number_input("Safety cap (max_steps)", min_value=1, value=int(default_request.steps), step=100))
        else:
            steps = int(_number_input("Steps", min_value=1, value=int(default_request.steps), step=100))

    base_grad_norm_tol = float(default_request.grad_norm_tol)
    base_objective_tol = float(default_request.objective_tol)
    base_divergence_norm_threshold = float(default_request.divergence_norm_threshold)
    if base_stopping_mode == "run_to_convergence":
        with st.expander("Advanced stopping", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                base_grad_norm_tol = float(_number_input("base grad_norm_tol", min_value=1e-16, value=float(default_request.grad_norm_tol), format="%.2e"))
            with c2:
                base_objective_tol = float(_number_input("base objective_tol", min_value=1e-16, value=float(default_request.objective_tol), format="%.2e"))
            with c3:
                base_divergence_norm_threshold = float(_number_input("base divergence_norm_threshold", min_value=1.0, value=float(default_request.divergence_norm_threshold), format="%.2e"))


    st.divider()
    st.markdown("### Sweep limits / seeds")
    active_max_runs = max_runs_1d if sweep_kind == "1D sweep" else max_runs_2d
    st.caption(f"Max sweep configurations allowed: {active_max_runs}")

    seed_input_mode = _radio("Seed input", ["Generated seed set", "Manual seed list"], horizontal=True)
    if seed_input_mode == "Generated seed set":
        c1, c2, c3 = st.columns(3)
        with c1:
            start_seed = int(_number_input("seed start", min_value=0, value=int(default_request.start_seed or 0), step=1))
        with c2:
            num_seeds = int(_number_input("seeds per configuration", min_value=1, value=int(default_request.num_seeds or 1), step=1))
        with c3:
            seed_step = int(_number_input("seed increment", min_value=1, value=int(default_request.seed_step or 1), step=1))
        seeds_text = ""
        st.caption("Each sweep configuration is repeated once per seed in the generated seed set.")
    else:
        seeds_text = _text_input("manual seeds", value=str(default_request.seeds_text or "0"))
        start_seed = 0
        num_seeds = 1
        seed_step = 1

    base_request = service.build_base_request(
        problem_family=problem_family,
        data_source=data_source,
        seed_input_mode=seed_input_mode,
        start_seed=start_seed,
        num_seeds=num_seeds,
        seed_step=seed_step,
        seeds_text=seeds_text,
        n=int(n),
        n_cols=n_cols,
        r_star=int(r_star),
        m=int(m),
        factor_rank=int(factor_rank),
        init_scale=float(base_init),
        lr=float(base_lr),
        steps=int(steps),
        log_every=int(log_every),
        lr_schedule_type=str(base_lr_schedule_type),
        lr_decay_gamma=float(base_lr_decay_gamma),
        lr_step_size=int(base_lr_step_size),
        lr_min=float(base_lr_min),
        noise_std=float(base_noise),
        val_fraction=float(val_fraction),
        test_fraction=float(test_fraction),
        early_stopping_enabled=bool(early_stopping_enabled),
        early_stopping_metric=str(early_stopping_metric),
        early_stopping_patience=int(early_stopping_patience),
        early_stopping_min_delta=float(early_stopping_min_delta),
        psd_regime=str(psd_regime),
        noncommutativity_level=float(noncommutativity_level),
        coherence_level=float(coherence_level),
        sampling_regime=str(sampling_regime),
        mc_spectrum_type=str(mc_spectrum_type),
        mc_spectrum_decay=float(mc_spectrum_decay),
        mc_factor_balance=str(mc_factor_balance),
        mc_factor_imbalance_level=float(mc_factor_imbalance_level),
        optimizer=str(base_optimizer),
        execution_mode=str(base_execution_mode),
        device_preference=str(base_device_preference),
        dtype_preference=str(base_dtype_preference),
        batch_size=base_batch_size,
        parameterization=str(base_parameterization),
        explicit_reg_type=str(base_explicit_reg_type),
        explicit_reg_strength=float(base_explicit_reg_strength),
        stopping_mode=str(base_stopping_mode),
        grad_norm_tol=float(base_grad_norm_tol),
        objective_tol=float(base_objective_tol),
        divergence_norm_threshold=float(base_divergence_norm_threshold),
        custom_dataset_path=str(custom_dataset_path),
        custom_dataset_id=str(custom_dataset_id),
    )

    seeds = service.resolve_seeds(base_request)
    st.caption(f"Seeds: {format_display_sequence(seeds)}")

    st.divider()
    st.markdown("### Preview / launch")
    active_max_runs = max_runs_1d if sweep_kind == "1D sweep" else max_runs_2d
    st.caption(f"Max sweep configurations allowed: {active_max_runs}")

    if axis_state["kind"] == "1D":
        sweep_field = str(axis_state["sweep_field"])
        values = list(axis_state["values"])
        custom_error = axis_state.get("custom_error")
        coupling = str(axis_state.get("coupling", "grid"))
        total_runs = len(values) * len(seeds)
        if custom_error:
            st.error(str(custom_error))
        if len(values) > max_runs_1d:
            st.warning(f"This sweep has {len(values)} configurations, exceeding the 1D cap of {max_runs_1d} configurations.")
        st.caption(f"Configurations: {len(values)} · runs: {total_runs}")
        candidate_request = SweepRequest(
            kind="1D",
            preset=str(axis_state["preset"]),
            base_request=base_request,
            fields=(sweep_field,),
            values={sweep_field: values},
            coupling=coupling,
        )
        validation_issues = service.validate_request(candidate_request) if values and not custom_error and len(values) <= max_runs_1d else []
        for issue in validation_issues:
            st.error(issue)
        if values and not custom_error and len(values) <= max_runs_1d and not validation_issues:
            for message in _sweep_audit_messages(service.audit_request(candidate_request)):
                st.warning(message)
        submit_disabled = bool(custom_error) or bool(validation_issues) or not values or len(values) > max_runs_1d
        if st.button(submit_1d_label, type="primary", disabled=submit_disabled):
            plan = service.build_execution_plan(candidate_request)
            service.save_metadata(plan.metadata)
            prog = st.progress(0.0)
            status = st.empty()
            result = service.execute_plan(
                plan,
                on_status=status.write,
                on_progress=lambda current, total: prog.progress(current / total if total else 1.0),
            )
            analysis = build_sweep_analysis_from_execution_result(result, problem_family=problem_family, data_source=data_source)
            if result.failures:
                st.warning(f"Completed with {len(result.failures)} failed runs.")
                st.dataframe(pd.DataFrame([failure.__dict__ for failure in result.failures]), use_container_width=True)
            if analysis.raw_df.empty:
                st.error("No sweep runs completed successfully.")
                return
            _store_latest_sweep_analysis(analysis)
            st.rerun()
        _render_latest_sweep_analysis(service=service, expected_kind="1D")
        return

    field_x = str(axis_state["field_x"])
    field_y = str(axis_state["field_y"])
    x_vals = list(axis_state["x_vals"])
    y_vals = list(axis_state["y_vals"])
    coupling = str(axis_state.get("coupling", "grid"))
    custom_error = axis_state.get("custom_error")
    total_configurations = len(x_vals) * len(y_vals) if coupling == "grid" else (len(x_vals) if len(x_vals) == len(y_vals) else 0)
    total_runs = total_configurations * len(seeds)
    if custom_error:
        st.error(str(custom_error))
    if total_configurations > max_runs_2d:
        st.warning(f"This sweep has {total_configurations} configurations, exceeding the 2D cap of {max_runs_2d} configurations.")
    st.caption(f"Configurations: {total_configurations} · runs: {total_runs} · layout: {coupling}")
    candidate_request = SweepRequest(
        kind="2D",
        preset=str(axis_state["preset"]),
        base_request=base_request,
        fields=(field_x, field_y),
        values={field_x: x_vals, field_y: y_vals},
        coupling=coupling,
    )
    validation_issues = service.validate_request(candidate_request) if total_configurations > 0 and not custom_error and total_configurations <= max_runs_2d else []
    for issue in validation_issues:
        st.error(issue)
    if total_configurations > 0 and not custom_error and total_configurations <= max_runs_2d and not validation_issues:
        for message in _sweep_audit_messages(service.audit_request(candidate_request)):
            st.warning(message)
    submit_disabled = bool(custom_error) or bool(validation_issues) or total_configurations <= 0 or total_configurations > max_runs_2d
    if st.button(submit_2d_label, type="primary", disabled=submit_disabled):
        plan = service.build_execution_plan(candidate_request)
        service.save_metadata(plan.metadata)
        prog = st.progress(0.0)
        status = st.empty()
        result = service.execute_plan(
            plan,
            on_status=status.write,
            on_progress=lambda current, total_count: prog.progress(current / total_count if total_count else 1.0),
        )
        analysis = build_sweep_analysis_from_execution_result(result, problem_family=problem_family, data_source=data_source)
        if result.failures:
            st.warning(f"Completed with {len(result.failures)} failed runs.")
            st.dataframe(pd.DataFrame([failure.__dict__ for failure in result.failures]), use_container_width=True)
        if analysis.raw_df.empty:
            st.error("No sweep runs completed successfully.")
            return
        _store_latest_sweep_analysis(analysis)
        st.rerun()
    _render_latest_sweep_analysis(service=service, expected_kind="2D")
