from __future__ import annotations

import streamlit as st

from framework.core.device import (
    list_device_preferences,
    list_dtype_preferences,
)
from framework.core.experiments import list_optimizers, list_problem_families
from framework.core.initialization import list_init_types
from framework.core.parameterization import list_parameterizations
from framework.core.psd_regimes import list_psd_regimes
from framework.core.schedules import list_lr_schedule_types
from framework.core.regularization import explicit_reg_types_for_parameterization
from framework.core.stopping import list_stopping_modes
from framework.services.data_sources import (
    get_data_source_stats,
    is_real_data_source,
    list_data_sources,
    supports_early_stopping,
)
from framework.ui.page_services import NewRunPageService
from framework.services.new_run import (
    normalize_choice_state_value,
    normalize_optional_numeric_state_value,
    normalize_text_state_value,
)
from framework.services.ui_state import remove_conflicting_widget_defaults
from framework.services.serialization import make_json_safe
from framework.ui.formatting import format_display_sequence






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


def _slider(label, **kwargs):
    key = kwargs.get("key")
    return st.slider(label, **_state_safe_kwargs(key, kwargs, default_arg_names=("value",)))


def _checkbox(label, **kwargs):
    key = kwargs.get("key")
    return st.checkbox(label, **_state_safe_kwargs(key, kwargs, default_arg_names=("value",)))


def _text_input(label, **kwargs):
    key = kwargs.get("key")
    return st.text_input(label, **_state_safe_kwargs(key, kwargs, default_arg_names=("value",)))


def _text_area(label, **kwargs):
    key = kwargs.get("key")
    return st.text_area(label, **_state_safe_kwargs(key, kwargs, default_arg_names=("value",)))


def _normalize_choice_state(key: str, options: list[str] | tuple[str, ...], *, default: str) -> str:
    return normalize_choice_state_value(st.session_state, key, options=options, default=default)


def _normalize_text_state(key: str, *, default: str = "") -> str:
    return normalize_text_state_value(st.session_state, key, default=default)


def _normalize_bool_state(key: str, *, default: bool = False) -> bool:
    value = st.session_state.get(key, default)
    if not isinstance(value, bool):
        value = bool(default)
    st.session_state[key] = value
    return value

def _clamp_numeric_state(key: str, *, min_value: float | None = None, max_value: float | None = None, default: float | int | None = None) -> None:
    if default is None:
        return
    normalize_optional_numeric_state_value(
        st.session_state,
        key,
        default=default,
        cast_type=int if isinstance(default, int) else float,
        min_value=min_value,
        max_value=max_value,
    )


def render_new_run_page(
    service: NewRunPageService,
    *,
    show_title: bool = True,
    submit_label: str = "Run",
) -> None:
    if show_title:
        st.title("New Run")

    problem_family_options = list_problem_families()
    current_problem_family = _normalize_choice_state("nr_problem_family", problem_family_options, default="matrix_completion")
    current_data_source = str(st.session_state.get("nr_data_source", "synthetic")) if current_problem_family == "matrix_completion" else "synthetic"
    default_request = service.sync_default_state(
        st.session_state,
        prefix="nr",
        problem_family=current_problem_family,
        data_source=current_data_source,
    )
    current_problem_family = _normalize_choice_state("nr_problem_family", problem_family_options, default=current_problem_family)

    st.markdown("### Header")
    st.caption("Define one experiment configuration, preview it, then launch it directly.")
    _normalize_text_state("nr_run_name", default=str(default_request.run_name or ""))
    run_name = _text_input("run name", value=str(st.session_state["nr_run_name"]), key="nr_run_name")
    _normalize_text_state("nr_notes", default=str(default_request.notes or ""))
    notes = _text_area("notes", value=str(st.session_state["nr_notes"]), key="nr_notes", height=100, placeholder="Optional notes, rationale, or reminders for this run")

    st.divider()
    st.markdown("### Core experiment setup")
    problem_family = _selectbox("problem_family", problem_family_options, index=problem_family_options.index(current_problem_family), key="nr_problem_family")
    data_source = "synthetic"
    if problem_family == "matrix_completion":
        data_source_options = list_data_sources(problem_family)
        current_data_source = _normalize_choice_state("nr_data_source", data_source_options, default=current_data_source if current_data_source in data_source_options else data_source_options[0])
        data_source = _selectbox("data_source", data_source_options, index=data_source_options.index(current_data_source), key="nr_data_source")
    psd_regime_options = list_psd_regimes()
    if problem_family == "psd_sensing":
        current_psd_regime = _normalize_choice_state("nr_psd_regime", psd_regime_options, default=str(default_request.psd_regime))
        psd_regime = _selectbox("psd_regime", psd_regime_options, index=psd_regime_options.index(current_psd_regime), key="nr_psd_regime")
    else:
        psd_regime = "non_commuting"
    noncommutativity_level = 1.0
    coherence_level = 0.0
    sampling_regime = "observed" if is_real_data_source(problem_family, data_source) else "uniform"
    mc_spectrum_type = "flat"
    mc_spectrum_decay = 1.0
    mc_factor_balance = "balanced"
    mc_factor_imbalance_level = 0.0
    early_stopping_enabled = False
    early_stopping_metric = "val_rmse"
    early_stopping_patience = 100
    early_stopping_min_delta = 0.0

    custom_dataset_id = ""
    custom_stats = None
    custom_stats_error = ""
    custom_records = service.list_custom_datasets() if problem_family == "matrix_completion" and data_source == "custom_triplets_csv" else []
    if problem_family == "matrix_completion" and data_source == "custom_triplets_csv":
        selected_dataset_id = str(st.session_state.get("nr_custom_dataset_id", ""))
        dataset_options = [""] + [record.dataset_id for record in custom_records]
        if selected_dataset_id not in dataset_options:
            selected_dataset_id = ""
        st.session_state["nr_custom_dataset_id"] = selected_dataset_id
        custom_dataset_id = _selectbox(
            "custom_dataset_id",
            options=dataset_options,
            index=dataset_options.index(selected_dataset_id),
            format_func=lambda value: "(select registered dataset)" if value == "" else value,
            key="nr_custom_dataset_id",
        )
        st.session_state["nr_custom_dataset_path"] = ""
        if custom_dataset_id:
            try:
                record = service.get_custom_dataset(custom_dataset_id)
                custom_stats = dict(record.stats)
                st.caption(f"Registered dataset · file={record.source_filename} · fingerprint={record.fingerprint[:12]}")
            except Exception as exc:
                custom_stats_error = str(exc)
        else:
            custom_stats_error = "Select a registered dataset from the Data Sources page first."
    st.divider()
    st.markdown("### Seeds")
    execution_mode_options = ["Single seed", "Multi-seed"]
    current_exec_mode = _normalize_choice_state("nr_exec_mode", execution_mode_options, default=execution_mode_options[0])
    execution_mode = _radio("Run mode", execution_mode_options, horizontal=True, index=execution_mode_options.index(current_exec_mode), key="nr_exec_mode")
    seed_mode = "single"
    seed = 0
    start_seed = 0
    num_seeds = 5
    seed_step = 1
    seeds_text = ""
    normalize_optional_numeric_state_value(st.session_state, "nr_seed", default=0, cast_type=int, min_value=0)
    normalize_optional_numeric_state_value(st.session_state, "nr_start_seed", default=0, cast_type=int, min_value=0)
    normalize_optional_numeric_state_value(st.session_state, "nr_num_seeds", default=5, cast_type=int, min_value=1)
    normalize_optional_numeric_state_value(st.session_state, "nr_seed_step", default=1, cast_type=int, min_value=1)
    if execution_mode == "Single seed":
        seed = int(_number_input("seed", min_value=0, value=int(st.session_state["nr_seed"]), step=1, key="nr_seed"))
    else:
        seed_mode_options = ["Generated seed set", "Manual seed list"]
        current_seed_mode = _normalize_choice_state("nr_seed_mode", seed_mode_options, default=seed_mode_options[0])
        seed_input_mode = _radio("Seed input", seed_mode_options, horizontal=True, index=seed_mode_options.index(current_seed_mode), key="nr_seed_mode")
        if seed_input_mode == "Generated seed set":
            seed_mode = "generated"
            c1, c2, c3 = st.columns(3)
            with c1:
                start_seed = int(_number_input("start_seed", min_value=0, value=int(st.session_state["nr_start_seed"]), step=1, key="nr_start_seed"))
            with c2:
                num_seeds = int(_number_input("num_seeds", min_value=1, value=int(st.session_state["nr_num_seeds"]), step=1, key="nr_num_seeds"))
            with c3:
                seed_step = int(_number_input("seed_step", min_value=1, value=int(st.session_state["nr_seed_step"]), step=1, key="nr_seed_step"))
        else:
            seed_mode = "manual"
            _normalize_text_state("nr_seeds_text", default="0,1,2,3,4")
            seeds_text = _text_input("manual seeds", value="0,1,2,3,4", key="nr_seeds_text")

    _clamp_numeric_state("nr_r_star", min_value=1, default=int(default_request.r_star))
    _clamp_numeric_state("nr_factor_rank", min_value=1, default=int(default_request.factor_rank))
    _clamp_numeric_state("nr_init_scale", min_value=0.0, default=float(default_request.init_scale))
    _clamp_numeric_state("nr_lr", min_value=0.0, default=float(default_request.lr))
    _clamp_numeric_state("nr_steps", min_value=1, default=int(default_request.steps))
    _clamp_numeric_state("nr_log_every", min_value=1, default=int(default_request.log_every))
    _clamp_numeric_state("nr_lr_decay_gamma", min_value=1e-6, max_value=1.0, default=float(default_request.lr_decay_gamma))
    _clamp_numeric_state("nr_lr_step_size", min_value=1, default=int(default_request.lr_step_size))
    _clamp_numeric_state("nr_lr_min", min_value=0.0, default=float(default_request.lr_min))
    _clamp_numeric_state("nr_noise_std", min_value=0.0, default=float(default_request.noise_std))
    _clamp_numeric_state("nr_explicit_reg_strength", min_value=0.0, default=float(default_request.explicit_reg_strength))
    _clamp_numeric_state("nr_grad_norm_tol", min_value=1e-16, default=float(default_request.grad_norm_tol))
    _clamp_numeric_state("nr_objective_tol", min_value=1e-16, default=float(default_request.objective_tol))
    _clamp_numeric_state("nr_divergence_threshold", min_value=1.0, default=float(default_request.divergence_norm_threshold))
    stopping_mode_options = list_stopping_modes()
    current_stopping_mode = _normalize_choice_state("nr_stopping_mode", stopping_mode_options, default=str(default_request.stopping_mode))

    st.divider()
    st.markdown("### Data setup")
    data_left, data_right = st.columns(2)
    with data_left:
        st.markdown("#### Dimensions")
        if problem_family == "matrix_completion" and data_source == "movielens_100k":
            stats = get_data_source_stats(problem_family, data_source) or {"n_rows": 943, "n_cols": 1682, "m": 100000}
            st.caption(f"MovieLens 100K · users={stats['n_rows']} · items={stats['n_cols']} · ratings={stats['m']}")
            n = int(stats["n_rows"])
            n_cols = int(stats["n_cols"])
            m = int(stats["m"])
            normalize_optional_numeric_state_value(st.session_state, "nr_n", default=n, cast_type=int, min_value=2)
            normalize_optional_numeric_state_value(st.session_state, "nr_n_cols", default=n_cols, cast_type=int, min_value=2)
            normalize_optional_numeric_state_value(st.session_state, "nr_m", default=m, cast_type=int, min_value=1)
            _number_input("n", value=int(st.session_state["nr_n"]), disabled=True, key="nr_n")
            _number_input("n_cols", value=int(st.session_state["nr_n_cols"]), disabled=True, key="nr_n_cols")
            _number_input("m observed entries", value=int(st.session_state["nr_m"]), disabled=True, key="nr_m")
        elif problem_family == "matrix_completion" and data_source == "custom_triplets_csv":
            if custom_stats is not None:
                st.caption(f"Custom triplets CSV · rows={custom_stats['n_rows']} · cols={custom_stats['n_cols']} · observations={custom_stats['m']}")
                n = int(custom_stats["n_rows"])
                n_cols = int(custom_stats["n_cols"])
                m = int(custom_stats["m"])
            else:
                if custom_stats_error:
                    st.error(custom_stats_error)
                st.caption("Select a registered dataset to infer matrix dimensions and observation count.")
                n = 2
                n_cols = 2
                m = 1
            normalize_optional_numeric_state_value(st.session_state, "nr_n", default=n, cast_type=int, min_value=2)
            normalize_optional_numeric_state_value(st.session_state, "nr_n_cols", default=n_cols, cast_type=int, min_value=2)
            normalize_optional_numeric_state_value(st.session_state, "nr_m", default=m, cast_type=int, min_value=1)
            _number_input("n", value=int(st.session_state["nr_n"]), disabled=True, key="nr_n")
            _number_input("n_cols", value=int(st.session_state["nr_n_cols"]), disabled=True, key="nr_n_cols")
            _number_input("m observed entries", value=int(st.session_state["nr_m"]), disabled=True, key="nr_m")
            if custom_stats is not None:
                st.caption(f"Value range: {custom_stats['rating_min']:.4g} to {custom_stats['rating_max']:.4g}")
        else:
            normalize_optional_numeric_state_value(st.session_state, "nr_n", default=int(st.session_state.get("nr_n", 40) or 40), cast_type=int, min_value=2)
            n = int(_number_input("n", min_value=2, value=int(st.session_state["nr_n"]), step=1, key="nr_n"))
            n_cols = None
            if problem_family == "matrix_completion":
                default_n_cols = int(st.session_state.get("nr_n_cols") or st.session_state["nr_n"])
                normalize_optional_numeric_state_value(st.session_state, "nr_n_cols", default=default_n_cols, cast_type=int, min_value=2)
                n_cols = int(_number_input("n_cols", min_value=2, value=int(st.session_state["nr_n_cols"]), step=1, key="nr_n_cols"))
            label = "m measurements" if problem_family == "psd_sensing" else "m observed entries"
            normalize_optional_numeric_state_value(st.session_state, "nr_m", default=int(st.session_state.get("nr_m", 800) or 800), cast_type=int, min_value=1)
            m = int(_number_input(label, min_value=1, value=int(st.session_state["nr_m"]), step=50, key="nr_m"))
    with data_right:
        st.markdown("#### Splits / noise")
        if problem_family == "matrix_completion" and is_real_data_source(problem_family, data_source):
            _clamp_numeric_state("nr_val_fraction", min_value=0.0, max_value=0.49, default=float(default_request.val_fraction))
            val_fraction = float(_number_input("val_fraction", min_value=0.0, max_value=0.49, value=float(st.session_state["nr_val_fraction"]), step=0.05, format="%.2f", key="nr_val_fraction"))
            _clamp_numeric_state("nr_test_fraction", min_value=0.0, max_value=0.49, default=float(default_request.test_fraction))
            test_fraction = float(_number_input("test_fraction", min_value=0.0, max_value=0.49, value=float(st.session_state["nr_test_fraction"]), step=0.05, format="%.2f", key="nr_test_fraction"))
            _normalize_bool_state("nr_early_stopping_enabled", default=True)
            early_stopping_enabled = bool(_checkbox("early_stopping_enabled", value=True, key="nr_early_stopping_enabled"))
            if early_stopping_enabled:
                early_metric_options = ["val_rmse", "val_loss"]
                current_early_metric = _normalize_choice_state("nr_early_stopping_metric", early_metric_options, default=str(default_request.early_stopping_metric))
                early_stopping_metric = _selectbox("early_stopping_metric", early_metric_options, index=early_metric_options.index(current_early_metric), key="nr_early_stopping_metric")
                c1, c2 = st.columns(2)
                with c1:
                    normalize_optional_numeric_state_value(st.session_state, "nr_early_stopping_patience", default=100, cast_type=int, min_value=1)
                    early_stopping_patience = int(_number_input("early_stopping_patience", min_value=1, value=int(st.session_state["nr_early_stopping_patience"]), step=1, key="nr_early_stopping_patience"))
                with c2:
                    normalize_optional_numeric_state_value(st.session_state, "nr_early_stopping_min_delta", default=1e-4, cast_type=float, min_value=0.0)
                    early_stopping_min_delta = float(_number_input("early_stopping_min_delta", min_value=0.0, value=float(st.session_state["nr_early_stopping_min_delta"]), format="%.6f", key="nr_early_stopping_min_delta"))
            st.session_state["nr_noise_std"] = 0.0
            noise_std = 0.0
            st.caption("Noise is disabled for real/data-backed matrix completion sources.")
        else:
            val_fraction = 0.1
            test_fraction = 0.1
            noise_std = float(_number_input("noise_std", min_value=0.0, value=float(st.session_state["nr_noise_std"]), format="%.6f", key="nr_noise_std"))

    with st.expander("Structure / sampling controls", expanded=False):
        if problem_family == "psd_sensing":
            if psd_regime == "commuting":
                _clamp_numeric_state("nr_noncommutativity_level", min_value=0.0, max_value=0.0, default=0.0)
                noncommutativity_level = float(_number_input("noncommutativity_level", min_value=0.0, max_value=0.0, value=0.0, step=0.05, disabled=True, key="nr_noncommutativity_level"))
            else:
                _clamp_numeric_state("nr_noncommutativity_level", min_value=0.0, max_value=1.0, default=1.0)
                noncommutativity_level = float(_slider("noncommutativity_level", min_value=0.0, max_value=1.0, value=1.0, step=0.05, key="nr_noncommutativity_level"))
        elif data_source == "synthetic":
            _clamp_numeric_state("nr_coherence_level", min_value=0.0, max_value=1.0, default=0.0)
            coherence_level = float(_slider("coherence_level", min_value=0.0, max_value=1.0, value=0.0, step=0.05, key="nr_coherence_level"))
            sampling_options = ["uniform", "leverage_biased"]
            current_sampling_regime = _normalize_choice_state("nr_sampling_regime", sampling_options, default=str(default_request.sampling_regime))
            sampling_regime = _selectbox("sampling_regime", sampling_options, index=sampling_options.index(current_sampling_regime), key="nr_sampling_regime")
            spectrum_options = ["flat", "power_decay", "exponential_decay"]
            current_mc_spectrum_type = _normalize_choice_state("nr_mc_spectrum_type", spectrum_options, default=str(default_request.mc_spectrum_type))
            mc_spectrum_type = _selectbox("mc_spectrum_type", spectrum_options, index=spectrum_options.index(current_mc_spectrum_type), key="nr_mc_spectrum_type")
            if mc_spectrum_type == "flat":
                _clamp_numeric_state("nr_mc_spectrum_decay", min_value=0.0, max_value=0.0, default=0.0)
                mc_spectrum_decay = float(_number_input("mc_spectrum_decay", min_value=0.0, max_value=0.0, value=0.0, step=0.1, disabled=True, key="nr_mc_spectrum_decay"))
            else:
                _clamp_numeric_state("nr_mc_spectrum_decay", min_value=0.0, default=1.0)
                mc_spectrum_decay = float(_number_input("mc_spectrum_decay", min_value=0.0, value=1.0, step=0.1, key="nr_mc_spectrum_decay"))
        if problem_family == "matrix_completion":
            factor_balance_options = ["balanced", "unbalanced"]
            current_factor_balance = _normalize_choice_state("nr_mc_factor_balance", factor_balance_options, default=str(default_request.mc_factor_balance))
            mc_factor_balance = _selectbox("mc_factor_balance", factor_balance_options, index=factor_balance_options.index(current_factor_balance), key="nr_mc_factor_balance")
            if mc_factor_balance == "balanced":
                _clamp_numeric_state("nr_mc_factor_imbalance_level", min_value=0.0, max_value=0.0, default=0.0)
                mc_factor_imbalance_level = float(_number_input("mc_factor_imbalance_level", min_value=0.0, max_value=0.0, value=0.0, step=0.05, disabled=True, key="nr_mc_factor_imbalance_level"))
            else:
                _clamp_numeric_state("nr_mc_factor_imbalance_level", min_value=0.0, max_value=1.0, default=0.5)
                mc_factor_imbalance_level = float(_slider("mc_factor_imbalance_level", min_value=0.0, max_value=1.0, value=0.5, step=0.05, key="nr_mc_factor_imbalance_level"))

    st.divider()
    st.markdown("### Model / parameterization")
    model_left, model_right = st.columns(2)
    with model_left:
        parameterization_options = list_parameterizations(problem_family)
        preferred_parameterization = "factorized_uv" if problem_family == "matrix_completion" and "factorized_uv" in parameterization_options else ("factorized_psd" if "factorized_psd" in parameterization_options else parameterization_options[0])
        current_parameterization = _normalize_choice_state("nr_parameterization", parameterization_options, default=preferred_parameterization)
        parameterization = _selectbox("parameterization", parameterization_options, index=parameterization_options.index(current_parameterization), key="nr_parameterization")
        init_type_options = list_init_types()
        current_init_type = _normalize_choice_state("nr_init_type", init_type_options, default=str(default_request.init_type))
        _selectbox("init_type", init_type_options, index=init_type_options.index(current_init_type), disabled=True, key="nr_init_type")
    with model_right:
        r_star = int(_number_input("r_star", min_value=1, value=int(st.session_state["nr_r_star"]), step=1, key="nr_r_star"))
        factor_rank = int(_number_input("factor_rank", min_value=1, value=int(st.session_state["nr_factor_rank"]), step=1, key="nr_factor_rank"))
        init_scale = float(_number_input("init_scale", min_value=0.0, value=float(st.session_state["nr_init_scale"]), format="%.6f", key="nr_init_scale"))
    with st.expander("Explicit regularization", expanded=False):
        allowed_reg_types = explicit_reg_types_for_parameterization(parameterization)
        default_reg_type = str(default_request.explicit_reg_type) if str(default_request.explicit_reg_type) in allowed_reg_types else ("none" if "none" in allowed_reg_types else allowed_reg_types[0])
        current_reg_type = _normalize_choice_state("nr_explicit_reg_type", allowed_reg_types, default=default_reg_type)
        explicit_reg_type = _selectbox("explicit_reg_type", allowed_reg_types, index=allowed_reg_types.index(current_reg_type), key="nr_explicit_reg_type")
        explicit_reg_strength = float(_number_input("explicit_reg_strength", min_value=0.0, value=float(st.session_state["nr_explicit_reg_strength"]), format="%.6f", key="nr_explicit_reg_strength", disabled=(explicit_reg_type == "none")))
        if explicit_reg_type == "none":
            explicit_reg_strength = 0.0

    st.divider()
    st.markdown("### Optimization")
    opt_left, opt_right = st.columns(2)
    with opt_left:
        optimizer_options = list_optimizers(problem_family)
        current_optimizer = _normalize_choice_state("nr_optimizer", optimizer_options, default=str(default_request.optimizer))
        optimizer = _selectbox("optimizer", optimizer_options, index=optimizer_options.index(current_optimizer), key="nr_optimizer")
        lr = float(_number_input("lr", min_value=0.0, value=float(st.session_state["nr_lr"]), format="%.6f", key="nr_lr"))
    with opt_right:
        batch_size = None
        if optimizer in {"sgd", "momentum"}:
            max_batch = int(m if problem_family != "matrix_completion" or not is_real_data_source(problem_family, data_source) else int(m * (1.0 - val_fraction - test_fraction)))
            default_batch_size = min(64, max(1, max_batch))
            normalize_optional_numeric_state_value(st.session_state, "nr_batch_size", default=default_batch_size, cast_type=int, min_value=1, max_value=max(1, max_batch))
            batch_size = int(_number_input("batch_size", min_value=1, max_value=max(1, max_batch), value=int(st.session_state["nr_batch_size"]), step=1, key="nr_batch_size"))

    lr_decay_gamma = 0.5
    lr_step_size = 100
    lr_min = 0.0
    with st.expander("LR schedule", expanded=False):
        lr_schedule_options = list_lr_schedule_types()
        current_lr_schedule = _normalize_choice_state("nr_lr_schedule_type", lr_schedule_options, default=str(default_request.lr_schedule_type))
        lr_schedule_type = _selectbox("lr_schedule_type", lr_schedule_options, index=lr_schedule_options.index(current_lr_schedule), key="nr_lr_schedule_type")
        if lr_schedule_type == "step_decay":
            c1, c2 = st.columns(2)
            with c1:
                lr_decay_gamma = float(_number_input("lr_decay_gamma", min_value=1e-6, max_value=1.0, value=float(st.session_state["nr_lr_decay_gamma"]), format="%.6f", key="nr_lr_decay_gamma"))
            with c2:
                lr_step_size = int(_number_input("lr_step_size", min_value=1, value=int(st.session_state["nr_lr_step_size"]), step=1, key="nr_lr_step_size"))
        elif lr_schedule_type == "cosine":
            lr_min = float(_number_input("lr_min", min_value=0.0, value=float(st.session_state["nr_lr_min"]), format="%.6f", key="nr_lr_min"))

    st.divider()
    st.markdown("### Runtime / logging")
    runtime_left, runtime_right = st.columns(2)
    with runtime_left:
        log_every = int(_number_input("log_every", min_value=1, value=int(st.session_state["nr_log_every"]), step=1, key="nr_log_every"))
    with runtime_right:
        device_options = list_device_preferences()
        current_device_preference = _normalize_choice_state("nr_device_preference", device_options, default=str(default_request.device_preference))
        device_preference = _selectbox("device", device_options, index=device_options.index(current_device_preference), key="nr_device_preference")
        st.caption("auto uses CPU for interactive runs; choose cuda/mps explicitly for accelerator tests.")
        dtype_options = list_dtype_preferences()
        current_dtype_preference = _normalize_choice_state("nr_dtype_preference", dtype_options, default=str(default_request.dtype_preference))
        dtype_preference = _selectbox("dtype", dtype_options, index=dtype_options.index(current_dtype_preference), key="nr_dtype_preference")
    execution_mode_value = str(default_request.execution_mode)

    st.divider()
    st.markdown("### Stopping")
    stop_left, stop_right = st.columns(2)
    with stop_left:
        stopping_mode = _selectbox("stopping_strategy", stopping_mode_options, index=stopping_mode_options.index(current_stopping_mode), key="nr_stopping_mode")
    with stop_right:
        if stopping_mode == "run_to_convergence":
            st.caption("Stops when convergence is reached, but this cap prevents endless runs.")
            steps = int(_number_input("Safety cap (max_steps)", min_value=1, value=int(st.session_state["nr_steps"]), step=100, key="nr_steps"))
        else:
            steps = int(_number_input("Steps", min_value=1, value=int(st.session_state["nr_steps"]), step=100, key="nr_steps"))
    if stopping_mode == "run_to_convergence":
        with st.expander("Advanced stopping", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                grad_norm_tol = float(_number_input("grad_norm_tol", min_value=1e-16, value=float(st.session_state["nr_grad_norm_tol"]), format="%.2e", key="nr_grad_norm_tol"))
            with c2:
                objective_tol = float(_number_input("objective_tol", min_value=1e-16, value=float(st.session_state["nr_objective_tol"]), format="%.2e", key="nr_objective_tol"))
            with c3:
                divergence_norm_threshold = float(_number_input("divergence_norm_threshold", min_value=1.0, value=float(st.session_state["nr_divergence_threshold"]), format="%.2e", key="nr_divergence_threshold"))
    else:
        grad_norm_tol = float(st.session_state["nr_grad_norm_tol"])
        objective_tol = float(st.session_state["nr_objective_tol"])
        divergence_norm_threshold = float(st.session_state["nr_divergence_threshold"])

    st.divider()
    st.markdown("### Preview / launch")
    request = service.get_request_from_state(
        st.session_state,
        prefix="nr",
        problem_family=problem_family,
        data_source=data_source,
    )

    validation = service.validate_request(request)
    if execution_mode == "Multi-seed" and validation.seeds:
        st.caption(f"Seeds: {format_display_sequence(validation.seeds)}")
    for error in validation.errors:
        st.error(error)
    for message in service.audit_request(request).messages:
        st.warning(message)

    if not st.button(submit_label, type="primary", key="nr_run_button"):
        return
    if not validation.is_valid or validation.plan is None:
        st.error("Fix the run configuration first.")
        return

    plan = validation.plan
    progress = st.progress(0.0)
    status = st.empty()

    execution = service.execute_plan(
        plan,
        on_status=status.write,
        on_progress=lambda current, total: progress.progress(current / total),
    )

    if execution.failures:
        if execution.is_complete_failure:
            st.error(f"All {execution.failed_count} runs failed.")
        else:
            st.warning(f"Saved {execution.succeeded_count} runs, but {execution.failed_count} failed.")
        for failure in execution.failures:
            st.error(f"seed={failure.seed} · stage={failure.stage} · {failure.error_type}: {failure.error_message}")

    if execution.is_complete_failure:
        return

    if len(execution.seeds) == 1:
        st.success(f"Run saved: {execution.saved_run_ids[0]}")
        st.json(make_json_safe(execution.items[0].summary))
        return

    if execution.is_complete_success:
        st.success(f"Saved {len(execution.saved_run_ids)} runs under seed_group_id={execution.seed_group_id}")
    if execution.group is not None:
        st.dataframe(service.aggregate_seed_group_final_metrics(execution.group), use_container_width=True)
