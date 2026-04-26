from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Callable
from datetime import datetime
import traceback

from framework.core.experiments import list_optimizers
from framework.core.models import RunSpec
from framework.services.run_repository import RunRepository
from framework.core.parameterization import list_parameterizations
from framework.services.local_runs import resolve_seed_values
from framework.services.custom_dataset_registry import LocalCustomDatasetRegistry
from framework.services.experiment_protocols import ExperimentProtocol, normalize_experiment_protocol, protocol_metadata
from framework.core.regularization import explicit_reg_types_for_parameterization
from framework.services.data_sources import (
    DEFAULT_MOVIELENS_DATASET_VERSION,
    DEFAULT_MOVIELENS_PREPROCESSING_VERSION,
    DEFAULT_MOVIELENS_SPLIT_VERSION,
    DEFAULT_SYNTHETIC_DATASET_VERSION,
    DEFAULT_SYNTHETIC_PREPROCESSING_VERSION,
    DEFAULT_SYNTHETIC_SPLIT_VERSION,
    canonical_data_source,
    derive_custom_triplets_dataset_version,
    get_custom_triplets_stats,
    get_data_source_default_provenance,
    get_default_request_values,
    is_real_data_source,
    supports_early_stopping,
)


DEFAULT_INIT_TYPE = "near_zero_gaussian"


@dataclass(frozen=True)
class NewRunRequest:
    seed_mode: str = "single"
    seed: int | None = 0
    start_seed: int | None = 0
    num_seeds: int | None = 5
    seed_step: int = 1
    seeds_text: str = ""
    n: int = 40
    n_cols: int | None = None
    r_star: int = 5
    m: int = 800
    factor_rank: int = 10
    init_scale: float = 1e-2
    lr: float = 1e-3
    steps: int = 2000
    log_every: int = 20
    lr_schedule_type: str = "constant"
    lr_decay_gamma: float = 0.5
    lr_step_size: int = 100
    lr_min: float = 0.0
    noise_std: float = 0.0
    problem_family: str = "psd_sensing"
    data_source: str = "synthetic"
    custom_dataset_path: str = ""
    custom_dataset_id: str = ""
    dataset_version: str = DEFAULT_SYNTHETIC_DATASET_VERSION
    split_version: str = DEFAULT_SYNTHETIC_SPLIT_VERSION
    preprocessing_version: str = DEFAULT_SYNTHETIC_PREPROCESSING_VERSION
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    early_stopping_enabled: bool = False
    early_stopping_metric: str = "val_rmse"
    early_stopping_patience: int = 100
    early_stopping_min_delta: float = 0.0
    psd_regime: str = "non_commuting"
    noncommutativity_level: float = 1.0
    coherence_level: float = 0.0
    sampling_regime: str = "uniform"
    mc_spectrum_type: str = "flat"
    mc_spectrum_decay: float = 1.0
    mc_factor_balance: str = "balanced"
    mc_factor_imbalance_level: float = 0.0
    optimizer: str = "gd"
    execution_mode: str = "accuracy"
    device_preference: str = "auto"
    dtype_preference: str = "auto"
    batch_size: int | None = None
    parameterization: str = "factorized_psd"
    explicit_reg_type: str = "none"
    explicit_reg_strength: float = 0.0
    stopping_mode: str = "run_to_convergence"
    grad_norm_tol: float = 1e-10
    objective_tol: float = 1e-12
    divergence_norm_threshold: float = 1e6
    run_name: str | None = None
    notes: str | None = ""
    init_type: str = DEFAULT_INIT_TYPE
    num_eigs: int = 5
    protocol: ExperimentProtocol | None = None





@dataclass(frozen=True)
class NewRunExecutionPlan:
    seeds: list[int]
    seed_group_id: str | None
    specs: list[RunSpec]
    protocol: ExperimentProtocol | None = None




@dataclass(frozen=True)
class NewRunSavedItem:
    run_id: str
    seed: int
    config: dict[str, Any]
    summary: dict[str, Any]
    metrics: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class NewRunFailure:
    index: int
    seed: int
    stage: str
    error_type: str
    error_message: str
    traceback_text: str


@dataclass(frozen=True)
class NewRunExecutionResult:
    seed_group_id: str | None
    seeds: list[int]
    saved_run_ids: list[str]
    items: list[NewRunSavedItem]
    failures: list[NewRunFailure]
    group: dict[str, Any] | None

    @property
    def succeeded_count(self) -> int:
        return len(self.items)

    @property
    def failed_count(self) -> int:
        return len(self.failures)

    @property
    def is_complete_success(self) -> bool:
        return self.failed_count == 0

    @property
    def is_complete_failure(self) -> bool:
        return self.succeeded_count == 0 and self.failed_count > 0


@dataclass(frozen=True)
class NewRunValidationResult:
    errors: list[str]
    seeds: list[int]
    plan: NewRunExecutionPlan | None

    @property
    def is_valid(self) -> bool:
        return not self.errors




_REQUEST_FIELD_NAMES = tuple(field.name for field in fields(NewRunRequest) if field.name != "protocol")
_UI_SINGLE_SEED_MODE = "Single seed"
_UI_MULTI_SEED_MODE = "Multi-seed"
_UI_GENERATED_SEEDS = "Generated seed set"
_UI_MANUAL_SEEDS = "Manual seed list"




def normalize_choice_state_value(state: dict, key: str, *, options: list[str] | tuple[str, ...], default: str) -> str:
    valid_options = [str(option) for option in options]
    fallback = str(default) if str(default) in valid_options else (valid_options[0] if valid_options else "")
    current = state.get(key, fallback)
    if current is None:
        current = fallback
    current = str(current)
    if current not in valid_options:
        current = fallback
    state[key] = current
    return current


def normalize_text_state_value(state: dict, key: str, *, default: str = "") -> str:
    value = state.get(key, default)
    if value is None:
        value = default
    value = str(value)
    state[key] = value
    return value

def normalize_optional_numeric_state_value(
    state: dict,
    key: str,
    *,
    default: float | int,
    cast_type: type[int] | type[float],
    min_value: float | None = None,
    max_value: float | None = None,
) -> float | int:
    value = state.get(key, default)
    if value is None:
        value = default
    try:
        numeric = cast_type(value)
    except (TypeError, ValueError):
        numeric = cast_type(default)
    if min_value is not None and numeric < min_value:
        numeric = cast_type(min_value)
    if max_value is not None and numeric > max_value:
        numeric = cast_type(max_value)
    state[key] = numeric
    return numeric


def bind_new_run_request_state(state: dict, *, prefix: str, request: NewRunRequest) -> None:
    for field_name in _REQUEST_FIELD_NAMES:
        state[f"{prefix}_{field_name}"] = getattr(request, field_name)


def sync_default_new_run_request_state(state: dict, *, prefix: str, problem_family: str, data_source: str) -> NewRunRequest:
    default_request = get_default_new_run_request(problem_family, data_source)
    prev_key = f"{prefix}_defaults_key"
    state_key = (problem_family, data_source)
    if state.get(prev_key) == state_key:
        return default_request

    current_optimizer = state.get(f"{prefix}_optimizer")
    if current_optimizer in list_optimizers(problem_family):
        default_request = replace(default_request, optimizer=str(current_optimizer))

    current_parameterization = state.get(f"{prefix}_parameterization")
    if current_parameterization in list_parameterizations(problem_family):
        default_request = replace(default_request, parameterization=str(current_parameterization))

    bind_new_run_request_state(state, prefix=prefix, request=default_request)
    state[f"{prefix}_problem_family"] = str(problem_family)
    state[f"{prefix}_data_source"] = str(data_source)
    state[f"{prefix}_exec_mode"] = _UI_SINGLE_SEED_MODE if default_request.seed_mode == "single" else _UI_MULTI_SEED_MODE
    state[f"{prefix}_seed_mode"] = _UI_GENERATED_SEEDS if default_request.seed_mode != "manual" else _UI_MANUAL_SEEDS
    state[prev_key] = state_key
    return default_request


def get_new_run_request_from_state(state: dict, *, prefix: str, problem_family: str, data_source: str) -> NewRunRequest:
    request = get_default_new_run_request(problem_family, data_source)
    updates: dict[str, object] = {}
    for field_name in _REQUEST_FIELD_NAMES:
        key = f"{prefix}_{field_name}"
        if key in state:
            updates[field_name] = state[key]

    updates["problem_family"] = str(problem_family)
    updates["data_source"] = str(data_source)

    execution_mode = state.get(f"{prefix}_exec_mode", _UI_SINGLE_SEED_MODE)
    if execution_mode == _UI_MULTI_SEED_MODE:
        seed_input_mode = state.get(f"{prefix}_seed_mode", _UI_GENERATED_SEEDS)
        updates["seed_mode"] = "manual" if seed_input_mode == _UI_MANUAL_SEEDS else "generated"
    else:
        updates["seed_mode"] = "single"

    return replace(request, **updates)


def _default_data_provenance(problem_family: str, data_source: str) -> dict[str, str]:
    return get_data_source_default_provenance(problem_family, data_source)


def _default_optimizer(problem_family: str) -> str:
    options = list_optimizers(problem_family)
    return options[0]


def _default_parameterization(problem_family: str) -> str:
    options = list_parameterizations(problem_family)
    preferred = {
        "psd_sensing": "factorized_psd",
        "matrix_completion": "factorized_uv",
    }.get(str(problem_family))
    if preferred in options:
        return str(preferred)
    return options[0]



def get_default_new_run_request(problem_family: str, data_source: str) -> NewRunRequest:
    canonical_source = canonical_data_source(problem_family, data_source)
    return NewRunRequest(
        problem_family=str(problem_family),
        data_source=str(canonical_source),
        optimizer=_default_optimizer(problem_family),
        parameterization=_default_parameterization(problem_family),
        **get_default_request_values(problem_family, canonical_source),
    )


def _build_seed_group_id() -> str:
    return f"seedgrp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def resolve_new_run_seeds(request: NewRunRequest) -> list[int]:
    return resolve_seed_values(
        request.seed_mode,
        seed=request.seed,
        start_seed=request.start_seed,
        num_seeds=request.num_seeds,
        seed_step=request.seed_step,
        seeds_text=request.seeds_text,
    )


def _measurement_type(problem_family: str, data_source: str, sampling_regime: str) -> str:
    if problem_family == "psd_sensing":
        return "gaussian_symmetric"
    if data_source in {"movielens_100k", "custom_triplets_csv"}:
        return "observed_ratings"
    return "uniform_entries" if sampling_regime == "uniform" else "leverage_biased_entries"


def normalize_new_run_request(raw: NewRunRequest, *, dataset_registry: LocalCustomDatasetRegistry | None = None) -> NewRunRequest:
    request = raw
    problem_family = str(request.problem_family)
    data_source = canonical_data_source(problem_family, str(request.data_source))
    parameterization = str(request.parameterization)
    optimizer = str(request.optimizer)

    defaults = _default_data_provenance(problem_family, data_source)
    custom_dataset_path = str(request.custom_dataset_path or "").strip()
    custom_dataset_id = str(request.custom_dataset_id or "").strip()
    if data_source != "custom_triplets_csv":
        custom_dataset_path = ""
        custom_dataset_id = ""
    custom_stats: dict[str, object] | None = None
    registry_record = None
    if problem_family == "matrix_completion" and data_source == "custom_triplets_csv":
        if dataset_registry is not None and (custom_dataset_id or custom_dataset_path):
            registry_record = dataset_registry.resolve_triplets_csv(dataset_id=custom_dataset_id, source_path=custom_dataset_path)
            custom_dataset_id = registry_record.dataset_id
            custom_dataset_path = registry_record.source_path
            custom_stats = dict(registry_record.stats)
            defaults = {
                **defaults,
                "dataset_version": registry_record.dataset_version,
                "split_version": registry_record.split_version,
                "preprocessing_version": registry_record.preprocessing_version,
            }
        else:
            if not custom_dataset_path:
                raise ValueError("custom_triplets_csv requires custom_dataset_path")
            custom_stats = get_custom_triplets_stats(custom_dataset_path)
            if not request.dataset_version or request.dataset_version in {
                defaults["dataset_version"],
                DEFAULT_SYNTHETIC_DATASET_VERSION,
            }:
                defaults = {
                    **defaults,
                    "dataset_version": derive_custom_triplets_dataset_version(custom_dataset_path),
                }
    dataset_version = str(request.dataset_version or defaults["dataset_version"])
    split_version = str(request.split_version or defaults["split_version"])
    preprocessing_version = str(request.preprocessing_version or defaults["preprocessing_version"])
    if problem_family == "matrix_completion" and data_source == "custom_triplets_csv":
        if dataset_version in {DEFAULT_SYNTHETIC_DATASET_VERSION, get_data_source_default_provenance(problem_family, data_source)["dataset_version"]}:
            dataset_version = str(defaults["dataset_version"])
    explicit_reg_type = str(request.explicit_reg_type)
    allowed_reg_types = explicit_reg_types_for_parameterization(parameterization)
    if explicit_reg_type not in allowed_reg_types:
        explicit_reg_type = "none"
    updates: dict[str, object] = {
        "run_name": request.run_name or None,
        "notes": request.notes or "",
        "init_type": str(request.init_type or DEFAULT_INIT_TYPE),
        "custom_dataset_path": custom_dataset_path or None,
        "custom_dataset_id": custom_dataset_id or None,
        "dataset_version": dataset_version,
        "split_version": split_version,
        "preprocessing_version": preprocessing_version,
        "noncommutativity_level": float(request.noncommutativity_level) if problem_family == "psd_sensing" else 1.0,
        "early_stopping_enabled": bool(request.early_stopping_enabled) if supports_early_stopping(problem_family, data_source) else False,
        "noise_std": 0.0 if is_real_data_source(problem_family, data_source) else float(request.noise_std),
        "batch_size": request.batch_size if optimizer in {"sgd", "momentum"} else None,
        "explicit_reg_type": explicit_reg_type,
        "explicit_reg_strength": 0.0 if explicit_reg_type == "none" else float(request.explicit_reg_strength),
        "num_eigs": max(1, min(int(request.num_eigs), int(request.n))),
        "protocol": normalize_experiment_protocol(request.protocol),
    }

    if problem_family == "psd_sensing":
        updates.update(
            n_cols=None,
            data_source="synthetic",
            custom_dataset_path=None,
            custom_dataset_id=None,
            sampling_regime="uniform",
            coherence_level=0.0,
            mc_spectrum_type="flat",
            mc_spectrum_decay=1.0,
            mc_factor_balance="balanced",
            mc_factor_imbalance_level=0.0,
        )
    else:
        sampling_regime = "observed" if is_real_data_source(problem_family, data_source) else str(request.sampling_regime)
        updates.update(
            sampling_regime=sampling_regime,
            coherence_level=float(request.coherence_level) if data_source == "synthetic" else 0.0,
            mc_spectrum_type=str(request.mc_spectrum_type) if data_source == "synthetic" else "flat",
            mc_spectrum_decay=float(request.mc_spectrum_decay) if data_source == "synthetic" else 1.0,
        )
        if parameterization == "factorized_uv":
            updates.update(
                mc_factor_balance=str(request.mc_factor_balance),
                mc_factor_imbalance_level=float(request.mc_factor_imbalance_level),
            )
        else:
            updates.update(
                mc_factor_balance="balanced",
                mc_factor_imbalance_level=0.0,
            )
        if custom_stats is not None:
            updates.update(
                n=int(custom_stats["n_rows"]),
                n_cols=int(custom_stats["n_cols"]),
                m=int(custom_stats["m"]),
            )

    updates["num_eigs"] = max(1, min(int(request.num_eigs), int(updates.get("n", request.n))))
    return replace(request, **updates)


def build_run_spec_kwargs_from_request(request: NewRunRequest, *, dataset_registry: LocalCustomDatasetRegistry | None = None) -> dict[str, object]:
    normalized = normalize_new_run_request(request, dataset_registry=dataset_registry)
    spec = asdict(normalized)
    for key in {"seed_mode", "seed", "start_seed", "num_seeds", "seed_step", "seeds_text", "protocol"}:
        spec.pop(key, None)
    sampling_regime = str(spec["sampling_regime"])
    spec["measurement_type"] = _measurement_type(
        str(spec["problem_family"]),
        str(spec["data_source"]),
        sampling_regime,
    )
    return spec


def _build_new_run_execution_plan(request: NewRunRequest, seeds: list[int], *, dataset_registry: LocalCustomDatasetRegistry | None = None) -> NewRunExecutionPlan:
    spec_kwargs = build_run_spec_kwargs_from_request(request, dataset_registry=dataset_registry)
    seed_group_id = _build_seed_group_id() if len(seeds) > 1 else None
    specs = [RunSpec(seed=int(seed_value), seed_group_id=seed_group_id, **spec_kwargs) for seed_value in seeds]
    return NewRunExecutionPlan(seeds=seeds, seed_group_id=seed_group_id, specs=specs, protocol=normalize_experiment_protocol(request.protocol))


def validate_new_run_request(request: NewRunRequest, *, dataset_registry: LocalCustomDatasetRegistry | None = None) -> NewRunValidationResult:
    errors: list[str] = []
    seeds: list[int] = []
    plan: NewRunExecutionPlan | None = None

    try:
        seeds = resolve_new_run_seeds(request)
    except Exception as exc:
        errors.append(str(exc))
        return NewRunValidationResult(errors=errors, seeds=seeds, plan=None)

    try:
        plan = _build_new_run_execution_plan(request, seeds, dataset_registry=dataset_registry)
        for spec in plan.specs:
            spec.validate()
    except Exception as exc:
        errors.append(str(exc))
        plan = None

    return NewRunValidationResult(errors=errors, seeds=seeds, plan=plan)


def build_new_run_execution_plan(request: NewRunRequest, *, dataset_registry: LocalCustomDatasetRegistry | None = None) -> NewRunExecutionPlan:
    return _build_new_run_execution_plan(request, resolve_new_run_seeds(request), dataset_registry=dataset_registry)



def _saved_item_from_result(result: Any, run_id: str) -> NewRunSavedItem:
    return NewRunSavedItem(
        run_id=str(run_id),
        seed=int(result.spec.seed),
        config=result.spec.to_dict(),
        summary=dict(result.summary),
        metrics=dict(result.metrics),
        metadata=dict(getattr(result, "metadata", {}) or {}),
    )


def _new_run_group_payload(
    *,
    seed_group_id: str | None,
    seeds: list[int],
    items: list[NewRunSavedItem],
    failures: list[NewRunFailure],
) -> dict[str, Any] | None:
    if len(seeds) <= 1 or not items:
        return None
    return {
        "seed_group_id": seed_group_id,
        "group_id": None,
        "items": [
            {
                "run_id": item.run_id,
                "config": item.config,
                "summary": item.summary,
                "metrics": item.metrics,
                "metadata": item.metadata,
            }
            for item in items
        ],
        "seed_count": len(seeds),
        "run_count": len(items),
        "failure_count": len(failures),
        "seeds": list(seeds),
        "config": items[0].config,
    }


def execute_new_run_plan(
    plan: NewRunExecutionPlan,
    *,
    run_experiment_fn: Callable[[RunSpec], Any] | None = None,
    run_repository: RunRepository | None = None,
    on_status: Callable[[str], None] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> NewRunExecutionResult:
    if run_experiment_fn is None:
        raise ValueError("run_experiment_fn is required for direct execution")
    if run_repository is None:
        raise ValueError("run_repository is required for direct execution")

    metadata_patch = protocol_metadata(plan.protocol)
    items: list[NewRunSavedItem] = []
    failures: list[NewRunFailure] = []
    total = len(plan.specs)
    for index, spec in enumerate(plan.specs, start=1):
        if on_status is not None:
            on_status(f"Running seed={spec.seed} ({index}/{total})")
        try:
            result = run_experiment_fn(spec)
            if hasattr(run_repository, "save_result_with_metadata"):
                run_id = run_repository.save_result_with_metadata(result, metadata_patch)
            else:
                run_id = run_repository.save_result(result)
            items.append(_saved_item_from_result(result, run_id))
        except Exception as exc:
            failures.append(NewRunFailure(
                index=index,
                seed=int(spec.seed),
                stage="run_experiment",
                error_type=type(exc).__name__,
                error_message=str(exc),
                traceback_text=traceback.format_exc(),
            ))
        if on_progress is not None:
            on_progress(index, total)

    return NewRunExecutionResult(
        seed_group_id=plan.seed_group_id,
        seeds=list(plan.seeds),
        saved_run_ids=[item.run_id for item in items],
        items=items,
        failures=failures,
        group=_new_run_group_payload(
            seed_group_id=plan.seed_group_id,
            seeds=list(plan.seeds),
            items=items,
            failures=failures,
        ),
    )
