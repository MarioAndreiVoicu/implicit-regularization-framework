from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import traceback
from itertools import product
from math import log10, prod
from typing import Any, Callable

import pandas as pd

from framework.core.models import RunSpec, SweepAxisSpec, SweepSpec
from framework.services.custom_dataset_registry import LocalCustomDatasetRegistry
from framework.services.new_run import (
    NewRunRequest,
    build_run_spec_kwargs_from_request,
    get_default_new_run_request,
    normalize_new_run_request,
    resolve_new_run_seeds,
)
from framework.services.run_repository import RunRepository
from framework.services.sweep_repository import SweepRepository
from framework.services.local_runs import sweep_metric_options
from framework.services.experiment_protocols import ExperimentProtocol, normalize_experiment_protocol, protocol_metadata
from framework.core.runtime_info import build_runtime_snapshot, RUNTIME_PROVENANCE_SCHEMA_VERSION
from framework.core.regularization import explicit_reg_types_for_parameterization


@dataclass(frozen=True)
class SweepRequest:
    kind: str
    preset: str
    base_request: NewRunRequest
    fields: tuple[str, ...]
    values: dict[str, list[Any]]
    protocol: ExperimentProtocol | None = None
    coupling: str = 'grid'


@dataclass(frozen=True)
class SweepPlannedRun:
    index: int
    spec: RunSpec
    sweep_values: dict[str, Any]


@dataclass(frozen=True)
class SweepExecutionPlan:
    group_id: str
    kind: str
    preset: str
    fields: tuple[str, ...]
    values: dict[str, list[Any]]
    seeds: list[int]
    sweep_spec: SweepSpec
    metadata: dict[str, Any]
    runs: list[SweepPlannedRun]
    protocol: ExperimentProtocol | None = None


_UI_SWEEP_GENERATED_SEEDS = "Generated seed set"
_UI_SWEEP_MANUAL_SEEDS = "Manual seed list"


def generate_numeric_sweep_values(
    *,
    start: float,
    stop: float,
    num: int,
    spacing: str,
    as_int: bool = False,
) -> list[int] | list[float]:
    if int(num) < 1:
        raise ValueError("num must be >= 1")
    if str(spacing) not in {'linear', 'log'}:
        raise ValueError("spacing must be 'linear' or 'log'")
    if int(num) == 1:
        values = [float(start)]
    elif spacing == 'linear':
        step = (float(stop) - float(start)) / (int(num) - 1)
        values = [float(start) + idx * step for idx in range(int(num))]
    else:
        if float(start) <= 0.0 or float(stop) <= 0.0:
            raise ValueError("log spacing requires positive start and stop")
        log_start = log10(float(start))
        log_stop = log10(float(stop))
        step = (log_stop - log_start) / (int(num) - 1)
        values = [float(10 ** (log_start + idx * step)) for idx in range(int(num))]
    if not as_int:
        return values
    deduped: list[int] = []
    for value in [int(round(item)) for item in values]:
        if not deduped or deduped[-1] != value:
            deduped.append(value)
    if not deduped:
        raise ValueError("generated integer sweep values are empty")
    return deduped


def parse_sweep_values_text(raw: str, *, value_type: str) -> list[Any]:
    tokens = [part.strip() for part in str(raw).replace('\n', ',').split(',') if part.strip()]
    if not tokens:
        raise ValueError("value list must be non-empty")
    if value_type == 'int':
        return [int(token) for token in tokens]
    if value_type == 'float':
        return [float(token) for token in tokens]
    return tokens


def get_default_sweep_base_request(problem_family: str, data_source: str) -> NewRunRequest:
    default_request = get_default_new_run_request(problem_family, data_source)
    return replace(
        default_request,
        seed_mode='generated',
        seed=0,
        start_seed=0,
        num_seeds=1,
        seed_step=1,
        seeds_text='',
        run_name=None,
        notes='',
    )


def resolve_sweep_seeds(request: NewRunRequest) -> list[int]:
    return resolve_new_run_seeds(request)



def build_sweep_base_request(
    *,
    problem_family: str,
    data_source: str,
    seed_input_mode: str,
    start_seed: int,
    num_seeds: int,
    seed_step: int,
    seeds_text: str,
    n: int,
    n_cols: int | None,
    r_star: int,
    m: int,
    factor_rank: int,
    init_scale: float,
    lr: float,
    steps: int,
    log_every: int,
    lr_schedule_type: str,
    lr_decay_gamma: float,
    lr_step_size: int,
    lr_min: float,
    noise_std: float,
    val_fraction: float,
    test_fraction: float,
    early_stopping_enabled: bool,
    early_stopping_metric: str,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    psd_regime: str,
    noncommutativity_level: float,
    coherence_level: float,
    sampling_regime: str,
    mc_spectrum_type: str,
    mc_spectrum_decay: float,
    mc_factor_balance: str,
    mc_factor_imbalance_level: float,
    optimizer: str,
    execution_mode: str,
    device_preference: str,
    dtype_preference: str,
    batch_size: int | None,
    parameterization: str,
    explicit_reg_type: str,
    explicit_reg_strength: float,
    stopping_mode: str,
    grad_norm_tol: float,
    objective_tol: float,
    divergence_norm_threshold: float,
    custom_dataset_path: str = "",
    custom_dataset_id: str = "",
    dataset_registry: LocalCustomDatasetRegistry | None = None,
) -> NewRunRequest:
    seed_mode = 'manual' if seed_input_mode == _UI_SWEEP_MANUAL_SEEDS else 'generated'
    request = replace(
        get_default_sweep_base_request(problem_family, data_source),
        seed_mode=seed_mode,
        seed=0,
        start_seed=int(start_seed),
        num_seeds=int(num_seeds),
        seed_step=int(seed_step),
        seeds_text=str(seeds_text),
        n=int(n),
        n_cols=None if n_cols is None else int(n_cols),
        r_star=int(r_star),
        m=int(m),
        factor_rank=int(factor_rank),
        init_scale=float(init_scale),
        lr=float(lr),
        steps=int(steps),
        log_every=int(log_every),
        lr_schedule_type=str(lr_schedule_type),
        lr_decay_gamma=float(lr_decay_gamma),
        lr_step_size=int(lr_step_size),
        lr_min=float(lr_min),
        noise_std=float(noise_std),
        problem_family=str(problem_family),
        data_source=str(data_source),
        custom_dataset_path=str(custom_dataset_path),
        custom_dataset_id=str(custom_dataset_id),
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
        optimizer=str(optimizer),
        execution_mode=str(execution_mode),
        device_preference=str(device_preference),
        dtype_preference=str(dtype_preference),
        batch_size=None if batch_size is None else int(batch_size),
        parameterization=str(parameterization),
        explicit_reg_type=str(explicit_reg_type),
        explicit_reg_strength=float(explicit_reg_strength),
        stopping_mode=str(stopping_mode),
        grad_norm_tol=float(grad_norm_tol),
        objective_tol=float(objective_tol),
        divergence_norm_threshold=float(divergence_norm_threshold),
        run_name=None,
        notes='',
    )
    try:
        return normalize_new_run_request(request, dataset_registry=dataset_registry)
    except Exception:
        return request


@dataclass(frozen=True)
class SavedSweepView:
    group_id: str
    metadata: dict[str, Any]
    raw_df: pd.DataFrame
    metric_options: dict[str, str]

    @property
    def sweep_spec(self) -> SweepSpec:
        payload = self.metadata.get("sweep_spec")
        if isinstance(payload, dict):
            return SweepSpec.from_dict(payload)
        legacy_payload = dict(self.metadata)
        legacy_payload.setdefault("group_id", self.group_id)
        return SweepSpec.from_dict(legacy_payload)

    @property
    def kind(self) -> str | None:
        return self.sweep_spec.kind

    @property
    def fields(self) -> tuple[str, ...]:
        return self.sweep_spec.fields


@dataclass(frozen=True)
class SweepFailure:
    index: int
    seed: int
    stage: str
    error_type: str
    error_message: str
    traceback_text: str


@dataclass(frozen=True)
class SweepExecutionResult:
    group_id: str
    metadata: dict[str, Any]
    raw_df: pd.DataFrame
    saved_run_ids: list[str]
    failures: list[SweepFailure]

    @property
    def succeeded_count(self) -> int:
        return int(len(self.raw_df.index))

    @property
    def failed_count(self) -> int:
        return len(self.failures)


def _build_group_id(kind: str, fields: tuple[str, ...]) -> str:
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if kind == '1D':
        return f"sweep1d_{fields[0]}_{stamp}"
    return f"sweep2d_{fields[0]}_{fields[1]}_{stamp}"


def _measurement_type_from_sampling(sampling_regime: str) -> str:
    return 'uniform_entries' if sampling_regime == 'uniform' else 'leverage_biased_entries'


def _coerce_sweep_spec_kwargs(spec_kwargs: dict[str, Any], *, swept_fields: tuple[str, ...] = ()) -> dict[str, Any]:
    kwargs = dict(spec_kwargs)
    swept = {str(field) for field in swept_fields}
    problem_family = str(kwargs.get('problem_family'))
    data_source = str(kwargs.get('data_source'))
    parameterization = str(kwargs.get('parameterization'))
    optimizer = str(kwargs.get('optimizer'))

    allowed_reg_types = explicit_reg_types_for_parameterization(parameterization)
    if str(kwargs.get('explicit_reg_type')) not in allowed_reg_types and 'explicit_reg_type' not in swept and 'explicit_reg_strength' not in swept:
        kwargs['explicit_reg_type'] = 'none'
        kwargs['explicit_reg_strength'] = 0.0

    if parameterization == 'direct_matrix':
        if 'mc_factor_balance' not in swept:
            kwargs['mc_factor_balance'] = 'balanced'
        if 'mc_factor_imbalance_level' not in swept:
            kwargs['mc_factor_imbalance_level'] = 0.0

    if problem_family == 'matrix_completion':
        if str(kwargs.get('mc_factor_balance')) == 'balanced' and 'mc_factor_imbalance_level' not in swept:
            kwargs['mc_factor_imbalance_level'] = 0.0
        if str(kwargs.get('mc_spectrum_type')) == 'flat' and 'mc_spectrum_decay' not in swept:
            kwargs['mc_spectrum_decay'] = 0.0
        if data_source == 'synthetic':
            current_sampling = str(kwargs.get('sampling_regime', 'uniform'))
            kwargs['measurement_type'] = _measurement_type_from_sampling(current_sampling)

    if optimizer in {'sgd', 'momentum'} and kwargs.get('batch_size') is None and 'batch_size' not in swept:
        train_est = int(float(kwargs['m']) * (1.0 - float(kwargs['val_fraction']) - float(kwargs['test_fraction'])))
        kwargs['batch_size'] = min(64, max(1, train_est))

    if optimizer == 'preconditioned_gd' and parameterization not in {'factorized_psd', 'factorized_uv'} and 'parameterization' not in swept:
        kwargs['parameterization'] = 'factorized_psd' if problem_family == 'psd_sensing' else 'factorized_uv'

    return kwargs


def build_sweep_spec(request: SweepRequest, *, dataset_registry: LocalCustomDatasetRegistry | None = None) -> SweepSpec:
    seeds = resolve_new_run_seeds(request.base_request)
    group_id = _build_group_id(request.kind, request.fields)
    created_at = datetime.now().isoformat(timespec='seconds')
    base_spec_kwargs = build_run_spec_kwargs_from_request(request.base_request, dataset_registry=dataset_registry)
    axes = tuple(
        SweepAxisSpec(field=str(field), values=tuple(request.values[field]))
        for field in request.fields
    )
    fixed_factors = {key: value for key, value in base_spec_kwargs.items() if key not in request.fields}
    spec = SweepSpec(
        group_id=group_id,
        kind=request.kind,
        preset=request.preset,
        axes=axes,
        fixed_factors=fixed_factors,
        seeds=tuple(int(seed) for seed in seeds),
        coupling=str(request.coupling or 'grid'),
        created_at=created_at,
    )
    spec.validate()
    return spec




def _iter_sweep_combinations(request: SweepRequest):
    field_names = tuple(str(field) for field in request.fields)
    field_values = [list(request.values[field]) for field in field_names]
    if str(request.kind).upper() == '2D' and str(request.coupling or 'grid') == 'paired':
        yield from zip(*field_values, strict=True)
    else:
        yield from product(*field_values)


def _semantic_sweep_issues(raw_kwargs: dict[str, Any], swept_fields: tuple[str, ...]) -> list[str]:
    fields = {str(field) for field in swept_fields}
    issues: list[str] = []
    problem_family = str(raw_kwargs.get('problem_family'))
    data_source = str(raw_kwargs.get('data_source'))

    if 'explicit_reg_strength' in fields and str(raw_kwargs.get('explicit_reg_type')) == 'none' and float(raw_kwargs.get('explicit_reg_strength', 0.0)) != 0.0:
        issues.append("explicit_reg_strength can only be swept when explicit_reg_type is not 'none'.")

    if problem_family == 'psd_sensing':
        if 'psd_regime' in fields and str(raw_kwargs.get('psd_regime')) == 'commuting' and float(raw_kwargs.get('noncommutativity_level', 0.0)) != 0.0:
            issues.append("psd_regime='commuting' requires noncommutativity_level=0.0; use a paired 2D sweep if you need both.")
        if 'noncommutativity_level' in fields and str(raw_kwargs.get('psd_regime')) == 'commuting' and float(raw_kwargs.get('noncommutativity_level', 0.0)) != 0.0:
            issues.append("noncommutativity_level cannot be nonzero while psd_regime='commuting'.")

    if problem_family == 'matrix_completion':
        if data_source == 'synthetic':
            if 'mc_spectrum_decay' in fields and str(raw_kwargs.get('mc_spectrum_type')) == 'flat' and float(raw_kwargs.get('mc_spectrum_decay', 0.0)) != 0.0:
                issues.append("mc_spectrum_decay is only meaningful when mc_spectrum_type is not 'flat'.")
            if 'mc_spectrum_type' in fields and 'mc_spectrum_decay' in fields and str(raw_kwargs.get('mc_spectrum_type')) == 'flat' and float(raw_kwargs.get('mc_spectrum_decay', 0.0)) != 0.0:
                issues.append("2D spectrum sweeps cannot pair mc_spectrum_type='flat' with nonzero mc_spectrum_decay.")
        if 'mc_factor_imbalance_level' in fields and str(raw_kwargs.get('mc_factor_balance')) == 'balanced' and float(raw_kwargs.get('mc_factor_imbalance_level', 0.0)) != 0.0:
            issues.append("mc_factor_imbalance_level requires mc_factor_balance='unbalanced'.")
        if 'mc_factor_balance' in fields and 'mc_factor_imbalance_level' in fields and str(raw_kwargs.get('mc_factor_balance')) == 'balanced' and float(raw_kwargs.get('mc_factor_imbalance_level', 0.0)) != 0.0:
            issues.append("2D factor-balance sweeps cannot pair balanced with nonzero mc_factor_imbalance_level.")

    return issues


def validate_sweep_request(request: SweepRequest, *, dataset_registry: LocalCustomDatasetRegistry | None = None, max_issues: int = 8) -> list[str]:
    try:
        sweep_spec = build_sweep_spec(request, dataset_registry=dataset_registry)
    except Exception as exc:
        return [str(exc)]
    field_names = tuple(str(field) for field in request.fields)
    try:
        base_spec_kwargs = build_run_spec_kwargs_from_request(request.base_request, dataset_registry=dataset_registry)
    except Exception as exc:
        return [str(exc)]
    representative_seed = int(sweep_spec.seeds[0]) if sweep_spec.seeds else 0
    issues: list[str] = []
    seen: set[str] = set()
    for combo in _iter_sweep_combinations(request):
        combo_values = dict(zip(field_names, combo, strict=True))
        raw_kwargs = dict(base_spec_kwargs)
        raw_kwargs.update(combo_values)
        raw_kwargs.setdefault('seed', representative_seed)
        combo_label = ", ".join(f"{key}={value!r}" for key, value in combo_values.items())
        for issue in _semantic_sweep_issues(raw_kwargs, field_names):
            message = f"Invalid sweep configuration ({combo_label}): {issue}"
            if message not in seen:
                issues.append(message)
                seen.add(message)
        spec_kwargs = _coerce_sweep_spec_kwargs(raw_kwargs, swept_fields=field_names)
        spec_kwargs.setdefault('seed', representative_seed)
        try:
            spec = RunSpec(**spec_kwargs)
            spec.validate()
        except Exception as exc:
            message = f"Invalid sweep configuration ({combo_label}): {exc}"
            if message not in seen:
                issues.append(message)
                seen.add(message)
        if len(issues) >= max_issues:
            issues.append("Additional invalid configurations omitted.")
            break
    return issues

def build_sweep_execution_plan(request: SweepRequest, *, dataset_registry: LocalCustomDatasetRegistry | None = None) -> SweepExecutionPlan:
    validation_issues = validate_sweep_request(request, dataset_registry=dataset_registry)
    if validation_issues:
        raise ValueError("Invalid sweep request: " + "; ".join(validation_issues[:4]))
    sweep_spec = build_sweep_spec(request, dataset_registry=dataset_registry)
    seeds = list(sweep_spec.seeds)
    base_spec_kwargs = build_run_spec_kwargs_from_request(request.base_request, dataset_registry=dataset_registry)
    group_id = sweep_spec.group_id
    created_at = sweep_spec.created_at or datetime.now().isoformat(timespec='seconds')
    expected_cell_count = int(len(sweep_spec.axes[0].values)) if sweep_spec.kind == '2D' and sweep_spec.coupling == 'paired' else int(prod(len(axis.values) for axis in sweep_spec.axes))
    expected_run_count = int(expected_cell_count * len(seeds))

    runs: list[SweepPlannedRun] = []
    index = 0
    combo_idx = 0
    field_names = tuple(request.fields)
    field_values = [list(request.values[field]) for field in field_names]

    combo_iterable = zip(*field_values, strict=True) if sweep_spec.kind == '2D' and sweep_spec.coupling == 'paired' else product(*field_values)

    for combo in combo_iterable:
        combo_idx += 1
        combo_values = dict(zip(field_names, combo, strict=True))
        seed_group_id = f"{group_id}_seedgrp_{combo_idx:04d}"
        for seed in seeds:
            index += 1
            spec_kwargs = dict(base_spec_kwargs)
            spec_kwargs.update(combo_values)
            spec_kwargs['seed'] = int(seed)
            spec_kwargs['group_id'] = group_id
            spec_kwargs['seed_group_id'] = seed_group_id
            spec_kwargs['run_name'] = None
            spec_kwargs['notes'] = request.preset
            spec_kwargs = _coerce_sweep_spec_kwargs(spec_kwargs, swept_fields=field_names)
            spec = RunSpec(**spec_kwargs)
            spec.validate()
            runs.append(SweepPlannedRun(index=index, spec=spec, sweep_values=combo_values))

    protocol = normalize_experiment_protocol(request.protocol)
    runtime_snapshot = build_runtime_snapshot()
    metadata = {
        'group_id': group_id,
        'provenance_schema_version': int(RUNTIME_PROVENANCE_SCHEMA_VERSION),
        'runtime_snapshot': runtime_snapshot,
        'python_version': runtime_snapshot.get('python_version'),
        'numpy_version': runtime_snapshot.get('numpy_version'),
        'pandas_version': runtime_snapshot.get('pandas_version'),
        'torch_version': runtime_snapshot.get('torch_version'),
        'kind': sweep_spec.kind,
        'fields': list(field_names),
        'values': sweep_spec.values,
        'preset': sweep_spec.preset,
        'coupling': sweep_spec.coupling,
        'created_at': created_at,
        'base_config': base_spec_kwargs,
        'fixed_factors': dict(sweep_spec.fixed_factors),
        'fixed_factor_keys': sorted(str(key) for key in sweep_spec.fixed_factors.keys()),
        'varied_factors': list(field_names),
        'seeds': list(seeds),
        'expected_seed_count': int(len(seeds)),
        'expected_cell_count': expected_cell_count,
        'expected_run_count': expected_run_count,
        'sweep_spec': sweep_spec.to_dict(),
        **protocol_metadata(protocol),
    }

    return SweepExecutionPlan(
        group_id=group_id,
        kind=sweep_spec.kind,
        preset=sweep_spec.preset,
        fields=field_names,
        values=sweep_spec.values,
        seeds=list(seeds),
        sweep_spec=sweep_spec,
        metadata=metadata,
        runs=runs,
        protocol=protocol,
    )


def execute_sweep_plan(
    plan: SweepExecutionPlan,
    *,
    run_experiment_fn: Callable[[RunSpec], Any] | None = None,
    run_repository: RunRepository | None = None,
    on_status: Callable[[str], None] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> SweepExecutionResult:
    if run_experiment_fn is None:
        raise ValueError("run_experiment_fn is required for direct sweep execution")
    if run_repository is None:
        raise ValueError("run_repository is required for direct sweep execution")

    rows: list[dict[str, Any]] = []
    saved_run_ids: list[str] = []
    failures: list[SweepFailure] = []
    metadata_patch = protocol_metadata(plan.protocol)
    total = len(plan.runs)
    for position, planned in enumerate(plan.runs, start=1):
        spec = planned.spec
        if on_status is not None:
            on_status(f"Running sweep cell {position}/{total} · seed={spec.seed}")
        try:
            result = run_experiment_fn(spec)
            if hasattr(run_repository, "save_result_with_metadata"):
                run_id = run_repository.save_result_with_metadata(result, metadata_patch)
            else:
                run_id = run_repository.save_result(result)
            saved_run_ids.append(run_id)
            summary = dict(result.summary)
            rows.append({
                'run': run_id,
                'seed': spec.seed,
                'seed_group_id': spec.seed_group_id,
                'status': summary.get('status'),
                'source_kind': summary.get('source_kind'),
                **planned.sweep_values,
                'final_rel_recon_error': summary.get('final_rel_recon_error'),
                'final_objective': summary.get('final_objective'),
                'final_trace_X': summary.get('final_trace_X'),
                'final_subspace_alignment': summary.get('final_subspace_alignment'),
                'final_subspace_projection_distance': summary.get('final_subspace_projection_distance'),
                'final_mean_subspace_alignment': summary.get('final_mean_subspace_alignment'),
                'final_mean_target_commutator': summary.get('final_mean_target_commutator'),
                'final_max_target_commutator': summary.get('final_max_target_commutator'),
                'final_frobenius_norm_X': summary.get('final_frobenius_norm_X'),
                'final_val_rmse': summary.get('final_val_rmse'),
                'final_test_rmse': summary.get('final_test_rmse'),
            })
        except Exception as exc:
            failures.append(SweepFailure(
                index=int(planned.index),
                seed=int(spec.seed),
                stage="run_experiment",
                error_type=type(exc).__name__,
                error_message=str(exc),
                traceback_text=traceback.format_exc(),
            ))
        if on_progress is not None:
            on_progress(position, total)

    raw_df = pd.DataFrame(rows)
    sort_cols = [field for field in plan.fields if field in raw_df.columns]
    sort_cols.extend([col for col in ['seed', 'run'] if col in raw_df.columns])
    if sort_cols and not raw_df.empty:
        raw_df = raw_df.sort_values(sort_cols).reset_index(drop=True)

    metadata = dict(plan.metadata)
    return SweepExecutionResult(
        group_id=plan.group_id,
        metadata=metadata,
        raw_df=raw_df,
        saved_run_ids=saved_run_ids,
        failures=failures,
    )


def save_sweep_metadata(repository: SweepRepository, metadata: dict[str, Any]) -> None:
    repository.save_metadata(metadata)


def load_saved_sweep_view(group_id: str, repository: SweepRepository) -> SavedSweepView | None:
    metadata = repository.load_metadata(group_id)
    if metadata is None:
        return None

    rows: list[dict[str, Any]] = []
    fields = tuple(str(field) for field in metadata.get("fields", []))
    loaded_runs = repository.load_group_runs(group_id)
    for item in loaded_runs:
        cfg = item.get("config", {})
        summary = item.get("summary", {})
        rows.append({
            "run": cfg.get("run_id") or item.get("run_id") or cfg.get("id") or item.get("id"),
            "seed": cfg.get("seed"),
            "seed_group_id": cfg.get("seed_group_id"),
            "status": summary.get("status"),
            "source_kind": summary.get("source_kind"),
            "final_rel_recon_error": summary.get("final_rel_recon_error"),
            "final_objective": summary.get("final_objective"),
            "final_trace_X": summary.get("final_trace_X"),
            "final_subspace_alignment": summary.get("final_subspace_alignment"),
            "final_subspace_projection_distance": summary.get("final_subspace_projection_distance"),
            "final_mean_subspace_alignment": summary.get("final_mean_subspace_alignment"),
            "final_mean_target_commutator": summary.get("final_mean_target_commutator"),
            "final_max_target_commutator": summary.get("final_max_target_commutator"),
            "final_frobenius_norm_X": summary.get("final_frobenius_norm_X"),
            "final_val_rmse": summary.get("final_val_rmse"),
            "final_test_rmse": summary.get("final_test_rmse"),
            **{field: cfg.get(field) for field in fields},
        })

    raw_df = pd.DataFrame(rows)
    if not raw_df.empty:
        sort_cols = [field for field in fields if field in raw_df.columns]
        sort_cols.extend([col for col in ["seed", "run"] if col in raw_df.columns])
        if sort_cols:
            raw_df = raw_df.sort_values(sort_cols).reset_index(drop=True)

    sweep_spec = SweepSpec.from_dict(metadata.get("sweep_spec") or metadata)
    base_cfg = metadata.get("base_config") or dict(sweep_spec.fixed_factors)
    metric_options = sweep_metric_options(base_cfg.get("problem_family"), base_cfg.get("data_source"))
    if not raw_df.empty:
        metric_options = {key: label for key, label in metric_options.items() if key in raw_df.columns}

    return SavedSweepView(
        group_id=group_id,
        metadata=metadata,
        raw_df=raw_df,
        metric_options=metric_options,
    )


__all__ = [
    "SweepRequest",
    "SweepPlannedRun",
    "SweepExecutionPlan",
    "SavedSweepView",
    "SweepExecutionResult",
    "get_default_sweep_base_request",
    "resolve_sweep_seeds",
    "build_sweep_base_request",
    "build_sweep_spec",
    "validate_sweep_request",
    "build_sweep_execution_plan",
    "execute_sweep_plan",
    "save_sweep_metadata",
    "load_saved_sweep_view",
    "generate_numeric_sweep_values",
    "parse_sweep_values_text",
]
