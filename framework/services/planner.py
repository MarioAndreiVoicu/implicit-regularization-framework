from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import log10
from typing import Any

from framework.services.experiment_critic import audit_new_run_request, audit_sweep_request
from framework.services.new_run import (
    NewRunRequest,
    get_default_new_run_request,
    normalize_new_run_request,
    resolve_new_run_seeds,
)
from framework.services.sweeps import SweepRequest


class PlanError(ValueError):
    pass


@dataclass(frozen=True)
class SeedPlan:
    mode: str = "single"
    seed: int = 0
    start_seed: int = 0
    num_seeds: int = 1
    seed_step: int = 1
    seeds_text: str = ""


@dataclass(frozen=True)
class NewRunPlan:
    problem_family: str
    data_source: str
    overrides: dict[str, Any] = field(default_factory=dict)
    seed_plan: SeedPlan = field(default_factory=SeedPlan)


@dataclass(frozen=True)
class SweepAxisPlan:
    field: str
    mode: str = "explicit"
    values: list[Any] = field(default_factory=list)
    start: float | None = None
    stop: float | None = None
    num: int | None = None


@dataclass(frozen=True)
class SweepPlan:
    kind: str
    preset: str
    problem_family: str
    data_source: str
    axes: tuple[SweepAxisPlan, ...]
    base_overrides: dict[str, Any] = field(default_factory=dict)
    seed_plan: SeedPlan = field(default_factory=SeedPlan)


@dataclass(frozen=True)
class PlannedRequest:
    request: NewRunRequest | SweepRequest
    run_count: int
    warnings: list[str]


def _request_with_seed_plan(request: NewRunRequest, seed_plan: SeedPlan) -> NewRunRequest:
    return replace(
        request,
        seed_mode=str(seed_plan.mode),
        seed=int(seed_plan.seed),
        start_seed=int(seed_plan.start_seed),
        num_seeds=int(seed_plan.num_seeds),
        seed_step=int(seed_plan.seed_step),
        seeds_text=str(seed_plan.seeds_text),
    )


def build_new_run_request_from_plan(plan: NewRunPlan) -> NewRunRequest:
    request = get_default_new_run_request(plan.problem_family, plan.data_source)
    request = _request_with_seed_plan(request, plan.seed_plan)
    if plan.overrides:
        request = replace(request, **dict(plan.overrides))
    return normalize_new_run_request(request)


def _linear_values(start: float, stop: float, num: int) -> list[float]:
    if num < 2:
        return [float(start)]
    step = (stop - start) / (num - 1)
    return [float(start + idx * step) for idx in range(num)]


def _log_values(start: float, stop: float, num: int) -> list[float]:
    if start <= 0.0 or stop <= 0.0:
        raise PlanError("log sweep requires start > 0 and stop > 0")
    if num < 2:
        return [float(start)]
    log_start = log10(start)
    log_stop = log10(stop)
    step = (log_stop - log_start) / (num - 1)
    return [float(10 ** (log_start + idx * step)) for idx in range(num)]


def resolve_axis_values(axis: SweepAxisPlan) -> list[Any]:
    mode = str(axis.mode)
    if mode == "explicit":
        if not axis.values:
            raise PlanError(f"Axis {axis.field!r} requires non-empty values.")
        return list(axis.values)
    if axis.start is None or axis.stop is None or axis.num is None:
        raise PlanError(f"Axis {axis.field!r} requires start, stop, and num for mode={mode!r}.")
    if int(axis.num) < 1:
        raise PlanError(f"Axis {axis.field!r} requires num >= 1.")
    if mode == "linear":
        return _linear_values(float(axis.start), float(axis.stop), int(axis.num))
    if mode == "log":
        return _log_values(float(axis.start), float(axis.stop), int(axis.num))
    raise PlanError(f"Unsupported axis mode={mode!r}.")


def build_sweep_request_from_plan(plan: SweepPlan) -> SweepRequest:
    if not plan.axes:
        raise PlanError("Sweep plan requires at least one axis.")
    request = build_new_run_request_from_plan(
        NewRunPlan(
            problem_family=plan.problem_family,
            data_source=plan.data_source,
            overrides=dict(plan.base_overrides),
            seed_plan=plan.seed_plan,
        )
    )
    values = {axis.field: resolve_axis_values(axis) for axis in plan.axes}
    return SweepRequest(
        kind=str(plan.kind),
        preset=str(plan.preset),
        base_request=request,
        fields=tuple(axis.field for axis in plan.axes),
        values=values,
    )


def _warnings_for_run_count(run_count: int) -> list[str]:
    warnings: list[str] = []
    if run_count >= 100:
        warnings.append(f"Plan expands to {run_count} runs.")
    return warnings


def plan_new_run(plan: NewRunPlan) -> PlannedRequest:
    request = build_new_run_request_from_plan(plan)
    seeds = resolve_new_run_seeds(request)
    warnings = _warnings_for_run_count(len(seeds))
    warnings.extend(audit_new_run_request(request).messages)
    return PlannedRequest(request=request, run_count=len(seeds), warnings=warnings)


def plan_sweep(plan: SweepPlan) -> PlannedRequest:
    request = build_sweep_request_from_plan(plan)
    seed_count = len(resolve_new_run_seeds(request.base_request))
    axis_count = 1
    for field in request.fields:
        axis_count *= len(request.values[field])
    run_count = seed_count * axis_count
    warnings = _warnings_for_run_count(run_count)
    if len(request.fields) != len(set(request.fields)):
        warnings.append("Sweep plan contains duplicate axis fields.")
    warnings.extend(audit_sweep_request(request).messages)
    return PlannedRequest(request=request, run_count=run_count, warnings=warnings)


def seed_plan_from_dict(data: dict[str, Any] | None) -> SeedPlan:
    payload = dict(data or {})
    return SeedPlan(
        mode=str(payload.get("mode", "single")),
        seed=int(payload.get("seed", 0)),
        start_seed=int(payload.get("start_seed", 0)),
        num_seeds=int(payload.get("num_seeds", 1)),
        seed_step=int(payload.get("seed_step", 1)),
        seeds_text=str(payload.get("seeds_text", "")),
    )


def new_run_plan_from_dict(data: dict[str, Any]) -> NewRunPlan:
    return NewRunPlan(
        problem_family=str(data.get("problem_family", "psd_sensing")),
        data_source=str(data.get("data_source", "synthetic")),
        overrides=dict(data.get("overrides", {})),
        seed_plan=seed_plan_from_dict(data.get("seed_plan")),
    )


def sweep_axis_plan_from_dict(data: dict[str, Any]) -> SweepAxisPlan:
    return SweepAxisPlan(
        field=str(data.get("field")),
        mode=str(data.get("mode", "explicit")),
        values=list(data.get("values", [])),
        start=float(data["start"]) if data.get("start") is not None else None,
        stop=float(data["stop"]) if data.get("stop") is not None else None,
        num=int(data["num"]) if data.get("num") is not None else None,
    )


def sweep_plan_from_dict(data: dict[str, Any]) -> SweepPlan:
    axes_payload = data.get("axes")
    if not isinstance(axes_payload, list) or not axes_payload:
        raise PlanError("Sweep plan requires a non-empty 'axes' list.")
    return SweepPlan(
        kind=str(data.get("kind", "1D")),
        preset=str(data.get("preset", "custom")),
        problem_family=str(data.get("problem_family", "psd_sensing")),
        data_source=str(data.get("data_source", "synthetic")),
        axes=tuple(sweep_axis_plan_from_dict(item) for item in axes_payload),
        base_overrides=dict(data.get("base_overrides", {})),
        seed_plan=seed_plan_from_dict(data.get("seed_plan")),
    )
