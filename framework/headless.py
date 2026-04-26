from __future__ import annotations

import argparse
import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Sequence

from framework.core.experiments import run_experiment
from framework.services.custom_dataset_registry import LocalCustomDatasetRegistry
from framework.services.experiment_protocols import normalize_experiment_protocol
from framework.services.new_run import (
    NewRunExecutionResult,
    NewRunRequest,
    execute_new_run_plan,
    validate_new_run_request,
)
from framework.services.planner import (
    NewRunPlan,
    PlannedRequest,
    SweepPlan,
    new_run_plan_from_dict,
    plan_new_run,
    plan_sweep,
    sweep_plan_from_dict,
)
from framework.services.run_repository import LocalReadRunRepository, LocalRunRepository
from framework.services.sweep_repository import LocalSweepRepository
from framework.services.sweeps import (
    SweepExecutionResult,
    SweepRequest,
    build_sweep_execution_plan,
    execute_sweep_plan,
    save_sweep_metadata,
)
from framework.storage.local import load_run, read_run_header, save_run


_NEW_RUN_FIELD_NAMES = {field.name for field in fields(NewRunRequest)}


class HeadlessRequestError(ValueError):
    pass


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _maybe_protocol(payload: Any) -> Any:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise HeadlessRequestError("protocol must be an object when provided.")
    return normalize_experiment_protocol(payload)


def _protocol_id_for_request(protocol: Any) -> str | None:
    normalized = normalize_experiment_protocol(protocol)
    return normalized.protocol_id if normalized is not None else None


def _run_paths(runs_dir: str | Path, run_ids: list[str]) -> list[str]:
    base = Path(runs_dir)
    return [str(base / run_id) for run_id in run_ids]


def new_run_request_from_dict(data: dict[str, Any]) -> NewRunRequest:
    filtered = {key: value for key, value in data.items() if key in _NEW_RUN_FIELD_NAMES}
    if "protocol" in data:
        filtered["protocol"] = _maybe_protocol(data.get("protocol"))
    return NewRunRequest(**filtered)


def sweep_request_from_dict(data: dict[str, Any]) -> SweepRequest:
    base_payload = data.get("base_request")
    if not isinstance(base_payload, dict):
        raise HeadlessRequestError("Sweep request requires a 'base_request' object.")
    fields_payload = data.get("fields")
    if not isinstance(fields_payload, (list, tuple)) or not fields_payload:
        raise HeadlessRequestError("Sweep request requires a non-empty 'fields' list.")
    values_payload = data.get("values")
    if not isinstance(values_payload, dict):
        raise HeadlessRequestError("Sweep request requires a 'values' mapping.")
    return SweepRequest(
        kind=str(data.get("kind", "1D")),
        preset=str(data.get("preset", "custom")),
        base_request=new_run_request_from_dict(base_payload),
        fields=tuple(str(field) for field in fields_payload),
        values={str(key): list(value) for key, value in values_payload.items()},
        protocol=_maybe_protocol(data.get("protocol")),
        coupling=str(data.get("coupling", "grid")),
    )


def execute_headless_new_run(
    request: NewRunRequest,
    *,
    runs_dir: str | Path = "runs",
    run_experiment_fn=run_experiment,
    datasets_registry_dir: str | Path = "datasets_registry",
) -> dict[str, Any]:
    dataset_registry = LocalCustomDatasetRegistry(Path(datasets_registry_dir))
    validation = validate_new_run_request(request, dataset_registry=dataset_registry)
    if not validation.is_valid or validation.plan is None:
        raise HeadlessRequestError("; ".join(validation.errors) or "Invalid new-run request.")

    repository = LocalRunRepository(runs_dir=Path(runs_dir), save_run_fn=save_run)
    result: NewRunExecutionResult = execute_new_run_plan(
        validation.plan,
        run_experiment_fn=run_experiment_fn,
        run_repository=repository,
    )
    return {
        "request_type": "new_run",
        "execution_mode": "direct",
        "seed_group_id": result.seed_group_id,
        "seeds": list(result.seeds),
        "saved_run_ids": list(result.saved_run_ids),
        "run_paths": _run_paths(runs_dir, list(result.saved_run_ids)),
        "succeeded_count": int(result.succeeded_count),
        "failed_count": int(result.failed_count),
        "is_complete_success": bool(result.is_complete_success),
        "is_complete_failure": bool(result.is_complete_failure),
        "protocol_id": _protocol_id_for_request(request.protocol),
        "failures": [asdict(failure) for failure in result.failures],
    }


def execute_headless_sweep(
    request: SweepRequest,
    *,
    runs_dir: str | Path = "runs",
    sweeps_dir: str | Path = "sweeps",
    run_experiment_fn=run_experiment,
    datasets_registry_dir: str | Path = "datasets_registry",
) -> dict[str, Any]:
    run_repository = LocalRunRepository(runs_dir=Path(runs_dir), save_run_fn=save_run)
    read_repository = LocalReadRunRepository(runs_dir=Path(runs_dir), load_run_fn=load_run, read_run_header_fn=read_run_header)
    sweep_repository = LocalSweepRepository(sweeps_dir=Path(sweeps_dir), read_repository=read_repository)
    dataset_registry = LocalCustomDatasetRegistry(Path(datasets_registry_dir))
    plan = build_sweep_execution_plan(request, dataset_registry=dataset_registry)
    result: SweepExecutionResult = execute_sweep_plan(
        plan,
        run_experiment_fn=run_experiment_fn,
        run_repository=run_repository,
    )
    save_sweep_metadata(sweep_repository, result.metadata)
    return {
        "request_type": "sweep",
        "execution_mode": "direct",
        "group_id": result.group_id,
        "kind": str(result.metadata.get("kind")),
        "fields": list(result.metadata.get("fields", [])),
        "coupling": str(result.metadata.get("coupling", plan.sweep_spec.coupling)),
        "saved_run_ids": list(result.saved_run_ids),
        "run_paths": _run_paths(runs_dir, list(result.saved_run_ids)),
        "sweep_path": str(Path(sweeps_dir) / f"{result.group_id}.json"),
        "succeeded_count": int(result.succeeded_count),
        "failed_count": int(result.failed_count),
        "protocol_id": _protocol_id_for_request(request.protocol),
        "failures": [asdict(failure) for failure in result.failures],
    }


def plan_request_file(path: str | Path, *, request_type: str) -> PlannedRequest:
    data = _load_json(path)
    if request_type == "new-run-plan":
        return plan_new_run(new_run_plan_from_dict(data))
    if request_type == "sweep-plan":
        return plan_sweep(sweep_plan_from_dict(data))
    raise HeadlessRequestError(f"Unsupported plan request_type={request_type!r}")


def load_request_file(path: str | Path, *, request_type: str) -> NewRunRequest | SweepRequest | NewRunPlan | SweepPlan:
    data = _load_json(path)
    if request_type == "new-run":
        return new_run_request_from_dict(data)
    if request_type == "sweep":
        return sweep_request_from_dict(data)
    if request_type == "new-run-plan":
        return new_run_plan_from_dict(data)
    if request_type == "sweep-plan":
        return sweep_plan_from_dict(data)
    raise HeadlessRequestError(f"Unsupported request_type={request_type!r}")


def run_request_file(
    path: str | Path,
    *,
    request_type: str,
    runs_dir: str | Path = "runs",
    sweeps_dir: str | Path = "sweeps",
    run_experiment_fn=run_experiment,
    datasets_registry_dir: str | Path = "datasets_registry",
) -> dict[str, Any]:
    request = load_request_file(path, request_type=request_type)
    if isinstance(request, NewRunPlan):
        request = plan_new_run(request).request
    elif isinstance(request, SweepPlan):
        request = plan_sweep(request).request
    if isinstance(request, NewRunRequest):
        return execute_headless_new_run(request, runs_dir=runs_dir, run_experiment_fn=run_experiment_fn, datasets_registry_dir=datasets_registry_dir)
    return execute_headless_sweep(request, runs_dir=runs_dir, sweeps_dir=sweeps_dir, run_experiment_fn=run_experiment_fn, datasets_registry_dir=datasets_registry_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run framework requests headlessly through direct execution.")
    parser.add_argument("request_type", choices=["new-run", "sweep", "new-run-plan", "sweep-plan"])
    parser.add_argument("request_json", help="Path to a JSON request file.")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--sweeps-dir", default="sweeps")
    parser.add_argument("--datasets-registry-dir", default="datasets_registry")
    parser.add_argument("--output", default=None, help="Optional path to write the result summary JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    summary = run_request_file(
        args.request_json,
        request_type=args.request_type,
        runs_dir=args.runs_dir,
        sweeps_dir=args.sweeps_dir,
        datasets_registry_dir=args.datasets_registry_dir,
    )
    payload = json.dumps(summary, indent=2)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
