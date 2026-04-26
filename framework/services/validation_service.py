from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, nan
from typing import Any, Callable
import traceback

from framework.core.models import RunSpec
from framework.services.run_repository import RunRepository
from framework.services.new_run import build_run_spec_kwargs_from_request, get_default_new_run_request


@dataclass(frozen=True)
class ValidationCheckResult:
    check: str
    passed: bool
    value: float
    threshold: str
    run: str

    def to_row(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "pass": self.passed,
            "value": self.value,
            "threshold": self.threshold,
            "run": self.run,
        }


@dataclass(frozen=True)
class ValidationSavedRun:
    index: int
    seed: int
    run_id: str
    config: dict[str, Any]
    summary: dict[str, Any]
    metrics: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ValidationFailure:
    index: int
    seed: int
    stage: str
    error_type: str
    error_message: str
    traceback_text: str


@dataclass(frozen=True)
class ValidationSuiteResult:
    checks: list[ValidationCheckResult]
    saved_run_ids: list[str]
    items: list[ValidationSavedRun]
    failures: list[ValidationFailure]
    stable_run_id: str | None

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


def build_validation_specs(stable_lr: float) -> list[RunSpec]:
    base_request = get_default_new_run_request("psd_sensing", "synthetic")
    base_kwargs = build_run_spec_kwargs_from_request(base_request)
    specs: list[RunSpec] = []
    for run_name in ["validation_stable", "validation_stable_repeat"]:
        cfg = dict(base_kwargs)
        cfg["seed"] = 0
        cfg["lr"] = float(stable_lr)
        cfg["run_name"] = run_name
        cfg["notes"] = "validation"
        specs.append(RunSpec(**cfg))
    return specs


def _find_item(items_by_name: dict[str, ValidationSavedRun], name: str) -> ValidationSavedRun | None:
    return items_by_name.get(name)


def _final_rel_recon_error(item: ValidationSavedRun | None) -> float:
    if item is None:
        return nan
    try:
        return float(item.summary["final_rel_recon_error"])
    except Exception:
        return nan


def _metric_integrity_ratio(items: list[ValidationSavedRun]) -> float:
    required_summary_fields = ["status", "final_rel_recon_error", "final_objective"]
    finite_summary_fields = ["final_rel_recon_error", "final_objective"]
    total = 0
    passed = 0
    for item in items:
        for key in required_summary_fields:
            total += 1
            if item.summary.get(key) not in (None, ""):
                passed += 1
        for key in finite_summary_fields:
            total += 1
            try:
                value = float(item.summary.get(key))
            except Exception:
                continue
            if isfinite(value):
                passed += 1
        for _, series in item.metrics.items():
            if not isinstance(series, (list, tuple)) or not series:
                continue
            numeric_seen = False
            finite_seen = False
            for raw_value in series:
                try:
                    value = float(raw_value)
                except Exception:
                    continue
                numeric_seen = True
                finite_seen = finite_seen or isfinite(value)
            if numeric_seen:
                total += 1
                if finite_seen:
                    passed += 1
    if total == 0:
        return 0.0
    return passed / total


def summarize_validation_result(
    *,
    items: list[ValidationSavedRun],
    failures: list[ValidationFailure],
    err_good_thresh: float,
    repro_tol: float,
) -> ValidationSuiteResult:
    items_by_name = {str(item.config.get("run_name") or item.run_id): item for item in items}
    stable_name = "validation_stable"
    stable_repeat_name = "validation_stable_repeat"

    stable_item = _find_item(items_by_name, stable_name)
    stable_repeat_item = _find_item(items_by_name, stable_repeat_name)

    err_stable = _final_rel_recon_error(stable_item)
    err_stable_repeat = _final_rel_recon_error(stable_repeat_item)
    repro_diff = abs(err_stable - err_stable_repeat) if isfinite(err_stable) and isfinite(err_stable_repeat) else nan
    metric_integrity_ratio = _metric_integrity_ratio(items)

    checks = [
        ValidationCheckResult(
            check="Stable run reaches low error",
            passed=isfinite(err_stable) and err_stable < err_good_thresh,
            value=err_stable,
            threshold=f"< {err_good_thresh}",
            run=stable_item.run_id if stable_item is not None else stable_name,
        ),
        ValidationCheckResult(
            check="Reproducibility (same config/seed)",
            passed=isfinite(repro_diff) and repro_diff < repro_tol,
            value=repro_diff,
            threshold=f"< {repro_tol}",
            run=(
                f"{stable_item.run_id} vs {stable_repeat_item.run_id}"
                if stable_item is not None and stable_repeat_item is not None
                else f"{stable_name} vs {stable_repeat_name}"
            ),
        ),
        ValidationCheckResult(
            check="Metric integrity",
            passed=isfinite(metric_integrity_ratio) and metric_integrity_ratio == 1.0 and len(items) > 0,
            value=metric_integrity_ratio,
            threshold="all required metrics present and finite",
            run="all validation runs",
        ),
    ]

    return ValidationSuiteResult(
        checks=checks,
        saved_run_ids=[item.run_id for item in items],
        items=list(items),
        failures=list(failures),
        stable_run_id=stable_item.run_id if stable_item is not None else None,
    )


def execute_validation_suite(
    *,
    stable_lr: float,
    err_good_thresh: float,
    repro_tol: float,
    run_experiment_fn: Callable[[RunSpec], Any],
    run_repository: RunRepository,
    on_status: Callable[[str], None] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> ValidationSuiteResult:
    specs = build_validation_specs(stable_lr=stable_lr)
    items: list[ValidationSavedRun] = []
    failures: list[ValidationFailure] = []
    total = len(specs)
    for index, spec in enumerate(specs, start=1):
        if on_status is not None:
            on_status(f"Running validation case {index}/{total} · {spec.run_name}")
        try:
            result = run_experiment_fn(spec)
            run_id = run_repository.save_result(result)
            items.append(ValidationSavedRun(
                index=index,
                seed=int(spec.seed),
                run_id=str(run_id),
                config=result.spec.to_dict(),
                summary=dict(result.summary),
                metrics=dict(result.metrics),
                metadata=dict(getattr(result, "metadata", {}) or {}),
            ))
        except Exception as exc:
            failures.append(ValidationFailure(
                index=index,
                seed=int(spec.seed),
                stage="run_experiment",
                error_type=type(exc).__name__,
                error_message=str(exc),
                traceback_text=traceback.format_exc(),
            ))
        if on_progress is not None:
            on_progress(index, total)

    return summarize_validation_result(
        items=items,
        failures=failures,
        err_good_thresh=err_good_thresh,
        repro_tol=repro_tol,
    )
