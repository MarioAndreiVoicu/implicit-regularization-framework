from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from framework.core.models import SweepSpec
from framework.services.data_sources import canonical_data_source, is_real_data_source
from framework.services.new_run import NewRunRequest, resolve_new_run_seeds
from framework.services.sweeps import SweepRequest


@dataclass(frozen=True)
class CriticWarning:
    code: str
    message: str


@dataclass
class CriticReport:
    warnings: list[CriticWarning] = field(default_factory=list)

    def add(self, code: str, message: str) -> None:
        self.warnings.append(CriticWarning(code=code, message=message))

    def extend(self, warnings: Iterable[CriticWarning]) -> None:
        self.warnings.extend(list(warnings))

    @property
    def messages(self) -> list[str]:
        return [warning.message for warning in self.warnings]

    @property
    def is_empty(self) -> bool:
        return not self.warnings


REAL_DATA_UNAVAILABLE_METRICS_WARNING = (
    "Real-data matrix completion does not provide synthetic-only recovery metrics; "
    "interpret held-out RMSE and stability metrics instead."
)


def _seed_count_warning(seed_count: int) -> str | None:
    if seed_count <= 1:
        return None
    if seed_count < 3:
        return f"Only {seed_count} seeds are configured. Multi-seed comparisons will be weak."
    if seed_count < 5:
        return f"Only {seed_count} seeds are configured. Results may still be noisy."
    return None



def _custom_dataset_provenance_warning(data_source: str, custom_dataset_id: str | None, custom_dataset_path: str | None) -> str | None:
    if data_source != "custom_triplets_csv":
        return None
    if custom_dataset_id:
        return None
    if custom_dataset_path:
        return "Custom dataset is referenced by path only. Register it for stronger provenance."
    return None



def audit_new_run_request(request: NewRunRequest) -> CriticReport:
    report = CriticReport()
    problem_family = str(request.problem_family)
    data_source = canonical_data_source(problem_family, str(request.data_source))

    try:
        seed_count = len(resolve_new_run_seeds(request))
    except Exception:
        seed_count = 0
    seed_warning = _seed_count_warning(seed_count)
    if seed_warning:
        report.add("low_seed_count", seed_warning)

    if problem_family == "matrix_completion" and is_real_data_source(problem_family, data_source):
        report.add("real_data_metric_scope", REAL_DATA_UNAVAILABLE_METRICS_WARNING)
        if not bool(request.early_stopping_enabled):
            report.add(
                "real_data_no_early_stopping",
                "Real-data matrix completion is running without early stopping. Held-out monitoring is available but disabled.",
            )

    provenance_warning = _custom_dataset_provenance_warning(
        data_source,
        getattr(request, "custom_dataset_id", None),
        getattr(request, "custom_dataset_path", None),
    )
    if provenance_warning:
        report.add("custom_dataset_path_only", provenance_warning)

    if int(request.steps) <= int(request.log_every) * 2:
        report.add(
            "sparse_logging",
            "steps is close to log_every. You will get very little trajectory information.",
        )

    return report



def audit_sweep_request(request: SweepRequest) -> CriticReport:
    report = CriticReport()
    report.extend(audit_new_run_request(request.base_request).warnings)

    seen: set[str] = set()
    for field in request.fields:
        values = list(request.values.get(field, []))
        if len(values) < 2:
            report.add("degenerate_axis", f"Sweep axis '{field}' has fewer than 2 values.")
        if len(values) != len({repr(value) for value in values}):
            report.add("duplicate_axis_values", f"Sweep axis '{field}' contains duplicate values.")
        if field in seen:
            report.add("duplicate_axis_field", f"Sweep field '{field}' appears more than once.")
        seen.add(field)

    return report



def audit_sweep_spec(spec: SweepSpec) -> CriticReport:
    report = CriticReport()
    seed_warning = _seed_count_warning(len(spec.seeds))
    if seed_warning:
        report.add("low_seed_count", seed_warning)

    data_source = str(spec.fixed_factors.get("data_source", ""))
    problem_family = str(spec.fixed_factors.get("problem_family", ""))
    if problem_family == "matrix_completion" and is_real_data_source(problem_family, data_source):
        report.add("real_data_metric_scope", REAL_DATA_UNAVAILABLE_METRICS_WARNING)

    if data_source == "custom_triplets_csv" and not spec.fixed_factors.get("custom_dataset_id") and spec.fixed_factors.get("custom_dataset_path"):
        report.add(
            "custom_dataset_path_only",
            "Custom dataset sweep is referenced by path only. Register it for stronger provenance.",
        )

    for axis in spec.axes:
        if len(axis.values) < 2:
            report.add("degenerate_axis", f"Sweep axis '{axis.field}' has fewer than 2 values.")
        if len(axis.values) != len({repr(value) for value in axis.values}):
            report.add("duplicate_axis_values", f"Sweep axis '{axis.field}' contains duplicate values.")

    return report


__all__ = [
    "CriticWarning",
    "CriticReport",
    "audit_new_run_request",
    "audit_sweep_request",
    "audit_sweep_spec",
]
