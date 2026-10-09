from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from framework.services.experiment_protocols import normalize_experiment_protocol
from framework.services.local_runs import MATCH_AXIS_FIELDS, NON_EXPERIMENTAL_CONFIG_FIELDS

CORE_CONTEXT_FIELDS = [
    "problem_family",
    "data_source",
    "measurement_type",
]

PROTOCOL_AXIS_ALIASES = {
    "lr": "learning_rate",
    "learning_rate": "learning_rate",
    "step_size": "learning_rate",
    "learning_rate_schedule": "learning_rate_schedule",
    "lr_schedule_type": "learning_rate_schedule",
    "lr_decay_gamma": "learning_rate_schedule",
    "lr_step_size": "learning_rate_schedule",
    "lr_min": "learning_rate_schedule",
    "steps": "training_budget",
    "training_steps": "training_budget",
    "training_budget": "training_budget",
    "grad_norm_tol": "training_budget",
    "objective_tol": "training_budget",
    "seed": "random_seed",
    "random_seed": "random_seed",
    "batch_size": "batch_size",
}


def _normalize_axis_name(axis: Any) -> str:
    key = str(axis or "").strip()
    return PROTOCOL_AXIS_ALIASES.get(key, key)


@dataclass(frozen=True)
class ComparisonAudit:
    status: str
    severity: str
    validity_label: str
    summary: str
    varying_axis: str
    protocol_varying_axis: str | None
    reference_axis_value: str
    candidate_axis_value: str
    allowed_differences: tuple[str, ...]
    unexpected_differences: tuple[str, ...]
    secondary_confounds: tuple[str, ...]
    core_mismatches: tuple[str, ...]
    held_fixed_violations: tuple[str, ...]
    protocol_conflicts: tuple[str, ...]
    seed_alignment: str
    paired_seed_expected: bool


def _normalize_value(value: Any) -> Any:
    if isinstance(value, float):
        return float(value)
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_value(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((str(k), _normalize_value(v)) for k, v in value.items()))
    return value


def _config_diff_fields(reference_cfg: dict[str, Any], candidate_cfg: dict[str, Any]) -> list[str]:
    fields = sorted(
        field
        for field in set(reference_cfg.keys()) | set(candidate_cfg.keys())
        if field not in NON_EXPERIMENTAL_CONFIG_FIELDS
    )
    return [
        field
        for field in fields
        if _normalize_value(reference_cfg.get(field)) != _normalize_value(candidate_cfg.get(field))
    ]


def _axis_value(cfg: dict[str, Any], varying_axis: str) -> str:
    fields = MATCH_AXIS_FIELDS[varying_axis]
    if len(fields) == 1:
        return str(cfg.get(fields[0], None))
    return " | ".join(f"{field}={cfg.get(field, None)}" for field in fields)


def _seed_alignment(reference_cfg: dict[str, Any], candidate_cfg: dict[str, Any]) -> str:
    ref_seed = reference_cfg.get("seed")
    cand_seed = candidate_cfg.get("seed")
    if ref_seed is None and cand_seed is None:
        return "both_missing"
    if ref_seed is None:
        return "reference_missing"
    if cand_seed is None:
        return "candidate_missing"
    return "matched" if int(ref_seed) == int(cand_seed) else "different"


def _extract_protocol(metadata: dict[str, Any] | None):
    if not isinstance(metadata, dict):
        return None
    payload = metadata.get("protocol")
    if not isinstance(payload, dict):
        return None
    return normalize_experiment_protocol(payload)


def _paired_seed_expected(*protocols: Any) -> bool:
    for protocol in protocols:
        if protocol is None:
            continue
        if "paired" in str(protocol.seed_policy).lower():
            return True
    return False


def _build_summary(*, core_mismatches: tuple[str, ...], protocol_conflicts: tuple[str, ...], held_fixed_violations: tuple[str, ...], secondary_confounds: tuple[str, ...], allowed_differences: tuple[str, ...], varying_axis: str, seed_alignment: str, paired_seed_expected: bool) -> tuple[str, str, str]:
    if core_mismatches:
        return (
            "invalid_context",
            "error",
            f"Invalid comparison context: {', '.join(core_mismatches)} differ.",
        )
    if protocol_conflicts or held_fixed_violations:
        details = list(protocol_conflicts)
        if held_fixed_violations:
            details.append("held fixed violated: " + ", ".join(held_fixed_violations))
        return (
            "protocol_conflict",
            "error",
            "Protocol conflict: " + "; ".join(details) + ".",
        )
    if not allowed_differences:
        return (
            "weakly_controlled",
            "warning",
            f"Weak comparison: no change detected on varying axis '{varying_axis}'.",
        )
    seed_is_allowed_axis = varying_axis == "random_seed"
    seed_is_confound = seed_alignment != "matched" and not seed_is_allowed_axis
    if secondary_confounds or seed_is_confound:
        seed_note = ""
        if paired_seed_expected and seed_is_confound:
            seed_note = f" Seed alignment is {seed_alignment} under a paired-seed protocol."
        elif seed_is_confound:
            seed_note = f" Seed alignment is {seed_alignment}."
        confounds = f" Secondary confounds: {', '.join(secondary_confounds)}." if secondary_confounds else ""
        return (
            "weakly_controlled",
            "warning",
            f"Weak comparison beyond {varying_axis}.{confounds}{seed_note}",
        )
    return (
        "valid_single_factor",
        "success",
        f"Valid single-factor comparison: only {varying_axis} changes under the current audit.",
    )


def audit_run_pair(
    reference_cfg: dict[str, Any],
    candidate_cfg: dict[str, Any],
    varying_axis: str,
    *,
    reference_metadata: dict[str, Any] | None = None,
    candidate_metadata: dict[str, Any] | None = None,
) -> ComparisonAudit:
    if varying_axis not in MATCH_AXIS_FIELDS:
        raise ValueError(f"Unsupported varying axis: {varying_axis}")

    allowed_fields = set(MATCH_AXIS_FIELDS[varying_axis])
    differing_fields = _config_diff_fields(reference_cfg, candidate_cfg)
    allowed_differences = tuple(field for field in differing_fields if field in allowed_fields)
    unexpected_differences = tuple(field for field in differing_fields if field not in allowed_fields)
    core_mismatches = tuple(field for field in CORE_CONTEXT_FIELDS if field in unexpected_differences)

    reference_protocol = _extract_protocol(reference_metadata)
    candidate_protocol = _extract_protocol(candidate_metadata)
    protocol_axes = {
        _normalize_axis_name(protocol.manipulated_factor)
        for protocol in (reference_protocol, candidate_protocol)
        if protocol is not None
    }
    protocol_varying_axis = next(iter(protocol_axes)) if len(protocol_axes) == 1 else None
    protocol_conflicts: list[str] = []
    if len(protocol_axes) > 1:
        protocol_conflicts.append("reference and candidate protocol manipulated factors disagree")
    if protocol_varying_axis is not None and protocol_varying_axis != varying_axis:
        protocol_conflicts.append(
            f"selected varying axis '{varying_axis}' does not match protocol manipulated factor '{protocol_varying_axis}'"
        )

    held_fixed_expected = {
        field
        for protocol in (reference_protocol, candidate_protocol)
        if protocol is not None
        for field in protocol.held_fixed_factors
    }
    held_fixed_violations = tuple(sorted(field for field in unexpected_differences if field in held_fixed_expected))

    seed_alignment = _seed_alignment(reference_cfg, candidate_cfg)
    paired_seed_expected = _paired_seed_expected(reference_protocol, candidate_protocol)
    if paired_seed_expected and seed_alignment != "matched" and varying_axis != "random_seed":
        protocol_conflicts.append(
            f"paired seed policy requires matched seeds but alignment is '{seed_alignment}'"
        )

    secondary_confounds = tuple(
        field
        for field in unexpected_differences
        if field not in core_mismatches and field not in held_fixed_violations
    )
    status, severity, summary = _build_summary(
        core_mismatches=core_mismatches,
        protocol_conflicts=tuple(protocol_conflicts),
        held_fixed_violations=held_fixed_violations,
        secondary_confounds=secondary_confounds,
        allowed_differences=allowed_differences,
        varying_axis=varying_axis,
        seed_alignment=seed_alignment,
        paired_seed_expected=paired_seed_expected,
    )
    validity_label = {
        "valid_single_factor": "Valid single-factor comparison",
        "weakly_controlled": "Weakly controlled comparison",
        "protocol_conflict": "Protocol conflict",
        "invalid_context": "Invalid context",
    }[status]

    return ComparisonAudit(
        status=status,
        severity=severity,
        validity_label=validity_label,
        summary=summary,
        varying_axis=varying_axis,
        protocol_varying_axis=protocol_varying_axis,
        reference_axis_value=_axis_value(reference_cfg, varying_axis),
        candidate_axis_value=_axis_value(candidate_cfg, varying_axis),
        allowed_differences=allowed_differences,
        unexpected_differences=unexpected_differences,
        secondary_confounds=secondary_confounds,
        core_mismatches=core_mismatches,
        held_fixed_violations=held_fixed_violations,
        protocol_conflicts=tuple(protocol_conflicts),
        seed_alignment=seed_alignment,
        paired_seed_expected=paired_seed_expected,
    )


def comparison_audit_details_table(audit: ComparisonAudit) -> pd.DataFrame:
    rows = [
        {"category": "validity", "details": audit.validity_label},
        {"category": "severity", "details": audit.severity},
        {"category": "varying axis", "details": audit.varying_axis},
        {"category": "protocol varying axis", "details": audit.protocol_varying_axis or ""},
        {"category": "reference axis value", "details": audit.reference_axis_value},
        {"category": "candidate axis value", "details": audit.candidate_axis_value},
        {
            "category": "allowed differences",
            "details": ", ".join(audit.allowed_differences) if audit.allowed_differences else "",
        },
        {
            "category": "unexpected differences",
            "details": ", ".join(audit.unexpected_differences) if audit.unexpected_differences else "",
        },
        {
            "category": "secondary confounds",
            "details": ", ".join(audit.secondary_confounds) if audit.secondary_confounds else "",
        },
        {
            "category": "held-fixed violations",
            "details": ", ".join(audit.held_fixed_violations) if audit.held_fixed_violations else "",
        },
        {
            "category": "protocol conflicts",
            "details": "; ".join(audit.protocol_conflicts) if audit.protocol_conflicts else "",
        },
        {
            "category": "core mismatches",
            "details": ", ".join(audit.core_mismatches) if audit.core_mismatches else "",
        },
        {"category": "seed alignment", "details": audit.seed_alignment},
        {
            "category": "paired seed expected",
            "details": "yes" if audit.paired_seed_expected else "no",
        },
    ]
    return pd.DataFrame(rows)
