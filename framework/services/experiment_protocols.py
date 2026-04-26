from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExperimentProtocol:
    protocol_id: str
    title: str
    hypothesis: str
    manipulated_factor: str
    held_fixed_factors: tuple[str, ...]
    primary_metrics: tuple[str, ...]
    secondary_metrics: tuple[str, ...] = ()
    seed_policy: str = ""
    notes: str | None = None
    inclusion_notes: str | None = None
    exclusion_notes: str | None = None
    schema_version: int = 1

    def validate(self) -> None:
        if int(self.schema_version) < 1:
            raise ValueError("ExperimentProtocol.schema_version must be >= 1")
        if not str(self.protocol_id).strip():
            raise ValueError("ExperimentProtocol.protocol_id must be non-empty")
        if not str(self.title).strip():
            raise ValueError("ExperimentProtocol.title must be non-empty")
        if not str(self.hypothesis).strip():
            raise ValueError("ExperimentProtocol.hypothesis must be non-empty")
        if not str(self.manipulated_factor).strip():
            raise ValueError("ExperimentProtocol.manipulated_factor must be non-empty")
        if not str(self.seed_policy).strip():
            raise ValueError("ExperimentProtocol.seed_policy must be non-empty")
        if not self.held_fixed_factors:
            raise ValueError("ExperimentProtocol.held_fixed_factors must be non-empty")
        if not self.primary_metrics:
            raise ValueError("ExperimentProtocol.primary_metrics must be non-empty")
        held_fixed = tuple(str(item).strip() for item in self.held_fixed_factors)
        primary_metrics = tuple(str(item).strip() for item in self.primary_metrics)
        secondary_metrics = tuple(str(item).strip() for item in self.secondary_metrics)
        if any(not item for item in held_fixed):
            raise ValueError("ExperimentProtocol.held_fixed_factors cannot contain empty entries")
        if any(not item for item in primary_metrics):
            raise ValueError("ExperimentProtocol.primary_metrics cannot contain empty entries")
        if any(not item for item in secondary_metrics):
            raise ValueError("ExperimentProtocol.secondary_metrics cannot contain empty entries")
        if len(set(held_fixed)) != len(held_fixed):
            raise ValueError("ExperimentProtocol.held_fixed_factors must be distinct")
        if len(set(primary_metrics)) != len(primary_metrics):
            raise ValueError("ExperimentProtocol.primary_metrics must be distinct")
        if len(set(secondary_metrics)) != len(secondary_metrics):
            raise ValueError("ExperimentProtocol.secondary_metrics must be distinct")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": int(self.schema_version),
            "protocol_id": str(self.protocol_id),
            "title": str(self.title),
            "hypothesis": str(self.hypothesis),
            "manipulated_factor": str(self.manipulated_factor),
            "held_fixed_factors": list(self.held_fixed_factors),
            "primary_metrics": list(self.primary_metrics),
            "secondary_metrics": list(self.secondary_metrics),
            "seed_policy": str(self.seed_policy),
            "notes": self.notes,
            "inclusion_notes": self.inclusion_notes,
            "exclusion_notes": self.exclusion_notes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentProtocol":
        protocol = cls(
            protocol_id=str(payload.get("protocol_id") or ""),
            title=str(payload.get("title") or ""),
            hypothesis=str(payload.get("hypothesis") or ""),
            manipulated_factor=str(payload.get("manipulated_factor") or ""),
            held_fixed_factors=tuple(str(item) for item in payload.get("held_fixed_factors", [])),
            primary_metrics=tuple(str(item) for item in payload.get("primary_metrics", [])),
            secondary_metrics=tuple(str(item) for item in payload.get("secondary_metrics", [])),
            seed_policy=str(payload.get("seed_policy") or ""),
            notes=payload.get("notes"),
            inclusion_notes=payload.get("inclusion_notes"),
            exclusion_notes=payload.get("exclusion_notes"),
            schema_version=int(payload.get("schema_version", 1)),
        )
        protocol.validate()
        return protocol


def normalize_experiment_protocol(protocol: ExperimentProtocol | dict[str, Any] | None) -> ExperimentProtocol | None:
    if protocol is None:
        return None
    if isinstance(protocol, ExperimentProtocol):
        protocol.validate()
        return protocol
    if isinstance(protocol, dict):
        return ExperimentProtocol.from_dict(protocol)
    raise TypeError("protocol must be an ExperimentProtocol, dict, or None")


def protocol_metadata(protocol: ExperimentProtocol | dict[str, Any] | None) -> dict[str, Any]:
    normalized = normalize_experiment_protocol(protocol)
    if normalized is None:
        return {}
    payload = normalized.to_dict()
    return {
        "protocol": payload,
        "protocol_id": payload["protocol_id"],
        "protocol_title": payload["title"],
        "protocol_manipulated_factor": payload["manipulated_factor"],
        "protocol_primary_metrics": list(payload["primary_metrics"]),
        "protocol_held_fixed_factors": list(payload["held_fixed_factors"]),
        "protocol_seed_policy": payload["seed_policy"],
    }


__all__ = [
    "ExperimentProtocol",
    "normalize_experiment_protocol",
    "protocol_metadata",
]
