from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from framework.core.device import (
    list_device_preferences,
    list_dtype_preferences,
    list_execution_modes,
)
from framework.core.schedules import list_lr_schedule_types
from framework.core.initialization import list_init_types
from framework.core.optimizers import list_optimizers
from framework.core.parameterization import list_parameterizations
from framework.core.psd_regimes import list_psd_regimes
from framework.core.regularization import list_explicit_reg_types, explicit_reg_types_for_parameterization
from framework.core.stopping import list_stopping_modes
from framework.core.runtime_info import build_runtime_snapshot, RUNTIME_PROVENANCE_SCHEMA_VERSION


class RunStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    DIVERGED = "diverged"
    INVALID = "invalid"


VALID_RUN_STATUSES = {
    RunStatus.PENDING,
    RunStatus.RUNNING,
    RunStatus.COMPLETED,
    RunStatus.FAILED,
    RunStatus.DIVERGED,
    RunStatus.INVALID,
}


@dataclass
class RunSpec:
    seed: int
    n: int
    r_star: int
    m: int
    factor_rank: int
    init_scale: float
    lr: float
    steps: int
    log_every: int
    lr_schedule_type: str = "constant"
    lr_decay_gamma: float = 0.5
    lr_step_size: int = 100
    lr_min: float = 0.0
    n_cols: int | None = None
    noise_std: float = 0.0
    problem_family: str = "psd_sensing"
    data_source: str = "synthetic"
    custom_dataset_path: str | None = None
    custom_dataset_id: str | None = None
    dataset_version: str = "synthetic_default_v1"
    split_version: str = "synthetic_default_split_v1"
    preprocessing_version: str = "none"
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    early_stopping_enabled: bool = False
    early_stopping_metric: str = "val_rmse"
    early_stopping_patience: int = 100
    early_stopping_min_delta: float = 0.0
    measurement_type: str = "gaussian_symmetric"
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
    momentum_beta: float = 0.9
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    preconditioner_eps: float = 1e-8
    parameterization: str = "factorized_psd"
    init_type: str = "near_zero_gaussian"
    explicit_reg_type: str = "none"
    explicit_reg_strength: float = 0.0
    stopping_mode: str = "run_to_convergence"
    grad_norm_tol: float = 1e-10
    objective_tol: float = 1e-12
    divergence_norm_threshold: float = 1e6
    track_eigs: bool = False
    num_eigs: int = 5
    run_name: str | None = None
    notes: str | None = ""
    group_id: str | None = None
    seed_group_id: str | None = None
    schema_version: int = 1
    framework_version: str = "0.1.0"
    git_commit: str | None = None

    def validate(self) -> None:
        if self.schema_version < 1:
            raise ValueError("schema_version must be >= 1")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")
        if self.n < 2:
            raise ValueError("n must be >= 2")
        if self.r_star < 1:
            raise ValueError("r_star must be >= 1")
        if self.m < 1:
            raise ValueError("m must be >= 1")
        if self.factor_rank < 1:
            raise ValueError("factor_rank must be >= 1")
        if self.problem_family == "psd_sensing":
            if self.r_star > self.n:
                raise ValueError("r_star cannot exceed n for psd_sensing")
            if self.factor_rank > self.n:
                raise ValueError("factor_rank cannot exceed n for psd_sensing")
            if self.psd_regime not in list_psd_regimes():
                raise ValueError(f"Unsupported psd_regime={self.psd_regime!r}")
            if not (0.0 <= self.noncommutativity_level <= 1.0):
                raise ValueError("noncommutativity_level must be in [0, 1] for psd_sensing")
            if self.psd_regime == "commuting" and self.noncommutativity_level != 0.0:
                raise ValueError("psd_regime='commuting' requires noncommutativity_level=0.0")
        elif self.problem_family == "matrix_completion":
            if self.data_source not in {"synthetic", "movielens_100k", "custom_triplets_csv"}:
                raise ValueError("matrix_completion requires data_source in {'synthetic', 'movielens_100k', 'custom_triplets_csv'}")
            if not (0.0 <= self.coherence_level <= 1.0):
                raise ValueError("coherence_level must be in [0, 1] for matrix_completion")
            if self.sampling_regime not in {"uniform", "leverage_biased", "observed"}:
                raise ValueError("sampling_regime must be one of {'uniform', 'leverage_biased', 'observed'}")
            if self.mc_spectrum_type not in {"flat", "power_decay", "exponential_decay"}:
                raise ValueError("mc_spectrum_type must be one of {'flat', 'power_decay', 'exponential_decay'} for matrix_completion")
            if self.mc_spectrum_decay < 0.0:
                raise ValueError("mc_spectrum_decay must be >= 0 for matrix_completion")
            if self.mc_factor_balance not in {"balanced", "unbalanced"}:
                raise ValueError("mc_factor_balance must be one of {'balanced', 'unbalanced'} for matrix_completion")
            if not (0.0 <= self.mc_factor_imbalance_level <= 1.0):
                raise ValueError("mc_factor_imbalance_level must be in [0, 1] for matrix_completion")
            if self.mc_factor_balance == "balanced" and self.mc_factor_imbalance_level != 0.0:
                raise ValueError("mc_factor_balance='balanced' requires mc_factor_imbalance_level=0.0")
            n_cols = self.n_cols if self.n_cols is not None else self.n
            if n_cols < 2:
                raise ValueError("n_cols must be >= 2 for matrix_completion")
            if self.r_star > min(self.n, n_cols):
                raise ValueError("r_star cannot exceed min(n, n_cols) for matrix_completion")
            if self.factor_rank > min(self.n, n_cols):
                raise ValueError("factor_rank cannot exceed min(n, n_cols) for matrix_completion")
            if self.m > self.n * n_cols:
                raise ValueError("m cannot exceed n * n_cols for matrix_completion")
        else:
            raise ValueError(f"Unsupported problem_family={self.problem_family!r}")
        if self.problem_family == "psd_sensing" and self.data_source != "synthetic":
            raise ValueError("psd_sensing requires data_source='synthetic'")
        if not str(self.dataset_version).strip():
            raise ValueError("dataset_version must be non-empty")
        if not str(self.split_version).strip():
            raise ValueError("split_version must be non-empty")
        if not str(self.preprocessing_version).strip():
            raise ValueError("preprocessing_version must be non-empty")
        if self.init_scale < 0.0:
            raise ValueError("init_scale must be >= 0")
        if self.lr < 0.0:
            raise ValueError("lr must be >= 0")
        if self.lr_schedule_type not in list_lr_schedule_types():
            raise ValueError(f"Unsupported lr_schedule_type={self.lr_schedule_type!r}")
        if not (0.0 < self.lr_decay_gamma <= 1.0):
            raise ValueError("lr_decay_gamma must be in (0, 1]")
        if self.lr_step_size < 1:
            raise ValueError("lr_step_size must be >= 1")
        if self.lr_min < 0.0:
            raise ValueError("lr_min must be >= 0")
        if self.lr_min > self.lr:
            raise ValueError("lr_min cannot exceed lr")
        if self.explicit_reg_strength < 0.0:
            raise ValueError("explicit_reg_strength must be >= 0")
        if self.steps < 1:
            raise ValueError("steps must be >= 1")
        if self.log_every < 1:
            raise ValueError("log_every must be >= 1")
        if self.log_every > self.steps:
            raise ValueError("log_every cannot exceed steps")
        if self.noise_std < 0.0:
            raise ValueError("noise_std must be >= 0")
        if not (0.0 <= self.val_fraction < 1.0):
            raise ValueError("val_fraction must be in [0, 1)")
        if not (0.0 <= self.test_fraction < 1.0):
            raise ValueError("test_fraction must be in [0, 1)")
        if self.val_fraction + self.test_fraction >= 1.0:
            raise ValueError("val_fraction + test_fraction must be < 1")
        if self.early_stopping_metric not in {"val_rmse", "val_loss"}:
            raise ValueError("early_stopping_metric must be one of {'val_rmse', 'val_loss'}")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be >= 1")
        if self.early_stopping_min_delta < 0.0:
            raise ValueError("early_stopping_min_delta must be >= 0")
        if self.early_stopping_enabled and not (self.problem_family == "matrix_completion" and self.data_source in {"movielens_100k", "custom_triplets_csv"}):
            raise ValueError("early_stopping_enabled is only supported for matrix_completion with a real observed-rating data source")
        if self.num_eigs < 1:
            raise ValueError("num_eigs must be >= 1")
        if self.num_eigs > self.n:
            raise ValueError("num_eigs cannot exceed n")
        if self.grad_norm_tol <= 0.0:
            raise ValueError("grad_norm_tol must be > 0")
        if self.objective_tol <= 0.0:
            raise ValueError("objective_tol must be > 0")
        if self.divergence_norm_threshold <= 0.0:
            raise ValueError("divergence_norm_threshold must be > 0")
        if self.batch_size is not None:
            if self.batch_size < 1:
                raise ValueError("batch_size must be >= 1 when provided")
            if self.batch_size > self.m:
                raise ValueError("batch_size cannot exceed m")
        if not (0.0 <= self.momentum_beta < 1.0):
            raise ValueError("momentum_beta must be in [0, 1)")
        if not (0.0 < self.adam_beta1 < 1.0):
            raise ValueError("adam_beta1 must be in (0, 1)")
        if not (0.0 < self.adam_beta2 < 1.0):
            raise ValueError("adam_beta2 must be in (0, 1)")
        if self.adam_eps <= 0.0:
            raise ValueError("adam_eps must be > 0")
        if self.preconditioner_eps <= 0.0:
            raise ValueError("preconditioner_eps must be > 0")
        if not str(self.problem_family).strip():
            raise ValueError("problem_family must be non-empty")
        if not str(self.measurement_type).strip():
            raise ValueError("measurement_type must be non-empty")
        if self.problem_family == "psd_sensing" and self.measurement_type != "gaussian_symmetric":
            raise ValueError("psd_sensing requires measurement_type='gaussian_symmetric'")
        if self.problem_family == "matrix_completion" and self.psd_regime != "non_commuting":
            raise ValueError("matrix_completion requires psd_regime='non_commuting'")
        if self.problem_family == "matrix_completion" and self.noncommutativity_level != 1.0:
            raise ValueError("matrix_completion requires noncommutativity_level=1.0")
        if self.problem_family == "psd_sensing" and self.coherence_level != 0.0:
            raise ValueError("psd_sensing requires coherence_level=0.0")
        if self.problem_family == "psd_sensing" and self.sampling_regime != "uniform":
            raise ValueError("psd_sensing requires sampling_regime='uniform'")
        if self.problem_family == "psd_sensing" and self.mc_spectrum_type != "flat":
            raise ValueError("psd_sensing requires mc_spectrum_type='flat'")
        if self.problem_family == "psd_sensing" and self.mc_spectrum_decay != 1.0:
            raise ValueError("psd_sensing requires mc_spectrum_decay=1.0")
        if self.problem_family == "psd_sensing" and self.mc_factor_balance != "balanced":
            raise ValueError("psd_sensing requires mc_factor_balance='balanced'")
        if self.problem_family == "psd_sensing" and self.mc_factor_imbalance_level != 0.0:
            raise ValueError("psd_sensing requires mc_factor_imbalance_level=0.0")
        if self.problem_family == "matrix_completion":
            allowed_measurements = {"uniform_entries", "leverage_biased_entries", "observed_ratings"}
            if self.measurement_type not in allowed_measurements:
                raise ValueError(f"matrix_completion requires measurement_type in {allowed_measurements}")
            if self.data_source == "synthetic" and self.measurement_type not in {"uniform_entries", "leverage_biased_entries"}:
                raise ValueError("synthetic matrix_completion requires measurement_type in {'uniform_entries', 'leverage_biased_entries'}")
            if self.data_source == "synthetic" and self.sampling_regime not in {"uniform", "leverage_biased"}:
                raise ValueError("synthetic matrix_completion requires sampling_regime in {'uniform', 'leverage_biased'}")
            if self.data_source == "synthetic":
                expected = "uniform_entries" if self.sampling_regime == "uniform" else "leverage_biased_entries"
                if self.measurement_type != expected:
                    raise ValueError(f"measurement_type={self.measurement_type!r} is inconsistent with sampling_regime={self.sampling_regime!r}")
            if self.data_source in {"movielens_100k", "custom_triplets_csv"} and self.measurement_type != "observed_ratings":
                raise ValueError(f"{self.data_source} requires measurement_type='observed_ratings'")
            if self.data_source in {"movielens_100k", "custom_triplets_csv"} and self.sampling_regime not in {"observed", "uniform"}:
                raise ValueError(f"{self.data_source} requires sampling_regime in {'observed', 'uniform'}")
            if self.data_source == "custom_triplets_csv" and not str(self.custom_dataset_path or "").strip():
                raise ValueError("custom_triplets_csv requires custom_dataset_path")
        if self.execution_mode not in list_execution_modes():
            raise ValueError(f"Unsupported execution_mode={self.execution_mode!r}")
        if self.device_preference not in list_device_preferences():
            raise ValueError(f"Unsupported device_preference={self.device_preference!r}")
        if self.dtype_preference not in list_dtype_preferences():
            raise ValueError(f"Unsupported dtype_preference={self.dtype_preference!r}")
        if not str(self.optimizer).strip():
            raise ValueError("optimizer must be non-empty")
        if self.optimizer not in list_optimizers():
            raise ValueError(f"Unsupported optimizer={self.optimizer!r}")
        if self.optimizer in {"sgd", "momentum"} and self.batch_size is None:
            raise ValueError(f"optimizer={self.optimizer!r} requires batch_size")
        if not str(self.parameterization).strip():
            raise ValueError("parameterization must be non-empty")
        if self.parameterization not in list_parameterizations():
            raise ValueError(f"Unsupported parameterization={self.parameterization!r}")
        if self.problem_family == "psd_sensing" and self.parameterization not in {"factorized_psd", "symmetric_matrix"}:
            raise ValueError("psd_sensing requires parameterization in {'factorized_psd', 'symmetric_matrix'}")
        if self.problem_family == "matrix_completion" and self.parameterization not in {"factorized_uv", "direct_matrix"}:
            raise ValueError("matrix_completion requires parameterization in {'factorized_uv', 'direct_matrix'}")
        if self.problem_family == "matrix_completion" and self.parameterization == "direct_matrix":
            if self.mc_factor_balance != "balanced":
                raise ValueError("direct_matrix requires mc_factor_balance='balanced'")
            if self.mc_factor_imbalance_level != 0.0:
                raise ValueError("direct_matrix requires mc_factor_imbalance_level=0.0")
        if self.optimizer == "preconditioned_gd" and self.parameterization not in {"factorized_psd", "factorized_uv"}:
            raise ValueError("preconditioned_gd requires a factorized parameterization")
        if not str(self.init_type).strip():
            raise ValueError("init_type must be non-empty")
        if self.init_type not in list_init_types():
            raise ValueError(f"Unsupported init_type={self.init_type!r}")
        if not str(self.explicit_reg_type).strip():
            raise ValueError("explicit_reg_type must be non-empty")
        if self.explicit_reg_type not in list_explicit_reg_types():
            raise ValueError(f"Unsupported explicit_reg_type={self.explicit_reg_type!r}")
        if self.explicit_reg_type == "none" and self.explicit_reg_strength != 0.0:
            raise ValueError("explicit_reg_strength must be 0 when explicit_reg_type='none'")
        if self.explicit_reg_type == "factor_l2" and self.parameterization not in {"factorized_psd", "factorized_uv"}:
            raise ValueError("explicit_reg_type='factor_l2' requires factorized parameterization")
        if self.explicit_reg_type == "frobenius" and self.parameterization not in {"symmetric_matrix", "direct_matrix"}:
            raise ValueError("explicit_reg_type='frobenius' requires direct matrix parameterization")
        allowed_reg_types = explicit_reg_types_for_parameterization(self.parameterization)
        if self.explicit_reg_type not in allowed_reg_types:
            allowed = ", ".join(allowed_reg_types)
            raise ValueError(f"parameterization={self.parameterization!r} supports explicit_reg_type in {{{allowed}}}")
        if not str(self.stopping_mode).strip():
            raise ValueError("stopping_mode must be non-empty")
        if self.stopping_mode not in list_stopping_modes():
            raise ValueError(f"Unsupported stopping_mode={self.stopping_mode!r}")
        if not str(self.framework_version).strip():
            raise ValueError("framework_version must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": int(self.schema_version),
            "framework_version": str(self.framework_version),
            "git_commit": self.git_commit,
            "seed": int(self.seed),
            "problem_family": str(self.problem_family),
            "data_source": str(self.data_source),
            "custom_dataset_id": str(self.custom_dataset_id) if self.custom_dataset_id is not None else None,
            "dataset_version": str(self.dataset_version),
            "split_version": str(self.split_version),
            "preprocessing_version": str(self.preprocessing_version),
            "val_fraction": float(self.val_fraction),
            "test_fraction": float(self.test_fraction),
            "early_stopping_enabled": bool(self.early_stopping_enabled),
            "early_stopping_metric": str(self.early_stopping_metric),
            "early_stopping_patience": int(self.early_stopping_patience),
            "early_stopping_min_delta": float(self.early_stopping_min_delta),
            "measurement_type": str(self.measurement_type),
            "psd_regime": str(self.psd_regime),
            "noncommutativity_level": float(self.noncommutativity_level),
            "coherence_level": float(self.coherence_level),
            "sampling_regime": str(self.sampling_regime),
            "mc_spectrum_type": str(self.mc_spectrum_type),
            "mc_spectrum_decay": float(self.mc_spectrum_decay),
            "mc_factor_balance": str(self.mc_factor_balance),
            "mc_factor_imbalance_level": float(self.mc_factor_imbalance_level),
            "optimizer": str(self.optimizer),
            "execution_mode": str(self.execution_mode),
            "device_preference": str(self.device_preference),
            "dtype_preference": str(self.dtype_preference),
            "batch_size": self.batch_size,
            "momentum_beta": float(self.momentum_beta),
            "adam_beta1": float(self.adam_beta1),
            "adam_beta2": float(self.adam_beta2),
            "adam_eps": float(self.adam_eps),
            "preconditioner_eps": float(self.preconditioner_eps),
            "parameterization": str(self.parameterization),
            "init_type": str(self.init_type),
            "explicit_reg_type": str(self.explicit_reg_type),
            "explicit_reg_strength": float(self.explicit_reg_strength),
            "stopping_mode": str(self.stopping_mode),
            "grad_norm_tol": float(self.grad_norm_tol),
            "objective_tol": float(self.objective_tol),
            "divergence_norm_threshold": float(self.divergence_norm_threshold),
            "n": int(self.n),
            "n_cols": self.n_cols,
            "r_star": int(self.r_star),
            "m": int(self.m),
            "factor_rank": int(self.factor_rank),
            "init_scale": float(self.init_scale),
            "lr": float(self.lr),
            "steps": int(self.steps),
            "lr_schedule_type": str(self.lr_schedule_type),
            "lr_decay_gamma": float(self.lr_decay_gamma),
            "lr_step_size": int(self.lr_step_size),
            "lr_min": float(self.lr_min),
            "log_every": int(self.log_every),
            "noise_std": float(self.noise_std),
            "track_eigs": bool(self.track_eigs),
            "num_eigs": int(self.num_eigs),
            "run_name": self.run_name,
            "notes": self.notes,
            "group_id": self.group_id,
            "seed_group_id": self.seed_group_id,
        }


@dataclass(frozen=True)
class SweepAxisSpec:
    field: str
    values: tuple[Any, ...]

    def validate(self) -> None:
        if not str(self.field).strip():
            raise ValueError("SweepAxisSpec.field must be non-empty")
        if len(self.values) < 1:
            raise ValueError("SweepAxisSpec.values must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {"field": str(self.field), "values": list(self.values)}


@dataclass(frozen=True)
class SweepSpec:
    group_id: str
    kind: str
    preset: str
    axes: tuple[SweepAxisSpec, ...]
    fixed_factors: dict[str, Any]
    seeds: tuple[int, ...]
    coupling: str = "grid"
    created_at: str | None = None

    def validate(self) -> None:
        if not str(self.group_id).strip():
            raise ValueError("SweepSpec.group_id must be non-empty")
        if self.kind not in {"1D", "2D"}:
            raise ValueError("SweepSpec.kind must be '1D' or '2D'")
        expected_axes = 1 if self.kind == "1D" else 2
        if len(self.axes) != expected_axes:
            raise ValueError("SweepSpec.axes does not match kind")
        if self.kind == "1D":
            if self.coupling != "grid":
                raise ValueError("1D sweeps must use coupling='grid'")
        elif self.coupling not in {"grid", "paired"}:
            raise ValueError("2D sweeps must use coupling='grid' or 'paired'")
        axis_fields = []
        for axis in self.axes:
            axis.validate()
            axis_fields.append(axis.field)
        if len(set(axis_fields)) != len(axis_fields):
            raise ValueError("SweepSpec axes must have distinct fields")
        if not isinstance(self.fixed_factors, dict):
            raise ValueError("SweepSpec.fixed_factors must be a dict")
        if any(field in self.fixed_factors for field in axis_fields):
            raise ValueError("SweepSpec.fixed_factors cannot contain swept fields")
        if len(self.seeds) < 1:
            raise ValueError("SweepSpec.seeds must be non-empty")
        if self.kind == "2D" and self.coupling == "paired":
            lengths = {len(axis.values) for axis in self.axes}
            if len(lengths) != 1:
                raise ValueError("Paired 2D sweeps require axes with equal-length value lists")

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(axis.field for axis in self.axes)

    @property
    def values(self) -> dict[str, list[Any]]:
        return {axis.field: list(axis.values) for axis in self.axes}

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "group_id": str(self.group_id),
            "kind": str(self.kind),
            "preset": str(self.preset),
            "axes": [axis.to_dict() for axis in self.axes],
            "fields": list(self.fields),
            "values": self.values,
            "fixed_factors": dict(self.fixed_factors),
            "seeds": list(self.seeds),
            "coupling": str(self.coupling),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> 'SweepSpec':
        axes_payload = payload.get("axes")
        if axes_payload is None:
            fields = list(payload.get("fields", []))
            values = dict(payload.get("values", {}))
            axes_payload = [{"field": field, "values": list(values.get(field, [None]))} for field in fields]
        else:
            normalized_axes = []
            for axis in axes_payload:
                axis_values = list(axis.get("values", []))
                if not axis_values:
                    axis_values = [None]
                normalized_axes.append({"field": axis.get("field"), "values": axis_values})
            axes_payload = normalized_axes
        spec = cls(
            group_id=str(payload.get("group_id") or ""),
            kind=str(payload.get("kind") or ""),
            preset=str(payload.get("preset") or ""),
            axes=tuple(SweepAxisSpec(field=str(axis.get("field") or ""), values=tuple(axis.get("values", []))) for axis in axes_payload),
            fixed_factors=dict(payload.get("fixed_factors") or payload.get("base_config") or {}),
            seeds=tuple(int(seed) for seed in payload.get("seeds", [0])),
            coupling=str(payload.get("coupling") or "grid"),
            created_at=payload.get("created_at"),
        )
        spec.validate()
        return spec


@dataclass
class RunResult:
    spec: RunSpec
    metrics: dict[str, list[float]]
    summary: dict[str, Any]
    artifacts: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.spec.validate()
        status = self.summary.get("status")
        if status not in VALID_RUN_STATUSES:
            raise ValueError(f"Invalid run status: {status!r}")
        if not isinstance(self.metrics, dict) or "step" not in self.metrics:
            raise ValueError("metrics must be a dict containing 'step'")
        lengths = {len(v) for v in self.metrics.values() if isinstance(v, list)}
        if lengths and len(lengths) != 1:
            raise ValueError("all metric series must have the same length")

    def metadata_dict(self, saved_at: str | None = None, run_id: str | None = None) -> dict[str, Any]:
        self.validate()
        runtime_snapshot = build_runtime_snapshot()
        return {
            "provenance_schema_version": int(RUNTIME_PROVENANCE_SCHEMA_VERSION),
            "runtime_snapshot": runtime_snapshot,
            "python_version": runtime_snapshot.get("python_version"),
            "python_implementation": runtime_snapshot.get("python_implementation"),
            "platform_system": runtime_snapshot.get("platform_system"),
            "platform_release": runtime_snapshot.get("platform_release"),
            "platform_machine": runtime_snapshot.get("platform_machine"),
            "platform_version": runtime_snapshot.get("platform_version"),
            "executable": runtime_snapshot.get("executable"),
            "numpy_version": runtime_snapshot.get("numpy_version"),
            "pandas_version": runtime_snapshot.get("pandas_version"),
            "torch_version": runtime_snapshot.get("torch_version"),
            "schema_version": int(self.spec.schema_version),
            "framework_version": str(self.spec.framework_version),
            "git_commit": self.spec.git_commit,
            "saved_at": saved_at,
            "run_id": run_id,
            "seed": int(self.spec.seed),
            "problem_family": str(self.spec.problem_family),
            "data_source": str(self.spec.data_source),
            "custom_dataset_id": str(self.spec.custom_dataset_id) if self.spec.custom_dataset_id is not None else None,
            "dataset_version": str(self.spec.dataset_version),
            "split_version": str(self.spec.split_version),
            "preprocessing_version": str(self.spec.preprocessing_version),
            "val_fraction": float(self.spec.val_fraction),
            "test_fraction": float(self.spec.test_fraction),
            "early_stopping_enabled": bool(self.spec.early_stopping_enabled),
            "early_stopping_metric": str(self.spec.early_stopping_metric),
            "early_stopping_patience": int(self.spec.early_stopping_patience),
            "early_stopping_min_delta": float(self.spec.early_stopping_min_delta),
            "measurement_type": str(self.spec.measurement_type),
            "psd_regime": str(self.spec.psd_regime),
            "noncommutativity_level": float(self.spec.noncommutativity_level),
            "coherence_level": float(self.spec.coherence_level),
            "sampling_regime": str(self.spec.sampling_regime),
            "mc_spectrum_type": str(self.spec.mc_spectrum_type),
            "mc_spectrum_decay": float(self.spec.mc_spectrum_decay),
            "mc_factor_balance": str(self.spec.mc_factor_balance),
            "mc_factor_imbalance_level": float(self.spec.mc_factor_imbalance_level),
            "optimizer": str(self.spec.optimizer),
            "lr_schedule_type": str(self.spec.lr_schedule_type),
            "lr_decay_gamma": float(self.spec.lr_decay_gamma),
            "lr_step_size": int(self.spec.lr_step_size),
            "lr_min": float(self.spec.lr_min),
            "execution_mode": str(self.spec.execution_mode),
            "device_preference": str(self.spec.device_preference),
            "dtype_preference": str(self.spec.dtype_preference),
            "resolved_device": self.summary.get("resolved_device"),
            "resolved_dtype": self.summary.get("resolved_dtype"),
            "batch_size": self.spec.batch_size,
            "momentum_beta": float(self.spec.momentum_beta),
            "adam_beta1": float(self.spec.adam_beta1),
            "adam_beta2": float(self.spec.adam_beta2),
            "adam_eps": float(self.spec.adam_eps),
            "preconditioner_eps": float(self.spec.preconditioner_eps),
            "parameterization": str(self.spec.parameterization),
            "init_type": str(self.spec.init_type),
            "explicit_reg_type": str(self.spec.explicit_reg_type),
            "explicit_reg_strength": float(self.spec.explicit_reg_strength),
            "stopping_mode": str(self.spec.stopping_mode),
            "grad_norm_tol": float(self.spec.grad_norm_tol),
            "objective_tol": float(self.spec.objective_tol),
            "divergence_norm_threshold": float(self.spec.divergence_norm_threshold),
            "group_id": self.spec.group_id,
            "seed_group_id": self.spec.seed_group_id,
            "status": str(self.summary.get("status")),
        }
