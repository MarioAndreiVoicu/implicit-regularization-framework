import torch

from framework.core.device import (
    resolve_dtype_preference,
)
from framework.core.experiments import run_experiment
from framework.core.models import RunSpec


def test_resolve_dtype_prefers_float64_for_accuracy_cpu():
    dtype, name = resolve_dtype_preference("auto", "accuracy", torch.device("cpu"))
    assert dtype == torch.float64
    assert name == "float64"


def test_resolve_dtype_prefers_float32_for_speed():
    dtype, name = resolve_dtype_preference("auto", "speed", torch.device("cpu"))
    assert dtype == torch.float32
    assert name == "float32"


def test_run_experiment_records_resolved_dtype():
    spec = RunSpec(
        seed=0,
        n=20,
        r_star=3,
        m=100,
        factor_rank=5,
        init_scale=1e-2,
        lr=1e-3,
        steps=10,
        log_every=5,
        problem_family="psd_sensing",
        data_source="synthetic",
        measurement_type="gaussian_symmetric",
        optimizer="gd",
        execution_mode="speed",
        device_preference="cpu",
        dtype_preference="auto",
    )
    result = run_experiment(spec)
    assert result.summary["resolved_device"] == "cpu"
    assert result.summary["resolved_dtype"] == "float32"
    assert result.summary["execution_mode"] == "speed"
