from __future__ import annotations

import pytest
import torch

from framework.core.device import list_device_preferences, resolve_device_preference
from framework.core.experiments import run_experiment
from framework.core.models import RunSpec


def test_list_device_preferences_contains_expected_values():
    assert list_device_preferences() == ["auto", "cpu", "cuda", "mps"]


def test_resolve_cpu_device():
    device, name = resolve_device_preference("cpu")
    assert str(device) == "cpu"
    assert name == "cpu"


def test_runs_record_resolved_device():
    spec = RunSpec(
        seed=0,
        n=10,
        r_star=2,
        m=50,
        factor_rank=3,
        init_scale=1e-2,
        lr=1e-3,
        steps=20,
        log_every=10,
        device_preference="cpu",
    )
    result = run_experiment(spec)
    assert result.summary["resolved_device"] == "cpu"
    assert result.metadata_dict()["device_preference"] == "cpu"
    assert result.metadata_dict()["resolved_device"] == "cpu"


def test_invalid_device_preference_rejected():
    with pytest.raises(ValueError, match="Unsupported device_preference"):
        RunSpec(
            seed=0,
            n=10,
            r_star=2,
            m=50,
            factor_rank=3,
            init_scale=1e-2,
            lr=1e-3,
            steps=20,
            log_every=10,
            device_preference="tpu",
        ).validate()
