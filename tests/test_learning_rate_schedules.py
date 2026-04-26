import math

import pytest

from framework.core.experiments import run_experiment
from framework.core.models import RunSpec
from framework.core.schedules import learning_rate_at_step, list_lr_schedule_types


def test_list_lr_schedule_types_contains_expected_values():
    assert list_lr_schedule_types() == ["constant", "step_decay", "cosine"]


def test_constant_schedule_keeps_lr_fixed():
    assert learning_rate_at_step(
        base_lr=0.1,
        schedule_type="constant",
        step=7,
        total_steps=100,
        decay_gamma=0.5,
        step_size=10,
        lr_min=0.0,
    ) == pytest.approx(0.1)


def test_step_decay_schedule_changes_at_boundaries():
    assert learning_rate_at_step(
        base_lr=0.1,
        schedule_type="step_decay",
        step=0,
        total_steps=100,
        decay_gamma=0.5,
        step_size=10,
        lr_min=0.0,
    ) == pytest.approx(0.1)
    assert learning_rate_at_step(
        base_lr=0.1,
        schedule_type="step_decay",
        step=10,
        total_steps=100,
        decay_gamma=0.5,
        step_size=10,
        lr_min=0.0,
    ) == pytest.approx(0.05)


def test_cosine_schedule_reaches_lr_min_at_end():
    assert learning_rate_at_step(
        base_lr=0.1,
        schedule_type="cosine",
        step=100,
        total_steps=100,
        decay_gamma=0.5,
        step_size=10,
        lr_min=0.01,
    ) == pytest.approx(0.01)


def test_runspec_rejects_invalid_schedule():
    spec = RunSpec(
        seed=0,
        n=20,
        r_star=3,
        m=200,
        factor_rank=5,
        init_scale=1e-2,
        lr=1e-3,
        steps=10,
        log_every=5,
        lr_schedule_type="weird",
    )
    with pytest.raises(ValueError, match="Unsupported lr_schedule_type"):
        spec.validate()


def test_run_experiment_logs_learning_rate_metric_for_step_decay():
    spec = RunSpec(
        seed=0,
        n=20,
        r_star=3,
        m=200,
        factor_rank=5,
        init_scale=1e-2,
        lr=1e-2,
        steps=20,
        log_every=5,
        problem_family="psd_sensing",
        optimizer="gd",
        lr_schedule_type="step_decay",
        lr_decay_gamma=0.5,
        lr_step_size=5,
        stopping_mode="fixed_steps",
    )
    result = run_experiment(spec)
    assert "lr" in result.metrics
    assert result.metrics["lr"][0] == pytest.approx(1e-2)
    assert result.metrics["lr"][1] == pytest.approx(5e-3)
    assert result.summary["final_lr"] == pytest.approx(result.metrics["lr"][-1])
