import pytest

from framework.core.experiments import (
    list_optimizers,
    list_problem_families,
    list_supported_runner_keys,
    run_experiment,
)
from framework.core.models import RunSpec


def test_registries_expose_optimizer_set():
    assert "psd_sensing" in list_problem_families()
    for name in ["gd", "adam", "sgd", "momentum", "coordinate_descent", "preconditioned_gd"]:
        assert name in list_optimizers()
        assert ("psd_sensing", name) in list_supported_runner_keys()


def test_runspec_rejects_unknown_optimizer():
    with pytest.raises(ValueError, match="Unsupported optimizer"):
        RunSpec(
            seed=0,
            n=20,
            r_star=3,
            m=200,
            factor_rank=5,
            init_scale=1e-2,
            lr=1e-3,
            steps=10,
            log_every=5,
            problem_family="psd_sensing",
            measurement_type="gaussian_symmetric",
            optimizer="not_an_optimizer",
        ).validate()


def test_dispatch_runs_adam():
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
        problem_family="psd_sensing",
        measurement_type="gaussian_symmetric",
        optimizer="adam",
    )
    result = run_experiment(spec)
    assert result.summary["status"] in {"completed", "diverged"}
