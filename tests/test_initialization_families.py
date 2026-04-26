import pytest
import torch

from framework.core.initialization import initialize_factor, list_init_types
from framework.core.models import RunSpec


def test_list_init_types_contains_theory_baseline_only():
    assert list_init_types() == ["near_zero_gaussian"]


def test_initialize_factor_is_reproducible_for_theory_baseline():
    u1 = initialize_factor(
        seed=123,
        n=12,
        factor_rank=4,
        init_scale=1e-2,
        init_type="near_zero_gaussian",
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    u2 = initialize_factor(
        seed=123,
        n=12,
        factor_rank=4,
        init_scale=1e-2,
        init_type="near_zero_gaussian",
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    assert torch.allclose(u1, u2)


def test_initialize_factor_rejects_unsupported_type():
    with pytest.raises(ValueError, match="Unsupported init_type"):
        initialize_factor(
            seed=123,
            n=12,
            factor_rank=4,
            init_scale=1e-2,
            init_type="orthogonal",
            device=torch.device("cpu"),
            dtype=torch.float64,
        )


def test_runspec_validation_rejects_unsupported_init_type():
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
        init_type="orthogonal",
    )

    with pytest.raises(ValueError, match="Unsupported init_type"):
        spec.validate()
