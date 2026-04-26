import pytest
import torch

from framework.core.models import RunSpec
from framework.core.regularization import explicit_reg_penalty, explicit_reg_types_for_parameterization, list_explicit_reg_types


def test_list_explicit_reg_types_contains_none_factor_l2_and_frobenius():
    assert list_explicit_reg_types() == ["factor_l2", "frobenius", "none"]


def test_explicit_reg_types_for_parameterization_filters_by_parameterization():
    assert explicit_reg_types_for_parameterization("factorized_psd") == ["none", "factor_l2"]
    assert explicit_reg_types_for_parameterization("symmetric_matrix") == ["none", "frobenius"]


def test_explicit_reg_penalty_is_zero_for_none():
    u = torch.ones((3, 2), dtype=torch.float64)
    val = explicit_reg_penalty(
        parameters={"U": u},
        parameterization="factorized_psd",
        explicit_reg_type="none",
        explicit_reg_strength=0.0,
    )
    assert float(val.item()) == 0.0


def test_explicit_reg_penalty_matches_factor_l2_definition():
    u = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    val = explicit_reg_penalty(
        parameters={"U": u},
        parameterization="factorized_psd",
        explicit_reg_type="factor_l2",
        explicit_reg_strength=0.5,
    )
    expected = 0.5 * 0.5 * float(torch.sum(u * u).item())
    assert float(val.item()) == pytest.approx(expected)


def test_runspec_rejects_nonzero_strength_for_none():
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
        explicit_reg_type="none",
        explicit_reg_strength=1e-3,
    )
    with pytest.raises(ValueError, match="explicit_reg_strength must be 0"):
        spec.validate()


def test_runspec_accepts_factor_l2_regularization():
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
        parameterization="factorized_psd",
        explicit_reg_type="factor_l2",
        explicit_reg_strength=1e-3,
    )
    spec.validate()


def test_explicit_reg_penalty_matches_frobenius_definition_for_direct_matrix():
    x = torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.float64)
    val = explicit_reg_penalty(
        parameters={"X": x},
        parameterization="direct_matrix",
        explicit_reg_type="frobenius",
        explicit_reg_strength=0.25,
    )
    expected = 0.5 * 0.25 * float(torch.sum(x * x).item())
    assert float(val.item()) == pytest.approx(expected)


def test_runspec_accepts_frobenius_regularization_for_direct_matrix():
    spec = RunSpec(
        seed=0,
        n=20,
        n_cols=18,
        r_star=3,
        m=120,
        factor_rank=5,
        init_scale=1e-2,
        lr=1e-3,
        steps=10,
        log_every=5,
        problem_family="matrix_completion",
        measurement_type="uniform_entries",
        parameterization="direct_matrix",
        explicit_reg_type="frobenius",
        explicit_reg_strength=1e-3,
    )
    spec.validate()
