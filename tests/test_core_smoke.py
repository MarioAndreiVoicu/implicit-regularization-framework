from framework.core.experiments import run_experiment
from framework.core.models import RunSpec, RunStatus


def test_run_experiment_smoke():
    spec = RunSpec(
        seed=0,
        n=20,
        r_star=3,
        m=200,
        factor_rank=5,
        init_scale=1e-2,
        lr=1e-3,
        steps=50,
        log_every=10,
        noise_std=0.0,
        problem_family="psd_sensing",
        measurement_type="gaussian_symmetric",
        optimizer="gd",
        parameterization="factorized_psd",
        init_type="near_zero_gaussian",
        explicit_reg_type="none",
        explicit_reg_strength=0.0,
        track_eigs=False,
        num_eigs=5,
        run_name="smoke",
        notes="test",
        group_id=None,
    )

    result = run_experiment(spec)

    assert result.summary["status"] in {RunStatus.COMPLETED, RunStatus.DIVERGED}
    assert "data_loss" in result.metrics
    assert "loss" in result.metrics
    assert "relative_reconstruction_error" in result.metrics
    assert "rel_recon_error" in result.metrics
    assert "trace_X" in result.metrics
    assert "update_norm" in result.metrics
    assert "effective_rank" in result.metrics
    assert len(result.metrics["step"]) > 0
    assert result.summary["stopping_mode"] == spec.stopping_mode
    assert "stop_reason" in result.summary


def test_run_experiment_with_factor_l2_regularization_smoke():
    spec = RunSpec(
        seed=1,
        n=20,
        r_star=3,
        m=200,
        factor_rank=5,
        init_scale=1e-2,
        lr=1e-3,
        steps=20,
        log_every=10,
        noise_std=0.0,
        problem_family="psd_sensing",
        measurement_type="gaussian_symmetric",
        optimizer="gd",
        parameterization="factorized_psd",
        init_type="near_zero_gaussian",
        explicit_reg_type="factor_l2",
        explicit_reg_strength=1e-3,
        track_eigs=False,
        num_eigs=5,
    )

    result = run_experiment(spec)

    assert "objective" in result.metrics
    assert "explicit_reg_term" in result.metrics
    assert "stable_rank" in result.metrics
    assert result.summary["final_explicit_reg_term"] >= 0.0
    assert result.summary["final_objective"] >= result.summary["final_loss"]
