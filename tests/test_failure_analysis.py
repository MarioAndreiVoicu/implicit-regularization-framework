import pandas as pd

from framework.core.models import RunStatus
from framework.services.failure_analysis import (
    analyze_run_instability,
    build_failure_table,
    instability_seed_group_table,
)


def test_analyze_run_instability_detects_stalled_run():
    config = {
        "problem_family": "matrix_completion",
        "data_source": "synthetic",
        "parameterization": "factorized_uv",
        "grad_norm_tol": 1e-10,
        "objective_tol": 1e-12,
    }
    summary = {
        "status": RunStatus.COMPLETED,
        "stop_reason": "max_steps",
        "converged": False,
        "final_gradient_norm": 1e-12,
        "final_update_norm": 1e-14,
        "final_objective_delta": 1e-14,
        "final_objective": 10.0,
        "final_rel_recon_error": 0.9999,
    }
    metrics = pd.DataFrame({
        "step": [0.0, 10.0],
        "objective": [10.0001, 10.0],
        "rel_recon_error": [1.0, 0.9999],
        "gradient_norm": [1e-12, 1e-12],
        "update_norm": [1e-14, 1e-14],
    })
    analysis = analyze_run_instability(config, summary, metrics)
    assert analysis["primary_issue"] == "stalled"
    assert "hit_cap" in analysis["issue_labels"]


def test_analyze_run_instability_detects_psd_violation():
    config = {
        "problem_family": "psd_sensing",
        "data_source": "synthetic",
        "parameterization": "symmetric_matrix",
        "grad_norm_tol": 1e-10,
        "objective_tol": 1e-12,
    }
    summary = {
        "status": RunStatus.COMPLETED,
        "stop_reason": "fixed_steps_complete",
        "converged": False,
        "final_gradient_norm": 1e-4,
        "final_update_norm": 1e-4,
        "final_objective_delta": 1e-4,
        "final_objective": 1.0,
        "final_rel_recon_error": 0.3,
        "final_min_eigenvalue_X": -0.2,
        "final_negative_eigen_mass": 0.2,
    }
    metrics = pd.DataFrame({"step": [0.0, 1.0], "objective": [2.0, 1.0], "rel_recon_error": [1.0, 0.3]})
    analysis = analyze_run_instability(config, summary, metrics)
    assert analysis["primary_issue"] == "psd_violation"


def test_build_failure_table_and_seed_group_summary():
    run_ok = {
        "run_id": "r1",
        "config": {"seed": 0, "seed_group_id": "sg1", "problem_family": "psd_sensing", "data_source": "synthetic", "optimizer": "gd", "parameterization": "factorized_psd", "lr": 0.1},
        "summary": {"status": RunStatus.COMPLETED, "stop_reason": "converged", "converged": True, "final_gradient_norm": 1e-12, "final_update_norm": 1e-12, "final_objective_delta": 1e-13, "final_objective": 0.1, "final_rel_recon_error": 0.1},
        "metrics": pd.DataFrame({"step": [0.0, 1.0], "objective": [1.0, 0.1], "rel_recon_error": [1.0, 0.1]}),
        "metadata": {"seed_group_id": "sg1"},
    }
    run_bad = {
        "run_id": "r2",
        "config": {"seed": 1, "seed_group_id": "sg1", "problem_family": "psd_sensing", "data_source": "synthetic", "optimizer": "gd", "parameterization": "factorized_psd", "lr": 0.1},
        "summary": {"status": RunStatus.DIVERGED, "stop_reason": "nonfinite_objective", "converged": False, "is_diverged": True, "final_gradient_norm": 1e9, "final_update_norm": 1e9, "final_objective_delta": 1.0, "final_objective": 1e9, "final_rel_recon_error": 1.0},
        "metrics": pd.DataFrame({"step": [0.0], "objective": [1.0], "rel_recon_error": [1.0]}),
        "metadata": {"seed_group_id": "sg1"},
    }
    table = build_failure_table([run_ok, run_bad])
    assert set(table["primary_issue"]) == {"ok", "diverged"}
    seed_table = instability_seed_group_table([run_ok, run_bad])
    assert seed_table.iloc[0]["has_problematic_seed"]
    assert "diverged:1" in seed_table.iloc[0]["issue_breakdown"]
