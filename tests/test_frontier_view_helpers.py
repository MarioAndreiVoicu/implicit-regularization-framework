import pandas as pd

from framework.services.frontier import frontier_metric_options, frontier_run_points, frontier_seed_group_points, pareto_frontier


def test_frontier_metric_options_include_psd_and_movielens_metrics():
    fit, complexity = frontier_metric_options([
        {"problem_family": "psd_sensing", "data_source": "synthetic"},
        {"problem_family": "matrix_completion", "data_source": "movielens_100k"},
    ])
    assert "final_rel_recon_error" in fit
    assert "final_val_rmse" in fit
    assert "final_trace_X" in complexity
    assert "final_stable_rank" in complexity


def test_frontier_run_points_extracts_summary_metrics():
    df = frontier_run_points([
        {
            "run_id": "r1",
            "config": {"problem_family": "psd_sensing", "optimizer": "gd", "parameterization": "factorized_psd", "explicit_reg_type": "none", "stopping_mode": "run_to_convergence", "init_scale": 0.01, "lr": 0.1},
            "summary": {"status": "completed", "final_objective": 1.2, "final_rel_recon_error": 0.3, "final_trace_X": 4.0},
            "metadata": {},
        }
    ])
    assert len(df) == 1
    assert df.iloc[0]["final_objective"] == 1.2
    assert df.iloc[0]["optimizer"] == "gd"


def test_pareto_frontier_keeps_nondominated_points():
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [4.0, 3.0, 3.5, 2.0],
        }
    )
    frontier = pareto_frontier(df, "x", "y")
    assert frontier[["x", "y"]].to_dict(orient="records") == [
        {"x": 1.0, "y": 4.0},
        {"x": 2.0, "y": 3.0},
        {"x": 4.0, "y": 2.0},
    ]


def test_frontier_seed_group_points_use_group_records():
    df = frontier_seed_group_points([
        {
            "run_id": "r1",
            "config": {"seed": 0, "seed_group_id": "sg1", "problem_family": "psd_sensing", "data_source": "synthetic", "optimizer": "gd", "parameterization": "factorized_psd", "explicit_reg_type": "none", "stopping_mode": "run_to_convergence", "init_scale": 0.01, "lr": 0.1},
            "summary": {"status": "completed", "final_objective": 1.0, "final_rel_recon_error": 0.2},
            "metrics": pd.DataFrame({"step": [0, 1], "objective": [2.0, 1.0], "rel_recon_error": [0.5, 0.2]}),
            "metadata": {"seed_group_id": "sg1"},
        },
        {
            "run_id": "r2",
            "config": {"seed": 1, "seed_group_id": "sg1", "problem_family": "psd_sensing", "data_source": "synthetic", "optimizer": "gd", "parameterization": "factorized_psd", "explicit_reg_type": "none", "stopping_mode": "run_to_convergence", "init_scale": 0.01, "lr": 0.1},
            "summary": {"status": "failed", "final_objective": 3.0, "final_rel_recon_error": 0.4},
            "metrics": pd.DataFrame({"step": [0, 1], "objective": [4.0, 3.0], "rel_recon_error": [0.6, 0.4]}),
            "metadata": {"seed_group_id": "sg1"},
        },
    ])
    assert len(df) == 1
    assert df.iloc[0]["seed_group_id"] == "sg1"
    assert df.iloc[0]["status"] == "mixed"
    assert float(df.iloc[0]["final_objective"]) == 2.0
