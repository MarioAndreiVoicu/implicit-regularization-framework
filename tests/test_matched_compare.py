import pandas as pd

from framework.services.local_runs import (
    build_match_key,
    find_matched_runs,
    matched_config_diff,
    matched_final_metrics_table,
)


def _run(run_id: str, **cfg):
    return {
        "run_id": run_id,
        "config": cfg,
        "summary": {
            "status": "completed",
            "stop_reason": "converged",
            "final_objective": cfg.get("final_objective", 1.0),
            "final_rel_recon_error": cfg.get("final_rel_recon_error", 0.1),
            "final_trace_X": cfg.get("final_trace_X", 2.0),
            "final_effective_rank": cfg.get("final_effective_rank", 1.5),
            "final_stable_rank": cfg.get("final_stable_rank", 1.2),
            "final_gradient_norm": cfg.get("final_gradient_norm", 1e-6),
        },
        "metrics": pd.DataFrame({"step": [0, 1], "objective": [1.0, 0.5]}),
    }


BASE_CFG = {
    "problem_family": "psd_sensing",
    "n": 20,
    "r_star": 3,
    "m": 200,
    "factor_rank": 5,
    "measurement_type": "gaussian_symmetric",
    "noise_std": 0.0,
    "seed": 0,
    "init_type": "near_zero_gaussian",
    "init_scale": 1e-3,
    "optimizer": "gd",
    "parameterization": "factorized_psd",
    "explicit_reg_type": "none",
    "explicit_reg_strength": 0.0,
    "stopping_mode": "run_to_convergence",
    "grad_norm_tol": 1e-10,
    "objective_tol": 1e-12,
    "divergence_norm_threshold": 1e6,
    "lr": 1e-2,
    "steps": 100,
}


def test_build_match_key_excludes_varying_axis_fields():
    key = dict(build_match_key(BASE_CFG, "optimizer"))
    assert "optimizer" not in key
    assert "seed" in key
    assert "n" in key



def test_find_matched_runs_keeps_only_one_axis_changes():
    reference = _run("ref", **BASE_CFG)
    good = _run("adam", **{**BASE_CFG, "optimizer": "adam", "adam_beta1": 0.9})
    bad = _run("bad_seed", **{**BASE_CFG, "optimizer": "adam", "seed": 2})

    ref, matches = find_matched_runs([reference, good, bad], "ref", "optimizer", strict_exact_match=True)

    assert ref is not None
    assert [item["run_id"] for item in matches] == ["adam"]



def test_matched_config_diff_marks_allowed_changes():
    candidate_cfg = {**BASE_CFG, "optimizer": "adam", "seed": 3}
    diff = matched_config_diff(BASE_CFG, candidate_cfg, "optimizer")

    allowed = {row["field"]: row["allowed_change"] for _, row in diff.iterrows()}
    assert allowed["optimizer"] is True
    assert allowed["seed"] is False



def test_matched_final_metrics_table_has_deltas():
    ref_summary = {
        "final_objective": 1.0,
        "final_rel_recon_error": 0.2,
    }
    cand_summary = {
        "final_objective": 0.5,
        "final_rel_recon_error": 0.1,
    }
    df = matched_final_metrics_table(ref_summary, cand_summary, metrics=["final_objective", "final_rel_recon_error"])
    assert list(df["metric"]) == ["final_objective", "final_rel_recon_error"]
    assert float(df.iloc[0]["delta"]) == -0.5
