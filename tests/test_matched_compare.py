import pandas as pd

from framework.services.local_runs import (
    config_diff,
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




def test_config_diffs_ignore_provenance_group_ids():
    reference_cfg = {**BASE_CFG, "seed_group_id": "seedgrp_ref", "group_id": "sweep_ref", "sweep_group_id": "sweep_a"}
    candidate_cfg = {**BASE_CFG, "lr": 0.2, "seed_group_id": "seedgrp_candidate", "group_id": "sweep_candidate", "sweep_group_id": "sweep_b"}

    free_diff = config_diff([reference_cfg, candidate_cfg])
    matched_diff = matched_config_diff(reference_cfg, candidate_cfg, "learning_rate")

    assert set(free_diff["field"]) == {"lr"}
    assert set(matched_diff["field"]) == {"lr"}


def test_match_axes_split_learning_rate_schedule_budget_seed_and_batch_size():
    lr_diff = matched_config_diff(BASE_CFG, {**BASE_CFG, "lr": 0.2}, "learning_rate")
    lr_allowed = {row["field"]: row["allowed_change"] for _, row in lr_diff.iterrows()}
    assert lr_allowed["lr"] is True

    schedule_diff = matched_config_diff(BASE_CFG, {**BASE_CFG, "lr_schedule_type": "cosine", "lr_min": 1e-5}, "learning_rate_schedule")
    schedule_allowed = {row["field"]: row["allowed_change"] for _, row in schedule_diff.iterrows()}
    assert schedule_allowed["lr_schedule_type"] is True
    assert schedule_allowed["lr_min"] is True

    budget_diff = matched_config_diff(BASE_CFG, {**BASE_CFG, "steps": 250}, "training_budget")
    budget_allowed = {row["field"]: row["allowed_change"] for _, row in budget_diff.iterrows()}
    assert budget_allowed["steps"] is True

    seed_diff = matched_config_diff(BASE_CFG, {**BASE_CFG, "seed": 99}, "random_seed")
    seed_allowed = {row["field"]: row["allowed_change"] for _, row in seed_diff.iterrows()}
    assert seed_allowed["seed"] is True

    batch_diff = matched_config_diff(BASE_CFG, {**BASE_CFG, "batch_size": 32}, "batch_size")
    batch_allowed = {row["field"]: row["allowed_change"] for _, row in batch_diff.iterrows()}
    assert batch_allowed["batch_size"] is True
