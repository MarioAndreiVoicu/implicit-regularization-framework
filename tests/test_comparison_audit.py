from framework.services.comparison_audit import audit_run_pair, comparison_audit_details_table
from framework.services.experiment_protocols import ExperimentProtocol, protocol_metadata


BASE_CFG = {
    "problem_family": "psd_sensing",
    "data_source": "synthetic",
    "measurement_type": "gaussian_symmetric",
    "n": 20,
    "r_star": 3,
    "m": 200,
    "factor_rank": 5,
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


def _protocol_meta(*, manipulated_factor: str = "optimizer", held_fixed: tuple[str, ...] = ("steps",), seed_policy: str = "paired-5-seed"):
    protocol = ExperimentProtocol(
        protocol_id="proto_compare",
        title="Controlled comparison",
        hypothesis="Optimizer change affects solution bias.",
        manipulated_factor=manipulated_factor,
        held_fixed_factors=held_fixed,
        primary_metrics=("final_rel_recon_error",),
        seed_policy=seed_policy,
    )
    return protocol_metadata(protocol)



def test_audit_run_pair_marks_valid_single_axis_change():
    candidate_cfg = {**BASE_CFG, "optimizer": "adam", "adam_beta1": 0.9}

    audit = audit_run_pair(BASE_CFG, candidate_cfg, "optimizer")

    assert audit.status == "valid_single_factor"
    assert audit.allowed_differences == ("adam_beta1", "optimizer")
    assert audit.unexpected_differences == ()
    assert audit.seed_alignment == "matched"
    assert audit.validity_label == "Valid single-factor comparison"



def test_audit_run_pair_marks_weakly_controlled_for_extra_changes():
    candidate_cfg = {**BASE_CFG, "optimizer": "adam", "adam_beta1": 0.9, "steps": 200}

    audit = audit_run_pair(BASE_CFG, candidate_cfg, "optimizer")

    assert audit.status == "weakly_controlled"
    assert audit.allowed_differences == ("adam_beta1", "optimizer")
    assert audit.unexpected_differences == ("steps",)
    assert audit.secondary_confounds == ("steps",)
    assert "Weak comparison" in audit.summary



def test_audit_run_pair_marks_invalid_context_for_core_mismatch():
    candidate_cfg = {**BASE_CFG, "optimizer": "adam", "data_source": "movielens_100k", "measurement_type": "observed_ratings"}

    audit = audit_run_pair(BASE_CFG, candidate_cfg, "optimizer")

    assert audit.status == "invalid_context"
    assert audit.core_mismatches == ("data_source", "measurement_type")



def test_audit_run_pair_uses_protocol_to_detect_held_fixed_violation():
    candidate_cfg = {**BASE_CFG, "optimizer": "adam", "adam_beta1": 0.9, "steps": 200}

    audit = audit_run_pair(
        BASE_CFG,
        candidate_cfg,
        "optimizer",
        reference_metadata=_protocol_meta(),
        candidate_metadata=_protocol_meta(),
    )

    assert audit.status == "protocol_conflict"
    assert audit.protocol_varying_axis == "optimizer"
    assert audit.held_fixed_violations == ("steps",)



def test_audit_run_pair_uses_protocol_to_detect_wrong_axis_and_seed_policy():
    candidate_cfg = {**BASE_CFG, "lr": 5e-3, "seed": 1}

    audit = audit_run_pair(
        BASE_CFG,
        candidate_cfg,
        "init_scale",
        reference_metadata=_protocol_meta(manipulated_factor="lr"),
        candidate_metadata=_protocol_meta(manipulated_factor="lr"),
    )

    assert audit.status == "protocol_conflict"
    assert audit.protocol_varying_axis == "learning_rate"
    assert any("selected varying axis 'init_scale'" in item for item in audit.protocol_conflicts)
    assert any("paired seed policy requires matched seeds" in item for item in audit.protocol_conflicts)



def test_comparison_audit_details_table_contains_richer_summary_fields():
    audit = audit_run_pair(BASE_CFG, {**BASE_CFG, "lr": 5e-3}, "init_scale")

    df = comparison_audit_details_table(audit)

    assert list(df["category"]) == [
        "validity",
        "severity",
        "varying axis",
        "protocol varying axis",
        "reference axis value",
        "candidate axis value",
        "allowed differences",
        "unexpected differences",
        "secondary confounds",
        "held-fixed violations",
        "protocol conflicts",
        "core mismatches",
        "seed alignment",
        "paired seed expected",
    ]


def test_audit_run_pair_supports_learning_rate_axis_and_protocol_alias():
    candidate_cfg = {**BASE_CFG, "lr": 5e-3}

    audit = audit_run_pair(
        BASE_CFG,
        candidate_cfg,
        "learning_rate",
        reference_metadata=_protocol_meta(manipulated_factor="lr", held_fixed=("steps",)),
        candidate_metadata=_protocol_meta(manipulated_factor="lr", held_fixed=("steps",)),
    )

    assert audit.status == "valid_single_factor"
    assert audit.allowed_differences == ("lr",)
    assert audit.protocol_varying_axis == "learning_rate"
    assert audit.unexpected_differences == ()




def test_audit_ignores_provenance_group_ids_for_learning_rate_change():
    reference_cfg = {**BASE_CFG, "seed_group_id": "seedgrp_ref", "group_id": "sweep_ref", "sweep_group_id": "sweep_a"}
    candidate_cfg = {**BASE_CFG, "lr": 5e-3, "seed_group_id": "seedgrp_candidate", "group_id": "sweep_candidate", "sweep_group_id": "sweep_b"}

    audit = audit_run_pair(reference_cfg, candidate_cfg, "learning_rate")

    assert audit.status == "valid_single_factor"
    assert audit.allowed_differences == ("lr",)
    assert "seed_group_id" not in audit.secondary_confounds
    assert "group_id" not in audit.secondary_confounds
    assert "sweep_group_id" not in audit.secondary_confounds
    assert audit.unexpected_differences == ()


def test_audit_run_pair_supports_training_budget_schedule_batch_and_seed_axes():
    assert audit_run_pair(BASE_CFG, {**BASE_CFG, "steps": 200}, "training_budget").status == "valid_single_factor"
    assert audit_run_pair(BASE_CFG, {**BASE_CFG, "lr_schedule_type": "cosine", "lr_min": 1e-5}, "learning_rate_schedule").status == "valid_single_factor"
    assert audit_run_pair(BASE_CFG, {**BASE_CFG, "batch_size": 16}, "batch_size").status == "valid_single_factor"

    seed_audit = audit_run_pair(BASE_CFG, {**BASE_CFG, "seed": 3}, "random_seed")
    assert seed_audit.status == "valid_single_factor"
    assert seed_audit.allowed_differences == ("seed",)
    assert seed_audit.seed_alignment == "different"
