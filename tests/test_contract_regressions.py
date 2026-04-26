import io
import json
import zipfile
from pathlib import Path

from framework.core.models import RunResult, RunSpec, RunStatus
from dataclasses import replace

from framework.services.analysis_records import load_analysis_records
from framework.services.compare_service import CompareService
from framework.services.experiment_protocols import ExperimentProtocol
from framework.services.export_service import LocalExportService
from framework.services.exports_local import (
    build_compare_bundle_zip_bytes,
    build_run_bundle_zip_bytes,
    build_sweep_bundle_zip_bytes,
    ensure_report_and_plots,
)
from framework.services.local_runs import config_diff
from framework.services.new_run import NewRunRequest, build_new_run_execution_plan, execute_new_run_plan
from framework.services.run_repository import LocalReadRunRepository, LocalRunRepository
from framework.services.sweep_analysis import build_sweep_analysis_from_execution_result
from framework.services.sweep_repository import LocalSweepRepository
from framework.services.sweeps import SweepRequest, build_sweep_execution_plan, execute_sweep_plan, save_sweep_metadata
from framework.storage.local import load_run, read_run_header, save_run


class _FakeExportService:
    def build_compare_bundle(self, *, run_dirs, include_run_reports: bool, include_artifacts: bool, filename: str = "compare_runs_bundle.zip", **kwargs):
        return type("Bundle", (), {
            "filename": filename,
            "content_type": "application/zip",
            "data": b"",
        })()



def _protocol(*, manipulated_factor: str, held_fixed_factors: tuple[str, ...]) -> ExperimentProtocol:
    return ExperimentProtocol(
        protocol_id=f"proto_{manipulated_factor}",
        title=f"{manipulated_factor} controlled comparison",
        hypothesis=f"Changing {manipulated_factor} changes the learned solution under fixed context.",
        manipulated_factor=manipulated_factor,
        held_fixed_factors=held_fixed_factors,
        primary_metrics=("final_objective", "final_rel_recon_error"),
        seed_policy="paired-2-seed",
    )



def _psd_request(*, init_scale: float = 1e-2, lr: float = 1e-3, protocol: ExperimentProtocol | None = None) -> NewRunRequest:
    return NewRunRequest(
        seed_mode="generated",
        start_seed=10,
        num_seeds=2,
        seed_step=1,
        n=18,
        n_cols=None,
        r_star=3,
        m=60,
        factor_rank=4,
        init_scale=init_scale,
        lr=lr,
        steps=20,
        log_every=5,
        problem_family="psd_sensing",
        data_source="synthetic",
        psd_regime="commuting",
        noncommutativity_level=0.0,
        optimizer="gd",
        execution_mode="accuracy",
        device_preference="auto",
        dtype_preference="auto",
        parameterization="factorized_psd",
        explicit_reg_type="none",
        explicit_reg_strength=0.0,
        stopping_mode="fixed_steps",
        grad_norm_tol=1e-10,
        objective_tol=1e-12,
        divergence_norm_threshold=1e6,
        protocol=protocol,
    )



def _fake_run_experiment(spec: RunSpec) -> RunResult:
    final_objective = float(spec.init_scale * 1000.0 + spec.lr * 100.0 + spec.seed * 1e-3)
    final_rel_recon_error = float(spec.init_scale * 100.0 + spec.seed * 1e-4)
    return RunResult(
        spec=spec,
        metrics={
            "step": [0, 10, 20],
            "objective": [final_objective * 2.0, final_objective * 1.25, final_objective],
            "rel_recon_error": [final_rel_recon_error * 2.0, final_rel_recon_error * 1.5, final_rel_recon_error],
            "gradient_norm": [1.0, 0.25, 0.05],
        },
        summary={
            "status": RunStatus.COMPLETED,
            "final_objective": final_objective,
            "final_rel_recon_error": final_rel_recon_error,
            "final_gradient_norm": 0.05,
            "converged": False,
            "stop_reason": "fixed_steps_complete",
        },
        artifacts={"weights": [1.0, 2.0]},
    )



def _with_seed_group_id(plan, seed_group_id: str):
    return replace(plan, seed_group_id=seed_group_id, specs=[replace(spec, seed_group_id=seed_group_id) for spec in plan.specs])


def _build_compare_service(runs_dir: Path) -> CompareService:
    read_repository = LocalReadRunRepository(runs_dir=runs_dir, load_run_fn=load_run, read_run_header_fn=read_run_header)
    return CompareService(read_repository=read_repository, config_diff_fn=config_diff, export_service=_FakeExportService())



def test_protocol_aware_compare_contract_survives_execution_round_trip(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    repository = LocalRunRepository(runs_dir=runs_dir, save_run_fn=save_run)
    protocol = _protocol(manipulated_factor="init_scale", held_fixed_factors=("lr", "steps"))

    reference_plan = build_new_run_execution_plan(_psd_request(init_scale=1e-2, protocol=protocol))
    candidate_plan = build_new_run_execution_plan(_psd_request(init_scale=2e-2, protocol=protocol))

    reference_result = execute_new_run_plan(reference_plan, run_experiment_fn=_fake_run_experiment, run_repository=repository)
    candidate_result = execute_new_run_plan(candidate_plan, run_experiment_fn=_fake_run_experiment, run_repository=repository)

    reference_by_seed = {item.seed: item.run_id for item in reference_result.items}
    candidate_by_seed = {item.seed: item.run_id for item in candidate_result.items}
    reference_run_id = reference_by_seed[10]
    candidate_run_id = candidate_by_seed[10]

    service = _build_compare_service(runs_dir)
    matched = service.matched_compare_result(
        reference_run_id=reference_run_id,
        varying_axis="init_scale",
        strict_exact_match=True,
        candidate_run_id=candidate_run_id,
    )

    assert matched is not None
    assert matched.audit.status == "valid_single_factor"
    assert matched.audit.seed_alignment == "matched"
    row = matched.candidates_df.loc[matched.candidates_df["run_id"] == candidate_run_id].iloc[0]
    assert row["audit_status"] == "valid_single_factor"
    assert row["audit_severity"] == "success"

    records = load_analysis_records(runs_dir)
    persisted = records.loc[records["run_id"].isin([reference_run_id, candidate_run_id])].set_index("run_id")
    assert persisted.loc[reference_run_id, "protocol_id"] == protocol.protocol_id
    assert persisted.loc[candidate_run_id, "protocol_manipulated_factor"] == "init_scale"
    assert persisted.loc[reference_run_id, "protocol_seed_policy"] == "paired-2-seed"
    assert persisted.loc[reference_run_id, "integrity_status"] == "valid"

    loaded = load_run(str(runs_dir / reference_run_id))
    assert int(loaded["metadata"]["runtime_snapshot"]["schema_version"]) >= 1
    assert loaded["integrity"]["status"] == "valid"



def test_paired_seed_group_compare_survives_execution_round_trip(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    repository = LocalRunRepository(runs_dir=runs_dir, save_run_fn=save_run)
    protocol = _protocol(manipulated_factor="init_scale", held_fixed_factors=("lr", "steps"))

    reference_plan = build_new_run_execution_plan(_psd_request(init_scale=1e-2, protocol=protocol))
    candidate_plan = _with_seed_group_id(
        build_new_run_execution_plan(_psd_request(init_scale=2e-2, protocol=protocol)),
        "seedgrp_candidate",
    )

    execute_new_run_plan(reference_plan, run_experiment_fn=_fake_run_experiment, run_repository=repository)
    execute_new_run_plan(candidate_plan, run_experiment_fn=_fake_run_experiment, run_repository=repository)

    service = _build_compare_service(runs_dir)
    paired = service.paired_seed_group_compare_result(
        reference_group_id=reference_plan.seed_group_id,
        varying_axis="init_scale",
        strict_exact_match=True,
        candidate_group_id=candidate_plan.seed_group_id,
    )

    assert paired is not None
    assert paired.audit.status == "valid_single_factor"
    assert paired.pairing_summary["paired_seed_count"] == 2
    assert paired.pairing_summary["complete"] is True
    assert set(paired.pairing_df["pairing_status"]) == {"paired"}
    summary = paired.delta_summary_df.set_index("metric")
    assert summary.loc["final_objective", "delta_direction"] == "positive"
    assert float(summary.loc["final_objective", "sign_consistency_ratio"]) == 1.0
    assert float(summary.loc["final_rel_recon_error", "mean_delta"]) > 0.0
    candidate_row = paired.candidates_df.loc[paired.candidates_df["seed_group_id"] == candidate_plan.seed_group_id].iloc[0]
    assert candidate_row["pairing_complete"] == True
    assert candidate_row["audit_status"] == "valid_single_factor"



def test_sweep_execution_export_preserves_protocol_and_provenance_metadata(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    sweeps_dir = tmp_path / "sweeps"
    repository = LocalRunRepository(runs_dir=runs_dir, save_run_fn=save_run)
    protocol = _protocol(manipulated_factor="lr", held_fixed_factors=("optimizer", "parameterization", "steps"))

    request = SweepRequest(
        kind="1D",
        preset="Sweep learning rate",
        base_request=_psd_request(protocol=protocol),
        fields=("lr",),
        values={"lr": [1e-4, 1e-3]},
        protocol=protocol,
    )
    plan = build_sweep_execution_plan(request)
    result = execute_sweep_plan(plan, run_experiment_fn=_fake_run_experiment, run_repository=repository)

    sweep_repository = LocalSweepRepository(
        sweeps_dir=sweeps_dir,
        read_repository=LocalReadRunRepository(runs_dir=runs_dir, load_run_fn=load_run, read_run_header_fn=read_run_header),
    )
    save_sweep_metadata(sweep_repository, result.metadata)
    analysis = build_sweep_analysis_from_execution_result(result, problem_family="psd_sensing", data_source="synthetic")

    assert analysis.contract["status"] == "controlled_single_factor"
    assert analysis.summary["expected_seed_count"] == 2
    assert analysis.summary["expected_cell_count"] == 2
    assert analysis.summary["seed_balance_status"] == "balanced"

    export_service = LocalExportService(
        ensure_report_and_plots_fn=ensure_report_and_plots,
        build_run_bundle_zip_bytes_fn=build_run_bundle_zip_bytes,
        build_compare_bundle_zip_bytes_fn=build_compare_bundle_zip_bytes,
        build_sweep_bundle_zip_bytes_fn=build_sweep_bundle_zip_bytes,
        load_run_fn=load_run,
    )
    bundle = export_service.build_sweep_bundle(
        analysis_view=analysis,
        repository=sweep_repository,
        include_run_reports=False,
        include_artifacts=False,
    )

    with zipfile.ZipFile(io.BytesIO(bundle.data), "r") as zf:
        meta = json.loads(zf.read(f"{plan.group_id}/meta.json").decode("utf-8"))
        summary_csv = zf.read(f"{plan.group_id}/summary.csv").decode("utf-8")

    assert meta["protocol_id"] == protocol.protocol_id
    assert meta["protocol_manipulated_factor"] == "lr"
    assert meta["expected_seed_count"] == 2
    assert meta["provenance_schema_version"] == 1
    assert meta["runtime_snapshot"]["python_version"] == meta["python_version"]
    assert "seed_balance_status" in summary_csv
    assert "expected_seed_count" in summary_csv
