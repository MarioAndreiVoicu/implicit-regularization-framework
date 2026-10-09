import io
import json
import zipfile
from pathlib import Path

from framework.core.models import RunResult, RunSpec, RunStatus
from framework.services.compare_service import CompareService
from framework.services.experiment_protocols import ExperimentProtocol, protocol_metadata
from framework.services.export_service import ExportBundle, LocalExportService
from framework.services.local_runs import config_diff
from framework.services.run_repository import LocalReadRunRepository
from framework.services.exports_local import build_compare_bundle_zip_bytes, build_run_bundle_zip_bytes, build_sweep_bundle_zip_bytes, ensure_report_and_plots
from framework.storage.local import load_run, read_run_header, save_run


class _FakeExportService:
    def build_compare_bundle(self, *, run_dirs, include_run_reports: bool, include_artifacts: bool, filename: str = "compare_runs_bundle.zip", **kwargs) -> ExportBundle:
        names = ",".join(sorted(Path(rd).name for rd in run_dirs))
        payload = f"reports={include_run_reports};artifacts={include_artifacts};runs={names}".encode()
        return ExportBundle(filename=filename, content_type="application/zip", data=payload)


def _make_result(*, seed: int, lr: float, init_scale: float = 1e-2, seed_group_id: str | None = None) -> RunResult:
    spec = RunSpec(
        seed=seed,
        n=10,
        r_star=2,
        m=30,
        factor_rank=3,
        init_scale=init_scale,
        lr=lr,
        steps=12,
        log_every=3,
        problem_family="psd_sensing",
        data_source="synthetic",
        measurement_type="gaussian_symmetric",
        psd_regime="commuting",
        noncommutativity_level=0.0,
        parameterization="factorized_psd",
        optimizer="gd",
        stopping_mode="fixed_steps",
        seed_group_id=seed_group_id,
    )
    return RunResult(
        spec=spec,
        metrics={"step": [0, 3, 6], "objective": [1.0, 0.6, 0.4], "rel_recon_error": [1.0, 0.5, 0.2], "gradient_norm": [1.0, 0.4, 0.1]},
        summary={
            "status": RunStatus.COMPLETED,
            "final_objective": 0.4,
            "final_rel_recon_error": 0.2,
            "final_gradient_norm": 0.1,
            "converged": False,
            "stop_reason": "fixed_steps_complete",
        },
        artifacts={},
    )


def test_compare_service_free_matched_and_seed_group_paths(tmp_path: Path):
    run1 = Path(save_run(_make_result(seed=1, lr=1e-3, init_scale=1e-2), runs_dir=str(tmp_path))).name
    run2 = Path(save_run(_make_result(seed=1, lr=1e-3, init_scale=2e-2), runs_dir=str(tmp_path))).name
    seed_group_id = "seedgrp_cmp"
    save_run(_make_result(seed=10, lr=1e-3, seed_group_id=seed_group_id), runs_dir=str(tmp_path))
    save_run(_make_result(seed=11, lr=1e-3, seed_group_id=seed_group_id), runs_dir=str(tmp_path))

    read_repository = LocalReadRunRepository(runs_dir=tmp_path, load_run_fn=load_run, read_run_header_fn=read_run_header)
    service = CompareService(read_repository=read_repository, config_diff_fn=config_diff, export_service=_FakeExportService())

    free = service.free_compare_result([run1, run2])
    assert list(free.summary_df["run"]) == [run1, run2]
    assert not free.diff_df.empty
    assert len(free.overlay_items) == 2

    bundle = service.build_free_compare_bundle(selected_run_ids=[run1, run2], include_run_reports=False, include_artifacts=True)
    assert bundle.filename == "compare_runs_bundle.zip"
    assert b"artifacts=True" in bundle.data

    matched = service.matched_compare_result(reference_run_id=run1, varying_axis="init_scale", strict_exact_match=True)
    assert matched is not None
    assert not matched.candidates_df.empty
    assert matched.candidate["run_id"] == run2
    assert not matched.final_metrics_df.empty

    seed_result = service.seed_aggregation_result(seed_group_id)
    assert seed_result is not None
    assert seed_result.group["seed_group_id"] == seed_group_id
    assert seed_result.agg_final_df is not None
    assert set(["sem", "cv", "range", "median"]).issubset(seed_result.agg_final_df.columns)
    assert seed_result.run_rows_df.shape[0] == 2


def test_compare_service_exposes_richer_audit_columns_and_protocol_conflict(tmp_path: Path):
    protocol = ExperimentProtocol(
        protocol_id="proto_lr",
        title="Learning-rate comparison",
        hypothesis="LR affects bias under fixed setup.",
        manipulated_factor="lr",
        held_fixed_factors=("steps",),
        primary_metrics=("final_rel_recon_error",),
        seed_policy="paired-2-seed",
    )
    metadata = protocol_metadata(protocol)

    run_ref = Path(save_run(_make_result(seed=1, lr=1e-3), runs_dir=str(tmp_path), metadata_overrides=metadata)).name
    run_candidate = Path(save_run(_make_result(seed=1, lr=1e-3, init_scale=2e-2), runs_dir=str(tmp_path), metadata_overrides=metadata)).name

    read_repository = LocalReadRunRepository(runs_dir=tmp_path, load_run_fn=load_run, read_run_header_fn=read_run_header)
    service = CompareService(read_repository=read_repository, config_diff_fn=config_diff, export_service=_FakeExportService())

    matched = service.matched_compare_result(reference_run_id=run_ref, varying_axis="init_scale", strict_exact_match=False, candidate_run_id=run_candidate)

    assert matched is not None
    assert matched.audit.status == "protocol_conflict"
    assert set(["audit_label", "audit_severity", "audit_seed_alignment", "audit_confound_count"]).issubset(matched.candidates_df.columns)
    row = matched.candidates_df.loc[matched.candidates_df["run_id"] == run_candidate].iloc[0]
    assert row["audit_status"] == "protocol_conflict"
    assert row["audit_severity"] == "error"


def test_compare_service_builds_paired_seed_group_deltas_and_missing_seed_summary(tmp_path: Path):
    protocol = ExperimentProtocol(
        protocol_id="proto_init",
        title="Init-scale seed group comparison",
        hypothesis="Init scale changes bias under fixed setup.",
        manipulated_factor="init_scale",
        held_fixed_factors=("lr", "steps"),
        primary_metrics=("final_rel_recon_error",),
        seed_policy="paired-2-seed",
    )
    metadata = protocol_metadata(protocol)

    save_run(_make_result(seed=10, lr=1e-3, init_scale=1e-2, seed_group_id="sg_ref"), runs_dir=str(tmp_path), metadata_overrides=metadata)
    save_run(_make_result(seed=11, lr=1e-3, init_scale=1e-2, seed_group_id="sg_ref"), runs_dir=str(tmp_path), metadata_overrides=metadata)
    save_run(_make_result(seed=10, lr=1e-3, init_scale=2e-2, seed_group_id="sg_cand"), runs_dir=str(tmp_path), metadata_overrides=metadata)
    save_run(_make_result(seed=12, lr=1e-3, init_scale=2e-2, seed_group_id="sg_cand"), runs_dir=str(tmp_path), metadata_overrides=metadata)

    read_repository = LocalReadRunRepository(runs_dir=tmp_path, load_run_fn=load_run, read_run_header_fn=read_run_header)
    service = CompareService(read_repository=read_repository, config_diff_fn=config_diff, export_service=_FakeExportService())

    result = service.paired_seed_group_compare_result(
        reference_group_id="sg_ref",
        varying_axis="init_scale",
        strict_exact_match=True,
        candidate_group_id="sg_cand",
    )

    assert result is not None
    assert result.candidate_group["seed_group_id"] == "sg_cand"
    assert result.audit.status == "valid_single_factor"
    assert result.pairing_summary["paired_seed_count"] == 1
    assert result.pairing_summary["reference_only_seed_count"] == 1
    assert result.pairing_summary["candidate_only_seed_count"] == 1
    assert list(result.pairing_df["pairing_status"]) == ["paired", "candidate_missing", "reference_missing"]
    assert set(["seed", "metric", "delta"]).issubset(result.per_seed_delta_df.columns)
    assert "final_loss" not in set(result.delta_summary_df["metric"])
    summary_row = result.delta_summary_df.loc[result.delta_summary_df["metric"] == "final_objective"].iloc[0]
    assert int(summary_row["paired_seed_count"]) == 1
    assert float(summary_row["mean_abs_delta"]) == 0.0
    assert float(summary_row["zero_delta_fraction"]) == 1.0
    assert summary_row["delta_direction"] == "flat"
    row = result.candidates_df.loc[result.candidates_df["seed_group_id"] == "sg_cand"].iloc[0]
    assert row["paired_seed_count"] == 1
    assert row["pairing_complete"] == False


def test_compare_service_builds_matched_and_paired_compare_export_bundles(tmp_path: Path):
    protocol = ExperimentProtocol(
        protocol_id="proto_init",
        title="Init-scale seed group comparison",
        hypothesis="Init scale changes bias under fixed setup.",
        manipulated_factor="init_scale",
        held_fixed_factors=("lr", "steps"),
        primary_metrics=("final_rel_recon_error",),
        seed_policy="paired-2-seed",
    )
    metadata = protocol_metadata(protocol)

    run_ref = Path(save_run(_make_result(seed=10, lr=1e-3, init_scale=1e-2, seed_group_id="sg_ref"), runs_dir=str(tmp_path), metadata_overrides=metadata)).name
    save_run(_make_result(seed=11, lr=1e-3, init_scale=1e-2, seed_group_id="sg_ref"), runs_dir=str(tmp_path), metadata_overrides=metadata)
    run_cand = Path(save_run(_make_result(seed=10, lr=1e-3, init_scale=2e-2, seed_group_id="sg_cand"), runs_dir=str(tmp_path), metadata_overrides=metadata)).name
    save_run(_make_result(seed=11, lr=1e-3, init_scale=2e-2, seed_group_id="sg_cand"), runs_dir=str(tmp_path), metadata_overrides=metadata)

    read_repository = LocalReadRunRepository(runs_dir=tmp_path, load_run_fn=load_run, read_run_header_fn=read_run_header)
    export_service = LocalExportService(
        ensure_report_and_plots_fn=ensure_report_and_plots,
        build_run_bundle_zip_bytes_fn=build_run_bundle_zip_bytes,
        build_compare_bundle_zip_bytes_fn=build_compare_bundle_zip_bytes,
        build_sweep_bundle_zip_bytes_fn=build_sweep_bundle_zip_bytes,
        load_run_fn=load_run,
    )
    service = CompareService(read_repository=read_repository, config_diff_fn=config_diff, export_service=export_service)

    matched = service.matched_compare_result(reference_run_id=run_ref, varying_axis="init_scale", strict_exact_match=True, candidate_run_id=run_cand)
    assert matched is not None
    matched_bundle = service.build_matched_compare_bundle(result=matched, include_run_reports=False, include_artifacts=False)
    with zipfile.ZipFile(io.BytesIO(matched_bundle.data), "r") as zf:
        matched_summary = json.loads(zf.read("compare/summary.json").decode("utf-8"))
        matched_metrics = zf.read("compare/final_metrics.csv").decode("utf-8")
    assert matched_summary["mode"] == "matched_compare"
    assert matched_summary["audit"]["status"] in {"valid_single_factor", "weakly_controlled"}
    assert "final_objective" in matched_metrics

    paired = service.paired_seed_group_compare_result(reference_group_id="sg_ref", varying_axis="init_scale", strict_exact_match=True, candidate_group_id="sg_cand")
    assert paired is not None
    paired_bundle = service.build_paired_seed_group_compare_bundle(result=paired, include_run_reports=False, include_artifacts=False)
    with zipfile.ZipFile(io.BytesIO(paired_bundle.data), "r") as zf:
        paired_summary = json.loads(zf.read("compare/summary.json").decode("utf-8"))
        delta_summary = zf.read("compare/delta_summary.csv").decode("utf-8")
        manifest = json.loads(zf.read("traceability_manifest.json").decode("utf-8"))
    assert paired_summary["mode"] == "paired_seed_group_compare"
    assert paired_summary["pairing_summary"]["paired_seed_count"] == 2
    assert "sign_consistency_ratio" in delta_summary
    assert manifest["context"]["compare_mode"] == "paired_seed_group_compare"
