import io
import json
import zipfile
from pathlib import Path

import pandas as pd

from framework.core.models import RunResult, RunSpec, RunStatus
from framework.services.export_service import LocalExportService
from framework.services.exports_local import (
    build_compare_bundle_zip_bytes,
    build_run_bundle_zip_bytes,
    build_sweep_bundle_zip_bytes,
    ensure_report_and_plots,
)
from framework.services.sweep_analysis import build_sweep_analysis_view
from framework.services.run_repository import LocalReadRunRepository
from framework.services.sweep_repository import LocalSweepRepository
from framework.storage.local import load_run, read_run_header, save_run


def _make_result(*, group_id: str | None = None) -> RunResult:
    spec = RunSpec(
        seed=3,
        n=10,
        r_star=2,
        m=30,
        factor_rank=3,
        init_scale=1e-2,
        lr=1e-3,
        steps=10,
        log_every=5,
        problem_family="psd_sensing",
        data_source="synthetic",
        dataset_version="synthetic_default_v1",
        split_version="synthetic_default_split_v1",
        preprocessing_version="none",
        measurement_type="gaussian_symmetric",
        group_id=group_id,
    )
    return RunResult(
        spec=spec,
        metrics={"step": [0, 5, 10], "loss": [2.0, 1.2, 0.8], "rel_recon_error": [1.0, 0.6, 0.3], "trace_X": [0.1, 0.2, 0.3]},
        summary={
            "status": RunStatus.COMPLETED,
            "final_objective": 0.8,
            "final_rel_recon_error": 0.3,
            "converged": True,
            "stop_reason": "objective_tol",
        },
        artifacts={"u": [1.0, 2.0]},
    )


def _make_service() -> LocalExportService:
    return LocalExportService(
        ensure_report_and_plots_fn=ensure_report_and_plots,
        build_run_bundle_zip_bytes_fn=build_run_bundle_zip_bytes,
        build_compare_bundle_zip_bytes_fn=build_compare_bundle_zip_bytes,
        build_sweep_bundle_zip_bytes_fn=build_sweep_bundle_zip_bytes,
        load_run_fn=load_run,
    )


def test_local_export_service_builds_run_bundle(tmp_path: Path):
    run_dir = Path(save_run(_make_result(), runs_dir=str(tmp_path)))
    service = _make_service()

    bundle = service.build_run_bundle(run_dir=run_dir, run_id=run_dir.name, include_artifacts=True, include_report=True)
    assert bundle.filename == f"{run_dir.name}.zip"
    assert bundle.content_type == "application/zip"
    assert len(bundle.data) > 0

    with zipfile.ZipFile(io.BytesIO(bundle.data), "r") as zf:
        manifest = json.loads(zf.read("traceability_manifest.json").decode("utf-8"))
    assert manifest["run_ids"] == [run_dir.name]
    assert manifest["context"]["export_kind"] == "run"


def test_local_export_service_builds_compare_bundle_with_summary_artifacts(tmp_path: Path):
    run_dir_a = Path(save_run(_make_result(), runs_dir=str(tmp_path / "runs")))
    run_dir_b = Path(save_run(_make_result(group_id="g2"), runs_dir=str(tmp_path / "runs")))
    service = _make_service()

    bundle = service.build_compare_bundle(
        run_dirs=[run_dir_a, run_dir_b],
        include_run_reports=False,
        include_artifacts=False,
        compare_summary={"mode": "matched_compare", "reference_run_id": run_dir_a.name, "candidate_run_id": run_dir_b.name},
        compare_tables={
            "summary": pd.DataFrame([{"run": run_dir_a.name}, {"run": run_dir_b.name}]),
            "final_metrics": pd.DataFrame([{"metric": "final_rel_recon_error", "reference": 0.3, "candidate": 0.3}]),
        },
        context={"compare_mode": "matched_compare"},
    )
    assert bundle.filename == "compare_runs_bundle.zip"
    assert bundle.content_type == "application/zip"

    with zipfile.ZipFile(io.BytesIO(bundle.data), "r") as zf:
        manifest = json.loads(zf.read("traceability_manifest.json").decode("utf-8"))
        summary = json.loads(zf.read("compare/summary.json").decode("utf-8"))
        final_metrics_csv = zf.read("compare/final_metrics.csv").decode("utf-8")

    assert manifest["context"]["compare_mode"] == "matched_compare"
    assert set(manifest["run_ids"]) == {run_dir_a.name, run_dir_b.name}
    assert summary["mode"] == "matched_compare"
    assert "final_rel_recon_error" in final_metrics_csv


def test_local_export_service_builds_sweep_bundle(tmp_path: Path):
    group_id = "sweep1d_lr_test"
    run_dir = Path(save_run(_make_result(group_id=group_id), runs_dir=str(tmp_path / 'runs')))
    sweeps_dir = tmp_path / 'sweeps'

    read_repository = LocalReadRunRepository(runs_dir=tmp_path / 'runs', load_run_fn=load_run, read_run_header_fn=read_run_header)
    repository = LocalSweepRepository(
        sweeps_dir=sweeps_dir,
        read_repository=read_repository,
    )
    repository.save_metadata({
        "group_id": group_id,
        "kind": "1D",
        "fields": ["lr"],
        "preset": "Sweep learning rate",
        "base_config": {"problem_family": "psd_sensing", "data_source": "synthetic"},
        "seeds": [3],
        "protocol_id": "proto_lr",
        "protocol_manipulated_factor": "lr",
    })

    raw_df = pd.DataFrame([
        {"group_id": group_id, "run": run_dir.name, "seed": 3, "lr": 1e-3, "final_rel_recon_error": 0.3, "loss": 0.8}
    ])
    analysis_view = build_sweep_analysis_view(
        group_id=group_id,
        metadata=repository.load_metadata(group_id) or {},
        raw_df=raw_df,
        metric_options={"final_rel_recon_error": "Final rel recon error", "loss": "Loss"},
    )
    service = _make_service()

    bundle = service.build_sweep_bundle(
        analysis_view=analysis_view,
        repository=repository,
        include_run_reports=False,
        include_artifacts=True,
    )
    assert bundle.filename == f"{group_id}.zip"
    assert bundle.content_type == "application/zip"
    assert len(bundle.data) > 0

    with zipfile.ZipFile(io.BytesIO(bundle.data), "r") as zf:
        contract = json.loads(zf.read(f"{group_id}/contract.json").decode("utf-8"))
        summary = json.loads(zf.read(f"{group_id}/summary.json").decode("utf-8"))
        warnings = json.loads(zf.read(f"{group_id}/confound_warnings.json").decode("utf-8"))
        manifest = json.loads(zf.read(f"{group_id}/traceability_manifest.json").decode("utf-8"))

    assert contract["status"] == "controlled_single_factor"
    assert summary["group_id"] == group_id
    assert isinstance(warnings["warnings"], list)
    assert manifest["context"]["protocol_id"] == "proto_lr"


def test_local_export_service_includes_all_existing_derived_outputs(tmp_path: Path):
    run_dir = Path(save_run(_make_result(), runs_dir=str(tmp_path)))
    (run_dir / "plots" / "custom_metric.png").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots" / "custom_metric.png").write_bytes(b"png-bytes")
    (run_dir / "report" / "assets" / "details.json").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "report" / "assets" / "details.json").write_text('{"ok": true}', encoding="utf-8")
    service = _make_service()

    bundle = service.build_run_bundle(run_dir=run_dir, run_id=run_dir.name, include_artifacts=False, include_report=True)

    with zipfile.ZipFile(io.BytesIO(bundle.data), "r") as zf:
        names = set(zf.namelist())

    assert f"{run_dir.name}/plots/custom_metric.png" in names
    assert f"{run_dir.name}/report/assets/details.json" in names


def test_local_export_service_deduplicates_compare_run_dirs(tmp_path: Path):
    run_dir = Path(save_run(_make_result(), runs_dir=str(tmp_path / "runs")))
    service = _make_service()

    bundle = service.build_compare_bundle(
        run_dirs=[run_dir, run_dir],
        include_run_reports=False,
        include_artifacts=False,
    )

    with zipfile.ZipFile(io.BytesIO(bundle.data), "r") as zf:
        manifest = json.loads(zf.read("traceability_manifest.json").decode("utf-8"))
        config_entries = [name for name in zf.namelist() if name.endswith("/config.json")]

    assert manifest["run_ids"] == [run_dir.name]
    assert config_entries == [f"{run_dir.name}/config.json"]
