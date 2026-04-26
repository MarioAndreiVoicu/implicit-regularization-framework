from pathlib import Path

from framework.core.models import RunResult, RunSpec, RunStatus
from framework.services.analysis_records import build_run_analysis_record, load_analysis_items, load_analysis_records
from framework.services.run_repository import LocalReadRunRepository
from framework.storage.local import load_run, read_run_header, save_run


def _make_result(seed: int = 0) -> RunResult:
    spec = RunSpec(
        seed=seed,
        n=12,
        r_star=3,
        m=50,
        factor_rank=4,
        init_scale=1e-2,
        lr=1e-3,
        steps=20,
        log_every=5,
        problem_family="matrix_completion",
        data_source="movielens_100k",
        dataset_version="movielens_100k_v1",
        split_version="split_v2",
        preprocessing_version="prep_v3",
        measurement_type="observed_ratings",
        sampling_regime="observed",
        early_stopping_enabled=True,
        parameterization="factorized_uv",
        run_name="recorded",
        notes="analysis-record-test",
        seed_group_id="seedgrp_test",
    )
    return RunResult(
        spec=spec,
        metrics={"step": [0, 5, 10], "objective": [1.0, 0.5, 0.25], "val_rmse": [1.2, 1.0, 0.9]},
        summary={
            "status": RunStatus.COMPLETED,
            "final_objective": 0.25,
            "final_val_rmse": 0.9,
            "resolved_device": "cpu",
            "resolved_dtype": "float64",
            "converged": True,
            "stop_reason": "objective_tol",
        },
        artifacts={"weights": [1.0, 2.0]},
    )


def test_build_run_analysis_record_prefers_metadata_for_provenance():
    record = build_run_analysis_record(
        run_id="run_1",
        config={"data_source": "synthetic", "dataset_version": "cfg_v1", "split_version": "cfg_split", "preprocessing_version": "cfg_prep"},
        summary={"status": RunStatus.COMPLETED},
        metadata={"data_source": "movielens_100k", "dataset_version": "meta_v2", "split_version": "meta_split", "preprocessing_version": "meta_prep"},
        artifact_manifest={"count": 3},
    )
    assert record["run_id"] == "run_1"
    assert record["data_source"] == "movielens_100k"
    assert record["dataset_version"] == "meta_v2"
    assert record["split_version"] == "meta_split"
    assert record["preprocessing_version"] == "meta_prep"
    assert record["artifact_count"] == 3


def test_load_analysis_records_and_items_round_trip(tmp_path: Path):
    run_path = save_run(_make_result(seed=7), runs_dir=str(tmp_path))
    records_df = load_analysis_records(tmp_path)

    assert list(records_df["run_id"]) == [Path(run_path).name]
    row = records_df.iloc[0].to_dict()
    assert row["dataset_version"] == "movielens_100k_v1"
    assert row["split_version"] == "split_v2"
    assert row["preprocessing_version"] == "prep_v3"
    assert row["status"] == RunStatus.COMPLETED
    assert row["artifact_count"] == 1
    assert row["metrics_rows"] == 3

    read_repository = LocalReadRunRepository(runs_dir=tmp_path, load_run_fn=load_run, read_run_header_fn=read_run_header)

    items = load_analysis_items(read_repository, include_metrics=True)
    assert len(items) == 1
    item = items[0]
    assert item["record"]["run_id"] == Path(run_path).name
    assert item["record"]["final_val_rmse"] == 0.9
    assert list(item["metrics"]["step"]) == [0, 5, 10]


def test_build_run_analysis_record_exposes_protocol_fields():
    record = build_run_analysis_record(
        run_id="run_proto",
        config={"data_source": "synthetic"},
        summary={"status": RunStatus.COMPLETED},
        metadata={
            "protocol_id": "proto_compare",
            "protocol_title": "Compare protocol",
            "protocol_manipulated_factor": "lr",
            "protocol_held_fixed_factors": ["steps", "seed"],
            "protocol_primary_metrics": ["final_rel_recon_error"],
            "protocol_seed_policy": "paired-5-seed",
        },
    )

    assert record["protocol_id"] == "proto_compare"
    assert record["protocol_title"] == "Compare protocol"
    assert record["protocol_manipulated_factor"] == "lr"
    assert record["protocol_held_fixed_factors"] == ("steps", "seed")
    assert record["protocol_primary_metrics"] == ("final_rel_recon_error",)
    assert record["protocol_seed_policy"] == "paired-5-seed"
