from __future__ import annotations

from pathlib import Path

from framework.core.models import RunSpec
from framework.services.local_runs import (
    aggregate_seed_group_final_metrics,
    aggregate_seed_group_trajectories,
    group_runs_by_seed,
)
from framework.storage.local import load_run
from tests.golden_helpers import (
    assert_loaded_contract,
    assert_run_folder_complete,
    run_and_save,
    write_fake_movielens,
)


def test_golden_psd_baseline_end_to_end(tmp_path: Path) -> None:
    spec = RunSpec(
        seed=0,
        n=20,
        r_star=3,
        m=200,
        factor_rank=5,
        init_scale=1e-2,
        lr=1e-3,
        steps=200,
        log_every=20,
        problem_family="psd_sensing",
        measurement_type="gaussian_symmetric",
        optimizer="gd",
        parameterization="factorized_psd",
        explicit_reg_type="none",
    )

    _, folder, loaded = run_and_save(spec, tmp_path / "runs")
    assert_run_folder_complete(folder)
    assert_loaded_contract(
        loaded,
        problem_family="psd_sensing",
        data_source="synthetic",
        required_summary_keys=["final_objective", "final_rel_recon_error", "resolved_device", "resolved_dtype"],
        required_metric_keys=["objective", "rel_recon_error", "trace_X", "gradient_norm", "stable_rank"],
    )



def test_golden_psd_nondefault_optimizer_end_to_end(tmp_path: Path) -> None:
    spec = RunSpec(
        seed=1,
        n=20,
        r_star=3,
        m=200,
        factor_rank=5,
        init_scale=1e-2,
        lr=1e-2,
        steps=200,
        log_every=20,
        problem_family="psd_sensing",
        measurement_type="gaussian_symmetric",
        optimizer="adam",
        parameterization="factorized_psd",
        explicit_reg_type="none",
    )

    _, folder, loaded = run_and_save(spec, tmp_path / "runs")
    assert_run_folder_complete(folder)
    assert_loaded_contract(
        loaded,
        problem_family="psd_sensing",
        data_source="synthetic",
        required_summary_keys=["final_objective", "final_rel_recon_error", "best_objective"],
        required_metric_keys=["objective", "rel_recon_error", "gradient_norm", "update_norm"],
    )



def test_golden_synthetic_matrix_completion_end_to_end(tmp_path: Path) -> None:
    spec = RunSpec(
        seed=0,
        n=20,
        n_cols=18,
        r_star=3,
        m=180,
        factor_rank=5,
        init_scale=0.1,
        lr=0.1,
        steps=500,
        log_every=50,
        problem_family="matrix_completion",
        data_source="synthetic",
        measurement_type="uniform_entries",
        optimizer="gd",
        parameterization="factorized_uv",
        explicit_reg_type="factor_l2",
        explicit_reg_strength=1e-4,
    )

    _, folder, loaded = run_and_save(spec, tmp_path / "runs")
    assert_run_folder_complete(folder)
    assert_loaded_contract(
        loaded,
        problem_family="matrix_completion",
        data_source="synthetic",
        required_summary_keys=["final_objective", "final_rel_recon_error", "final_frobenius_norm_X"],
        required_metric_keys=["objective", "rel_recon_error", "frobenius_norm_X", "stable_rank"],
    )



def test_golden_movielens_matrix_completion_end_to_end(tmp_path: Path, monkeypatch) -> None:
    datasets_dir = tmp_path / "datasets"
    write_fake_movielens(datasets_dir)
    monkeypatch.setenv("FRAMEWORK_DATASETS_DIR", str(datasets_dir))

    spec = RunSpec(
        seed=0,
        n=5,
        n_cols=4,
        r_star=2,
        m=10,
        factor_rank=3,
        init_scale=0.1,
        lr=0.05,
        steps=100,
        log_every=10,
        problem_family="matrix_completion",
        data_source="movielens_100k",
        val_fraction=0.2,
        test_fraction=0.2,
        measurement_type="observed_ratings",
        optimizer="gd",
        parameterization="factorized_uv",
        explicit_reg_type="none",
    )

    _, folder, loaded = run_and_save(spec, tmp_path / "runs")
    assert_run_folder_complete(folder)
    assert_loaded_contract(
        loaded,
        problem_family="matrix_completion",
        data_source="movielens_100k",
        required_summary_keys=["final_train_loss", "final_val_rmse", "final_test_rmse"],
        required_metric_keys=["train_loss", "val_loss", "test_loss", "val_rmse", "test_rmse"],
    )



def test_golden_multiseed_group_end_to_end(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    seed_group_id = "golden_seed_group"
    for seed in [0, 1, 2]:
        spec = RunSpec(
            seed=seed,
            n=18,
            r_star=3,
            m=180,
            factor_rank=5,
            init_scale=1e-2,
            lr=1e-3,
            steps=120,
            log_every=20,
            problem_family="psd_sensing",
            measurement_type="gaussian_symmetric",
            optimizer="gd",
            parameterization="factorized_psd",
            explicit_reg_type="none",
            seed_group_id=seed_group_id,
        )
        _, folder, loaded = run_and_save(spec, runs_dir)
        assert_run_folder_complete(folder)
        assert_loaded_contract(
            loaded,
            problem_family="psd_sensing",
            data_source="synthetic",
            required_summary_keys=["final_objective", "final_rel_recon_error"],
            required_metric_keys=["objective", "rel_recon_error", "gradient_norm"],
        )

    loaded_runs = []
    for folder in sorted(runs_dir.iterdir()):
        loaded = load_run(str(folder))
        loaded_runs.append(
            {
                "run_id": folder.name,
                "config": loaded["config"],
                "summary": loaded["summary"],
                "metrics": loaded["metrics"],
                "metadata": loaded["metadata"],
            }
        )

    groups = group_runs_by_seed(loaded_runs)
    assert len(groups) == 1
    group = groups[0]
    assert group["seed_group_id"] == seed_group_id
    assert group["seed_count"] == 3
    assert group["run_count"] == 3

    final_agg = aggregate_seed_group_final_metrics(group)
    traj_agg = aggregate_seed_group_trajectories(group)
    assert not final_agg.empty
    assert not traj_agg.empty
    assert set(["metric", "mean", "std", "min", "max", "seed_count"]).issubset(final_agg.columns)
    assert set(["metric", "step", "mean", "std", "min", "max", "seed_count"]).issubset(traj_agg.columns)
    assert int(final_agg["seed_count"].max()) == 3
    assert int(traj_agg["seed_count"].max()) == 3



def test_golden_load_run_backfills_missing_metadata(tmp_path: Path) -> None:
    spec = RunSpec(
        seed=0,
        n=16,
        r_star=2,
        m=120,
        factor_rank=4,
        init_scale=1e-2,
        lr=1e-3,
        steps=80,
        log_every=20,
        problem_family="psd_sensing",
        measurement_type="gaussian_symmetric",
        optimizer="gd",
        parameterization="factorized_psd",
        explicit_reg_type="none",
    )
    _, folder, _ = run_and_save(spec, tmp_path / "runs")
    (folder / "metadata.json").unlink()

    loaded = load_run(str(folder))
    assert loaded["metadata"]["problem_family"] == "psd_sensing"
    assert loaded["metadata"]["optimizer"] == "gd"
    assert loaded["metadata"]["parameterization"] == "factorized_psd"
    assert loaded["metadata"]["config_hash"]
