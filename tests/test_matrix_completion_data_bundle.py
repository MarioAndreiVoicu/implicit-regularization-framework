from __future__ import annotations

from pathlib import Path

from framework.core.problems.matrix_completion import (
    make_matrix_completion_data_bundle,
    make_matrix_completion_problem,
    matrix_completion_has_heldout_observations,
    matrix_completion_supports_ground_truth_metrics,
)


def _write_fake_movielens(root: Path) -> None:
    data_dir = root / "ml-100k"
    data_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        "1\t1\t5\t874965758",
        "1\t2\t4\t874965759",
        "2\t1\t3\t874965760",
        "2\t3\t2\t874965761",
        "3\t1\t4\t874965762",
        "3\t2\t5\t874965763",
        "4\t3\t1\t874965764",
        "4\t4\t2\t874965765",
        "5\t1\t5\t874965766",
        "5\t4\t3\t874965767",
    ]
    (data_dir / "u.data").write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_synthetic_matrix_completion_bundle_has_canonical_observation_contract() -> None:
    bundle = make_matrix_completion_data_bundle(
        seed=0,
        n_rows=12,
        n_cols=15,
        r_star=3,
        m=40,
        data_source="synthetic",
        measurement_type="uniform_entries",
        coherence_level=0.2,
        sampling_regime="uniform",
        mc_spectrum_type="power_decay",
        mc_spectrum_decay=1.3,
        dataset_version="synthetic_custom_v2",
        split_version="synthetic_split_custom_v2",
        preprocessing_version="none",
    )

    assert bundle.data_source == "synthetic"
    assert bundle.source_kind == "synthetic"
    assert bundle.split_strategy == "observed_only"
    assert bundle.has_ground_truth is True
    assert bundle.observed_y.numel() == 40
    assert bundle.train_y.numel() == 40
    assert bundle.val_y.numel() == 0
    assert bundle.test_y.numel() == 0
    assert int(bundle.observed_mask.sum().item()) == 40
    assert bundle.X_star is not None
    assert matrix_completion_supports_ground_truth_metrics(bundle) is True
    assert matrix_completion_has_heldout_observations(bundle) is False
    assert bundle.U_star is not None
    assert bundle.V_star is not None
    assert bundle.dataset_version == "synthetic_custom_v2"
    assert bundle.split_version == "synthetic_split_custom_v2"
    assert bundle.preprocessing_version == "none"


def test_movielens_matrix_completion_bundle_uses_same_contract(tmp_path: Path, monkeypatch) -> None:
    _write_fake_movielens(tmp_path)
    monkeypatch.setenv("FRAMEWORK_DATASETS_DIR", str(tmp_path))

    bundle = make_matrix_completion_data_bundle(
        seed=0,
        n_rows=5,
        n_cols=4,
        r_star=2,
        m=10,
        data_source="movielens_100k",
        measurement_type="observed_ratings",
        val_fraction=0.2,
        test_fraction=0.2,
        dataset_version="movielens_100k",
        split_version="movielens_custom_split_v2",
        preprocessing_version="movielens_custom_prep_v2",
    )

    assert bundle.data_source == "movielens_100k"
    assert bundle.source_kind == "real"
    assert bundle.split_strategy == "heldout_observed_entries"
    assert bundle.has_ground_truth is False
    assert bundle.observed_y.numel() == 10
    assert bundle.train_y.numel() + bundle.val_y.numel() + bundle.test_y.numel() == 10
    assert int(bundle.observed_mask.sum().item()) == 10
    assert bundle.X_star is None
    assert matrix_completion_supports_ground_truth_metrics(bundle) is False
    assert matrix_completion_has_heldout_observations(bundle) is True
    assert bundle.dataset_version == "movielens_100k"
    assert bundle.split_version == "movielens_custom_split_v2"
    assert bundle.preprocessing_version == "movielens_custom_prep_v2"


def test_matrix_completion_problem_is_built_from_bundle_contract(tmp_path: Path, monkeypatch) -> None:
    _write_fake_movielens(tmp_path)
    monkeypatch.setenv("FRAMEWORK_DATASETS_DIR", str(tmp_path))

    problem = make_matrix_completion_problem(
        seed=0,
        n_rows=5,
        n_cols=4,
        r_star=2,
        m=10,
        data_source="movielens_100k",
        measurement_type="observed_ratings",
        val_fraction=0.2,
        test_fraction=0.2,
        dataset_version="movielens_100k",
        split_version="movielens_custom_split_v2",
        preprocessing_version="movielens_custom_prep_v2",
    )

    assert problem.data_bundle is not None
    assert problem.source_kind == problem.data_bundle.source_kind
    assert problem.split_strategy == problem.data_bundle.split_strategy
    assert problem.has_ground_truth == problem.data_bundle.has_ground_truth
    assert problem.data_bundle.data_source == problem.data_source
    assert problem.data_bundle.n_rows == problem.n_rows
    assert problem.data_bundle.n_cols == problem.n_cols
    assert problem.data_bundle.train_y.shape == problem.train_y.shape
    assert problem.data_bundle.split_version == problem.split_version



def _write_custom_triplets_csv(path: Path) -> Path:
    csv_path = path / "ratings.csv"
    csv_path.write_text(
        "row,col,value\n"
        "0,0,5\n"
        "0,1,4\n"
        "1,0,3\n"
        "1,2,2\n"
        "2,0,4\n"
        "2,1,5\n"
        "3,2,1\n"
        "3,3,2\n",
        encoding="utf-8",
    )
    return csv_path


def test_custom_triplets_matrix_completion_bundle_uses_real_source_contract(tmp_path: Path) -> None:
    csv_path = _write_custom_triplets_csv(tmp_path)

    bundle = make_matrix_completion_data_bundle(
        seed=0,
        n_rows=1,
        n_cols=1,
        r_star=2,
        m=1,
        data_source="custom_triplets_csv",
        custom_dataset_path=str(csv_path),
        measurement_type="observed_ratings",
        val_fraction=0.25,
        test_fraction=0.25,
        dataset_version="custom_triplets_csv:ratings.csv",
        split_version="custom_split_v1",
        preprocessing_version="custom_raw_v1",
    )

    assert bundle.data_source == "custom_triplets_csv"
    assert bundle.source_kind == "real"
    assert bundle.has_ground_truth is False
    assert bundle.n_rows == 4
    assert bundle.n_cols == 4
    assert bundle.observed_y.numel() == 8
    assert bundle.train_y.numel() + bundle.val_y.numel() + bundle.test_y.numel() == 8
    assert bundle.custom_dataset_path == str(csv_path)
    assert bundle.split_strategy == "heldout_observed_entries"


def test_custom_triplets_matrix_completion_problem_preserves_source_aware_bundle(tmp_path: Path) -> None:
    csv_path = _write_custom_triplets_csv(tmp_path)

    problem = make_matrix_completion_problem(
        seed=0,
        n_rows=1,
        n_cols=1,
        r_star=2,
        m=1,
        data_source="custom_triplets_csv",
        custom_dataset_path=str(csv_path),
        measurement_type="observed_ratings",
        val_fraction=0.25,
        test_fraction=0.25,
        dataset_version="custom_triplets_csv:ratings.csv",
        split_version="custom_split_v1",
        preprocessing_version="custom_raw_v1",
    )

    assert problem.data_source == "custom_triplets_csv"
    assert problem.source_kind == "real"
    assert problem.data_bundle is not None
    assert problem.data_bundle.custom_dataset_path == str(csv_path)
    assert problem.has_ground_truth is False
