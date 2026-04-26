from framework.services.data_sources import (
    DEFAULT_MOVIELENS_DATASET_VERSION,
    DEFAULT_MOVIELENS_PREPROCESSING_VERSION,
    DEFAULT_MOVIELENS_SPLIT_VERSION,
    DEFAULT_SYNTHETIC_DATASET_VERSION,
    DEFAULT_SYNTHETIC_PREPROCESSING_VERSION,
    DEFAULT_SYNTHETIC_SPLIT_VERSION,
    canonical_data_source,
    get_data_source_default_provenance,
    get_data_source_stats,
    get_default_request_values,
    list_data_sources,
    supports_early_stopping,
)


def test_psd_sensing_forces_synthetic_data_source_contract():
    assert list_data_sources("psd_sensing") == ["synthetic"]
    assert canonical_data_source("psd_sensing", "movielens_100k") == "synthetic"
    assert supports_early_stopping("psd_sensing", "synthetic") is False
    assert get_data_source_stats("psd_sensing", "synthetic") is None
    assert get_data_source_default_provenance("psd_sensing", "synthetic") == {
        "dataset_version": DEFAULT_SYNTHETIC_DATASET_VERSION,
        "split_version": DEFAULT_SYNTHETIC_SPLIT_VERSION,
        "preprocessing_version": DEFAULT_SYNTHETIC_PREPROCESSING_VERSION,
    }


def test_movielens_data_source_defaults_include_stats_and_provenance():
    defaults = get_default_request_values("matrix_completion", "movielens_100k")

    assert defaults["dataset_version"] == DEFAULT_MOVIELENS_DATASET_VERSION
    assert defaults["split_version"] == DEFAULT_MOVIELENS_SPLIT_VERSION
    assert defaults["preprocessing_version"] == DEFAULT_MOVIELENS_PREPROCESSING_VERSION
    assert defaults["n"] == 943
    assert defaults["n_cols"] == 1682
    assert defaults["m"] == 100000
    assert defaults["early_stopping_enabled"] is True
    assert supports_early_stopping("matrix_completion", "movielens_100k") is True
