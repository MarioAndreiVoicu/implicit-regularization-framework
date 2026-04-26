from __future__ import annotations

from typing import Any

from pathlib import Path

from framework.core.datasets import custom_triplets_csv_stats
from framework.core.datasets.movielens import movielens_100k_stats

DEFAULT_SYNTHETIC_DATASET_VERSION = "synthetic_default_v1"
DEFAULT_SYNTHETIC_SPLIT_VERSION = "synthetic_default_split_v1"
DEFAULT_SYNTHETIC_PREPROCESSING_VERSION = "none"
DEFAULT_MOVIELENS_DATASET_VERSION = "movielens_100k"
DEFAULT_MOVIELENS_SPLIT_VERSION = "movielens_100k_default_split_v1"
DEFAULT_MOVIELENS_PREPROCESSING_VERSION = "movielens_100k_default_preprocessing_v1"
DEFAULT_CUSTOM_TRIPLETS_DATASET_VERSION = "custom_triplets_csv"
DEFAULT_CUSTOM_TRIPLETS_SPLIT_VERSION = "custom_triplets_csv_default_split_v1"
DEFAULT_CUSTOM_TRIPLETS_PREPROCESSING_VERSION = "custom_triplets_csv_raw_v1"


def canonical_data_source(problem_family: str, data_source: str) -> str:
    family = str(problem_family)
    source = str(data_source or "synthetic")
    if family == "psd_sensing":
        return "synthetic"
    if family == "matrix_completion" and source in {"synthetic", "movielens_100k", "custom_triplets_csv"}:
        return source
    return "synthetic"


def list_data_sources(problem_family: str) -> list[str]:
    family = str(problem_family)
    if family == "matrix_completion":
        return ["synthetic", "movielens_100k", "custom_triplets_csv"]
    return ["synthetic"]


def is_synthetic_data_source(problem_family: str, data_source: str) -> bool:
    return canonical_data_source(problem_family, data_source) == "synthetic"


def data_source_kind(problem_family: str, data_source: str) -> str:
    source = canonical_data_source(problem_family, data_source)
    if source == "synthetic":
        return "synthetic"
    return "real"


def is_real_data_source(problem_family: str, data_source: str) -> bool:
    return data_source_kind(problem_family, data_source) == "real"


def supports_early_stopping(problem_family: str, data_source: str) -> bool:
    return str(problem_family) == "matrix_completion" and is_real_data_source(problem_family, data_source)


def get_data_source_stats(problem_family: str, data_source: str) -> dict[str, int] | None:
    family = str(problem_family)
    source = canonical_data_source(family, data_source)
    if family == "matrix_completion" and source == "movielens_100k":
        return movielens_100k_stats()
    return None


def get_custom_triplets_stats(path: str | Path) -> dict[str, int | float | str]:
    return custom_triplets_csv_stats(path)


def derive_custom_triplets_dataset_version(path: str | Path) -> str:
    resolved = Path(path).expanduser()
    return f"custom_triplets_csv:{resolved.name}"


def get_data_source_default_provenance(problem_family: str, data_source: str) -> dict[str, str]:
    source = canonical_data_source(problem_family, data_source)
    if source == "movielens_100k":
        return {
            "dataset_version": DEFAULT_MOVIELENS_DATASET_VERSION,
            "split_version": DEFAULT_MOVIELENS_SPLIT_VERSION,
            "preprocessing_version": DEFAULT_MOVIELENS_PREPROCESSING_VERSION,
        }
    if source == "custom_triplets_csv":
        return {
            "dataset_version": DEFAULT_CUSTOM_TRIPLETS_DATASET_VERSION,
            "split_version": DEFAULT_CUSTOM_TRIPLETS_SPLIT_VERSION,
            "preprocessing_version": DEFAULT_CUSTOM_TRIPLETS_PREPROCESSING_VERSION,
        }
    return {
        "dataset_version": DEFAULT_SYNTHETIC_DATASET_VERSION,
        "split_version": DEFAULT_SYNTHETIC_SPLIT_VERSION,
        "preprocessing_version": DEFAULT_SYNTHETIC_PREPROCESSING_VERSION,
    }


def get_default_request_values(problem_family: str, data_source: str) -> dict[str, Any]:
    family = str(problem_family)
    source = canonical_data_source(family, data_source)
    if family == "matrix_completion" and source == "movielens_100k":
        stats = get_data_source_stats(family, source) or {"n_rows": 943, "n_cols": 1682, "m": 100000}
        return {
            "n": int(stats["n_rows"]),
            "n_cols": int(stats["n_cols"]),
            "m": int(stats["m"]),
            "r_star": 5,
            "factor_rank": 20,
            "init_scale": 1e-1,
            "lr": 5e-2,
            "steps": 2000,
            "log_every": 50,
            "noise_std": 0.0,
            "early_stopping_enabled": True,
            "early_stopping_patience": 100,
            "early_stopping_min_delta": 1e-4,
            **get_data_source_default_provenance(family, source),
        }
    if family == "matrix_completion" and source == "custom_triplets_csv":
        return {
            "n": 10,
            "n_cols": 10,
            "m": 10,
            "r_star": 5,
            "factor_rank": 10,
            "init_scale": 1e-1,
            "lr": 5e-2,
            "steps": 2000,
            "log_every": 50,
            "noise_std": 0.0,
            "early_stopping_enabled": True,
            "early_stopping_patience": 100,
            "early_stopping_min_delta": 1e-4,
            **get_data_source_default_provenance(family, source),
        }
    if family == "matrix_completion":
        return {
            "n": 40,
            "n_cols": 50,
            "m": 800,
            "r_star": 5,
            "factor_rank": 10,
            "init_scale": 1e-1,
            "lr": 2e-1,
            "steps": 3000,
            "log_every": 50,
            "noise_std": 0.0,
            "early_stopping_enabled": False,
            "early_stopping_patience": 100,
            "early_stopping_min_delta": 0.0,
            **get_data_source_default_provenance(family, source),
        }
    return {
        "n": 40,
        "n_cols": None,
        "m": 800,
        "r_star": 5,
        "factor_rank": 10,
        "init_scale": 1e-2,
        "lr": 1e-3,
        "steps": 2000,
        "log_every": 20,
        "noise_std": 0.0,
        "early_stopping_enabled": False,
        "early_stopping_patience": 100,
        "early_stopping_min_delta": 0.0,
        **get_data_source_default_provenance(family, source),
    }
