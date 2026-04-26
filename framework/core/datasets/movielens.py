from __future__ import annotations

import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

MOVIELENS_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
MOVIELENS_100K_DIRNAME = "ml-100k"
MOVIELENS_100K_FILENAME = "u.data"
MOVIELENS_100K_NUM_USERS = 943
MOVIELENS_100K_NUM_ITEMS = 1682
MOVIELENS_100K_NUM_RATINGS = 100000


@dataclass
class MovieLensSplit:
    n_rows: int
    n_cols: int
    total_observations: int
    train_row_idx: torch.Tensor
    train_col_idx: torch.Tensor
    train_y: torch.Tensor
    val_row_idx: torch.Tensor
    val_col_idx: torch.Tensor
    val_y: torch.Tensor
    test_row_idx: torch.Tensor
    test_col_idx: torch.Tensor
    test_y: torch.Tensor
    rating_min: float
    rating_max: float


def movielens_100k_stats() -> dict[str, int]:
    return {
        "n_rows": MOVIELENS_100K_NUM_USERS,
        "n_cols": MOVIELENS_100K_NUM_ITEMS,
        "m": MOVIELENS_100K_NUM_RATINGS,
    }


def _datasets_root(root: str | Path | None = None) -> Path:
    if root is not None:
        return Path(root)
    env = os.environ.get("FRAMEWORK_DATASETS_DIR")
    if env:
        return Path(env)
    return Path("datasets_cache")


def ensure_movielens_100k(root: str | Path | None = None) -> Path:
    base = _datasets_root(root)
    base.mkdir(parents=True, exist_ok=True)
    dataset_dir = base / MOVIELENS_100K_DIRNAME
    data_file = dataset_dir / MOVIELENS_100K_FILENAME
    if data_file.exists():
        return dataset_dir

    archive_path = base / f"{MOVIELENS_100K_DIRNAME}.zip"
    if not archive_path.exists():
        try:
            urllib.request.urlretrieve(MOVIELENS_100K_URL, archive_path)
        except Exception as exc:
            raise RuntimeError(
                "Could not download MovieLens 100K automatically. "
                "Set FRAMEWORK_DATASETS_DIR to a folder containing ml-100k/u.data, "
                "or make sure internet access is available."
            ) from exc

    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(base)
    except Exception as exc:
        raise RuntimeError("Could not extract MovieLens 100K archive.") from exc

    if not data_file.exists():
        raise RuntimeError("MovieLens 100K archive extracted, but ml-100k/u.data was not found.")
    return dataset_dir


def load_movielens_100k_dataframe(root: str | Path | None = None) -> pd.DataFrame:
    dataset_dir = ensure_movielens_100k(root)
    df = pd.read_csv(
        dataset_dir / MOVIELENS_100K_FILENAME,
        sep="\t",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )
    if df.empty:
        raise RuntimeError("MovieLens 100K dataset is empty.")
    return df


def make_movielens_100k_split(
    *,
    seed: int,
    val_fraction: float,
    test_fraction: float,
    root: str | Path | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> MovieLensSplit:
    if device is None:
        device = torch.device("cpu")
    if val_fraction < 0.0 or test_fraction < 0.0 or val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction and test_fraction must be >= 0 and sum to < 1.")

    df = load_movielens_100k_dataframe(root)
    user_codes = pd.Categorical(df["user_id"])
    item_codes = pd.Categorical(df["item_id"])

    row_idx_np = user_codes.codes.astype(np.int64)
    col_idx_np = item_codes.codes.astype(np.int64)
    ratings_np = df["rating"].astype(float).to_numpy()

    total = int(len(df))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(total)
    test_n = int(round(total * test_fraction))
    val_n = int(round(total * val_fraction))
    train_n = total - val_n - test_n
    if train_n <= 0:
        raise ValueError("Train split is empty. Reduce val_fraction/test_fraction.")

    train_idx = perm[:train_n]
    val_idx = perm[train_n : train_n + val_n]
    test_idx = perm[train_n + val_n :]

    def _take(ix: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = torch.tensor(row_idx_np[ix], device=device, dtype=torch.long)
        cols = torch.tensor(col_idx_np[ix], device=device, dtype=torch.long)
        y = torch.tensor(ratings_np[ix], device=device, dtype=dtype)
        return rows, cols, y

    train_rows, train_cols, train_y = _take(train_idx)
    val_rows, val_cols, val_y = _take(val_idx)
    test_rows, test_cols, test_y = _take(test_idx)

    return MovieLensSplit(
        n_rows=int(len(user_codes.categories)),
        n_cols=int(len(item_codes.categories)),
        total_observations=total,
        train_row_idx=train_rows,
        train_col_idx=train_cols,
        train_y=train_y,
        val_row_idx=val_rows,
        val_col_idx=val_cols,
        val_y=val_y,
        test_row_idx=test_rows,
        test_col_idx=test_cols,
        test_y=test_y,
        rating_min=float(np.min(ratings_np)),
        rating_max=float(np.max(ratings_np)),
    )
