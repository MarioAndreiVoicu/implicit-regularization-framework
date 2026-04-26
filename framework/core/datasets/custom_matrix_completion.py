from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


@dataclass
class CustomTripletSplit:
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
    source_path: str


_ROW_ALIASES = ("row", "row_idx", "user", "user_id")
_COL_ALIASES = ("col", "col_idx", "item", "item_id")
_VALUE_ALIASES = ("value", "rating", "y", "score")


def _resolve_triplet_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    lowered = {str(col).strip().lower(): str(col) for col in df.columns}

    def _pick(aliases: tuple[str, ...]) -> str | None:
        for alias in aliases:
            if alias in lowered:
                return lowered[alias]
        return None

    row_col = _pick(_ROW_ALIASES)
    col_col = _pick(_COL_ALIASES)
    value_col = _pick(_VALUE_ALIASES)
    if row_col and col_col and value_col:
        return row_col, col_col, value_col

    if len(df.columns) == 3:
        return str(df.columns[0]), str(df.columns[1]), str(df.columns[2])

    raise ValueError(
        "Custom matrix-completion CSV must contain triplet columns row/col/value "
        "(or user/item/rating aliases)."
    )


def load_custom_triplets_csv_dataframe(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"Custom dataset CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Custom matrix-completion CSV is empty.")

    row_col, col_col, value_col = _resolve_triplet_columns(df)
    out = df[[row_col, col_col, value_col]].copy()
    out.columns = ["row", "col", "value"]
    out["row"] = pd.to_numeric(out["row"], errors="coerce")
    out["col"] = pd.to_numeric(out["col"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    if out[["row", "col", "value"]].isna().any().any():
        raise ValueError("Custom matrix-completion CSV contains non-numeric row/col/value entries.")

    row_min = float(out["row"].min())
    col_min = float(out["col"].min())
    if row_min >= 1.0 and col_min >= 1.0:
        out["row"] = out["row"] - 1.0
        out["col"] = out["col"] - 1.0

    out["row"] = out["row"].astype(np.int64)
    out["col"] = out["col"].astype(np.int64)
    if (out["row"] < 0).any() or (out["col"] < 0).any():
        raise ValueError("Custom matrix-completion CSV requires non-negative row/col indices.")

    if out.duplicated(subset=["row", "col"]).any():
        raise ValueError("Custom matrix-completion CSV contains duplicate (row, col) observations.")

    return out.reset_index(drop=True)


def custom_triplets_csv_stats(path: str | Path) -> dict[str, int | float | str]:
    df = load_custom_triplets_csv_dataframe(path)
    return {
        "n_rows": int(df["row"].max()) + 1,
        "n_cols": int(df["col"].max()) + 1,
        "m": int(len(df)),
        "rating_min": float(df["value"].min()),
        "rating_max": float(df["value"].max()),
        "source_path": str(Path(path).expanduser()),
    }


def make_custom_triplets_csv_split(
    *,
    path: str | Path,
    seed: int,
    val_fraction: float,
    test_fraction: float,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> CustomTripletSplit:
    if device is None:
        device = torch.device("cpu")
    if val_fraction < 0.0 or test_fraction < 0.0 or val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction and test_fraction must be >= 0 and sum to < 1.")

    df = load_custom_triplets_csv_dataframe(path)
    total = int(len(df))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(total)
    test_n = int(round(total * test_fraction))
    val_n = int(round(total * val_fraction))
    train_n = total - val_n - test_n
    if train_n <= 0:
        raise ValueError("Train split is empty. Reduce val_fraction/test_fraction.")

    row_idx_np = df["row"].to_numpy(dtype=np.int64)
    col_idx_np = df["col"].to_numpy(dtype=np.int64)
    values_np = df["value"].to_numpy(dtype=float)

    train_idx = perm[:train_n]
    val_idx = perm[train_n : train_n + val_n]
    test_idx = perm[train_n + val_n :]

    def _take(ix: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = torch.tensor(row_idx_np[ix], device=device, dtype=torch.long)
        cols = torch.tensor(col_idx_np[ix], device=device, dtype=torch.long)
        y = torch.tensor(values_np[ix], device=device, dtype=dtype)
        return rows, cols, y

    train_rows, train_cols, train_y = _take(train_idx)
    val_rows, val_cols, val_y = _take(val_idx)
    test_rows, test_cols, test_y = _take(test_idx)

    return CustomTripletSplit(
        n_rows=int(df["row"].max()) + 1,
        n_cols=int(df["col"].max()) + 1,
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
        rating_min=float(df["value"].min()),
        rating_max=float(df["value"].max()),
        source_path=str(Path(path).expanduser()),
    )
