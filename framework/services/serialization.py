from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _is_missing_scalar(value: Any) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, float):
        return math.isnan(value) or math.isinf(value)
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(result, (bool, np.bool_)):
        return bool(result)
    return False


def make_json_safe(value: Any) -> Any:
    """Convert common scientific-Python objects into clean JSON/display values."""
    if _is_missing_scalar(value):
        return None

    if isinstance(value, np.generic):
        return make_json_safe(value.item())

    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())

    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Decimal):
        return make_json_safe(float(value))

    if is_dataclass(value) and not isinstance(value, type):
        return make_json_safe(asdict(value))

    if isinstance(value, pd.Series):
        return make_json_safe(value.to_dict())

    if isinstance(value, pd.DataFrame):
        return make_json_safe(value.to_dict(orient="records"))

    if isinstance(value, Mapping):
        return {str(make_json_safe(key)): make_json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set, frozenset)):
        return [make_json_safe(item) for item in value]

    if isinstance(value, (str, int, bool)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    return str(value)


def json_dumps(value: Any, **kwargs: Any) -> str:
    kwargs.setdefault("indent", 2)
    kwargs.setdefault("allow_nan", False)
    return json.dumps(make_json_safe(value), **kwargs)
