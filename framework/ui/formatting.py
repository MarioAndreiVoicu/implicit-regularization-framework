from __future__ import annotations

import math
from typing import Any, Iterable

from framework.services.serialization import make_json_safe


def format_display_value(value: Any) -> str:
    clean = make_json_safe(value)
    if clean is None:
        return "—"
    if isinstance(clean, bool):
        return "true" if clean else "false"
    if isinstance(clean, int):
        return str(clean)
    if isinstance(clean, float):
        if not math.isfinite(clean):
            return "—"
        return f"{clean:.6g}"
    return str(clean)


def format_display_sequence(values: Iterable[Any], *, limit: int | None = None) -> str:
    clean_values = list(make_json_safe(list(values)))
    shown = clean_values if limit is None else clean_values[:limit]
    text = ", ".join(format_display_value(item) for item in shown)
    if limit is not None and len(clean_values) > limit:
        text += ", ..."
    return text
