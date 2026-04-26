from __future__ import annotations

from typing import Any, Mapping


def remove_conflicting_widget_defaults(
    state: Mapping[str, Any],
    key: str | None,
    kwargs: dict[str, Any],
    *,
    default_arg_names: tuple[str, ...],
) -> dict[str, Any]:
    """Remove default-value kwargs when Streamlit state already owns a widget key.

    Streamlit warns when a widget is created with both an explicit default
    argument, such as ``value`` or ``index``, and an existing ``st.session_state``
    value for the same key. Conditional forms hit this often after mode switches.
    This helper keeps defaults for first render, but lets existing state own later
    renders.
    """
    cleaned = dict(kwargs)
    if key is not None and key in state:
        for name in default_arg_names:
            cleaned.pop(name, None)
    return cleaned
