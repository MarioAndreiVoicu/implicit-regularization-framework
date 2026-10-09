from __future__ import annotations

from html import escape
from typing import Any, Iterable, Sequence

import streamlit as st

from framework.ui.formatting import format_display_value

_VALUE_COLUMN_STYLE = """
<style>
.irf-value-column-label {
    font-size: 0.875rem;
    font-weight: 600;
    line-height: 1.25;
    margin-bottom: 0.25rem;
}
.irf-value-column-value {
    font-size: 1.35rem;
    font-weight: 600;
    line-height: 1.25;
    white-space: normal;
    overflow: visible;
    overflow-wrap: anywhere;
    word-break: normal;
}
</style>
"""


def _inject_value_column_style() -> None:
    # Streamlit rebuilds the DOM on every rerun, so the style must be emitted
    # whenever these value columns are rendered.
    st.markdown(_VALUE_COLUMN_STYLE, unsafe_allow_html=True)


def render_value_columns(items: Iterable[tuple[str, Any]]) -> None:
    """Render short label/value summaries without Streamlit metric truncation.

    ``st.metric`` is designed for compact numeric KPI cards and can ellipsize
    longer text values inside narrow columns. These lightweight value columns
    keep text visible and wrapping, which is better for categorical statuses
    such as "Weakly controlled" or "matrix_completion".
    """

    item_list: Sequence[tuple[str, Any]] = list(items)
    if not item_list:
        return
    _inject_value_column_style()
    cols = st.columns(len(item_list))
    for col, (label, value) in zip(cols, item_list, strict=False):
        with col:
            label_text = escape(str(label))
            value_text = escape(format_display_value(value))
            st.markdown(
                f'<div class="irf-value-column-label">{label_text}</div>'
                f'<div class="irf-value-column-value">{value_text}</div>',
                unsafe_allow_html=True,
            )
