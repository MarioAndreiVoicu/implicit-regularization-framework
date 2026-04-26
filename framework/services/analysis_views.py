from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ANALYSIS_VIEWS_PATH = Path('analysis_views.json')


def _empty_namespace() -> dict[str, Any]:
    return {"views": {}, "order": []}


def load_analysis_views() -> dict[str, Any]:
    default = {
        "compare_runs": _empty_namespace(),
        "sweeps": _empty_namespace(),
    }
    if not ANALYSIS_VIEWS_PATH.exists():
        return default
    try:
        raw = json.loads(ANALYSIS_VIEWS_PATH.read_text(encoding='utf-8'))
    except Exception:
        return default
    if not isinstance(raw, dict):
        return default
    for ns in list(default.keys()):
        if ns not in raw or not isinstance(raw[ns], dict):
            raw[ns] = _empty_namespace()
        raw[ns].setdefault('views', {})
        raw[ns].setdefault('order', [])
    return raw


def save_analysis_views(meta: dict[str, Any]) -> None:
    ANALYSIS_VIEWS_PATH.write_text(json.dumps(meta, indent=2), encoding='utf-8')


def upsert_analysis_view(meta: dict[str, Any], namespace: str, name: str, state: dict[str, Any]) -> dict[str, Any]:
    out = load_analysis_views() if not isinstance(meta, dict) else meta
    ns = out.setdefault(namespace, _empty_namespace())
    ns.setdefault('views', {})
    ns.setdefault('order', [])
    ns['views'][name] = state
    if name not in ns['order']:
        ns['order'].append(name)
    return out


def delete_analysis_view(meta: dict[str, Any], namespace: str, name: str) -> dict[str, Any]:
    out = load_analysis_views() if not isinstance(meta, dict) else meta
    ns = out.setdefault(namespace, _empty_namespace())
    ns.setdefault('views', {})
    ns.setdefault('order', [])
    ns['views'].pop(name, None)
    ns['order'] = [item for item in ns['order'] if item != name]
    return out


PENDING_ANALYSIS_VIEW_STATE_KEY = "_pending_analysis_view_state"


def apply_analysis_view_state(view_state: dict[str, Any] | None) -> None:
    import streamlit as st

    st.session_state[PENDING_ANALYSIS_VIEW_STATE_KEY] = dict(view_state or {})


def consume_pending_analysis_view_state() -> None:
    import streamlit as st

    pending = st.session_state.pop(PENDING_ANALYSIS_VIEW_STATE_KEY, None)
    if not isinstance(pending, dict):
        return
    for key, value in pending.items():
        st.session_state[key] = value
