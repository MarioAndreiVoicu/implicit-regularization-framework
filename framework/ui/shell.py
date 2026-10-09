from __future__ import annotations

import streamlit as st


PAGES = [
    "Experiments",
    "Results",
    "Compare",
    "Data Sources",
    "Validation",
    "Docs / Theory",
]

_NAV_KEY = "nav_page"
_PENDING_NAV_KEY = "_pending_nav_page"


def request_navigation(page: str) -> None:
    """Request a page change for the next Streamlit rerun.

    The sidebar radio owns ``st.session_state['nav_page']`` once it has been
    instantiated during a render. Updating that key from a later button
    callback/body path raises a StreamlitAPIException. Store the intended page
    separately and let ``render_sidebar_navigation`` consume it before creating
    the radio on the next rerun.
    """
    if page in PAGES:
        st.session_state[_PENDING_NAV_KEY] = page


def apply_navigation_handoff(updates: dict) -> None:
    """Apply session-state updates that may include a page navigation request."""
    for key, value in updates.items():
        if key == _NAV_KEY:
            request_navigation(str(value))
        else:
            st.session_state[key] = value


def render_sidebar_navigation() -> str:
    st.sidebar.title("Navigation")
    pending_page = st.session_state.pop(_PENDING_NAV_KEY, None)
    if pending_page in PAGES:
        st.session_state[_NAV_KEY] = pending_page
    if _NAV_KEY not in st.session_state or st.session_state[_NAV_KEY] not in PAGES:
        st.session_state[_NAV_KEY] = PAGES[0]
    st.sidebar.radio("Go to", PAGES, key=_NAV_KEY)
    return str(st.session_state[_NAV_KEY])
