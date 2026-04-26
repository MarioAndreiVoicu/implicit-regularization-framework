from __future__ import annotations

import streamlit as st


PAGES = [
    "Experiments",
    "Results",
    "Compare",
    "Datasets",
    "Validation",
    "Docs / Theory",
]

_NAV_KEY = "nav_page"

def render_sidebar_navigation() -> str:
    st.sidebar.title("Navigation")
    if _NAV_KEY not in st.session_state or st.session_state[_NAV_KEY] not in PAGES:
        st.session_state[_NAV_KEY] = PAGES[0]
    st.sidebar.radio("Go to", PAGES, key=_NAV_KEY)
    return str(st.session_state[_NAV_KEY])
