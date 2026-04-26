from pathlib import Path

import streamlit as st

from framework.ui.bootstrap import run_streamlit_app

st.set_page_config(page_title="Implicit Regularization Lab", layout="wide")

RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(exist_ok=True)

run_streamlit_app(RUNS_DIR)