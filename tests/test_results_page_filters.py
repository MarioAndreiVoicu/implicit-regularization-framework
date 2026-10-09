import importlib
import sys
import types

import pandas as pd


def _results_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "streamlit", types.SimpleNamespace(session_state={}))
    sys.modules.pop("framework.ui.pages.results", None)
    return importlib.import_module("framework.ui.pages.results")


def test_run_search_treats_special_characters_literally(monkeypatch):
    results = _results_module(monkeypatch)
    df = pd.DataFrame(
        {
            "run_id": ["run[abc]", "run-plain"],
            "problem_family": ["psd_sensing", "psd_sensing"],
            "favorite": [False, False],
        }
    )

    filtered = results._filter_runs(df, search="[abc]", artifact_status="(all)", family="(all)", favorite_only=False)

    assert filtered["run_id"].tolist() == ["run[abc]"]


def test_sweep_search_treats_special_characters_literally(monkeypatch):
    results = _results_module(monkeypatch)
    df = pd.DataFrame(
        {
            "group_id": ["sweep(lr)", "sweep-optimizer"],
            "kind": ["1D", "1D"],
        }
    )

    filtered = results._filter_sweeps(df, search="(lr)", kind="(all)")

    assert filtered["group_id"].tolist() == ["sweep(lr)"]


def test_favorite_filter_accepts_bool_and_string_values(monkeypatch):
    results = _results_module(monkeypatch)
    df = pd.DataFrame(
        {
            "run_id": ["bool", "string", "one", "false"],
            "problem_family": ["psd_sensing"] * 4,
            "favorite": [True, "true", "1", False],
        }
    )

    filtered = results._filter_runs(df, search="", artifact_status="(all)", family="(all)", favorite_only=True)

    assert filtered["run_id"].tolist() == ["bool", "string", "one"]


def test_multiselect_state_is_pruned_when_options_change(monkeypatch):
    results = _results_module(monkeypatch)
    results.st.session_state["selection"] = ["old", "keep"]

    value = results._ensure_valid_multiselect_state("selection", ["keep", "new"], default=["new"])

    assert value == ["keep"]
    assert results.st.session_state["selection"] == ["keep"]


def test_multiselect_state_preserves_intentional_empty_selection(monkeypatch):
    results = _results_module(monkeypatch)
    results.st.session_state["selection"] = []

    value = results._ensure_valid_multiselect_state("selection", ["a", "b"], default=["a"])

    assert value == []
    assert results.st.session_state["selection"] == []
