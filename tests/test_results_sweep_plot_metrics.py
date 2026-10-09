import importlib
import sys
import types

import pandas as pd


def _results_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "streamlit", types.SimpleNamespace(session_state={}))
    sys.modules.pop("framework.ui.pages.results", None)
    return importlib.import_module("framework.ui.pages.results")


def test_sweep_plot_metric_options_include_aggregate_mean_columns(monkeypatch):
    results = _results_module(monkeypatch)
    df = pd.DataFrame(
        {
            "lr": [0.01, 0.02],
            "seed_count": [3, 3],
            "run_count": [3, 3],
            "final_rel_recon_error_mean": [0.4, 0.2],
            "final_rel_recon_error_std": [0.01, 0.03],
        }
    )

    options = results._preferred_sweep_metric_columns(
        df,
        ["lr"],
        metric_options={"final_rel_recon_error": "Final relative reconstruction error"},
    )

    assert options[0] == "final_rel_recon_error_mean"
    assert "lr" not in options
    assert "seed_count" not in options
    assert "run_count" not in options


def test_sweep_plot_metric_options_include_raw_metric_columns_when_no_aggregate(monkeypatch):
    results = _results_module(monkeypatch)
    df = pd.DataFrame(
        {
            "optimizer": ["gd", "adam"],
            "run": ["a", "b"],
            "seed": [0, 1],
            "final_objective": [1.2, 0.8],
        }
    )

    options = results._preferred_sweep_metric_columns(
        df,
        ["optimizer"],
        metric_options={"final_objective": "Final objective"},
    )

    assert options == ["final_objective"]


def test_sweep_plot_metric_label_for_aggregate_columns_is_readable(monkeypatch):
    results = _results_module(monkeypatch)
    assert results._metric_label("final_rel_recon_error_mean") == "Mean final relative reconstruction error"
    assert results._metric_label("final_val_rmse_median") == "Median final validation RMSE"


def test_sweep_heatmap_axis_order_uses_declared_sweep_values(monkeypatch):
    results = _results_module(monkeypatch)
    analysis_view = types.SimpleNamespace(
        metadata={"values": {"optimizer": ["adam", "gd"], "parameterization": ["factorized", "direct"]}}
    )

    assert results._declared_sweep_axis_values(analysis_view, "optimizer", ["gd", "adam"]) == ["adam", "gd"]
    assert results._declared_sweep_axis_values(analysis_view, "parameterization", ["direct", "factorized"]) == ["factorized", "direct"]
