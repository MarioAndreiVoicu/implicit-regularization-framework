from __future__ import annotations

import json
import sys
import types

import numpy as np
import pandas as pd

from framework.services.serialization import json_dumps, make_json_safe
from framework.ui.formatting import format_display_sequence


def test_make_json_safe_converts_numpy_and_pandas_scalars() -> None:
    payload = {
        "steps": np.int64(2000),
        "score": np.float64(0.25),
        "done": np.bool_(True),
        "missing": np.nan,
        "series": pd.Series({"seed": np.int64(3)}),
    }

    clean = make_json_safe(payload)

    assert clean["steps"] == 2000
    assert type(clean["steps"]) is int
    assert clean["score"] == 0.25
    assert type(clean["score"]) is float
    assert clean["done"] is True
    assert clean["missing"] is None
    assert clean["series"] == {"seed": 3}


def test_json_dumps_does_not_expose_numpy_repr() -> None:
    text = json_dumps({"steps": np.int64(100), "values": [np.int64(1), np.float64(2.5)]})

    assert "np." not in text
    assert "int64" not in text
    assert json.loads(text) == {"steps": 100, "values": [1, 2.5]}


def test_format_display_sequence_cleans_numpy_repr() -> None:
    assert format_display_sequence([np.int64(1), np.int64(2), np.float64(3.5)]) == "1, 2, 3.5"
    assert format_display_sequence([np.int64(1), np.int64(2), np.int64(3)], limit=2) == "1, 2, ..."


def test_results_technical_details_json_is_expanded_once_expander_is_open() -> None:
    source = __import__("pathlib").Path("framework/ui/pages/results.py").read_text()

    assert "st.json(make_json_safe(compact), expanded=True)" in source
    assert "st.json(make_json_safe(detail_payload), expanded=True)" in source
    assert "st.json(make_json_safe(compact), expanded=False)" not in source
    assert "st.json(make_json_safe(detail_payload), expanded=False)" not in source


def test_results_run_plot_metric_defaults_are_context_aware(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "streamlit", types.SimpleNamespace())
    from framework.ui.pages.results import _default_run_plot_metric

    options = ["objective", "rel_recon_error", "gradient_norm"]
    assert _default_run_plot_metric(options, {"problem_family": "psd_sensing", "data_source": "synthetic"}) == "rel_recon_error"
    assert _default_run_plot_metric(options, {"problem_family": "matrix_completion", "data_source": "synthetic"}) == "rel_recon_error"

    real_options = ["objective", "train_loss", "val_rmse", "test_rmse"]
    assert _default_run_plot_metric(real_options, {"problem_family": "matrix_completion", "data_source": "movielens_100k"}) == "val_rmse"
    assert _default_run_plot_metric(real_options, {"problem_family": "matrix_completion", "data_source": "custom_triplets_csv"}) == "val_rmse"


def test_results_run_plot_options_are_curated_and_grouped(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "streamlit", types.SimpleNamespace())
    from framework.ui.pages.results import _run_plot_metric_options, _run_metric_select_label

    metrics_df = pd.DataFrame(
        {
            "step": [0, 1],
            "rel_recon_error": [1.0, 0.5],
            "objective": [2.0, 1.0],
            "gradient_norm": [0.2, 0.1],
            "internal_counter": [10, 11],
        }
    )

    options = _run_plot_metric_options(metrics_df, {"problem_family": "psd_sensing", "data_source": "synthetic"})

    assert options[:3] == ["rel_recon_error", "objective", "gradient_norm"]
    assert "internal_counter" not in options
    assert _run_metric_select_label("rel_recon_error") == "Relative reconstruction error · Main fit"
