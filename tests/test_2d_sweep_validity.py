from __future__ import annotations

import pandas as pd

from framework.services.sweep_analysis import aggregate_sweep_2d, build_sweep_analysis_view


def test_aggregate_sweep_2d_ignores_all_empty_metric_columns():
    raw_df = pd.DataFrame(
        {
            "lr": [0.001, 0.001, 0.002, 0.002],
            "init_scale": [0.01, 0.01, 0.01, 0.01],
            "seed": [0, 1, 0, 1],
            "run": ["a", "b", "c", "d"],
            "final_rel_recon_error": [0.5, 0.6, 0.3, 0.4],
            "final_val_rmse": [None, None, None, None],
        }
    )

    aggregate = aggregate_sweep_2d(
        raw_df,
        "lr",
        "init_scale",
        ["final_rel_recon_error", "final_val_rmse"],
        expected_seed_count=2,
    )

    assert "final_rel_recon_error_mean" in aggregate.columns
    assert "final_val_rmse_mean" not in aggregate.columns


def test_paired_2d_expected_cell_count_uses_pairs_when_metadata_count_missing():
    metadata = {
        "group_id": "paired_test",
        "kind": "2D",
        "preset": "paired",
        "fields": ["lr", "init_scale"],
        "values": {"lr": [0.001, 0.002], "init_scale": [0.01, 0.02]},
        "fixed_factors": {"problem_family": "psd_sensing", "data_source": "synthetic"},
        "seeds": [0],
        "coupling": "paired",
    }
    raw_df = pd.DataFrame(
        {
            "lr": [0.001, 0.002],
            "init_scale": [0.01, 0.02],
            "seed": [0, 0],
            "run": ["a", "b"],
            "final_rel_recon_error": [0.5, 0.4],
        }
    )

    view = build_sweep_analysis_view(
        group_id="paired_test",
        metadata=metadata,
        raw_df=raw_df,
        metric_options={"final_rel_recon_error": "Final relative reconstruction error"},
    )

    assert view.summary["expected_cell_count"] == 2
    assert view.summary["observed_cell_count"] == 2
    assert view.summary["seed_balance_status"] == "balanced"
