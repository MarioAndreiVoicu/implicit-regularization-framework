from pathlib import Path

from framework.services import analysis_views


def test_analysis_views_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis_views, 'ANALYSIS_VIEWS_PATH', tmp_path / 'analysis_views.json')

    meta = analysis_views.load_analysis_views()
    assert 'compare_runs' in meta
    assert 'sweeps' in meta

    meta = analysis_views.upsert_analysis_view(meta, 'compare_runs', 'frontier', {'compare_mode': 'Complexity vs Fit'})
    analysis_views.save_analysis_views(meta)

    loaded = analysis_views.load_analysis_views()
    assert loaded['compare_runs']['views']['frontier']['compare_mode'] == 'Complexity vs Fit'
    assert 'frontier' in loaded['compare_runs']['order']

    loaded = analysis_views.delete_analysis_view(loaded, 'compare_runs', 'frontier')
    analysis_views.save_analysis_views(loaded)
    loaded2 = analysis_views.load_analysis_views()
    assert 'frontier' not in loaded2['compare_runs']['views']
