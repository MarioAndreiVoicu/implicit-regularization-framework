import json
from pathlib import Path

import numpy as np

from framework.core.models import RunResult, RunSpec
from framework.services.analysis_views import save_analysis_views, load_analysis_views
from framework.services.sweep_repository import LocalSweepRepository
from framework.storage.local import save_run


class _EmptyReadRepository:
    def list_groups(self):
        return {}

    def load_run(self, run_dir):
        raise AssertionError("not used")


def _small_spec() -> RunSpec:
    return RunSpec(
        seed=0,
        n=4,
        r_star=1,
        m=4,
        factor_rank=1,
        init_scale=0.01,
        lr=0.001,
        steps=1,
        log_every=1,
        num_eigs=4,
    )


def test_save_run_normalizes_numpy_scalars_in_json_files(tmp_path: Path) -> None:
    result = RunResult(
        spec=_small_spec(),
        metrics={"step": [0], "objective": [0.25]},
        summary={"status": "completed", "steps": np.int64(1), "final_objective": np.float64(0.25)},
        artifacts={},
    )

    run_dir = Path(save_run(result, tmp_path / "runs", metadata_overrides={"np_seed": np.int64(7)}))

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert summary["steps"] == 1
    assert summary["final_objective"] == 0.25
    assert metadata["np_seed"] == 7


def test_sweep_metadata_normalizes_numpy_values(tmp_path: Path) -> None:
    repository = LocalSweepRepository(sweeps_dir=tmp_path / "sweeps", read_repository=_EmptyReadRepository())

    repository.save_metadata({"group_id": "sweep_np", "values": {"steps": [np.int64(5)]}})

    assert repository.load_metadata("sweep_np")["values"]["steps"] == [5]


def test_analysis_view_save_normalizes_numpy_values(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "analysis_views.json"
    monkeypatch.setattr("framework.services.analysis_views.ANALYSIS_VIEWS_PATH", path)

    save_analysis_views({"compare_runs": {"views": {"v": {"step": np.int64(3)}}, "order": ["v"]}})

    assert load_analysis_views()["compare_runs"]["views"]["v"]["step"] == 3
