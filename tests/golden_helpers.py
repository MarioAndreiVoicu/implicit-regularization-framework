from __future__ import annotations

from pathlib import Path
from typing import Iterable

from framework.core.experiments import run_experiment
from framework.core.models import RunSpec, RunStatus
from framework.storage.local import RUN_REQUIRED_FILENAMES, is_run_dir_complete, load_run, save_run


GOLDEN_OK_STATUSES = {RunStatus.COMPLETED, RunStatus.DIVERGED}


def write_fake_movielens(root: Path) -> None:
    data_dir = root / "ml-100k"
    data_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        "1\t1\t5\t874965758",
        "1\t2\t4\t874965759",
        "2\t1\t3\t874965760",
        "2\t3\t2\t874965761",
        "3\t1\t4\t874965762",
        "3\t2\t5\t874965763",
        "4\t3\t1\t874965764",
        "4\t4\t2\t874965765",
        "5\t1\t5\t874965766",
        "5\t4\t3\t874965767",
    ]
    (data_dir / "u.data").write_text("\n".join(rows) + "\n", encoding="utf-8")


def run_and_save(spec: RunSpec, runs_dir: Path) -> tuple[object, Path, dict]:
    result = run_experiment(spec)
    assert result.summary["status"] in GOLDEN_OK_STATUSES
    folder = Path(save_run(result, runs_dir=str(runs_dir)))
    loaded = load_run(str(folder))
    return result, folder, loaded


def assert_run_folder_complete(folder: Path) -> None:
    assert is_run_dir_complete(folder)
    for name in RUN_REQUIRED_FILENAMES:
        assert (folder / name).exists(), f"Missing required run file: {name}"


def assert_loaded_contract(
    loaded: dict,
    *,
    problem_family: str,
    data_source: str,
    required_summary_keys: Iterable[str],
    required_metric_keys: Iterable[str],
) -> None:
    cfg = loaded["config"]
    summary = loaded["summary"]
    metrics = loaded["metrics"]
    metadata = loaded["metadata"]
    artifact_manifest = loaded["artifact_manifest"]

    assert cfg["problem_family"] == problem_family
    assert cfg["data_source"] == data_source
    assert summary["status"] in GOLDEN_OK_STATUSES
    assert metadata["problem_family"] == problem_family
    assert metadata["status"] == summary["status"]
    assert metadata["run_id"]
    assert metadata["config_hash"]
    assert artifact_manifest["count"] >= 1
    assert len(metrics) > 0

    for key in required_summary_keys:
        assert key in summary, f"Missing summary key: {key}"
    for key in required_metric_keys:
        assert key in metrics.columns, f"Missing metric column: {key}"
