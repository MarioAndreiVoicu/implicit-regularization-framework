from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from framework.services.run_repository import ReadRunRepository
from framework.services.serialization import json_dumps


class SweepRepository(Protocol):
    def save_metadata(self, metadata: dict[str, Any]) -> None:
        ...

    def load_metadata(self, group_id: str) -> dict[str, Any] | None:
        ...

    def list_group_ids(self) -> list[str]:
        ...

    def load_group_runs(self, group_id: str) -> list[dict[str, Any]]:
        ...

    def load_group_run_items(self, group_id: str) -> list[tuple[Path, dict[str, Any], dict[str, Any], pd.DataFrame]]:
        ...


@dataclass(frozen=True)
class LocalSweepRepository:
    sweeps_dir: Path
    read_repository: ReadRunRepository

    def __post_init__(self) -> None:
        self.sweeps_dir.mkdir(parents=True, exist_ok=True)

    def save_metadata(self, metadata: dict[str, Any]) -> None:
        group_id = str(metadata['group_id'])
        path = self.sweeps_dir / f"{group_id}.json"
        path.write_text(json_dumps(metadata, indent=2), encoding='utf-8')

    def load_metadata(self, group_id: str) -> dict[str, Any] | None:
        path = self.sweeps_dir / f"{group_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding='utf-8'))

    def list_group_ids(self) -> list[str]:
        return list(self.read_repository.list_groups().keys())

    def load_group_runs(self, group_id: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for run_dir, cfg, summary, metrics in self.load_group_run_items(group_id):
            rows.append({
                'run_id': Path(run_dir).name,
                'config': cfg,
                'summary': summary,
                'metrics': metrics,
            })
        return rows

    def load_group_run_items(self, group_id: str) -> list[tuple[Path, dict[str, Any], dict[str, Any], pd.DataFrame]]:
        groups = self.read_repository.list_groups()
        run_dirs = groups.get(group_id, [])
        items: list[tuple[Path, dict[str, Any], dict[str, Any], pd.DataFrame]] = []
        for run_dir in run_dirs:
            loaded = self.read_repository.load_run(run_dir)
            metrics = loaded.get('metrics', pd.DataFrame())
            if not isinstance(metrics, pd.DataFrame):
                metrics = pd.DataFrame(metrics)
            items.append((Path(run_dir), dict(loaded.get('config', {})), dict(loaded.get('summary', {})), metrics))
        return items
