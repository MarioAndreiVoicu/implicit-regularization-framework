from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from framework.storage.local import read_run_header as _read_run_header
from framework.services.local_runs import list_run_dirs as _list_run_dirs


class RunRepository(Protocol):
    def save_result(self, result: Any) -> str:
        ...

    def save_result_with_metadata(self, result: Any, metadata_patch: dict[str, Any] | None = None) -> str:
        ...


class ReadRunRepository(Protocol):
    runs_dir: Path

    def resolve_run_dir(self, run_ref: str | Path) -> Path:
        ...

    def list_run_dirs(self) -> list[Path]:
        ...

    def list_run_ids(self) -> list[str]:
        ...

    def read_run_header(self, run_ref: str | Path) -> dict[str, Any]:
        ...

    def load_run(self, run_ref: str | Path) -> dict[str, Any]:
        ...

    def list_groups(self) -> dict[str, list[Path]]:
        ...


@dataclass(frozen=True)
class LocalRunRepository:
    runs_dir: Path
    save_run_fn: Callable[..., str]

    def save_result(self, result: Any) -> str:
        folder = self.save_run_fn(result, runs_dir=str(self.runs_dir))
        return Path(folder).name

    def save_result_with_metadata(self, result: Any, metadata_patch: dict[str, Any] | None = None) -> str:
        folder = self.save_run_fn(result, runs_dir=str(self.runs_dir), metadata_overrides=metadata_patch)
        return Path(folder).name


@dataclass(frozen=True)
class LocalReadRunRepository:
    runs_dir: Path
    load_run_fn: Callable[[str], dict[str, Any]]
    read_run_header_fn: Callable[[str], dict[str, Any]] = _read_run_header

    def resolve_run_dir(self, run_ref: str | Path) -> Path:
        path = Path(run_ref)
        if path.is_absolute():
            return path
        if path.parent != Path('.'):
            return path
        return self.runs_dir / path

    def list_run_dirs(self) -> list[Path]:
        return _list_run_dirs(self.runs_dir)

    def list_run_ids(self) -> list[str]:
        return [path.name for path in self.list_run_dirs()]

    def read_run_header(self, run_ref: str | Path) -> dict[str, Any]:
        return self.read_run_header_fn(str(self.resolve_run_dir(run_ref)))

    def load_run(self, run_ref: str | Path) -> dict[str, Any]:
        return self.load_run_fn(str(self.resolve_run_dir(run_ref)))

    def list_groups(self) -> dict[str, list[Path]]:
        groups: dict[str, list[Path]] = {}
        for run_dir in self.list_run_dirs():
            try:
                header = self.read_run_header(run_dir)
            except Exception:
                continue
            group_id = header.get("config", {}).get("group_id") or header.get("metadata", {}).get("group_id")
            if group_id:
                groups.setdefault(str(group_id), []).append(run_dir)
        return groups
