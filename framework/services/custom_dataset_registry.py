from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
import json
from typing import Any

from framework.core.datasets.custom_matrix_completion import custom_triplets_csv_stats
from framework.services.data_sources import (
    DEFAULT_CUSTOM_TRIPLETS_PREPROCESSING_VERSION,
    DEFAULT_CUSTOM_TRIPLETS_SPLIT_VERSION,
)


@dataclass(frozen=True)
class CustomDatasetRecord:
    dataset_id: str
    data_source: str
    source_kind: str
    source_path: str
    source_filename: str
    fingerprint: str
    dataset_version: str
    split_version: str
    preprocessing_version: str
    stats: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CustomDatasetRecord":
        return cls(
            dataset_id=str(payload["dataset_id"]),
            data_source=str(payload.get("data_source", "custom_triplets_csv")),
            source_kind=str(payload.get("source_kind", "real")),
            source_path=str(payload["source_path"]),
            source_filename=str(payload.get("source_filename") or Path(str(payload["source_path"])).name),
            fingerprint=str(payload["fingerprint"]),
            dataset_version=str(payload.get("dataset_version") or f"custom_triplets_csv:{payload['dataset_id']}"),
            split_version=str(payload.get("split_version") or DEFAULT_CUSTOM_TRIPLETS_SPLIT_VERSION),
            preprocessing_version=str(payload.get("preprocessing_version") or DEFAULT_CUSTOM_TRIPLETS_PREPROCESSING_VERSION),
            stats=dict(payload.get("stats", {})),
        )


class LocalCustomDatasetRegistry:
    def __init__(self, registry_dir: str | Path) -> None:
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def _manifest_path(self, dataset_id: str) -> Path:
        return self.registry_dir / f"{dataset_id}.json"

    def _file_fingerprint(self, path: str | Path) -> str:
        digest = sha256()
        with Path(path).expanduser().open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def compute_fingerprint(self, path: str | Path) -> str:
        return self._file_fingerprint(path)

    def list_records(self) -> list[CustomDatasetRecord]:
        records: list[CustomDatasetRecord] = []
        for manifest in sorted(self.registry_dir.glob("*.json")):
            try:
                records.append(CustomDatasetRecord.from_dict(json.loads(manifest.read_text(encoding="utf-8"))))
            except Exception:
                continue
        return records

    def get(self, dataset_id: str) -> CustomDatasetRecord:
        manifest = self._manifest_path(dataset_id)
        if not manifest.exists():
            raise KeyError(f"Unknown custom dataset id: {dataset_id}")
        return CustomDatasetRecord.from_dict(json.loads(manifest.read_text(encoding="utf-8")))

    def register_triplets_csv(self, path: str | Path) -> CustomDatasetRecord:
        resolved = str(Path(path).expanduser().resolve())
        stats = custom_triplets_csv_stats(resolved)
        fingerprint = self._file_fingerprint(resolved)
        dataset_id = f"custom_mc_{fingerprint[:12]}"
        record = CustomDatasetRecord(
            dataset_id=dataset_id,
            data_source="custom_triplets_csv",
            source_kind="real",
            source_path=resolved,
            source_filename=Path(resolved).name,
            fingerprint=fingerprint,
            dataset_version=f"custom_triplets_csv:{dataset_id}",
            split_version=DEFAULT_CUSTOM_TRIPLETS_SPLIT_VERSION,
            preprocessing_version=DEFAULT_CUSTOM_TRIPLETS_PREPROCESSING_VERSION,
            stats=dict(stats),
        )
        self._manifest_path(dataset_id).write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
        return record

    def resolve_triplets_csv(self, *, dataset_id: str = "", source_path: str = "") -> CustomDatasetRecord:
        dataset_id = str(dataset_id or "").strip()
        source_path = str(source_path or "").strip()
        if dataset_id:
            return self.get(dataset_id)
        if not source_path:
            raise ValueError("Custom dataset resolution requires dataset_id or source_path")
        resolved = str(Path(source_path).expanduser().resolve())
        try:
            fingerprint = self._file_fingerprint(resolved)
        except FileNotFoundError:
            raise
        for record in self.list_records():
            if record.fingerprint == fingerprint:
                return record
        return self.register_triplets_csv(resolved)
