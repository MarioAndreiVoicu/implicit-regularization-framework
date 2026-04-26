from pathlib import Path

from framework.services.custom_dataset_registry import LocalCustomDatasetRegistry


def _write_triplets_csv(path: Path) -> Path:
    csv_path = path / "ratings.csv"
    csv_path.write_text(
        "row,col,value\n0,0,5\n0,1,4\n1,0,3\n1,2,2\n2,0,4\n2,1,5\n",
        encoding="utf-8",
    )
    return csv_path


def test_register_triplets_csv_creates_stable_manifest(tmp_path: Path) -> None:
    csv_path = _write_triplets_csv(tmp_path)
    registry = LocalCustomDatasetRegistry(tmp_path / "registry")

    record = registry.register_triplets_csv(csv_path)

    assert record.dataset_id.startswith("custom_mc_")
    assert record.dataset_version == f"custom_triplets_csv:{record.dataset_id}"
    assert record.source_path == str(csv_path.resolve())
    assert (tmp_path / "registry" / f"{record.dataset_id}.json").exists()


def test_resolve_triplets_csv_reuses_existing_record_for_same_file(tmp_path: Path) -> None:
    csv_path = _write_triplets_csv(tmp_path)
    registry = LocalCustomDatasetRegistry(tmp_path / "registry")

    first = registry.register_triplets_csv(csv_path)
    second = registry.resolve_triplets_csv(source_path=str(csv_path))
    third = registry.resolve_triplets_csv(dataset_id=first.dataset_id)

    assert second.dataset_id == first.dataset_id
    assert third.dataset_id == first.dataset_id
