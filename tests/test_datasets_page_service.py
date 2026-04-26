from pathlib import Path

from framework.ui.page_services import CustomDatasetsPageService
from framework.services.custom_dataset_registry import LocalCustomDatasetRegistry


def _write_triplets_csv(path: Path) -> Path:
    csv_path = path / "ratings.csv"
    csv_path.write_text(
        "row,col,value\n0,0,5\n0,1,4\n1,0,3\n1,2,2\n2,0,4\n2,1,5\n",
        encoding="utf-8",
    )
    return csv_path


def test_validate_triplets_csv_returns_preview_and_provenance(tmp_path: Path) -> None:
    csv_path = _write_triplets_csv(tmp_path)
    service = CustomDatasetsPageService(dataset_registry=LocalCustomDatasetRegistry(tmp_path / "registry"))

    preview = service.validate_triplets_csv(str(csv_path), preview_rows=3)

    assert preview["dataset_id"].startswith("custom_mc_")
    assert preview["dataset_version"] == f"custom_triplets_csv:{preview['dataset_id']}"
    assert preview["stats"]["n_rows"] == 3
    assert len(preview["preview_rows"]) == 3
    assert preview["already_registered"] is False


def test_inspect_record_reports_missing_source_file(tmp_path: Path) -> None:
    csv_path = _write_triplets_csv(tmp_path)
    registry = LocalCustomDatasetRegistry(tmp_path / "registry")
    record = registry.register_triplets_csv(csv_path)
    csv_path.unlink()

    service = CustomDatasetsPageService(dataset_registry=registry)
    inspected = service.inspect_record(record.dataset_id)

    assert inspected["source_exists"] is False
    assert inspected["preview_rows"] == []


def test_list_builtin_sources_distinguishes_built_in_and_custom_registry(tmp_path: Path) -> None:
    csv_path = _write_triplets_csv(tmp_path)
    registry = LocalCustomDatasetRegistry(tmp_path / "registry")
    registry.register_triplets_csv(csv_path)
    service = CustomDatasetsPageService(dataset_registry=registry)

    sources = {row["data_source"]: row for row in service.list_builtin_sources()}

    assert sources["synthetic"]["kind"] == "synthetic"
    assert sources["movielens_100k"]["stats"]["m"] == 100000
    assert sources["custom_triplets_csv"]["stats"]["registered_datasets"] == 1


def test_resolve_selected_dataset_id_tracks_registry_changes(tmp_path: Path) -> None:
    first_csv = _write_triplets_csv(tmp_path)
    second_csv = tmp_path / "ratings_2.csv"
    second_csv.write_text(
        "row,col,value\n0,0,1\n0,1,2\n1,0,3\n1,1,4\n",
        encoding="utf-8",
    )
    registry = LocalCustomDatasetRegistry(tmp_path / "registry")
    first = registry.register_triplets_csv(first_csv)
    second = registry.register_triplets_csv(second_csv)
    service = CustomDatasetsPageService(dataset_registry=registry)

    assert service.resolve_selected_dataset_id(None) == first.dataset_id
    assert service.resolve_selected_dataset_id(first.dataset_id) == first.dataset_id
    assert service.resolve_selected_dataset_id(second.dataset_id) == second.dataset_id
    assert service.resolve_selected_dataset_id("missing") == first.dataset_id
