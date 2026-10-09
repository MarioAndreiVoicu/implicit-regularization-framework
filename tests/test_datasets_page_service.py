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


def test_list_dataset_sources_excludes_generators_and_custom_registry(tmp_path: Path) -> None:
    csv_path = _write_triplets_csv(tmp_path)
    registry = LocalCustomDatasetRegistry(tmp_path / "registry")
    registry.register_triplets_csv(csv_path)
    service = CustomDatasetsPageService(dataset_registry=registry)

    sources = {row["data_source"]: row for row in service.list_dataset_sources()}

    assert set(sources) == {"movielens_100k"}
    assert sources["movielens_100k"]["kind"] == "dataset"
    assert sources["movielens_100k"]["stats"]["m"] == 100000


def test_list_builtin_sources_matches_dataset_sources(tmp_path: Path) -> None:
    service = CustomDatasetsPageService(dataset_registry=LocalCustomDatasetRegistry(tmp_path / "registry"))

    assert service.list_builtin_sources() == service.list_dataset_sources()


def test_custom_registry_summary_reports_registered_datasets(tmp_path: Path) -> None:
    csv_path = _write_triplets_csv(tmp_path)
    registry = LocalCustomDatasetRegistry(tmp_path / "registry")
    registry.register_triplets_csv(csv_path)
    service = CustomDatasetsPageService(dataset_registry=registry)

    summary = service.custom_registry_summary()

    assert summary["data_source"] == "custom_triplets_csv"
    assert summary["registered_datasets"] == 1
    assert summary["dataset_version_template"] == "custom_triplets_csv:<registered-dataset-id>"


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

    fallback_id = service.list_records()[0].dataset_id

    assert fallback_id in {first.dataset_id, second.dataset_id}
    assert service.resolve_selected_dataset_id(None) == fallback_id
    assert service.resolve_selected_dataset_id(first.dataset_id) == first.dataset_id
    assert service.resolve_selected_dataset_id(second.dataset_id) == second.dataset_id
    assert service.resolve_selected_dataset_id("missing") == fallback_id


def test_stage_uploaded_triplets_csv_persists_upload_under_registry(tmp_path: Path) -> None:
    service = CustomDatasetsPageService(dataset_registry=LocalCustomDatasetRegistry(tmp_path / "registry"))

    staged_path = Path(service.stage_uploaded_triplets_csv("ratings export.csv", b"row,col,value\n0,0,5\n"))

    assert staged_path.exists()
    assert staged_path.parent.name == "uploaded_sources"
    assert staged_path.name.startswith("ratings_export_")
    assert staged_path.suffix == ".csv"
    assert staged_path.read_text(encoding="utf-8") == "row,col,value\n0,0,5\n"
