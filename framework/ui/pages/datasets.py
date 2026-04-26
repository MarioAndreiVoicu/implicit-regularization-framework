from __future__ import annotations

import pandas as pd
import streamlit as st

from framework.services.new_run import normalize_text_state_value
from framework.services.ui_state import remove_conflicting_widget_defaults
from framework.ui.page_services import CustomDatasetsPageService


def _state_safe_kwargs(key: str | None, kwargs: dict, *, default_arg_names: tuple[str, ...]) -> dict:
    return remove_conflicting_widget_defaults(st.session_state, key, kwargs, default_arg_names=default_arg_names)


def _text_input(label, **kwargs):
    key = kwargs.get("key")
    return st.text_input(label, **_state_safe_kwargs(key, kwargs, default_arg_names=("value",)))


def _selectbox(label, options, **kwargs):
    key = kwargs.get("key")
    return st.selectbox(label, options, **_state_safe_kwargs(key, kwargs, default_arg_names=("index",)))


def _metric_columns(stats: dict) -> None:
    cols = st.columns(4)
    items = [
        ("Rows", stats.get("n_rows")),
        ("Cols", stats.get("n_cols")),
        ("Observations", stats.get("m")),
        ("Range", None if stats.get("rating_min") is None else f"{stats.get('rating_min')} → {stats.get('rating_max')}"),
    ]
    for col, (label, value) in zip(cols, items, strict=False):
        col.metric(label, "—" if value is None else value)


def _source_rows(sources: list[dict]) -> list[dict]:
    rows = []
    for source in sources:
        stats = dict(source.get("stats") or {})
        rows.append(
            {
                "label": source["label"],
                "kind": source["kind"],
                "data_source": source["data_source"],
                "rows": stats.get("n_rows"),
                "cols": stats.get("n_cols"),
                "observations": stats.get("m") if stats else stats.get("registered_datasets"),
            }
        )
    return rows


def _record_rows(records) -> list[dict]:
    rows = []
    for record in records:
        stats = dict(record.stats or {})
        rows.append(
            {
                "dataset_id": record.dataset_id,
                "source_file": record.source_filename,
                "rows": stats.get("n_rows"),
                "cols": stats.get("n_cols"),
                "observations": stats.get("m"),
                "range": None if stats.get("rating_min") is None else f"{stats.get('rating_min')} → {stats.get('rating_max')}",
            }
        )
    return rows


def _filter_records(records, query: str):
    query = str(query or "").strip().lower()
    if not query:
        return list(records)
    return [
        record
        for record in records
        if query in record.dataset_id.lower()
        or query in record.source_filename.lower()
        or query in str(record.source_path).lower()
    ]


def render_datasets_page(service: CustomDatasetsPageService) -> None:
    st.title("Datasets")
    st.caption("Understand built-in sources and manage custom matrix-completion datasets.")

    builtin_tab, custom_tab, register_tab = st.tabs(["Built-in", "Custom", "Register / Edit"])

    with builtin_tab:
        st.subheader("Built-in datasets")
        sources = service.list_builtin_sources()
        st.dataframe(pd.DataFrame(_source_rows(sources)), use_container_width=True, hide_index=True)

        source_options = [source["label"] for source in sources]
        current_source = st.session_state.get("datasets_builtin_source", source_options[0] if source_options else "")
        if current_source not in source_options and source_options:
            current_source = source_options[0]
            st.session_state["datasets_builtin_source"] = current_source
        if source_options:
            selected_label = _selectbox(
                "Selected source",
                source_options,
                index=source_options.index(current_source),
                key="datasets_builtin_source",
            )
            selected_source = next(source for source in sources if source["label"] == selected_label)
            stats = dict(selected_source.get("stats") or {})
            st.markdown("#### Source details")
            st.write(selected_source["description"])
            if any(key in stats for key in ("n_rows", "n_cols", "m", "rating_min")):
                _metric_columns(stats)
            elif stats:
                st.json(stats, expanded=False)
            with st.expander("Provenance", expanded=False):
                st.json(
                    {
                        "data_source": selected_source["data_source"],
                        "dataset_version": selected_source["dataset_version"],
                        "split_version": selected_source["split_version"],
                        "preprocessing_version": selected_source["preprocessing_version"],
                    },
                    expanded=False,
                )

    with custom_tab:
        st.subheader("Custom datasets")
        records = service.list_records()
        if not records:
            st.info("No custom datasets registered yet.")
            return

        normalize_text_state_value(st.session_state, "datasets_custom_search", default="")
        search_query = _text_input(
            "Search custom datasets",
            value=str(st.session_state["datasets_custom_search"]),
            key="datasets_custom_search",
            placeholder="Search by dataset id, file name, or path",
        )
        filtered_records = _filter_records(records, search_query)
        if not filtered_records:
            st.warning("No custom datasets match the search.")
            return

        with st.expander("Show registered datasets table", expanded=False):
            st.dataframe(pd.DataFrame(_record_rows(filtered_records)), use_container_width=True, hide_index=True)

        dataset_options = [record.dataset_id for record in filtered_records]
        resolved_selected_id = service.resolve_selected_dataset_id(st.session_state.get("datasets_selected_id"))
        if resolved_selected_id not in dataset_options:
            resolved_selected_id = dataset_options[0]
        st.session_state["datasets_selected_id"] = resolved_selected_id
        st.session_state["datasets_selected_id_widget"] = resolved_selected_id

        selected_id = _selectbox(
            "Selected dataset",
            options=dataset_options,
            index=dataset_options.index(resolved_selected_id),
            key="datasets_selected_id_widget",
        )
        st.session_state["datasets_selected_id"] = selected_id

        inspected = service.inspect_record(selected_id, preview_rows=8)
        record = dict(inspected["record"])
        stats = dict(record.get("stats") or {})

        st.markdown("#### Inspect selected dataset")
        if not inspected.get("source_exists"):
            st.warning("The registered source file is no longer available at its recorded path.")
        if inspected.get("validation_error"):
            st.error(str(inspected["validation_error"]))
        _metric_columns(stats)

        preview_rows = list(inspected.get("preview_rows") or [])
        st.caption("Preview rows")
        if preview_rows:
            st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No preview available.")

        with st.expander("Provenance / manifest", expanded=False):
            st.json(
                {
                    "dataset_id": record.get("dataset_id"),
                    "data_source": record.get("data_source"),
                    "dataset_version": record.get("dataset_version"),
                    "split_version": record.get("split_version"),
                    "preprocessing_version": record.get("preprocessing_version"),
                    "source_path": record.get("source_path"),
                    "file_exists": inspected.get("source_exists"),
                    "file_size_bytes": inspected.get("file_size_bytes"),
                },
                expanded=False,
            )
            st.json(record, expanded=False)

    with register_tab:
        st.subheader("Register custom triplets CSV")
        normalize_text_state_value(st.session_state, "datasets_register_triplets_csv_path", default="")
        csv_path = _text_input(
            "CSV path",
            value=str(st.session_state["datasets_register_triplets_csv_path"]),
            help="CSV with triplet columns row,col,value, or user,item,rating aliases.",
            key="datasets_register_triplets_csv_path",
        ).strip()

        action_cols = st.columns([1, 1, 3])
        if action_cols[0].button("Validate CSV", key="datasets_validate_button"):
            if not csv_path:
                st.error("Provide a CSV path first.")
            else:
                try:
                    preview = service.validate_triplets_csv(csv_path)
                    st.session_state["datasets_validation_preview"] = preview
                    st.success("CSV looks valid.")
                except Exception as exc:
                    st.session_state.pop("datasets_validation_preview", None)
                    st.error(str(exc))

        if action_cols[1].button("Register", type="primary", key="datasets_register_button"):
            if not csv_path:
                st.error("Provide a CSV path first.")
            else:
                try:
                    record = service.register_triplets_csv(csv_path)
                    st.session_state["datasets_selected_id"] = record.dataset_id
                    st.session_state["datasets_selected_id_widget"] = record.dataset_id
                    st.session_state["datasets_validation_preview"] = service.validate_triplets_csv(csv_path)
                    st.success(f"Registered {record.dataset_id}")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

        preview = st.session_state.get("datasets_validation_preview")
        if isinstance(preview, dict):
            st.markdown("#### Validation preview")
            _metric_columns(dict(preview.get("stats") or {}))
            if preview.get("already_registered"):
                st.info("This CSV is already registered. Registering will reuse the same dataset id.")
            preview_rows = list(preview.get("preview_rows") or [])
            if preview_rows:
                st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)
            with st.expander("Resolved provenance", expanded=False):
                st.json(
                    {
                        "dataset_id": preview.get("dataset_id"),
                        "fingerprint": str(preview.get("fingerprint", ""))[:16],
                        "file_size_bytes": preview.get("file_size_bytes"),
                        "dataset_version": preview.get("dataset_version"),
                        "split_version": preview.get("split_version"),
                        "preprocessing_version": preview.get("preprocessing_version"),
                        "resolved_path": preview.get("resolved_path"),
                    },
                    expanded=False,
                )
