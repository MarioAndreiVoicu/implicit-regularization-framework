from __future__ import annotations

import pandas as pd
import streamlit as st

from framework.services.ui_state import remove_conflicting_widget_defaults
from framework.services.serialization import make_json_safe
from framework.ui.cards import render_value_columns
from framework.ui.page_services import CustomDatasetsPageService


def _state_safe_kwargs(key: str | None, kwargs: dict, *, default_arg_names: tuple[str, ...]) -> dict:
    return remove_conflicting_widget_defaults(st.session_state, key, kwargs, default_arg_names=default_arg_names)


def _text_input(label, **kwargs):
    key = kwargs.get("key")
    return st.text_input(label, **_state_safe_kwargs(key, kwargs, default_arg_names=("value",)))


def _selectbox(label, options, **kwargs):
    key = kwargs.get("key")
    return st.selectbox(label, options, **_state_safe_kwargs(key, kwargs, default_arg_names=("index",)))


def _text_state_value(key: str, *, default: str = "") -> str:
    value = st.session_state.get(key, default)
    if value is None:
        return default
    return str(value)


def _metric_columns(stats: dict) -> None:
    render_value_columns(
        [
            ("Rows", stats.get("n_rows")),
            ("Cols", stats.get("n_cols")),
            ("Observations", stats.get("m")),
            ("Range", None if stats.get("rating_min") is None else f"{stats.get('rating_min')} → {stats.get('rating_max')}"),
        ]
    )


def _source_stats_columns(stats: dict) -> None:
    render_value_columns(
        [
            ("Rows", stats.get("n_rows")),
            ("Cols", stats.get("n_cols")),
            ("Dimension", stats.get("n") if stats.get("n_rows") is None else None),
            ("Observations / measurements", stats.get("m")),
            ("Target rank", stats.get("r_star")),
            ("Factor rank", stats.get("factor_rank")),
            ("Range", None if stats.get("rating_min") is None else f"{stats.get('rating_min')} → {stats.get('rating_max')}"),
            ("Noise std", stats.get("noise_std")),
        ]
    )


def _source_rows(sources: list[dict]) -> list[dict]:
    rows = []
    for source in sources:
        stats = dict(source.get("stats") or {})
        rows.append(
            {
                "label": source["label"],
                "kind": source["kind"],
                "problem_family": source.get("problem_family"),
                "data_source": source["data_source"],
                "rows_or_dimension": stats.get("n_rows", stats.get("n")),
                "cols": stats.get("n_cols"),
                "observations_or_measurements": stats.get("m", stats.get("registered_datasets")),
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


def _render_source_detail(source: dict, *, default_caption: str | None = None) -> None:
    stats = dict(source.get("stats") or {})
    st.markdown("#### Source details")
    st.write(source["description"])
    if default_caption:
        st.caption(default_caption)
    if stats:
        if any(key in stats for key in ("n_rows", "n_cols", "n", "m", "r_star", "factor_rank", "rating_min", "noise_std")):
            _source_stats_columns(stats)
            extra_stats = {key: value for key, value in stats.items() if key not in {"n_rows", "n_cols", "n", "m", "r_star", "factor_rank", "rating_min", "rating_max", "noise_std"}}
            if extra_stats:
                st.json(make_json_safe(extra_stats), expanded=True)
        else:
            st.json(make_json_safe(stats), expanded=True)
    with st.expander("Provenance", expanded=False):
        st.json(
            make_json_safe({
                "problem_family": source.get("problem_family"),
                "data_source": source["data_source"],
                "dataset_version": source["dataset_version"],
                "split_version": source["split_version"],
                "preprocessing_version": source["preprocessing_version"],
            }),
            expanded=True,
        )


def render_datasets_page(service: CustomDatasetsPageService) -> None:
    st.title("Data Sources")
    st.caption("Inspect reusable matrix-completion data sources and register custom triplets CSV files.")

    sources_tab, register_tab = st.tabs(["Sources", "Register"])

    with sources_tab:
        st.subheader("Data sources")
        st.caption("Data sources are fixed or user-registered matrix-completion inputs.")

        st.markdown("#### Built-in sources")
        sources = service.list_dataset_sources()
        st.dataframe(pd.DataFrame(_source_rows(sources)), use_container_width=True, hide_index=True)

        source_options = [source["label"] for source in sources]
        current_source = st.session_state.get("datasets_dataset_source", source_options[0] if source_options else "")
        if current_source not in source_options and source_options:
            current_source = source_options[0]
            st.session_state["datasets_dataset_source"] = current_source
        if source_options:
            selected_label = _selectbox(
                "Selected built-in source",
                source_options,
                index=source_options.index(current_source),
                key="datasets_dataset_source",
            )
            selected_source = next(source for source in sources if source["label"] == selected_label)
            _render_source_detail(selected_source)

        st.markdown("#### Custom triplets CSV sources")
        records = service.list_records()
        registry_summary = service.custom_registry_summary()
        render_value_columns([("Registered datasets", registry_summary.get("registered_datasets"))])
        if not records:
            st.info("No custom datasets registered yet. Use Register to add one.")
        else:
            search_query = _text_input(
                "Search custom datasets",
                value=_text_state_value("datasets_custom_search"),
                key="datasets_custom_search",
                placeholder="Search by dataset id, file name, or path",
            )
            filtered_records = _filter_records(records, search_query)
            if not filtered_records:
                st.warning("No custom datasets match the search. Clear the search or use Register to add one.")
            else:
                with st.expander("Show registered datasets table", expanded=False):
                    st.dataframe(pd.DataFrame(_record_rows(filtered_records)), use_container_width=True, hide_index=True)

                dataset_options = [record.dataset_id for record in filtered_records]
                resolved_selected_id = service.resolve_selected_dataset_id(st.session_state.get("datasets_selected_id"))
                if resolved_selected_id not in dataset_options:
                    resolved_selected_id = dataset_options[0]
                st.session_state["datasets_selected_id"] = resolved_selected_id

                selected_id = _selectbox(
                    "Selected custom dataset",
                    options=dataset_options,
                    index=dataset_options.index(resolved_selected_id),
                    key="datasets_selected_id_widget",
                )
                st.session_state["datasets_selected_id"] = selected_id

                inspected = service.inspect_record(selected_id, preview_rows=8)
                record = dict(inspected["record"])
                stats = dict(record.get("stats") or {})

                st.markdown("##### Inspect selected custom dataset")
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

                with st.expander("Technical details", expanded=False):
                    st.json(
                        make_json_safe({
                            "dataset_id": record.get("dataset_id"),
                            "data_source": record.get("data_source"),
                            "dataset_version": record.get("dataset_version"),
                            "split_version": record.get("split_version"),
                            "preprocessing_version": record.get("preprocessing_version"),
                            "source_path": record.get("source_path"),
                            "file_exists": inspected.get("source_exists"),
                            "file_size_bytes": inspected.get("file_size_bytes"),
                        }),
                        expanded=True,
                    )

    with register_tab:
        st.subheader("Register custom triplets CSV")
        st.caption("Upload a CSV file or register one from a local filesystem path.")
        uploaded_file = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            key="datasets_register_uploaded_csv",
            help="CSV with triplet columns row,col,value, or user,item,rating aliases.",
        )
        csv_path = _text_input(
            "Or use local CSV path",
            value=_text_state_value("datasets_register_triplets_csv_path"),
            help="Local path on the machine running Streamlit. Ignored when a CSV file is uploaded above.",
            key="datasets_register_triplets_csv_path",
        ).strip()

        if uploaded_file is not None:
            st.info("The uploaded CSV will be staged in the dataset registry when you validate or register it.")

        def _resolve_csv_path_for_action() -> str:
            if uploaded_file is not None:
                content = uploaded_file.getvalue()
                if not content:
                    raise ValueError("The uploaded CSV is empty.")
                return service.stage_uploaded_triplets_csv(uploaded_file.name, content)
            if not csv_path:
                raise ValueError("Upload a CSV file or provide a local CSV path first.")
            return csv_path

        action_cols = st.columns([1, 1, 3])
        if action_cols[0].button("Validate CSV", key="datasets_validate_button"):
            try:
                source_path = _resolve_csv_path_for_action()
                preview = service.validate_triplets_csv(source_path)
                st.session_state["datasets_validation_preview"] = preview
                st.success("CSV looks valid.")
            except Exception as exc:
                st.session_state.pop("datasets_validation_preview", None)
                st.error(str(exc))

        if action_cols[1].button("Register", type="primary", key="datasets_register_button"):
            try:
                source_path = _resolve_csv_path_for_action()
                record = service.register_triplets_csv(source_path)
                st.session_state["datasets_selected_id"] = record.dataset_id
                st.session_state["datasets_validation_preview"] = service.validate_triplets_csv(source_path)
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
            with st.expander("Technical details", expanded=False):
                st.json(
                    make_json_safe({
                        "dataset_id": preview.get("dataset_id"),
                        "fingerprint": str(preview.get("fingerprint", ""))[:16],
                        "file_size_bytes": preview.get("file_size_bytes"),
                        "dataset_version": preview.get("dataset_version"),
                        "split_version": preview.get("split_version"),
                        "preprocessing_version": preview.get("preprocessing_version"),
                        "resolved_path": preview.get("resolved_path"),
                    }),
                    expanded=True,
                )
