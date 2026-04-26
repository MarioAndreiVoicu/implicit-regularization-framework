from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd

from framework.services.run_manager_local import (
    apply_run_filters,
    apply_run_manager_view_state,
    collection_add_runs,
    collection_delete,
    collection_remove_runs,
    collection_rename,
    consume_pending_run_manager_view_state,
    load_run_collections,
    load_run_manager_meta,
    load_run_manager_views,
    save_run_collections,
    save_run_manager_meta,
    save_run_manager_views,
    tags_str_to_list,
)


@dataclass(frozen=True)
class RunManagerBrowseContext:
    runs_df: pd.DataFrame
    view_names: list[str]
    gid_options: list[str]
    status_options: list[str]
    artifact_status_options: list[str]


@dataclass(frozen=True)
class RunManagerViewsContext:
    views_meta: dict
    views: dict
    order: list[str]


@dataclass(frozen=True)
class RunManagerCollectionsContext:
    collections_meta: dict
    collections: dict
    order: list[str]


class LocalRunManagerService:
    def __init__(
        self,
        *,
        runs_dir: Path,
        get_runs_df_for_manager_fn: Callable[[], pd.DataFrame],
        clear_runs_cache_fn: Callable[[], None],
        build_runs_collection_zip_bytes_fn: Callable[..., bytes],
    ) -> None:
        self.runs_dir = Path(runs_dir)
        self._get_runs_df_for_manager = get_runs_df_for_manager_fn
        self._clear_runs_cache = clear_runs_cache_fn
        self._build_runs_collection_zip_bytes = build_runs_collection_zip_bytes_fn

    def consume_pending_view_state(self) -> None:
        consume_pending_run_manager_view_state()

    def refresh_run_index(self) -> None:
        try:
            self._clear_runs_cache()
        except Exception:
            pass

    def build_browse_context(self) -> RunManagerBrowseContext:
        df_all = self._get_runs_df_for_manager()
        views_meta = load_run_manager_views()
        views = views_meta.get("views", {}) if isinstance(views_meta, dict) else {}
        order = views_meta.get("order", []) if isinstance(views_meta, dict) else []
        view_names = ["(none)"] + [name for name in order if name in views]

        if df_all.empty:
            return RunManagerBrowseContext(
                runs_df=df_all,
                view_names=view_names,
                gid_options=["(all)"],
                status_options=["(all)"],
                artifact_status_options=["(all)", "(unreviewed)", "exploratory", "keep", "thesis_candidate", "final", "invalid"],
            )

        gid_series = df_all["group_id"].fillna("(none)").astype(str)
        status_series = df_all["status"].fillna("completed").astype(str)
        artifact_status_series = df_all.get("artifact_status", pd.Series([], dtype=str)).fillna("").astype(str)
        artifact_status_values = [x for x in artifact_status_series.unique().tolist() if str(x).strip()]
        preferred_order = ["exploratory", "keep", "thesis_candidate", "final", "invalid"]
        ordered_artifact_status = [x for x in preferred_order if x in artifact_status_values]
        ordered_artifact_status += [x for x in sorted(artifact_status_values) if x not in ordered_artifact_status]
        return RunManagerBrowseContext(
            runs_df=df_all,
            view_names=view_names,
            gid_options=["(all)"] + sorted(gid_series.unique().tolist()),
            status_options=["(all)"] + sorted(status_series.unique().tolist()),
            artifact_status_options=["(all)", "(unreviewed)"] + ordered_artifact_status,
        )

    def filter_runs(self, df_all: pd.DataFrame, state: dict) -> pd.DataFrame:
        return apply_run_filters(df_all, state)

    def load_views_context(self) -> RunManagerViewsContext:
        views_meta = load_run_manager_views()
        views = views_meta.get("views", {}) if isinstance(views_meta, dict) else {}
        order = views_meta.get("order", []) if isinstance(views_meta, dict) else []
        return RunManagerViewsContext(views_meta=views_meta, views=views, order=order)

    def apply_view_state(self, view_state: dict) -> None:
        apply_run_manager_view_state(view_state)

    def save_view(self, name: str, view_state: dict) -> tuple[bool, str]:
        name = (name or "").strip()
        ctx = self.load_views_context()
        if not name:
            return False, "Give the view a name."
        if name in ctx.views:
            return False, "View already exists."
        views = dict(ctx.views)
        order = list(ctx.order)
        views[name] = dict(view_state)
        order.append(name)
        save_run_manager_views({"views": views, "order": order})
        return True, f"Saved view '{name}'."

    def delete_view(self, name: str) -> None:
        ctx = self.load_views_context()
        views = dict(ctx.views)
        views.pop(name, None)
        save_run_manager_views({"views": views, "order": [x for x in ctx.order if x != name]})

    def rename_view(self, old_name: str, new_name: str) -> tuple[bool, str]:
        new_name = (new_name or "").strip()
        ctx = self.load_views_context()
        if not new_name:
            return False, "Enter a new view name."
        if new_name in ctx.views:
            return False, "A view with that name already exists."
        if old_name not in ctx.views:
            return False, "Selected view does not exist."
        views = dict(ctx.views)
        views[new_name] = views.pop(old_name)
        order = [new_name if x == old_name else x for x in ctx.order]
        save_run_manager_views({"views": views, "order": order})
        return True, "Renamed."

    def load_meta(self) -> dict:
        return load_run_manager_meta()

    def mark_favorite(self, run_ids: list[str], *, favorite: bool) -> None:
        meta = load_run_manager_meta()
        meta.setdefault("runs", {})
        for rid in [str(x) for x in run_ids]:
            meta["runs"].setdefault(rid, {})
            meta["runs"][rid]["favorite"] = favorite
        save_run_manager_meta(meta)

    def set_tags(self, run_ids: list[str], tags_text: str) -> None:
        meta = load_run_manager_meta()
        meta.setdefault("runs", {})
        new_tags = tags_str_to_list(tags_text)
        for rid in [str(x) for x in run_ids]:
            meta["runs"].setdefault(rid, {})
            meta["runs"][rid]["tags"] = new_tags
        save_run_manager_meta(meta)


    def set_artifact_status(self, run_ids: list[str], artifact_status: str) -> None:
        meta = load_run_manager_meta()
        meta.setdefault("runs", {})
        artifact_status = str(artifact_status or "").strip()
        for rid in [str(x) for x in run_ids]:
            meta["runs"].setdefault(rid, {})
            if artifact_status:
                meta["runs"][rid]["artifact_status"] = artifact_status
            else:
                meta["runs"][rid].pop("artifact_status", None)
        save_run_manager_meta(meta)

    def build_free_compare_handoff(self, run_ids: list[str]) -> dict:
        selected = [str(x) for x in run_ids]
        return {
            "nav_page": "Compare",
            "compare_mode": "Free Compare",
            "free_compare_selected_runs": selected,
        }

    def build_matched_compare_handoff(self, run_ids: list[str]) -> dict:
        selected = [str(x) for x in run_ids]
        if len(selected) != 2:
            raise ValueError("Matched compare handoff requires exactly two runs.")
        return {
            "nav_page": "Compare",
            "compare_mode": "Matched Compare",
            "matched_reference_run": selected[0],
            "matched_candidate_run": selected[1],
        }

    def build_selected_runs_export_target(self, run_ids: list[str]) -> dict:
        selected = [str(x) for x in run_ids]
        return {
            "nav_page": "Results",
            "results_export_target": {
                "kind": "selected_runs",
                "run_ids": selected,
                "label": f"{len(selected)} selected runs",
            },
        }

    def load_collections_context(self) -> RunManagerCollectionsContext:
        meta = load_run_collections()
        collections = meta.get("collections", {}) if isinstance(meta, dict) else {}
        order = meta.get("order", []) if isinstance(meta, dict) else []
        return RunManagerCollectionsContext(collections_meta=meta, collections=collections, order=order)

    def create_collection(self, name: str) -> tuple[bool, str]:
        name = (name or "").strip()
        if not name:
            return False, "Give the collection a name."
        meta = collection_add_runs(load_run_collections(), name, [])
        save_run_collections(meta)
        return True, f"Created '{name}'."

    def add_runs_to_collection(self, *, selected_ids: list[str], target: str, new_name: str) -> tuple[bool, str]:
        meta = load_run_collections()
        if target != "(choose)":
            meta = collection_add_runs(meta, target, selected_ids)
            save_run_collections(meta)
            return True, f"Added to '{target}'."
        name = (new_name or "").strip()
        if not name:
            return False, "Pick a collection or enter a new name."
        meta = collection_add_runs(meta, name, selected_ids)
        save_run_collections(meta)
        return True, f"Created '{name}' and added runs."

    def remove_runs_from_collection(self, collection_name: str, run_ids: list[str]) -> None:
        meta = collection_remove_runs(load_run_collections(), collection_name, run_ids)
        save_run_collections(meta)

    def delete_collection(self, collection_name: str) -> None:
        meta = collection_delete(load_run_collections(), collection_name)
        save_run_collections(meta)

    def rename_collection(self, old_name: str, new_name: str) -> tuple[bool, str]:
        meta = collection_rename(load_run_collections(), old_name, new_name)
        save_run_collections(meta)
        return True, "Renamed collection."

    def selected_run_dirs(self, run_ids: list[str]) -> list[Path]:
        ordered: list[Path] = []
        seen: set[str] = set()
        for rid in run_ids:
            rid_str = str(rid)
            if rid_str in seen:
                continue
            seen.add(rid_str)
            ordered.append(self.runs_dir / rid_str)
        return ordered

    def build_bundle_zip(self, *, run_ids: list[str], include_run_reports: bool, include_artifacts: bool) -> bytes:
        return self._build_runs_collection_zip_bytes(
            run_dirs=self.selected_run_dirs(run_ids),
            include_run_reports=include_run_reports,
            include_artifacts=include_artifacts,
        )

    def selected_file_name(self) -> str:
        return f"runs_selected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"

    def collection_file_name(self, collection_name: str) -> str:
        return f"{collection_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
