from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from framework.services.local_runs import list_run_dirs
from framework.storage.local import read_run_header


RUN_MANAGER_META_PATH = Path("run_manager_meta.json")
RUN_MANAGER_VIEWS_PATH = Path("run_manager_views.json")
RUN_COLLECTIONS_PATH = Path("run_collections.json")


def _runs_dir_signature(runs_dir: Path) -> tuple[tuple[str, int], ...]:
    """Small cache key that changes when run folders are added/updated."""
    if not runs_dir.exists():
        return ()
    signature: list[tuple[str, int]] = []
    for run_dir in list_run_dirs(runs_dir):
        try:
            signature.append((run_dir.name, int(run_dir.stat().st_mtime_ns)))
        except OSError:
            signature.append((run_dir.name, 0))
    return tuple(signature)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def tags_str_to_list(tags_text: str) -> list[str]:
    parts = [x.strip() for x in (tags_text or "").split(",")]
    out: list[str] = []
    seen = set()
    for p in parts:
        if not p:
            continue
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def load_run_manager_meta() -> dict:
    return _read_json(RUN_MANAGER_META_PATH, {"runs": {}})


def save_run_manager_meta(meta: dict) -> None:
    _write_json(RUN_MANAGER_META_PATH, meta)


def load_run_manager_views() -> dict:
    return _read_json(RUN_MANAGER_VIEWS_PATH, {"views": {}, "order": []})


def save_run_manager_views(meta: dict) -> None:
    _write_json(RUN_MANAGER_VIEWS_PATH, meta)


PENDING_RUN_MANAGER_VIEW_STATE_KEY = "_pending_run_manager_view_state"


def apply_run_manager_view_state(view_state: dict) -> None:
    import streamlit as st

    st.session_state[PENDING_RUN_MANAGER_VIEW_STATE_KEY] = dict(view_state or {})


def consume_pending_run_manager_view_state() -> None:
    import streamlit as st

    pending = st.session_state.pop(PENDING_RUN_MANAGER_VIEW_STATE_KEY, None)
    if not isinstance(pending, dict):
        return
    for k, v in pending.items():
        st.session_state[f"rm_{k}"] = v


def load_run_collections() -> dict:
    return _read_json(RUN_COLLECTIONS_PATH, {"collections": {}, "order": []})


def save_run_collections(meta: dict) -> None:
    _write_json(RUN_COLLECTIONS_PATH, meta)


def collection_add_runs(meta: dict, collection_name: str, run_ids: list[str]) -> dict:
    meta = meta or {"collections": {}, "order": []}
    meta.setdefault("collections", {})
    meta.setdefault("order", [])
    if collection_name not in meta["collections"]:
        meta["collections"][collection_name] = {"description": "", "run_ids": []}
        meta["order"].append(collection_name)
    current = [str(x) for x in meta["collections"][collection_name].get("run_ids", [])]
    merged = current[:]
    for rid in run_ids:
        rid = str(rid)
        if rid not in merged:
            merged.append(rid)
    meta["collections"][collection_name]["run_ids"] = merged
    return meta


def collection_remove_runs(meta: dict, collection_name: str, run_ids: list[str]) -> dict:
    if not meta or collection_name not in meta.get("collections", {}):
        return meta
    remove_set = {str(x) for x in run_ids}
    current = [str(x) for x in meta["collections"][collection_name].get("run_ids", [])]
    meta["collections"][collection_name]["run_ids"] = [rid for rid in current if rid not in remove_set]
    return meta


def collection_rename(meta: dict, old_name: str, new_name: str) -> dict:
    new_name = (new_name or "").strip()
    if not new_name or old_name == new_name:
        return meta
    if old_name not in meta.get("collections", {}):
        return meta
    if new_name in meta.get("collections", {}):
        return meta
    meta["collections"][new_name] = meta["collections"].pop(old_name)
    meta["order"] = [new_name if x == old_name else x for x in meta.get("order", [])]
    return meta


def collection_delete(meta: dict, collection_name: str) -> dict:
    if collection_name in meta.get("collections", {}):
        meta["collections"].pop(collection_name, None)
        meta["order"] = [x for x in meta.get("order", []) if x != collection_name]
    return meta


def _infer_sweep_kind_for_gid(gid: str | None) -> str | None:
    if not gid:
        return None
    sweeps_dir = Path("sweeps")
    meta_path = sweeps_dir / f"{gid}.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            kind = meta.get("kind")
            if kind == "1D":
                return "1D"
            if kind == "2D":
                return "2D"
        except Exception:
            pass
    gid = str(gid)
    if gid.startswith("sweep1d_"):
        return "1D"
    if gid.startswith("sweep2d_"):
        return "2D"
    return None


def _merge_run_manager_meta_into_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    meta = load_run_manager_meta().get("runs", {})
    out = df.copy()
    favorites = []
    tags = []
    artifact_statuses = []
    sweep_kinds = []
    for _, row in out.iterrows():
        rid = str(row["run_id"])
        rmeta = meta.get(rid, {})
        favorites.append(bool(rmeta.get("favorite", False)))
        rtags = rmeta.get("tags", [])
        tags.append(", ".join(str(x) for x in rtags) if isinstance(rtags, list) else "")
        artifact_statuses.append(str(rmeta.get("artifact_status", "") or ""))
        sweep_kinds.append(_infer_sweep_kind_for_gid(row.get("group_id")))
    out["favorite"] = favorites
    out["tags"] = tags
    out["artifact_status"] = artifact_statuses
    out["sweep_kind"] = sweep_kinds
    return out


@lru_cache(maxsize=1)
def scan_runs_table_base(runs_dir_str: str, signature: tuple[tuple[str, int], ...] | None = None) -> pd.DataFrame:
    runs_dir = Path(runs_dir_str)
    rows = []
    for rd in list_run_dirs(runs_dir):
        try:
            header = read_run_header(str(rd))
        except Exception:
            continue
        cfg = header["config"]
        summary = header["summary"]
        metadata = header["metadata"]
        artifact_manifest = header["artifact_manifest"]
        integrity = header.get("integrity", {}) or {}
        rows.append(
            {
                "run_id": rd.name,
                "group_id": cfg.get("group_id"),
                "seed_group_id": cfg.get("seed_group_id") or metadata.get("seed_group_id"),
                "run_name": cfg.get("run_name"),
                "notes": cfg.get("notes"),
                "problem_family": metadata.get("problem_family", cfg.get("problem_family", "psd_sensing")),
                "data_source": metadata.get("data_source", cfg.get("data_source", "synthetic")),
                "dataset_version": metadata.get("dataset_version", cfg.get("dataset_version")),
                "split_version": metadata.get("split_version", cfg.get("split_version")),
                "preprocessing_version": metadata.get("preprocessing_version", cfg.get("preprocessing_version")),
                "optimizer": metadata.get("optimizer", cfg.get("optimizer")),
                "lr_schedule_type": metadata.get("lr_schedule_type", cfg.get("lr_schedule_type", "constant")),
                "lr_decay_gamma": metadata.get("lr_decay_gamma", cfg.get("lr_decay_gamma", 0.5)),
                "lr_step_size": metadata.get("lr_step_size", cfg.get("lr_step_size", 100)),
                "lr_min": metadata.get("lr_min", cfg.get("lr_min", 0.0)),
                "execution_mode": metadata.get("execution_mode", cfg.get("execution_mode", "accuracy")),
                "device_preference": metadata.get("device_preference", cfg.get("device_preference", "auto")),
                "dtype_preference": metadata.get("dtype_preference", cfg.get("dtype_preference", "auto")),
                "resolved_device": metadata.get("resolved_device", summary.get("resolved_device")),
                "resolved_dtype": metadata.get("resolved_dtype", summary.get("resolved_dtype")),
                "integrity_status": integrity.get("status", "valid"),
                "integrity_warning_count": int(integrity.get("warning_count", 0) or 0),
                "integrity_error_count": int(integrity.get("error_count", 0) or 0),
                "parameterization": metadata.get("parameterization", cfg.get("parameterization", "factorized_psd")),
                "init_type": metadata.get("init_type", cfg.get("init_type", "near_zero_gaussian")),
                "explicit_reg_type": metadata.get("explicit_reg_type", cfg.get("explicit_reg_type", "none")),
                "explicit_reg_strength": metadata.get("explicit_reg_strength", cfg.get("explicit_reg_strength", 0.0)),
                "stopping_mode": metadata.get("stopping_mode", cfg.get("stopping_mode", "run_to_convergence")),
                "grad_norm_tol": metadata.get("grad_norm_tol", cfg.get("grad_norm_tol", 1e-10)),
                "objective_tol": metadata.get("objective_tol", cfg.get("objective_tol", 1e-12)),
                "divergence_norm_threshold": metadata.get("divergence_norm_threshold", cfg.get("divergence_norm_threshold", 1e6)),
                "measurement_type": metadata.get("measurement_type", cfg.get("measurement_type")),
                "psd_regime": metadata.get("psd_regime", cfg.get("psd_regime", "non_commuting")),
                "noncommutativity_level": metadata.get("noncommutativity_level", cfg.get("noncommutativity_level", 1.0)),
                "coherence_level": metadata.get("coherence_level", cfg.get("coherence_level", 0.0)),
                "sampling_regime": metadata.get("sampling_regime", cfg.get("sampling_regime", "uniform")),
                "mc_spectrum_type": metadata.get("mc_spectrum_type", cfg.get("mc_spectrum_type", "flat")),
                "mc_spectrum_decay": metadata.get("mc_spectrum_decay", cfg.get("mc_spectrum_decay", 1.0)),
                "mc_factor_balance": metadata.get("mc_factor_balance", cfg.get("mc_factor_balance", "balanced")),
                "mc_factor_imbalance_level": metadata.get("mc_factor_imbalance_level", cfg.get("mc_factor_imbalance_level", 0.0)),
                "seed": metadata.get("seed", cfg.get("seed")),
                "schema_version": metadata.get("schema_version", cfg.get("schema_version", 1)),
                "framework_version": metadata.get("framework_version", cfg.get("framework_version")),
                "git_commit": metadata.get("git_commit", cfg.get("git_commit")),
                "config_hash": metadata.get("config_hash"),
                "artifact_count": artifact_manifest.get("count", 0),
                "n": cfg.get("n"),
                "r_star": cfg.get("r_star"),
                "m": cfg.get("m"),
                "factor_rank": cfg.get("factor_rank"),
                "init_scale": cfg.get("init_scale"),
                "lr": cfg.get("lr"),
                "steps": cfg.get("steps"),
                "noise_std": cfg.get("noise_std"),
                "status": metadata.get("status", summary.get("status")),
                "final_loss": summary.get("final_loss"),
                "final_rel_recon_error": summary.get("final_rel_recon_error"),
                "best_rel_recon_error": summary.get("best_rel_recon_error"),
                "final_trace_X": summary.get("final_trace_X"),
                "final_mean_target_commutator": summary.get("final_mean_target_commutator"),
                "final_max_target_commutator": summary.get("final_max_target_commutator"),
                "final_target_row_coherence": summary.get("final_target_row_coherence"),
                "final_target_col_coherence": summary.get("final_target_col_coherence"),
                "final_target_condition_number": summary.get("final_target_condition_number"),
                "final_target_effective_rank": summary.get("final_target_effective_rank"),
                "final_target_stable_rank": summary.get("final_target_stable_rank"),
                "final_explicit_reg_term": summary.get("final_explicit_reg_term"),
                "final_objective": summary.get("final_objective"),
                "converged": summary.get("converged"),
                "stop_reason": summary.get("stop_reason"),
            }
        )
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def get_runs_df_for_manager(runs_dir: Path) -> pd.DataFrame:
    base = scan_runs_table_base(str(runs_dir), _runs_dir_signature(runs_dir))
    return base if base.empty else _merge_run_manager_meta_into_df(base.copy())


def apply_run_filters(df_all: pd.DataFrame, state: dict) -> pd.DataFrame:
    if df_all.empty:
        return df_all.copy()

    def _list_like(value):
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return [value]

    df = df_all.copy()

    gid_choice = str(state.get("gid_choice", "") or "").strip()
    if gid_choice and gid_choice != "(all)" and "group_id" in df.columns:
        if gid_choice == "(none)":
            df = df[df["group_id"].isna() | (df["group_id"].astype(str) == "")]
        else:
            df = df[df["group_id"].fillna("(none)").astype(str) == gid_choice]

    status_choice = str(state.get("status_choice", "") or "").strip()
    if status_choice and status_choice != "(all)" and "status" in df.columns:
        df = df[df["status"].fillna("").astype(str) == status_choice]

    q_name = str(state.get("q_name", "") or "").strip().lower()
    if q_name and "run_name" in df.columns:
        df = df[df["run_name"].astype(str).str.lower().str.contains(q_name, na=False)]

    q_notes = str(state.get("q_notes", "") or "").strip().lower()
    if q_notes and "notes" in df.columns:
        df = df[df["notes"].astype(str).str.lower().str.contains(q_notes, na=False)]

    sweep_kind = str(state.get("sweep_kind", "") or "").strip()
    if sweep_kind and sweep_kind != "(all)" and "sweep_kind" in df.columns:
        if sweep_kind == "(none)":
            df = df[df["sweep_kind"].isna() | (df["sweep_kind"].astype(str) == "")]
        else:
            df = df[df["sweep_kind"].fillna("(none)").astype(str) == sweep_kind]

    favorite = str(state.get("favorite", "") or "").strip()
    if favorite == "only" and "favorite" in df.columns:
        df = df[df["favorite"] == True]
    elif favorite == "exclude" and "favorite" in df.columns:
        df = df[df["favorite"] != True]

    artifact_status = str(state.get("artifact_status", "") or "").strip()
    if artifact_status and artifact_status != "(all)" and "artifact_status" in df.columns:
        if artifact_status == "(unreviewed)":
            df = df[df["artifact_status"].fillna("").astype(str).str.strip() == ""]
        else:
            df = df[df["artifact_status"].fillna("").astype(str) == artifact_status]

    integrity_scope = str(state.get("integrity_scope", "") or "").strip()
    if integrity_scope and integrity_scope != "(all)" and "integrity_status" in df.columns:
        statuses = df["integrity_status"].fillna("valid").astype(str)
        if integrity_scope == "clean_only":
            df = df[statuses == "valid"]
        elif integrity_scope == "problematic_only":
            df = df[statuses != "valid"]
        elif integrity_scope == "invalid_only":
            df = df[statuses == "invalid"]

    tags_any = [t.strip().lower() for t in str(state.get("tags_any", "") or "").split(",") if t.strip()]
    if tags_any and "tags" in df.columns:
        tags_series = df["tags"].astype(str).str.lower()
        mask = pd.Series(False, index=df.index)
        for tag in tags_any:
            mask = mask | tags_series.str.contains(tag, na=False)
        df = df[mask]

    search = str(state.get("search", "") or "").strip().lower()
    if search:
        mask = pd.Series(False, index=df.index)
        for col in ["run_id", "run_name", "notes", "optimizer", "parameterization", "tags", "group_id", "seed_group_id"]:
            if col in df.columns:
                mask = mask | df[col].astype(str).str.lower().str.contains(search, na=False)
        df = df[mask]

    status_vals = [x for x in _list_like(state.get("status")) if str(x)]
    if status_vals and "status" in df.columns:
        df = df[df["status"].isin(status_vals)]

    optimizer_vals = [x for x in _list_like(state.get("optimizer")) if str(x)]
    if optimizer_vals and "optimizer" in df.columns:
        df = df[df["optimizer"].isin(optimizer_vals)]

    parameterization_vals = [x for x in _list_like(state.get("parameterization")) if str(x)]
    if parameterization_vals and "parameterization" in df.columns:
        df = df[df["parameterization"].isin(parameterization_vals)]

    sweep_kind_vals = [x for x in _list_like(state.get("sweep_kind_vals")) if str(x)]
    if sweep_kind_vals and "sweep_kind" in df.columns:
        df = df[df["sweep_kind"].isin(sweep_kind_vals)]

    artifact_status_vals = [x for x in _list_like(state.get("artifact_status_vals")) if str(x)]
    if artifact_status_vals and "artifact_status" in df.columns:
        df = df[df["artifact_status"].fillna("").isin(artifact_status_vals)]

    return df.reset_index(drop=True)
