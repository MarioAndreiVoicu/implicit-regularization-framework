from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from framework.services.analysis_records import load_analysis_items, load_analysis_records
from framework.services.run_repository import ReadRunRepository
from framework.services.export_service import ExportBundle, LocalExportService
from framework.services.comparison_audit import audit_run_pair
from framework.services.failure_analysis import build_failure_table, diagnostics_table, instability_seed_group_table
from framework.services.frontier import frontier_matched_points, frontier_metric_options, frontier_run_points, frontier_seed_group_points, match_axis_options
from framework.services.local_runs import find_matched_runs, matched_candidates_table, matched_config_diff, matched_final_metrics_table, metrics_for_config
from framework.services.seed_group_records import (
    find_matched_seed_groups,
    load_seed_group_analysis_items,
    paired_seed_delta_summary,
    paired_seed_final_metric_deltas,
    paired_seed_pairing_summary,
    seed_group_final_metrics,
    seed_group_pairing_table,
    seed_group_records_table,
    seed_group_trajectories,
)


@dataclass(frozen=True)
class FreeCompareResult:
    summary_df: pd.DataFrame
    diff_df: pd.DataFrame
    overlay_items: list[tuple[str, dict, dict, pd.DataFrame]]
    selected_run_ids: list[str]
    overlay_metrics: list[str]


@dataclass(frozen=True)
class MatchedCompareContext:
    labels: list[str]
    varying_axis_options: list[str]


@dataclass(frozen=True)
class MatchedCompareResult:
    reference: dict
    matches: list[dict]
    candidates_df: pd.DataFrame
    candidate: dict
    audit: object
    config_diff_df: pd.DataFrame
    final_metrics_df: pd.DataFrame
    trajectory_metrics: list[str]




@dataclass(frozen=True)
class PairedSeedGroupCompareResult:
    reference_group: dict
    matches: list[dict]
    candidates_df: pd.DataFrame
    candidate_group: dict
    audit: object
    pairing_summary: dict
    pairing_df: pd.DataFrame
    per_seed_delta_df: pd.DataFrame
    delta_summary_df: pd.DataFrame

@dataclass(frozen=True)
class SeedAggregationResult:
    groups_df: pd.DataFrame
    group: dict
    agg_final_df: pd.DataFrame
    run_rows_df: pd.DataFrame
    trajectory_df: pd.DataFrame
    trajectory_metrics: list[str]


@dataclass(frozen=True)
class FailureAnalysisContext:
    labels: list[str]
    varying_axis_options: list[str]


@dataclass(frozen=True)
class FailureAnalysisResult:
    filtered_df: pd.DataFrame
    selected_run: dict | None
    diagnostics_df: pd.DataFrame
    seed_instability_df: pd.DataFrame
    overlay_items: list[tuple[str, dict, dict, pd.DataFrame]]


@dataclass(frozen=True)
class FrontierContext:
    labels: list[str]
    varying_axis_options: list[str]


@dataclass(frozen=True)
class FrontierPointsResult:
    points_df: pd.DataFrame
    fit_options: list[str]
    complexity_options: list[str]




def _unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        value = str(value)
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _numeric_metric_columns(df: pd.DataFrame, *, excluded: set[str] | None = None) -> set[str]:
    excluded = excluded or set()
    if not isinstance(df, pd.DataFrame) or df.empty:
        return set()
    cols: set[str] = set()
    for col in df.columns:
        if col in excluded:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().any():
            cols.add(str(col))
    return cols


def _common_trajectory_metrics(items: list[dict]) -> list[str]:
    if not items:
        return []
    common: set[str] | None = None
    for item in items:
        numeric = _numeric_metric_columns(item.get("metrics"), excluded={"step", "epoch", "iteration"})
        common = numeric if common is None else common & numeric
    if not common:
        return []

    candidates: list[str] = []
    for item in items:
        candidates.extend(metrics_for_config(item.get("config"), "trajectory"))
    candidates.extend([
        "val_rmse",
        "test_rmse",
        "train_rmse",
        "rel_recon_error",
        "relative_reconstruction_error",
        "objective",
        "loss",
        "gradient_norm",
    ])
    ordered = _unique_preserve_order([*candidates, *sorted(common)])
    return [metric for metric in ordered if metric in common]

class CompareService:
    def __init__(
        self,
        *,
        read_repository: ReadRunRepository,
        config_diff_fn: Callable[[list[dict]], pd.DataFrame],
        export_service: LocalExportService | None = None,
    ) -> None:
        self._read_repository = read_repository
        self._config_diff_fn = config_diff_fn
        self._export_service = export_service

    def load_records(self) -> pd.DataFrame:
        return load_analysis_records(self._read_repository)

    def _load_items(self, *, include_metrics: bool = True) -> list[dict]:
        return load_analysis_items(self._read_repository, include_metrics=include_metrics)

    def free_compare_result(self, selected_run_ids: list[str]) -> FreeCompareResult:
        loaded_items = self._load_items(include_metrics=True)
        by_run_id = {item["run_id"]: item for item in loaded_items}
        chosen = [by_run_id[run_id] for run_id in selected_run_ids if run_id in by_run_id]
        summary_rows = []
        for item in chosen:
            row = {"run": item["run_id"]}
            row.update(item["summary"])
            summary_rows.append(row)
        summary_df = pd.DataFrame(summary_rows)
        diff_df = self._config_diff_fn([item["config"] for item in chosen]) if chosen else pd.DataFrame()
        overlay_metrics = _common_trajectory_metrics(chosen)
        overlay_items = [
            (item["run_id"], item["config"], item["summary"], item["metrics"]) for item in chosen
        ]
        return FreeCompareResult(
            summary_df=summary_df,
            diff_df=diff_df,
            overlay_items=overlay_items,
            selected_run_ids=[item["run_id"] for item in chosen],
            overlay_metrics=overlay_metrics,
        )


    def build_free_compare_bundle(
        self,
        *,
        selected_run_ids: list[str],
        include_run_reports: bool,
        include_artifacts: bool,
    ) -> ExportBundle:
        if self._export_service is None:
            raise RuntimeError("Compare export service is not configured.")
        result = self.free_compare_result(selected_run_ids)
        run_dirs = [self._read_repository.resolve_run_dir(run_id) for run_id in result.selected_run_ids]
        return self._export_service.build_compare_bundle(
            run_dirs=run_dirs,
            include_run_reports=include_run_reports,
            include_artifacts=include_artifacts,
            compare_summary={
                "mode": "free_compare",
                "selected_run_ids": result.selected_run_ids,
                "overlay_metrics": result.overlay_metrics,
                "run_count": len(result.selected_run_ids),
            },
            compare_tables={
                "summary": result.summary_df,
                "config_diff": result.diff_df,
            },
            context={"compare_mode": "free_compare"},
        )

    def build_matched_compare_bundle(
        self,
        *,
        result: MatchedCompareResult,
        include_run_reports: bool,
        include_artifacts: bool,
        filename: str = "matched_compare_bundle.zip",
    ) -> ExportBundle:
        if self._export_service is None:
            raise RuntimeError("Compare export service is not configured.")
        run_dirs = [
            self._read_repository.resolve_run_dir(result.reference["run_id"]),
            self._read_repository.resolve_run_dir(result.candidate["run_id"]),
        ]
        compare_summary = {
            "mode": "matched_compare",
            "reference_run_id": result.reference["run_id"],
            "candidate_run_id": result.candidate["run_id"],
            "audit": None if result.audit is None else {
                "status": result.audit.status,
                "label": result.audit.validity_label,
                "severity": result.audit.severity,
                "summary": result.audit.summary,
                "seed_alignment": result.audit.seed_alignment,
                "held_fixed_violations": list(result.audit.held_fixed_violations),
                "protocol_conflicts": list(result.audit.protocol_conflicts),
                "core_mismatches": list(result.audit.core_mismatches),
                "secondary_confounds": list(result.audit.secondary_confounds),
            },
        }
        return self._export_service.build_compare_bundle(
            run_dirs=run_dirs,
            include_run_reports=include_run_reports,
            include_artifacts=include_artifacts,
            filename=filename,
            compare_summary=compare_summary,
            compare_tables={
                "candidates": result.candidates_df,
                "config_diff": result.config_diff_df,
                "final_metrics": result.final_metrics_df,
            },
            context={
                "compare_mode": "matched_compare",
                "reference_run_id": result.reference["run_id"],
                "candidate_run_id": result.candidate["run_id"],
            },
        )

    def build_paired_seed_group_compare_bundle(
        self,
        *,
        result: PairedSeedGroupCompareResult,
        include_run_reports: bool,
        include_artifacts: bool,
        filename: str = "paired_seed_group_compare_bundle.zip",
    ) -> ExportBundle:
        if self._export_service is None:
            raise RuntimeError("Compare export service is not configured.")
        ref_run_ids = [item["run_id"] for item in result.reference_group.get("items", [])]
        cand_run_ids = [item["run_id"] for item in result.candidate_group.get("items", [])]
        run_dirs = [self._read_repository.resolve_run_dir(run_id) for run_id in [*ref_run_ids, *cand_run_ids]]
        compare_summary = {
            "mode": "paired_seed_group_compare",
            "reference_group_id": result.reference_group.get("seed_group_id"),
            "candidate_group_id": result.candidate_group.get("seed_group_id"),
            "pairing_summary": dict(result.pairing_summary),
            "audit": None if result.audit is None else {
                "status": result.audit.status,
                "label": result.audit.validity_label,
                "severity": result.audit.severity,
                "summary": result.audit.summary,
                "seed_alignment": result.audit.seed_alignment,
                "held_fixed_violations": list(result.audit.held_fixed_violations),
                "protocol_conflicts": list(result.audit.protocol_conflicts),
                "core_mismatches": list(result.audit.core_mismatches),
                "secondary_confounds": list(result.audit.secondary_confounds),
            },
        }
        return self._export_service.build_compare_bundle(
            run_dirs=run_dirs,
            include_run_reports=include_run_reports,
            include_artifacts=include_artifacts,
            filename=filename,
            compare_summary=compare_summary,
            compare_tables={
                "candidates": result.candidates_df,
                "pairing": result.pairing_df,
                "per_seed_deltas": result.per_seed_delta_df,
                "delta_summary": result.delta_summary_df,
            },
            context={
                "compare_mode": "paired_seed_group_compare",
                "reference_group_id": result.reference_group.get("seed_group_id"),
                "candidate_group_id": result.candidate_group.get("seed_group_id"),
            },
        )

    def matched_compare_context(self) -> MatchedCompareContext:
        items = self._load_items(include_metrics=True)
        return MatchedCompareContext(
            labels=[item["run_id"] for item in items],
            varying_axis_options=match_axis_options(),
        )

    def selected_pair_audit(self, *, reference_run_id: str, candidate_run_id: str, varying_axis: str):
        loaded_runs = self._load_items(include_metrics=False)
        by_run_id = {item["run_id"]: item for item in loaded_runs}
        reference = by_run_id.get(reference_run_id)
        candidate = by_run_id.get(candidate_run_id)
        if reference is None or candidate is None:
            return None
        return audit_run_pair(
            reference["config"],
            candidate["config"],
            varying_axis,
            reference_metadata=reference.get("metadata"),
            candidate_metadata=candidate.get("metadata"),
        )

    def matched_compare_result(self, *, reference_run_id: str, varying_axis: str, strict_exact_match: bool, candidate_run_id: str | None = None) -> MatchedCompareResult | None:
        loaded_runs = self._load_items(include_metrics=True)
        reference, matches = find_matched_runs(loaded_runs, reference_run_id, varying_axis, strict_exact_match)
        if reference is None:
            return None
        audits_by_run_id = {
            item["run_id"]: audit_run_pair(
                reference["config"],
                item["config"],
                varying_axis,
                reference_metadata=reference.get("metadata"),
                candidate_metadata=item.get("metadata"),
            )
            for item in matches
        }
        candidates_df = matched_candidates_table(reference, matches).copy()
        if not candidates_df.empty:
            candidates_df["audit_status"] = candidates_df["run_id"].map(lambda rid: audits_by_run_id[rid].status)
            candidates_df["audit_label"] = candidates_df["run_id"].map(lambda rid: audits_by_run_id[rid].validity_label)
            candidates_df["audit_severity"] = candidates_df["run_id"].map(lambda rid: audits_by_run_id[rid].severity)
            candidates_df["audit_seed_alignment"] = candidates_df["run_id"].map(lambda rid: audits_by_run_id[rid].seed_alignment)
            candidates_df["audit_confound_count"] = candidates_df["run_id"].map(
                lambda rid: len(audits_by_run_id[rid].secondary_confounds)
                + len(audits_by_run_id[rid].held_fixed_violations)
                + len(audits_by_run_id[rid].core_mismatches)
                + len(audits_by_run_id[rid].protocol_conflicts)
            )
            candidates_df["audit_summary"] = candidates_df["run_id"].map(lambda rid: audits_by_run_id[rid].summary)
        if not matches:
            return MatchedCompareResult(
                reference=reference,
                matches=matches,
                candidates_df=candidates_df,
                candidate={},
                audit=None,
                config_diff_df=pd.DataFrame(),
                final_metrics_df=pd.DataFrame(),
                trajectory_metrics=[],
            )
        selected_candidate_id = candidate_run_id or matches[0]["run_id"]
        candidate = next((item for item in matches if item["run_id"] == selected_candidate_id), matches[0])
        audit = audits_by_run_id[candidate["run_id"]]
        final_metrics = metrics_for_config(reference["config"], "final")
        final_df = matched_final_metrics_table(reference["summary"], candidate["summary"], metrics=final_metrics)
        return MatchedCompareResult(
            reference=reference,
            matches=matches,
            candidates_df=candidates_df,
            candidate=candidate,
            audit=audit,
            config_diff_df=matched_config_diff(reference["config"], candidate["config"], varying_axis),
            final_metrics_df=final_df,
            trajectory_metrics=metrics_for_config(reference["config"], "trajectory"),
        )

    def seed_aggregation_result(self, selected_group_id: str | None = None) -> SeedAggregationResult | None:
        groups = load_seed_group_analysis_items(self._read_repository, include_metrics=True)
        if not groups:
            return None
        groups_df = seed_group_records_table(groups)
        group_id = selected_group_id or str(groups_df.iloc[0]["seed_group_id"])
        group = next(item for item in groups if item["seed_group_id"] == group_id)
        final_metrics = metrics_for_config(group.get("config", {}), "final")
        agg_final_df = seed_group_final_metrics(group, final_metrics)
        run_rows_df = pd.DataFrame(
            [
                {
                    "run_id": item["run_id"],
                    "seed": item["config"].get("seed"),
                    "status": item["summary"].get("status"),
                    "final_objective": item["summary"].get("final_objective"),
                    "final_rel_recon_error": item["summary"].get("final_rel_recon_error"),
                    "final_val_rmse": item["summary"].get("final_val_rmse"),
                    "final_test_rmse": item["summary"].get("final_test_rmse"),
                }
                for item in group["items"]
            ]
        )
        traj_metrics = metrics_for_config(group.get("config", {}), "trajectory")
        trajectory_df = seed_group_trajectories(group, traj_metrics)
        return SeedAggregationResult(
            groups_df=groups_df,
            group=group,
            agg_final_df=agg_final_df,
            run_rows_df=run_rows_df,
            trajectory_df=trajectory_df,
            trajectory_metrics=traj_metrics,
        )



    def paired_seed_group_compare_result(
        self,
        *,
        reference_group_id: str,
        varying_axis: str,
        strict_exact_match: bool,
        candidate_group_id: str | None = None,
    ) -> PairedSeedGroupCompareResult | None:
        groups = load_seed_group_analysis_items(self._read_repository, include_metrics=True)
        if not groups:
            return None
        reference_group, matches = find_matched_seed_groups(groups, reference_group_id, varying_axis, strict_exact_match)
        if reference_group is None:
            return None
        audits_by_group_id = {
            item["seed_group_id"]: audit_run_pair(
                reference_group.get("compare_config", reference_group.get("config", {})),
                item.get("compare_config", item.get("config", {})),
                varying_axis,
                reference_metadata=reference_group.get("metadata"),
                candidate_metadata=item.get("metadata"),
            )
            for item in matches
        }
        rows: list[dict] = []
        for item in matches:
            pairing_df = seed_group_pairing_table(reference_group, item)
            pairing_summary = paired_seed_pairing_summary(pairing_df)
            rows.append(
                {
                    "seed_group_id": item.get("seed_group_id"),
                    "varying_value": item.get("varying_value"),
                    "seed_count": item.get("seed_count"),
                    "paired_seed_count": pairing_summary.get("paired_seed_count", 0),
                    "reference_only_seed_count": pairing_summary.get("reference_only_seed_count", 0),
                    "candidate_only_seed_count": pairing_summary.get("candidate_only_seed_count", 0),
                    "pairing_complete": pairing_summary.get("complete", False),
                    "audit_status": audits_by_group_id[item["seed_group_id"]].status,
                    "audit_label": audits_by_group_id[item["seed_group_id"]].validity_label,
                    "audit_severity": audits_by_group_id[item["seed_group_id"]].severity,
                    "audit_summary": audits_by_group_id[item["seed_group_id"]].summary,
                }
            )
        candidates_df = pd.DataFrame(rows)
        if not matches:
            return PairedSeedGroupCompareResult(
                reference_group=reference_group,
                matches=matches,
                candidates_df=candidates_df,
                candidate_group={},
                audit=None,
                pairing_summary={},
                pairing_df=pd.DataFrame(),
                per_seed_delta_df=pd.DataFrame(),
                delta_summary_df=pd.DataFrame(),
            )
        selected_candidate_id = candidate_group_id or matches[0]["seed_group_id"]
        candidate_group = next((item for item in matches if item["seed_group_id"] == selected_candidate_id), matches[0])
        pairing_df = seed_group_pairing_table(reference_group, candidate_group)
        final_metrics = metrics_for_config(reference_group.get("config", {}), "final")
        per_seed_delta_df = paired_seed_final_metric_deltas(reference_group, candidate_group, final_metrics)
        delta_summary_df = paired_seed_delta_summary(per_seed_delta_df, pairing_df)
        return PairedSeedGroupCompareResult(
            reference_group=reference_group,
            matches=matches,
            candidates_df=candidates_df,
            candidate_group=candidate_group,
            audit=audits_by_group_id[candidate_group["seed_group_id"]],
            pairing_summary=paired_seed_pairing_summary(pairing_df),
            pairing_df=pairing_df,
            per_seed_delta_df=per_seed_delta_df,
            delta_summary_df=delta_summary_df,
        )

    def failure_context(self) -> FailureAnalysisContext:
        items = self._load_items(include_metrics=True)
        return FailureAnalysisContext(labels=[item["run_id"] for item in items], varying_axis_options=match_axis_options())

    def failure_result(
        self,
        *,
        source: str,
        reference_run_id: str | None,
        varying_axis: str,
        strict_exact_match: bool,
        only_problematic: bool,
        selected_issues: list[str],
        selected_group: str,
        run_id: str | None,
    ) -> FailureAnalysisResult | None:
        loaded_runs = self._load_items(include_metrics=True)
        if not loaded_runs:
            return None
        subset = loaded_runs
        if source == "Matched set" and reference_run_id:
            reference, matches = find_matched_runs(loaded_runs, reference_run_id, varying_axis, strict_exact_match)
            subset = ([reference] if reference is not None else []) + matches
        failures_df = build_failure_table(subset)
        if failures_df.empty:
            return FailureAnalysisResult(pd.DataFrame(), None, pd.DataFrame(), pd.DataFrame(), [])
        filtered = failures_df.copy()
        if only_problematic:
            filtered = filtered[filtered["primary_issue"] != "ok"]
        if selected_issues:
            filtered = filtered[filtered["primary_issue"].isin(selected_issues)]
        if selected_group:
            filtered = filtered[filtered["group_id"].astype(str) == selected_group]
        if filtered.empty:
            return FailureAnalysisResult(filtered, None, pd.DataFrame(), pd.DataFrame(), [])
        selected_run_id = run_id or str(filtered.iloc[0]["run_id"])
        selected = next((item for item in subset if item["run_id"] == selected_run_id), None)
        diagnostics_df = pd.DataFrame()
        overlay_items: list[tuple[str, dict, dict, pd.DataFrame]] = []
        if selected is not None:
            diagnostics_df = diagnostics_table(selected["config"], selected["summary"], selected["metrics"])
            overlay_items = [(selected["run_id"], selected["config"], selected["summary"], selected["metrics"])]
        seed_df = instability_seed_group_table(subset)
        return FailureAnalysisResult(filtered, selected, diagnostics_df, seed_df, overlay_items)

    def frontier_context(self) -> FrontierContext:
        items = self._load_items(include_metrics=True)
        return FrontierContext(labels=[item["run_id"] for item in items], varying_axis_options=match_axis_options())

    def frontier_points_result(self, *, source_mode: str, reference_run_id: str | None, varying_axis: str, strict_exact_match: bool) -> FrontierPointsResult | None:
        loaded_runs = self._load_items(include_metrics=True)
        if not loaded_runs:
            return None
        if source_mode == "All runs":
            points_df = frontier_run_points(loaded_runs)
        elif source_mode == "Seed groups":
            points_df = frontier_seed_group_points(loaded_runs)
        else:
            if not reference_run_id:
                return FrontierPointsResult(pd.DataFrame(), [], [])
            points_df = frontier_matched_points(loaded_runs, reference_run_id, varying_axis, strict_exact_match)
        configs = [{"problem_family": row.get("problem_family"), "data_source": row.get("data_source")} for row in points_df.to_dict(orient="records")]
        fit_options, complexity_options = frontier_metric_options(configs)
        return FrontierPointsResult(points_df=points_df, fit_options=fit_options, complexity_options=complexity_options)


__all__ = [
    "CompareService",
    "FailureAnalysisContext",
    "FailureAnalysisResult",
    "FreeCompareResult",
    "FrontierContext",
    "FrontierPointsResult",
    "MatchedCompareContext",
    "MatchedCompareResult",
    "PairedSeedGroupCompareResult",
    "SeedAggregationResult",
]
