from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Iterable, Iterator

import matplotlib.pyplot as plt
import pandas as pd

from framework.services.local_runs import x_col
from framework.storage.local import load_run


def _safe_read_bytes(path: Path) -> bytes | None:
    try:
        if path.exists() and path.is_file():
            return path.read_bytes()
    except Exception:
        return None
    return None


def _safe_read_text(path: Path) -> str | None:
    try:
        if path.exists() and path.is_file():
            return path.read_text(encoding="utf-8")
    except Exception:
        return None
    return None


def _write_plot_png(df: pd.DataFrame, x: str, y: str, title: str, out_path: Path) -> None:
    if y not in df.columns or df[y].isna().all():
        return
    fig = plt.figure()
    try:
        plt.plot(df[x], df[y])
        plt.title(title)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)


def ensure_report_and_plots(
    run_dir: Path,
    cfg: dict | None = None,
    summary: dict | None = None,
    df: pd.DataFrame | None = None,
) -> None:
    run_dir = Path(run_dir)

    if cfg is None or summary is None or df is None:
        loaded = load_run(str(run_dir))
        cfg = loaded["config"]
        summary = loaded["summary"]
        df = loaded["metrics"]
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

    plots_dir = run_dir / "plots"
    report_dir = run_dir / "report"
    plots_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    x = x_col(df)

    problem_family = cfg.get("problem_family", "psd_sensing")
    data_source = cfg.get("data_source", "synthetic")
    loss_metric = "loss" if data_source == "synthetic" else "train_loss"
    recon_metric = "rel_recon_error" if data_source == "synthetic" else "val_rmse"
    recon_title = "Relative reconstruction error vs step" if data_source == "synthetic" else "Validation RMSE vs step"
    _write_plot_png(df, x, loss_metric, f"{loss_metric} vs step", plots_dir / f"{loss_metric}.png")
    _write_plot_png(df, x, recon_metric, recon_title, plots_dir / f"{recon_metric}.png")
    if problem_family == "psd_sensing":
        extra_metric = "trace_X"
        extra_title = "trace(UUᵀ) vs step"
    elif data_source in {"movielens_100k", "custom_triplets_csv"}:
        extra_metric = "test_rmse"
        extra_title = "Test RMSE vs step"
    else:
        extra_metric = "frobenius_norm_X"
        extra_title = "||X||_F vs step"
    extra_plot_name = f"{extra_metric}.png"
    _write_plot_png(df, x, extra_metric, extra_title, plots_dir / extra_plot_name)

    html = f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\">
  <title>Run report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    h1, h2 {{ margin-bottom: 8px; }}
    pre {{ background: #f5f5f5; padding: 12px; overflow-x: auto; }}
    img {{ max-width: 100%; margin-bottom: 20px; border: 1px solid #ddd; }}
  </style>
</head>
<body>
  <h1>Run report</h1>

  <h2>Summary</h2>
  <pre>{json.dumps(summary, indent=2)}</pre>

  <h2>Config</h2>
  <pre>{json.dumps(cfg, indent=2)}</pre>

  <h2>Plots</h2>
  <div><img src=\"../plots/{loss_metric}.png\" alt=\"{loss_metric}\"></div>
  <div><img src=\"../plots/{recon_metric}.png\" alt=\"{recon_metric}\"></div>
  <div><img src=\"../plots/{extra_plot_name}\" alt=\"{extra_metric}\"></div>
</body>
</html>
"""
    (report_dir / "report.html").write_text(html, encoding="utf-8")


def _zip_add_if_present(zf: zipfile.ZipFile, real_path: Path, arcname: str) -> None:
    data = _safe_read_bytes(real_path)
    if data is not None:
        zf.writestr(arcname, data)


def _iter_existing_relative_files(root: Path, relative_dir: str) -> Iterator[str]:
    base = root / relative_dir
    if not base.exists() or not base.is_dir():
        return
    for path in sorted(p for p in base.rglob("*") if p.is_file()):
        yield path.relative_to(root).as_posix()


def _iter_run_export_paths(run_dir: Path, *, include_run_reports: bool, include_artifacts: bool) -> Iterator[str]:
    for rel in [
        "config.json",
        "summary.json",
        "metadata.json",
        "metrics.csv",
        "artifacts_manifest.json",
    ]:
        yield rel
    if include_artifacts:
        yield "artifacts.npz"
    if include_run_reports:
        yield from _iter_existing_relative_files(run_dir, "plots")
        yield from _iter_existing_relative_files(run_dir, "report")


def _zip_add_run_dir(
    zf: zipfile.ZipFile,
    run_dir: Path,
    include_run_reports: bool,
    include_artifacts: bool,
) -> None:
    base = run_dir.name
    for rel in _iter_run_export_paths(
        run_dir,
        include_run_reports=include_run_reports,
        include_artifacts=include_artifacts,
    ):
        _zip_add_if_present(zf, run_dir / rel, f"{base}/{rel}")


def _build_run_dirs_bundle_zip_bytes(
    run_dirs: Iterable[Path],
    *,
    include_run_reports: bool,
    include_artifacts: bool,
) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rd in run_dirs:
            _zip_add_run_dir(
                zf,
                Path(rd),
                include_run_reports=include_run_reports,
                include_artifacts=include_artifacts,
            )
    return buf.getvalue()


def build_run_bundle_zip_bytes(
    selected_dir: Path,
    include_artifacts: bool,
    include_report: bool,
) -> bytes:
    return _build_run_dirs_bundle_zip_bytes(
        [Path(selected_dir)],
        include_run_reports=include_report,
        include_artifacts=include_artifacts,
    )


def build_runs_collection_zip_bytes(
    run_dirs: Iterable[Path],
    include_run_reports: bool,
    include_artifacts: bool,
) -> bytes:
    return _build_run_dirs_bundle_zip_bytes(
        run_dirs,
        include_run_reports=include_run_reports,
        include_artifacts=include_artifacts,
    )


def build_compare_bundle_zip_bytes(
    run_dirs: Iterable[Path],
    include_run_reports: bool,
    include_artifacts: bool,
) -> bytes:
    return _build_run_dirs_bundle_zip_bytes(
        run_dirs,
        include_run_reports=include_run_reports,
        include_artifacts=include_artifacts,
    )


def build_sweep_bundle_zip_bytes(
    group_id: str,
    loaded_runs: list[tuple[Path, dict, dict, pd.DataFrame]],
    sweep_field: str,
    sweep_summary_df: pd.DataFrame,
    include_run_reports: bool,
    include_artifacts: bool,
    sweep_meta: dict | None = None,
) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{group_id}/meta.json", json.dumps(sweep_meta or {"group_id": group_id}, indent=2))
        zf.writestr(f"{group_id}/summary.csv", sweep_summary_df.to_csv(index=False))

        if (
            sweep_field in sweep_summary_df.columns
            and "final_rel_recon_error" in sweep_summary_df.columns
            and len(sweep_summary_df) > 0
        ):
            fig = plt.figure()
            try:
                xs = sweep_summary_df[sweep_field]
                ys = sweep_summary_df["final_rel_recon_error"]
                plt.plot(xs, ys, marker="o")
                plt.title("final_rel_recon_error vs sweep value")
                plt.xlabel(sweep_field)
                plt.ylabel("final_rel_recon_error")
                plt.tight_layout()
                img = io.BytesIO()
                fig.savefig(img, format="png", dpi=150, bbox_inches="tight")
                zf.writestr(f"{group_id}/plots/sweep_curve.png", img.getvalue())
            finally:
                plt.close(fig)

        for rd, _, _, _ in loaded_runs:
            _zip_add_run_dir(
                zf,
                Path(rd),
                include_run_reports=include_run_reports,
                include_artifacts=include_artifacts,
            )

    return buf.getvalue()
