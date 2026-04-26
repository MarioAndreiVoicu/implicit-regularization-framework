from __future__ import annotations

import json
from pathlib import Path

from framework.core.models import RunResult
from framework.headless import (
    execute_headless_new_run,
    execute_headless_sweep,
    load_request_file,
    new_run_request_from_dict,
    plan_request_file,
    run_request_file,
    sweep_request_from_dict,
)
from framework.services.new_run import NewRunRequest
from framework.services.sweeps import SweepRequest


def _fake_run_experiment(spec):
    return RunResult(
        spec=spec,
        metrics={"step": [0], "objective": [0.0]},
        summary={"status": "completed", "resolved_device": "cpu", "resolved_dtype": "float64"},
        artifacts={},
    )


def test_new_run_request_from_dict_ignores_unknown_fields():
    request = new_run_request_from_dict({"n": 12, "seed": 3, "unknown": "ignored"})
    assert isinstance(request, NewRunRequest)
    assert request.n == 12
    assert request.seed == 3


def test_new_run_request_from_dict_accepts_protocol():
    request = new_run_request_from_dict(
        {
            "n": 12,
            "seed": 3,
            "protocol": {
                "protocol_id": "p1",
                "title": "Protocol 1",
                "hypothesis": "h",
                "manipulated_factor": "lr",
                "held_fixed_factors": ["seed", "optimizer"],
                "primary_metrics": ["final_rel_recon_error"],
                "seed_policy": "matched",
            },
        }
    )
    assert request.protocol is not None
    assert request.protocol.protocol_id == "p1"


def test_sweep_request_from_dict_builds_nested_request():
    request = sweep_request_from_dict(
        {
            "kind": "1D",
            "preset": "lr",
            "base_request": {"problem_family": "psd_sensing", "n": 8, "m": 20, "steps": 5, "log_every": 1},
            "fields": ["lr"],
            "values": {"lr": [1e-3, 1e-2]},
        }
    )
    assert isinstance(request, SweepRequest)
    assert request.fields == ("lr",)
    assert request.base_request.problem_family == "psd_sensing"


def test_sweep_request_from_dict_accepts_protocol_and_coupling():
    request = sweep_request_from_dict(
        {
            "kind": "2D",
            "preset": "custom",
            "base_request": {"problem_family": "psd_sensing", "n": 8, "m": 20, "factor_rank": 3, "steps": 5, "log_every": 1},
            "fields": ["lr", "init_scale"],
            "values": {"lr": [1e-3, 1e-2], "init_scale": [1e-2, 1e-1]},
            "coupling": "paired",
            "protocol": {
                "protocol_id": "p2",
                "title": "Protocol 2",
                "hypothesis": "h2",
                "manipulated_factor": "lr",
                "held_fixed_factors": ["seed", "optimizer"],
                "primary_metrics": ["final_rel_recon_error"],
                "seed_policy": "matched",
            },
        }
    )
    assert request.coupling == "paired"
    assert request.protocol is not None
    assert request.protocol.protocol_id == "p2"


def test_load_request_file_supports_new_run_and_sweep(tmp_path: Path):
    new_run_path = tmp_path / "new_run.json"
    new_run_path.write_text(json.dumps({"n": 9, "seed": 2}), encoding="utf-8")
    assert load_request_file(new_run_path, request_type="new-run").n == 9

    sweep_path = tmp_path / "sweep.json"
    sweep_path.write_text(
        json.dumps({
            "kind": "1D",
            "preset": "lr",
            "base_request": {"problem_family": "psd_sensing", "n": 8, "m": 20, "steps": 5, "log_every": 1},
            "fields": ["lr"],
            "values": {"lr": [1e-3, 1e-2]},
        }),
        encoding="utf-8",
    )
    assert load_request_file(sweep_path, request_type="sweep").fields == ("lr",)


def test_plan_request_file_supports_planner_payload(tmp_path: Path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"problem_family": "psd_sensing", "base_overrides": {"n": 8}}), encoding="utf-8")
    planned = plan_request_file(path, request_type="new-run-plan")
    assert planned.request.problem_family == "psd_sensing"


def test_execute_headless_new_run_writes_summary(tmp_path: Path):
    request = NewRunRequest(n=6, r_star=2, m=12, factor_rank=3, steps=2, log_every=1, num_eigs=3)
    summary = execute_headless_new_run(request, runs_dir=tmp_path / "runs", run_experiment_fn=_fake_run_experiment)
    assert summary["request_type"] == "new_run"
    assert summary["execution_mode"] == "direct"
    assert summary["saved_run_ids"]
    assert summary["run_paths"] == [str(tmp_path / "runs" / summary["saved_run_ids"][0])]


def test_execute_headless_new_run_includes_protocol_id(tmp_path: Path):
    request = new_run_request_from_dict({
        "n": 6,
        "r_star": 2,
        "m": 12,
        "factor_rank": 3,
        "steps": 2,
        "log_every": 1,
        "num_eigs": 3,
        "protocol": {
            "protocol_id": "proto_run",
            "title": "Proto Run",
            "hypothesis": "h",
            "manipulated_factor": "lr",
            "held_fixed_factors": ["seed", "optimizer"],
            "primary_metrics": ["final_rel_recon_error"],
            "seed_policy": "matched",
        },
    })
    summary = execute_headless_new_run(request, runs_dir=tmp_path / "runs", run_experiment_fn=_fake_run_experiment)
    assert summary["protocol_id"] == "proto_run"


def test_execute_headless_sweep_writes_sweep_metadata(tmp_path: Path):
    request = SweepRequest(
        kind="1D",
        preset="lr",
        base_request=NewRunRequest(n=6, r_star=2, m=12, factor_rank=3, steps=2, log_every=1, num_eigs=3),
        fields=("lr",),
        values={"lr": [1e-3, 1e-2]},
    )
    summary = execute_headless_sweep(
        request,
        runs_dir=tmp_path / "runs",
        sweeps_dir=tmp_path / "sweeps",
        run_experiment_fn=_fake_run_experiment,
    )
    assert summary["request_type"] == "sweep"
    assert summary["execution_mode"] == "direct"
    assert summary["succeeded_count"] == 2
    assert Path(summary["sweep_path"]).exists()


def test_run_request_file_supports_direct_sweep_plan(tmp_path: Path):
    plan_path = tmp_path / "sweep_plan.json"
    plan_path.write_text(json.dumps({
        "kind": "1D",
        "preset": "lr",
        "problem_family": "psd_sensing",
        "data_source": "synthetic",
        "seed_plan": {"mode": "generated", "start_seed": 0, "num_seeds": 1, "seed_step": 1},
        "base_overrides": {"n": 8, "r_star": 2, "m": 16, "factor_rank": 3, "steps": 2, "log_every": 1, "num_eigs": 3},
        "axes": [{"field": "lr", "mode": "explicit", "values": [1e-3, 1e-2]}]
    }), encoding="utf-8")

    summary = run_request_file(
        plan_path,
        request_type="sweep-plan",
        runs_dir=tmp_path / "runs",
        sweeps_dir=tmp_path / "sweeps",
        run_experiment_fn=_fake_run_experiment,
    )
    assert summary["request_type"] == "sweep"
    assert summary["succeeded_count"] == 2
