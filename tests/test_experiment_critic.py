from framework.services.experiment_critic import audit_new_run_request, audit_sweep_request, audit_sweep_spec
from framework.services.new_run import NewRunRequest
from framework.services.sweeps import SweepRequest, build_sweep_spec


def test_audit_new_run_request_warns_on_real_data_metrics_and_low_seed_count():
    request = NewRunRequest(
        problem_family="matrix_completion",
        data_source="movielens_100k",
        seed_mode="generated",
        start_seed=0,
        num_seeds=2,
        early_stopping_enabled=False,
        parameterization="factorized_uv",
        optimizer="gd",
    )
    messages = audit_new_run_request(request).messages
    assert any("Only 2 seeds" in message for message in messages)
    assert any("synthetic-only recovery metrics" in message for message in messages)
    assert any("early stopping" in message for message in messages)


def test_audit_sweep_request_warns_on_multifactor_and_duplicate_values():
    request = SweepRequest(
        kind="2D",
        preset="grid",
        base_request=NewRunRequest(problem_family="psd_sensing", data_source="synthetic", seed_mode="generated", start_seed=0, num_seeds=2),
        fields=("lr", "init_scale"),
        values={"lr": [1e-3, 1e-3], "init_scale": [1e-2, 1e-1]},
    )
    messages = audit_sweep_request(request).messages
    assert any("interaction study" in message for message in messages)
    assert any("duplicate values" in message for message in messages)


def test_audit_sweep_spec_warns_on_real_data_metric_scope(tmp_path):
    csv_path = tmp_path / "ratings.csv"
    csv_path.write_text("row,col,value\n0,0,5.0\n0,1,3.0\n1,0,4.0\n1,1,2.0\n", encoding="utf-8")
    request = SweepRequest(
        kind="1D",
        preset="lr",
        base_request=NewRunRequest(
            problem_family="matrix_completion",
            data_source="custom_triplets_csv",
            custom_dataset_path=str(csv_path),
            seed_mode="generated",
            start_seed=0,
            num_seeds=2,
            parameterization="factorized_uv",
            optimizer="gd",
        ),
        fields=("lr",),
        values={"lr": [1e-3, 1e-2]},
    )
    spec = build_sweep_spec(request)
    messages = audit_sweep_spec(spec).messages
    assert any("synthetic-only recovery metrics" in message for message in messages)
    assert any("Only 2 seeds" in message for message in messages)
