import pytest

from framework.services.experiment_protocols import ExperimentProtocol, normalize_experiment_protocol, protocol_metadata


def test_experiment_protocol_round_trip_and_metadata_projection():
    protocol = ExperimentProtocol(
        protocol_id="proto_compare_lr",
        title="Learning-rate comparison",
        hypothesis="Lower learning rate prefers lower-complexity solutions at matched fit.",
        manipulated_factor="lr",
        held_fixed_factors=("dataset_version", "split_version", "parameterization", "optimizer"),
        primary_metrics=("final_rel_recon_error",),
        secondary_metrics=("final_objective",),
        seed_policy="paired-10-seed",
        notes="Match seeds exactly.",
    )

    payload = protocol.to_dict()
    restored = ExperimentProtocol.from_dict(payload)
    meta = protocol_metadata(restored)

    assert restored == protocol
    assert meta["protocol_id"] == "proto_compare_lr"
    assert meta["protocol_title"] == "Learning-rate comparison"
    assert meta["protocol_manipulated_factor"] == "lr"
    assert meta["protocol_primary_metrics"] == ["final_rel_recon_error"]
    assert meta["protocol_seed_policy"] == "paired-10-seed"


def test_normalize_experiment_protocol_rejects_invalid_payload():
    with pytest.raises(ValueError):
        normalize_experiment_protocol({
            "protocol_id": "",
            "title": "Bad",
            "hypothesis": "x",
            "manipulated_factor": "lr",
            "held_fixed_factors": ["optimizer"],
            "primary_metrics": ["final_rel_recon_error"],
            "seed_policy": "paired",
        })
