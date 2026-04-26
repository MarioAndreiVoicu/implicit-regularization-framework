from __future__ import annotations

_STOPPING_MODE_REGISTRY: dict[str, dict[str, str]] = {
    "fixed_steps": {
        "label": "Fixed steps",
        "note": "Run for exactly the configured number of steps.",
    },
    "run_to_convergence": {
        "label": "Run to convergence",
        "note": "Stop when both gradient norm and objective change are below tolerance, or at max_steps.",
    },
}


def list_stopping_modes() -> list[str]:
    return sorted(_STOPPING_MODE_REGISTRY.keys())
