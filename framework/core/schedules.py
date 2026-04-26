from __future__ import annotations

import math


_SCHEDULES = {
    "constant": {"label": "Constant"},
    "step_decay": {"label": "Step decay"},
    "cosine": {"label": "Cosine"},
}


def list_lr_schedule_types() -> list[str]:
    return list(_SCHEDULES.keys())


def learning_rate_at_step(*, base_lr: float, schedule_type: str, step: int, total_steps: int, decay_gamma: float, step_size: int, lr_min: float) -> float:
    if step < 0:
        raise ValueError("step must be >= 0")
    if total_steps < 1:
        raise ValueError("total_steps must be >= 1")
    if schedule_type == "constant":
        return float(base_lr)
    if schedule_type == "step_decay":
        if step_size < 1:
            raise ValueError("step_size must be >= 1")
        factor = decay_gamma ** (step // step_size)
        return float(base_lr) * float(factor)
    if schedule_type == "cosine":
        progress = min(max(step / max(1, total_steps), 0.0), 1.0)
        return float(lr_min) + 0.5 * (float(base_lr) - float(lr_min)) * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"Unsupported lr_schedule_type={schedule_type!r}")
