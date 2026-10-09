from __future__ import annotations

AUTO_AUDIT_AXIS = "__auto__"
NO_ADVANCED_AUDIT_AXIS = "__none__"

COMPARE_AXIS_LABELS = {
    "learning_rate": "Learning rate",
    "learning_rate_schedule": "Learning-rate schedule",
    "training_budget": "Training budget",
    "random_seed": "Random seed",
    "batch_size": "Batch size",
    "optimizer": "Optimizer",
    "parameterization": "Parameterization",
    "explicit_regularization": "Regularization",
    "init_scale": "Initialization",
    "stopping_mode": "Stopping mode",
    "psd_regime": "PSD regime",
    "noncommutativity_level": "Noncommutativity level",
    "coherence_level": "Coherence level",
    "sampling_regime": "Sampling regime",
    "mc_spectrum_type": "Matrix-completion spectrum type",
    "mc_spectrum_decay": "Matrix-completion spectrum decay",
    "mc_factor_balance": "Matrix-completion factor balance",
    "mc_factor_imbalance_level": "Matrix-completion factor imbalance",
}

COMMON_AUDIT_AXES = [
    "optimizer",
    "learning_rate",
    "parameterization",
    "explicit_regularization",
    "init_scale",
    "training_budget",
    "random_seed",
]

ADVANCED_AUDIT_AXES = [
    "learning_rate_schedule",
    "batch_size",
    "stopping_mode",
    "psd_regime",
    "noncommutativity_level",
    "coherence_level",
    "sampling_regime",
    "mc_spectrum_type",
    "mc_spectrum_decay",
    "mc_factor_balance",
    "mc_factor_imbalance_level",
]


def axis_label(axis: str) -> str:
    axis = str(axis)
    if axis == AUTO_AUDIT_AXIS:
        return "Auto-detect"
    if axis == NO_ADVANCED_AUDIT_AXIS:
        return "None"
    return COMPARE_AXIS_LABELS.get(axis, axis.replace("_", " ").strip().title())


def split_audit_axis_options(axis_options: list[str]) -> tuple[list[str], list[str]]:
    """Split configured audit axes into default/common and advanced UI choices."""
    available = [str(axis) for axis in axis_options]
    common = [axis for axis in COMMON_AUDIT_AXES if axis in available]
    advanced = [axis for axis in ADVANCED_AUDIT_AXES if axis in available]
    advanced += [axis for axis in available if axis not in common and axis not in advanced]
    return common, advanced
