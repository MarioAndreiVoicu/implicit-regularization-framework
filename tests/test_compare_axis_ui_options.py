from framework.ui.compare_axes import (
    AUTO_AUDIT_AXIS,
    NO_ADVANCED_AUDIT_AXIS,
    axis_label,
    split_audit_axis_options,
)
from framework.services.local_runs import match_axis_options


def test_compare_axis_options_keep_common_choices_short():
    common, advanced = split_audit_axis_options(match_axis_options())

    assert common == [
        "optimizer",
        "learning_rate",
        "parameterization",
        "explicit_regularization",
        "init_scale",
        "training_budget",
        "random_seed",
    ]
    assert "learning_rate_schedule" in advanced
    assert "batch_size" in advanced
    assert "psd_regime" in advanced
    assert set(common).isdisjoint(advanced)


def test_compare_axis_labels_include_auto_and_advanced_none():
    assert axis_label(AUTO_AUDIT_AXIS) == "Auto-detect"
    assert axis_label(NO_ADVANCED_AUDIT_AXIS) == "None"
    assert axis_label("explicit_regularization") == "Regularization"
    assert axis_label("learning_rate_schedule") == "Learning-rate schedule"
