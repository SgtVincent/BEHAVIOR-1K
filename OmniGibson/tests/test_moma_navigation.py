"""CPU determinism and limit tests for the MoMa navigation feedback math.

Ported scope: CPU-3 (test_navigation_same_input). The module is numpy-only;
this test does not import omnigibson or torch.
"""
import numpy as np
import pytest

from omnigibson.moma.navigation import (
    navigation_feedback_enabled,
    navigation_command,
    apply_coupled_base_scale,
    base_point_twist,
)

LIMITS = {
    "input_limits": np.array([[-1.0] * 3, [1.0] * 3]),
    "output_limits": np.array([[-1.0] * 3, [1.0] * 3]),
    "controller_velocity_limits": np.array([[-0.5] * 3, [0.5] * 3]),
}


def _control(pos):
    return {
        "root_pos": np.asarray(pos, dtype=float),
        "root_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "canonical_pos": np.zeros(3),
        "canonical_quat": np.array([0.0, 0.0, 0.0, 1.0]),
    }


def _target(pos):
    m = np.eye(4)
    m[:3, 3] = pos
    return m


def test_feedback_boundary_indices():
    assert navigation_feedback_enabled(5, 5) is True
    assert navigation_feedback_enabled(4, 5) is False
    with pytest.raises(ValueError):
        navigation_feedback_enabled(True, 5)


def test_zero_command_when_already_at_target():
    control = _control([0.1, 0.0, 0.0])
    command, record = navigation_command(control, _target([0.1, 0.0, 0.0]), **LIMITS)
    assert np.allclose(command, 0.0, atol=1e-6)
    assert np.isfinite(command).all()
    assert record["source_waypoint_position_error_m"] < 1e-9


def test_deterministic_same_input_same_command():
    control = _control([0.1, 0.0, 0.0])
    tgt = _target([0.25, 0.06, 0.0])
    c1, r1 = navigation_command(control, tgt, **LIMITS)
    c2, r2 = navigation_command(control, tgt, **LIMITS)
    assert np.array_equal(c1, c2)
    assert np.allclose(r1["predicted_sdk_controller_goal"], r2["predicted_sdk_controller_goal"])


def test_world_linear_speed_within_linear_limit():
    control = _control([0.1, 0.0, 0.0])
    tgt = _target([0.5, 0.2, 0.0])
    command, record = navigation_command(control, tgt, **LIMITS, linear_limit=0.12)
    linear = record["predicted_base_linear_world_mps"]
    assert np.linalg.norm(linear) <= 0.12 * (1 + 1e-6)
    assert np.isfinite(command).all()
    assert record["base_slowdown_scale"] <= 1.0 + 1e-9


def test_point_twist_within_limit_at_finger_tcp():
    control = _control([0.1, 0.0, 0.0])
    tgt = _target([0.5, 0.2, 0.0])
    _, record = navigation_command(control, tgt, **LIMITS, points_world=[[0.3, 0.1, 0.0]])
    twist = base_point_twist(record, np.array([0.3, 0.1, 0.0]))
    assert np.linalg.norm(twist[:3]) <= 0.12 * (1 + 1e-6)
    assert np.isfinite(twist).all()


def test_coupled_scale_only_slows_command():
    control = _control([0.1, 0.0, 0.0])
    tgt = _target([0.5, 0.2, 0.0])
    _, record = navigation_command(control, tgt, **LIMITS)
    slowed, record2 = apply_coupled_base_scale(record, 0.5)
    assert np.isfinite(slowed).all()
    assert record2["coupled_base_scale"] == 0.5
    with pytest.raises(ValueError, match="invalid_coupled_base_scale"):
        apply_coupled_base_scale(record, 1.5)


def test_invalid_limits_rejected():
    bad = {**LIMITS, "controller_velocity_limits": np.array([[0.5] * 3, [-0.5] * 3])}
    with pytest.raises(ValueError, match="invalid_base_controller_range"):
        navigation_command(_control([0.1, 0, 0]), _target([0.2, 0, 0]), **bad)
