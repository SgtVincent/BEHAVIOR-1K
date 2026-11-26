"""CPU unit tests for MoMa action schema labels and tagged-action validation.

Sim-free: no omnigibson, torch, or simulator import.
"""
import numpy as np
import pytest

from omnigibson.moma.schema import (
    JOINT_NATIVE23,
    IK_TARGET23,
    EXECUTION23_SE3,
    TaggedAction,
    native23_action,
    ACTION_DIM,
)


def test_native23_tagged_action_accepts_finite_vector():
    vec = np.zeros(ACTION_DIM)
    action = native23_action(vec, meta={"step": 7})
    assert action.label == JOINT_NATIVE23
    assert action.meta["step"] == 7
    assert action.to_list() == [0.0] * ACTION_DIM


def test_rejects_wrong_shape():
    with pytest.raises(ValueError, match="action23_invalid_shape"):
        TaggedAction(JOINT_NATIVE23, np.zeros(22))


def test_rejects_nonfinite():
    vec = np.zeros(ACTION_DIM)
    vec[5] = np.nan
    with pytest.raises(ValueError, match="action23_nonfinite"):
        TaggedAction(JOINT_NATIVE23, vec)


def test_rejects_unknown_label():
    with pytest.raises(ValueError, match="unknown_action_schema"):
        TaggedAction("radio-policy23", np.zeros(ACTION_DIM))


def test_same_width_different_schema_cannot_bridge():
    native = native23_action(np.zeros(ACTION_DIM))
    with pytest.raises(ValueError, match="schema_mismatch"):
        native.assert_compatible(IK_TARGET23)
    with pytest.raises(ValueError, match="schema_mismatch"):
        native.assert_compatible(EXECUTION23_SE3)


def test_same_schema_compatible():
    a = native23_action(np.zeros(ACTION_DIM))
    b = native23_action(np.ones(ACTION_DIM))
    a.assert_compatible(b.label)  # does not raise


def test_labels_match_integration_contract():
    assert JOINT_NATIVE23 == "joint-native23"
    assert IK_TARGET23 == "IK-target23"
