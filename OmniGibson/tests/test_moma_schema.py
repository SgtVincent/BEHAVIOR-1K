"""CPU unit tests for MoMa action schema labels and tagged-action validation.

Sim-free: no omnigibson, torch, or simulator import.
"""
import numpy as np
import pytest

from omnigibson.moma.schema import (
    NATIVE23_JOINT,
    BRIDGE_DECODED23_IK_POSE,
    EXECUTION23_SE3_DELTA,
    LEGACY_LABEL_ALIASES,
    normalize_label,
    TaggedAction,
    native23_action,
    ACTION_DIM,
)


def test_canonical_labels_match_integration_contract():
    assert NATIVE23_JOINT == "native23_joint"
    assert BRIDGE_DECODED23_IK_POSE == "bridge_decoded23_ik_pose"
    assert EXECUTION23_SE3_DELTA == "execution23_se3_delta"


def test_native23_tagged_action_accepts_finite_vector():
    vec = np.zeros(ACTION_DIM)
    action = native23_action(vec, meta={"step": 7})
    assert action.label == NATIVE23_JOINT
    assert action.meta["step"] == 7
    assert action.to_list() == [0.0] * ACTION_DIM


def test_rejects_wrong_shape():
    with pytest.raises(ValueError, match="action23_invalid_shape"):
        TaggedAction(NATIVE23_JOINT, np.zeros(22))


def test_rejects_nonfinite():
    vec = np.zeros(ACTION_DIM)
    vec[5] = np.nan
    with pytest.raises(ValueError, match="action23_nonfinite"):
        TaggedAction(NATIVE23_JOINT, vec)


def test_rejects_unknown_label():
    with pytest.raises(ValueError, match="unknown_action_schema"):
        TaggedAction("radio-policy23", np.zeros(ACTION_DIM))


def test_legacy_aliases_normalize_to_canonical():
    for legacy, canonical in LEGACY_LABEL_ALIASES.items():
        assert normalize_label(legacy) == canonical
        action = TaggedAction(legacy, np.zeros(ACTION_DIM))
        assert action.label == canonical


def test_alias_and_canonical_are_compatible():
    action = TaggedAction("joint-native23", np.zeros(ACTION_DIM))
    action.assert_compatible(NATIVE23_JOINT)  # does not raise
    action.assert_compatible("joint-native23")  # does not raise


def test_same_width_different_schema_cannot_bridge():
    native = native23_action(np.zeros(ACTION_DIM))
    with pytest.raises(ValueError, match="schema_mismatch"):
        native.assert_compatible(BRIDGE_DECODED23_IK_POSE)
    with pytest.raises(ValueError, match="schema_mismatch"):
        native.assert_compatible(EXECUTION23_SE3_DELTA)


def test_legacy_alias_cross_schema_still_rejected():
    native = TaggedAction("joint-native23", np.zeros(ACTION_DIM))
    with pytest.raises(ValueError, match="schema_mismatch"):
        native.assert_compatible("IK-target23")
    with pytest.raises(ValueError, match="schema_mismatch"):
        native.assert_compatible("execution23_se3")


def test_misspelled_legacy_label_rejected():
    for bad in ("joint-native23 ", "ik-target23", "execution23-se3", "native23-joint"):
        with pytest.raises(ValueError, match="unknown_action_schema"):
            normalize_label(bad)


def test_same_schema_compatible():
    a = native23_action(np.zeros(ACTION_DIM))
    b = native23_action(np.ones(ACTION_DIM))
    a.assert_compatible(b.label)  # does not raise
