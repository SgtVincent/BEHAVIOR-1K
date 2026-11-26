"""Stdlib tests for the tagged-action step boundary.

Covers: model output rows tagged IK-target23 with the pinned physical layout,
the joint-native23 consumer guard, and the same-width physical23 negative
refusal. The guard must never relabel an IK-target row as a joint command.
No planner, simulator, torch, numpy, or omnigibson is imported.
"""
from brvla_bridge import action_tag, contracts
from brvla_bridge.errors import ControlledActionBlocked
from brvla_bridge.tests._assert import raises


def _row(value: float = 0.01) -> list:
    return [value] * contracts.ACTION23


def test_model_output_is_tagged_ik_target_with_physical_layout():
    tagged = action_tag.tag_model_output(_row())
    assert tagged.label == contracts.IK_TARGET23
    assert tagged.layout == contracts.SERVER_METADATA_LAYOUT
    assert len(tagged.vector) == contracts.ACTION23
    assert tagged.is_native_joint() is False


def test_model_output_rejects_wrong_width():
    with raises(ControlledActionBlocked):
        action_tag.tag_model_output(_row()[:22])


def test_model_output_rejects_non_finite():
    row = _row()
    row[4] = float("nan")
    with raises(ControlledActionBlocked):
        action_tag.tag_model_output(row)


def test_model_output_rejects_unpinned_layout():
    with raises(ControlledActionBlocked):
        action_tag.tag_model_output(_row(), layout="other_layout")


def test_native23_tag_passes_consumer_guard():
    tagged = action_tag.tag_native23(_row(0.02), provenance={"arm_left": "planner"})
    vector = action_tag.assert_native23_joint(tagged)
    assert len(vector) == contracts.ACTION23
    assert tagged.label == contracts.JOINT_NATIVE23
    assert tagged.is_native_joint() is True


def test_same_width_ik_target_is_refused_at_step_boundary():
    # Negative case: the physical23 row is 23-d like the native controller
    # vector, but its semantics are IK targets. The consumer guard refuses it
    # before env.step; the action is held and zero steps are taken.
    physical = action_tag.tag_model_output(_row())
    try:
        action_tag.assert_native23_joint(physical)
    except ControlledActionBlocked as exc:
        assert "joint-native23" in str(exc)
    else:
        raise AssertionError("IK-target row must not pass the native23 guard")


def test_untagged_action_is_refused():
    with raises(ControlledActionBlocked):
        action_tag.assert_native23_joint(_row())


def test_native_tag_rejects_non_finite():
    with raises(ControlledActionBlocked):
        action_tag.tag_native23([float("inf")] * contracts.ACTION23)
