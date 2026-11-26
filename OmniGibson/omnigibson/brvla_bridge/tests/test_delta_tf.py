"""Stdlib tests for the fixed-anchor SE3 reconstruction seam.

Covers the frozen producer semantics: right-multiply SE3 accumulation, fixed
anchor pre-multiplication, body-frame move velocity, gripper pass-through,
exhaustion without repeat, and the fail-closed width / finite / dt gates. No
torch, numpy, simulator, or planner is imported.
"""
import math

from brvla_bridge import delta_tf, transport
from brvla_bridge.errors import ContractViolation
from brvla_bridge.tests._assert import raises

DT = 1.0 / 30.0


def _row(*, left=(0.0,) * 6, right=(0.0,) * 6, gripper=(0.0, 0.0),
         move=(0.0, 0.0, 0.0), chest=(0.0,) * 6) -> list:
    row = [0.0] * 23
    row[0:6] = list(left)
    row[6:12] = list(right)
    row[12:14] = list(gripper)
    row[14:17] = list(move)
    row[17:23] = list(chest)
    return row


def _anchors() -> dict:
    return {
        "left": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
        "right": [-0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
        "chest": [0.0, 0.0, 0.9, 0.0, 0.0, 0.0],
    }


def _targets(rows, **kwargs) -> delta_tf.FixedAnchorTargets:
    kwargs.setdefault("anchors", _anchors())
    kwargs.setdefault("dt_seconds", DT)
    return delta_tf.FixedAnchorTargets(rows, **kwargs)


def test_euler_matrix_roundtrip():
    pose = [0.12, -0.34, 0.56, 0.4, -0.25, 0.7]
    mat = delta_tf.pose6_to_matrix(pose)
    back = delta_tf.matrix_to_pose6(mat)
    for a, b in zip(pose, back):
        assert abs(a - b) < 1e-9


def test_zero_deltas_target_equals_anchor():
    targets = _targets([_row(), _row()])
    target = targets.target(0)
    for limb in delta_tf.LIMBS:
        anchor = delta_tf.pose6_to_matrix(_anchors()[limb])
        assert target["pose_matrices"][limb] == anchor
    assert target["base_velocity_body"] == [0.0, 0.0, 0.0]
    assert target["continuous_gripper"] == [0.0, 0.0]


def test_anchor_pre_multiply_not_post():
    # Anchor: translate x=0.5 then Rz(90 deg). Cumulative: translate x=0.1 in
    # the anchor frame. anchor @ cum places that offset along anchor's rotated x.
    anchor = {"left": [0.5, 0.0, 0.0, 0.0, 0.0, math.pi / 2],
              "right": [0.0] * 6, "chest": [0.0] * 6}
    row = _row(left=(0.1, 0.0, 0.0, 0.0, 0.0, 0.0))
    targets = delta_tf.FixedAnchorTargets([row], anchor, DT)
    pose = targets.target(0)["pose_matrices"]["left"]
    assert abs(pose[0][3] - 0.5) < 1e-9
    assert abs(pose[1][3] - 0.1) < 1e-9


def test_pairwise_chain_composition():
    d1 = (0.1, 0.0, 0.0, 0.0, 0.0, 0.0)
    d2 = (0.0, 0.1, 0.0, 0.0, 0.0, 0.0)
    targets = _targets([_row(left=d1), _row(left=d2)])
    m1 = delta_tf.pose6_to_matrix(list(d1))
    m2 = delta_tf.pose6_to_matrix(list(d2))
    anchor = delta_tf.pose6_to_matrix(_anchors()["left"])
    expected = delta_tf.mat_mul(delta_tf.mat_mul(anchor, m1), m2)
    actual = targets.target(1)["pose_matrices"]["left"]
    for i in range(4):
        for j in range(4):
            assert abs(actual[i][j] - expected[i][j]) < 1e-9


def test_body_move_velocity_recovers_step_delta():
    row0 = _row(move=(0.1, 0.0, 0.01))
    targets = _targets([row0])
    vel = targets.target(0)["base_velocity_body"]
    assert abs(vel[0] - 0.1 / DT) < 1e-9
    assert abs(vel[1]) < 1e-9
    assert abs(vel[2] - 0.01 / DT) < 1e-9


def test_gripper_passes_through_without_accumulation():
    row0 = _row(gripper=(0.7, 0.3))
    row1 = _row(gripper=(0.2, 0.8))
    targets = _targets([row0, row1])
    assert targets.target(0)["continuous_gripper"] == [0.7, 0.3]
    assert targets.target(1)["continuous_gripper"] == [0.2, 0.8]


def test_cumulative_representation_skips_accumulator():
    row = _row(left=(0.2, 0.0, 0.1, 0.0, 0.0, 0.0))
    targets = delta_tf.FixedAnchorTargets(
        [row], _anchors(), DT, representation=delta_tf.REPRESENTATION_CUMULATIVE
    )
    anchor = delta_tf.pose6_to_matrix(_anchors()["left"])
    cum = delta_tf.pose6_to_matrix(row[0:6])
    expected = delta_tf.mat_mul(anchor, cum)
    actual = targets.target(0)["pose_matrices"]["left"]
    for i in range(4):
        for j in range(4):
            assert abs(actual[i][j] - expected[i][j]) < 1e-9


def test_target_exhausted_raises_without_repeat():
    targets = _targets([_row()])
    targets.target(0)
    with raises(delta_tf.TargetExhausted):
        targets.target(1)


def test_bad_width_finite_dt_representation_rejected():
    with raises(ContractViolation):
        delta_tf.FixedAnchorTargets([[0.0] * 22], _anchors(), DT)
    bad = _row()
    bad[0] = float("nan")
    with raises(ContractViolation):
        delta_tf.FixedAnchorTargets([bad], _anchors(), DT)
    with raises(ContractViolation):
        delta_tf.FixedAnchorTargets([_row()], _anchors(), 0.0)
    with raises(ContractViolation):
        delta_tf.FixedAnchorTargets([_row()], _anchors(), DT, representation="unknown")
    missing = {"left": [0.0] * 6, "right": [0.0] * 6}
    with raises(ContractViolation):
        delta_tf.FixedAnchorTargets([_row()], missing, DT)


def test_pose_matrix_matches_existing_transport_convention():
    pose = [0.2, -0.1, 0.4, 0.35, -0.2, 0.65]
    mine = delta_tf.pose6_to_matrix(pose)
    theirs = transport.pose_matrix(pose)
    for i in range(4):
        for j in range(4):
            assert abs(mine[i][j] - theirs[i][j]) < 1e-12


def test_receipt_records_anchors_and_dt():
    targets = _targets([_row(), _row()])
    receipt = targets.receipt()
    assert receipt["rows"] == 2
    assert receipt["dt_seconds"] == DT
    assert receipt["representation"] == delta_tf.REPRESENTATION_PAIRWISE
    assert receipt["anchors"]["left"] == _anchors()["left"]
    assert "rotation_util.py" in receipt["producer_source_sha256"]
