# Copyright (c) 2026 B1K2026 integration line
# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests: frozen native23 slices, tagged action, target parsing."""
from __future__ import annotations

import math
import pathlib
import sys
import unittest

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[2]))

from curobo_v2.contracts import (
    ACTION_DIM,
    NATIVE23_GROUPS,
    NATIVE23_JOINT,
    ContractViolation,
    TaggedAction,
    assemble_native23,
    native23_action,
    parse_target,
    validate_trajectory_shape,
)


def _identity_4x4(tx: float = 0.0) -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, tx],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.4],
        [0.0, 0.0, 0.0, 1.0],
    ]


class SlicesTest(unittest.TestCase):
    def test_group_widths_sum_to_23(self) -> None:
        self.assertEqual(sum(w for _, w in NATIVE23_GROUPS), ACTION_DIM)

    def test_assemble_slices_roundtrip(self) -> None:
        vec = assemble_native23(
            base_vel=[0.1, 0.2, 0.3],
            trunk=[0.0, 0.1, 0.2, 0.3],
            arm_left=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            gripper_left=0.02,
            arm_right=[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
            gripper_right=0.03,
        )
        self.assertEqual(len(vec), 23)
        self.assertTrue(all(math.isfinite(v) for v in vec))
        self.assertEqual(vec[:3], (0.1, 0.2, 0.3))
        self.assertEqual(vec[14:15], (0.02,))
        self.assertEqual(vec[22:23], (0.03,))

    def test_assemble_rejects_wrong_width(self) -> None:
        with self.assertRaises(ContractViolation):
            assemble_native23(
                base_vel=[0.0, 0.0],
                trunk=[0.0] * 4,
                arm_left=[0.0] * 7,
                gripper_left=0.0,
                arm_right=[0.0] * 7,
                gripper_right=0.0,
            )

    def test_assemble_rejects_nonfinite(self) -> None:
        with self.assertRaises(ContractViolation):
            assemble_native23(
                base_vel=[0.0, 0.0, 0.0],
                trunk=[math.nan] * 4,
                arm_left=[0.0] * 7,
                gripper_left=0.0,
                arm_right=[0.0] * 7,
                gripper_right=0.0,
            )


class TaggedActionTest(unittest.TestCase):
    def test_native23_action_to_list(self) -> None:
        act = native23_action([0.0] * 23)
        self.assertEqual(act.label, NATIVE23_JOINT)
        self.assertEqual(len(act.to_list()), 23)
        act.assert_compatible(NATIVE23_JOINT)

    def test_rejects_bad_label(self) -> None:
        with self.assertRaises(ContractViolation):
            TaggedAction("not_a_schema", [0.0] * 23)

    def test_rejects_wrong_width(self) -> None:
        with self.assertRaises(ContractViolation):
            native23_action([0.0] * 22)

    def test_rejects_nonfinite(self) -> None:
        with self.assertRaises(ContractViolation):
            native23_action([math.inf] + [0.0] * 22)

    def test_schema_mismatch(self) -> None:
        act = native23_action([0.0] * 23)
        with self.assertRaises(ContractViolation):
            act.assert_compatible("execution23_se3_delta")


class ParseTargetTest(unittest.TestCase):
    def test_provider_target(self) -> None:
        target = {
            "target_pose8x4_world": [_identity_4x4(0.3), _identity_4x4(0.5)],
            "grippers": [0.1, 0.2],
            "target_object": "can_of_soda_114",
        }
        goal = parse_target(target)
        self.assertEqual(goal.source, "object_provider")
        self.assertEqual(goal.arm, "both")
        self.assertEqual(goal.object_id, "can_of_soda_114")
        self.assertIsNotNone(goal.left_pose)
        self.assertIsNotNone(goal.right_pose)
        self.assertEqual(goal.grippers, (0.1, 0.2))

    def test_provider_left_only_target(self) -> None:
        target = {"target_pose8x4_world": [_identity_4x4(0.3)]}
        goal = parse_target(target)
        self.assertEqual(goal.arm, "left")
        self.assertIsNotNone(goal.left_pose)
        self.assertIsNone(goal.right_pose)
        self.assertIsNone(goal.base_vel)

    def test_provider_target_parses_base_vel(self) -> None:
        target = {
            "target_pose8x4_world": [_identity_4x4(0.3), _identity_4x4(0.5)],
            "base_vel": [0.1, -0.2, 0.0],
        }
        goal = parse_target(target)
        self.assertEqual(goal.base_vel, (0.1, -0.2, 0.0))

    def test_provider_base_vel_invalid_rejected(self) -> None:
        with self.assertRaises(ContractViolation):
            parse_target({"target_pose8x4_world": [_identity_4x4(0.3)], "base_vel": [0.1, 0.2]})
        with self.assertRaises(ContractViolation):
            parse_target({"target_pose8x4_world": [_identity_4x4(0.3)], "base_vel": [0.1, 0.2, math.nan]})

    def test_s2_args_target(self) -> None:
        goal = parse_target({"primitive": "pick_up", "target_object": "trash_can_116"})
        self.assertEqual(goal.source, "s2_args")
        self.assertEqual(goal.primitive, "pick_up")
        self.assertIsNone(goal.left_pose)

    def test_malformed_provider_target(self) -> None:
        with self.assertRaises(ContractViolation):
            parse_target({"target_pose8x4_world": [_identity_4x4(), "not-a-matrix"]})

    def test_unrecognized_shape(self) -> None:
        with self.assertRaises(ContractViolation):
            parse_target({"goal": "somewhere"})

    def test_non_dict_rejected(self) -> None:
        with self.assertRaises(ContractViolation):
            parse_target("nope")  # type: ignore[arg-type]


class TrajectoryShapeTest(unittest.TestCase):
    JOINTS = 28

    def test_2d_single_plan_positive(self) -> None:
        spec = validate_trajectory_shape((21, self.JOINTS), self.JOINTS)
        self.assertEqual(spec["batch"], 1)
        self.assertEqual(spec["horizon"], 1)
        self.assertEqual(spec["waypoints"], 21)
        self.assertEqual(spec["selector"], ())

    def test_3d_batch1_positive(self) -> None:
        spec = validate_trajectory_shape((1, 21, self.JOINTS), self.JOINTS)
        self.assertEqual(spec["selector"], (0,))
        self.assertEqual(spec["waypoints"], 21)

    def test_4d_batch1_horizon1_positive(self) -> None:
        spec = validate_trajectory_shape((1, 1, 21, self.JOINTS), self.JOINTS)
        self.assertEqual(spec["selector"], (0, 0))
        self.assertEqual(spec["waypoints"], 21)

    def test_3d_batch2_rejected(self) -> None:
        with self.assertRaises(ContractViolation) as cm:
            validate_trajectory_shape((2, 21, self.JOINTS), self.JOINTS)
        self.assertEqual(cm.exception.args[0], "trajectory_batch_gt1")

    def test_4d_horizon2_rejected(self) -> None:
        with self.assertRaises(ContractViolation) as cm:
            validate_trajectory_shape((1, 2, 21, self.JOINTS), self.JOINTS)
        self.assertEqual(cm.exception.args[0], "trajectory_horizon_gt1")

    def test_4d_batch2_rejected(self) -> None:
        with self.assertRaises(ContractViolation):
            validate_trajectory_shape((2, 1, 21, self.JOINTS), self.JOINTS)

    def test_joint_dim_mismatch_rejected(self) -> None:
        with self.assertRaises(ContractViolation):
            validate_trajectory_shape((21, 27), self.JOINTS)

    def test_unsupported_ndim_rejected(self) -> None:
        with self.assertRaises(ContractViolation):
            validate_trajectory_shape((21,), self.JOINTS)
        with self.assertRaises(ContractViolation):
            validate_trajectory_shape((1, 1, 1, 21, self.JOINTS), self.JOINTS)

    def test_empty_waypoints_rejected(self) -> None:
        with self.assertRaises(ContractViolation):
            validate_trajectory_shape((0, self.JOINTS), self.JOINTS)


if __name__ == "__main__":
    unittest.main()
