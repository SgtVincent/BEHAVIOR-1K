# Copyright (c) 2026 B1K2026 integration line
# SPDX-License-Identifier: Apache-2.0
"""CPU fail-closed tests: uncertified / unbound backends never emit an action."""
from __future__ import annotations

import pathlib
import sys
import unittest
from typing import Any

_HERE = pathlib.Path(__file__).resolve().parent
_OMNIGIBSON_DIR = _HERE.parents[2]
sys.path.insert(0, str(_OMNIGIBSON_DIR))

from curobo_v2.backend import CuroboV2Backend  # noqa: E402
from curobo_v2.contracts import PlannerGoal  # noqa: E402
from curobo_v2.errors import (  # noqa: E402
    AdapterNotCertified,
    RobotProfileUnbound,
)
from curobo_v2.executor import CuroboV2Executor  # noqa: E402

PROVIDER_TARGET = {
    "target_pose8x4_world": [
        [
            [1.0, 0.0, 0.0, 0.3],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.4],
            [0.0, 0.0, 0.0, 1.0],
        ],
        [
            [1.0, 0.0, 0.0, 0.5],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.4],
            [0.0, 0.0, 0.0, 1.0],
        ],
    ],
    "grippers": [0.1, 0.2],
    "target_object": "can_of_soda_114",
}


class FailClosedTest(unittest.TestCase):
    def test_unverified_executor_refuses_plan(self) -> None:
        executor = CuroboV2Executor(context={"robot_profile": "r1pro"})
        self.assertEqual(executor.status, "unverified")
        with self.assertRaises(AdapterNotCertified):
            executor.plan(PROVIDER_TARGET)

    def test_certify_rejects_incomplete_evidence(self) -> None:
        executor = CuroboV2Executor(context={"robot_profile": "r1pro"})
        with self.assertRaises(AdapterNotCertified):
            executor.certify({"canary_pass": True})
        with self.assertRaises(AdapterNotCertified):
            executor.certify({"receipt_sha256": "deadbeef"})

    def test_certified_but_unbound_backend_fails_closed(self) -> None:
        executor = CuroboV2Executor(context={"robot_profile": "r1pro"})
        executor.certify({"canary_pass": True, "receipt_sha256": "0" * 64})
        self.assertEqual(executor.status, "canary_pass")
        # No R1Pro robot config yaml / canary receipt on disk: binding must
        # raise classified blocker instead of fabricating a zero action.
        with self.assertRaises(RobotProfileUnbound):
            executor.plan(PROVIDER_TARGET)

    def test_malformed_target_raises_without_backend(self) -> None:
        executor = CuroboV2Executor(context={"robot_profile": "r1pro"})
        executor.certify({"canary_pass": True, "receipt_sha256": "0" * 64})
        with self.assertRaises(Exception):
            executor.plan({"goal": "unknown-shape"})

    def test_certificate_provenance_back_link(self) -> None:
        evidence = {
            "canary_pass": True,
            "receipt_sha256": "0" * 64,
            "config_yml_sha256": "a" * 64,
        }
        backend = CuroboV2Backend(
            robot_profile="r1pro", canary_evidence=evidence
        )
        provenance = backend._provenance()  # noqa: SLF001
        self.assertEqual(provenance["receipt"], "0" * 64)
        self.assertEqual(provenance["config_yml_sha256"], "a" * 64)
        empty = CuroboV2Backend(robot_profile="r1pro")
        self.assertIsNone(empty._provenance()["receipt"])  # noqa: SLF001

    @staticmethod
    def _goal(**kwargs: Any) -> PlannerGoal:
        defaults = {"source": "object_provider", "arm": "left"}
        defaults.update(kwargs)
        return PlannerGoal(**defaults)

    def test_dual_arm_goal_rejected_not_silently_left(self) -> None:
        backend = CuroboV2Backend(robot_profile="r1pro")
        both = self._goal(arm="both", left_pose=((0.3, 0.0, 0.4), (0, 0, 0, 1)),
                          right_pose=((0.5, 0.0, 0.4), (0, 0, 0, 1)))
        routed = backend._route_arm(both)  # noqa: SLF001
        self.assertIsNotNone(routed)
        self.assertEqual(routed["certificate"]["status"], "blocked")
        self.assertEqual(routed["certificate"]["reason"], "dual_arm_unproven")

    def test_right_only_goal_rejected(self) -> None:
        backend = CuroboV2Backend(robot_profile="r1pro")
        right = self._goal(arm="right", right_pose=((0.5, 0.0, 0.4), (0, 0, 0, 1)))
        routed = backend._route_arm(right)  # noqa: SLF001
        self.assertEqual(routed["certificate"]["reason"], "right_planner_unproven")

    def test_left_only_goal_routes_through(self) -> None:
        backend = CuroboV2Backend(robot_profile="r1pro")
        left = self._goal(arm="left", left_pose=((0.3, 0.0, 0.4), (0, 0, 0, 1)))
        self.assertIsNone(backend._route_arm(left))  # noqa: SLF001

    def test_certificate_records_left_only_scope(self) -> None:
        backend = CuroboV2Backend(robot_profile="r1pro")
        scope = backend._scope_info()  # noqa: SLF001
        self.assertEqual(scope["certified_scope"], "left_arm_only; self_collision; scene_model_null")
        self.assertFalse(scope["right_arm_planned"])
        self.assertEqual(scope["device_uuid_source"], "manifest_argument_not_live_probe")

    def test_seam_bindings_flag_missing_slots(self) -> None:
        backend = CuroboV2Backend(robot_profile="r1pro")
        missing = self._goal(arm="left", left_pose=((0.3, 0.0, 0.4), (0, 0, 0, 1)))
        seam = backend._seam_bindings(missing)  # noqa: SLF001
        self.assertFalse(seam["base_vel"]["supplied"])
        self.assertFalse(seam["gripper_left"]["supplied"])
        self.assertFalse(seam["gripper_right"]["supplied"])
        supplied = self._goal(
            arm="left", left_pose=((0.3, 0.0, 0.4), (0, 0, 0, 1)),
            grippers=(0.1, 0.2), base_vel=(0.0, 0.0, 0.1),
        )
        seam2 = backend._seam_bindings(supplied)  # noqa: SLF001
        self.assertTrue(seam2["base_vel"]["supplied"])
        self.assertEqual(seam2["base_vel"]["value"], [0.0, 0.0, 0.1])
        self.assertTrue(seam2["gripper_left"]["supplied"])
        self.assertTrue(seam2["gripper_right"]["supplied"])

    def test_profile_reports_unbound(self) -> None:
        executor = CuroboV2Executor(context={"robot_profile": "r1pro"})
        report = executor.profile()
        self.assertEqual(report["status"], "unverified")
        self.assertFalse(report["backend"]["bound"])
        self.assertFalse(report["backend"]["certified"])


if __name__ == "__main__":
    unittest.main()
