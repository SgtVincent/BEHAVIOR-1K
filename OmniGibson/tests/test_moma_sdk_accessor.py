"""CPU tests for the SDK robot accessor boundary adapter.

Sim-free: the live robot is a duck-typed fake; no omnigibson/torch import.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from omnigibson.moma.sdk import SdkRobotAccessor, AccessorUnboundError


class _GraspState:
    def __init__(self, name: str, value: int) -> None:
        self.name = name
        self.value = value

    def __int__(self) -> int:
        return self.value


class FakeRobot:
    arm_names = ["left", "right"]
    action_dim = 23
    controller_order = ["base", "trunk", "arm_left", "gripper_left", "arm_right", "gripper_right"]

    def __init__(self) -> None:
        self._prop = {
            "arm_left_qpos": np.zeros(7),
            "arm_left_qvel": np.zeros(7),
            "arm_right_qpos": np.zeros(7),
            "arm_right_qvel": np.zeros(7),
            "gripper_left_qpos": np.zeros(1),
            "gripper_left_qvel": np.zeros(1),
            "gripper_right_qpos": np.zeros(1),
            "gripper_right_qvel": np.zeros(1),
            "eef_left_pos": np.zeros(3),
            "eef_left_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "eef_right_pos": np.zeros(3),
            "eef_right_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "base_qvel": np.zeros(3),
            "trunk_qpos": np.zeros(4),
            "trunk_qvel": np.zeros(4),
        }
        self._ag_obj_in_hand = {"left": None, "right": None}
        self._ag_release_counter = {"left": None, "right": 0}
        self._ag_grasp_counter = {"left": 0, "right": 0}
        self._ag_obj_constraints = {"left": None, "right": None}
        self._ag_obj_constraint_params = {"left": None, "right": None}
        self._ag_freeze_gripper = {"left": False, "right": False}
        self.released: list[str] = []
        self.grasp_calls: list[tuple[str, object]] = []

    def _get_proprioception_dict(self):
        return self._prop

    def get_relative_eef_pose(self, arm="default"):
        return np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])

    def is_grasping(self, arm="default", candidate_obj=None):
        self.grasp_calls.append((arm, candidate_obj))
        held = self._ag_obj_in_hand[arm]
        if held is not None:
            return _GraspState("TRUE", 1)
        return _GraspState("FALSE", -1)

    def release_grasp_immediately(self, arm="default"):
        self.released.append(arm)

    def get_joint_positions(self):
        return np.arange(26, dtype=float)

    def controller_action_idx(self):
        return {"base": np.arange(3)}


def test_unbound_accessor_rejects():
    acc = SdkRobotAccessor()
    assert not acc.bound
    with pytest.raises(AccessorUnboundError):
        acc.arm_names()


def test_bound_reads_proprioception():
    robot = FakeRobot()
    acc = SdkRobotAccessor(robot)
    assert acc.arm_names() == ["left", "right"]
    assert acc.action_dim() == 23
    pos, quat = acc.eef_pose("left")
    assert pos.shape == (3,) and quat.shape == (4,)
    assert acc.base_qvel().shape == (3,)
    assert acc.trunk_qpos().shape == (4,)


def test_empty_hand_record_is_not_a_hold():
    acc = SdkRobotAccessor(FakeRobot())
    left = acc.hand_record("left")
    assert left["object_in_hand"] is None
    assert left["release_counter"] is None
    assert left["valid_sdk_hold"] is False
    assert left["constraint_present"] is False


def test_held_object_record_is_sdk_hold():
    robot = FakeRobot()
    robot._ag_obj_in_hand["right"] = SimpleNamespace(name="can_of_soda_114")
    robot._ag_obj_constraints["right"] = object()
    acc = SdkRobotAccessor(robot)
    right = acc.hand_record("right")
    assert right["object_in_hand"] == "can_of_soda_114"
    assert right["sdk_grasp_state"] == "TRUE"
    assert right["valid_sdk_hold"] is True
    assert right["constraint_present"] is True


def test_release_delegates_to_sdk():
    robot = FakeRobot()
    acc = SdkRobotAccessor(robot)
    acc.release("right")
    assert robot.released == ["right"]


def test_safe_hand_record_keeps_unknown_on_read_failure():
    robot = FakeRobot()

    def boom():
        raise RuntimeError("contact view missing")

    robot._ag_obj_in_hand["left"] = boom  # type: ignore[assignment]
    acc = SdkRobotAccessor(robot)
    record = acc.safe_hand_record("left")
    assert record.get("status") == "UNKNOWN"


def test_joint_qpos_reads_method_surface():
    robot = FakeRobot()
    acc = SdkRobotAccessor(robot)
    qpos = acc.joint_qpos()
    assert qpos.shape == (26,)
    assert np.allclose(qpos, np.arange(26, dtype=float))


def test_absent_freeze_field_stays_unknown():
    robot = FakeRobot()
    del robot._ag_freeze_gripper
    acc = SdkRobotAccessor(robot)
    record = acc.hand_record("left")
    freeze = record["freeze_gripper"]
    assert isinstance(freeze, dict) and freeze.get("status") == "UNKNOWN"
    assert freeze.get("error") == "field_absent_post1"
