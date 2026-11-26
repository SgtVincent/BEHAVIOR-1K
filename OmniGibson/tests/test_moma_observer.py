"""CPU predicate tests for the three-can observer layer.

Ported scope: CPU-1 (test_a19_empty_grasp_recovery_cpu). Fake robot/object
stubs stand in for the 3.7.2 SDK surface; no simulator is imported.
"""
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

from omnigibson.moma.observer.three_can import clean, read_value, observe, transitions


class _GraspState:
    def __init__(self, name: str, value: int):
        self.name = name
        self.value = value


@dataclass
class _FakeObj:
    name: str
    inside: bool = False
    contacts: List[str] = field(default_factory=list)
    velocity: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    held_by: Optional[str] = None

    def __post_init__(self):
        self.states = {
            "Inside": _FakeState(self.inside),
            "ContactBodies": _FakeState(self.contacts),
        }

    def get_linear_velocity(self):
        return list(self.velocity)

    def get_angular_velocity(self):
        return [0.0, 0.0, 0.0]


class _FakeState:
    def __init__(self, value):
        self._value = value

    def get_value(self, *args):
        return self._value


class _FakeRobot:
    arm_names = ["left", "right"]
    grasping_mode = "assisted"

    def __init__(self):
        self._ag_obj_in_hand: Dict[str, Optional[_FakeObj]] = {"left": None, "right": None}
        self._ag_release_counter = {"left": 0, "right": 0}
        self._ag_grasp_counter = {"left": 0, "right": 0}
        self._ag_obj_constraints = {"left": None, "right": None}
        self._ag_obj_constraint_params = {"left": None, "right": None}
        self._ag_freeze_gripper = {"left": False, "right": False}
        self._grasp: Dict[tuple, _GraspState] = {}

    def is_grasping(self, arm: str, candidate_obj=None):
        return self._grasp[(arm, candidate_obj.name)]


def _make_robot(cans: Dict[str, _FakeObj], bucket: _FakeObj):
    robot = _FakeRobot()
    for name, obj in cans.items():
        held = obj.held_by
        if held is not None:
            robot._ag_obj_in_hand[held] = obj
            robot._ag_grasp_counter[held] = 1
            robot._grasp[(held, name)] = _GraspState("TRUE", 1)
    for arm in robot.arm_names:
        other = "right" if arm == "left" else "left"
        for name, obj in cans.items():
            if (arm, name) not in robot._grasp:
                robot._grasp[(arm, name)] = _GraspState("FALSE", -1)
    return robot


def _pose_fn(obj):
    return [0.1, 0.2, 0.3]


def test_clean_strips_tensor_like_and_flags_nonfinite():
    class _TensorLike:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return [1.0, 2.0]

    assert clean({"a": _TensorLike()}) == {"a": [1.0, 2.0]}
    assert clean([float("nan")]) == [{"status": "UNKNOWN", "error": "nonfinite_value"}]
    assert clean(3) == 3


def test_read_value_unknown_on_exception():
    def boom():
        raise RuntimeError("sensor missing")

    out = read_value(boom)
    assert out["status"] == "UNKNOWN"
    assert "RuntimeError" in out["error"]


def test_observe_reports_held_can_and_inside():
    can = _FakeObj("can_of_soda_114", inside=True, held_by="left")
    bucket = _FakeObj("trash_can_116")
    robot = _make_robot({"can_of_soda_114": can}, bucket)

    obs = observe(robot, {"can_of_soda_114": can}, bucket, _pose_fn, "Inside", "ContactBodies")
    assert obs["objects"]["can_of_soda_114"]["inside_bucket"] is True
    left = obs["arms"]["left"]
    assert left["object_in_hand"] == "can_of_soda_114"
    assert left["constraint_present"] is False
    right = obs["arms"]["right"]
    assert right["object_in_hand"] is None


def test_transitions_inside_and_grasp_events():
    prev = {
        "objects": {
            "can_of_soda_114": {
                "inside_bucket": False,
                "grasp_by_arm": {"left": {"name": "TRUE", "value": 1}, "right": {"name": "FALSE", "value": -1}},
            }
        },
    }
    curr = {
        "objects": {
            "can_of_soda_114": {
                "inside_bucket": True,
                "grasp_by_arm": {"left": {"name": "FALSE", "value": -1}, "right": {"name": "FALSE", "value": -1}},
            }
        },
    }
    events = transitions(prev, curr, ["can_of_soda_114"])
    kinds = [(e["event"], e["object"], e.get("arm")) for e in events]
    assert ("inside_entered", "can_of_soda_114", None) in kinds
    assert ("sdk_grasp_released", "can_of_soda_114", "left") in kinds


def test_transitions_missing_predicate_keys_report_degraded_not_keyerror():
    prev = {"objects": {"can_of_soda_114": {"status": "UNKNOWN", "error": "read_failed"}}}
    curr = {"objects": {"can_of_soda_114": {"inside_bucket": True}}}
    events = transitions(prev, curr, ["can_of_soda_114"])
    assert events == [{"event": "observation_degraded", "object": "can_of_soda_114",
                       "code": "missing_predicate_keys"}]


def test_transitions_missing_object_entry_report_degraded():
    events = transitions({"objects": {}}, {"objects": {}}, ["can_of_soda_115"])
    assert events[0]["event"] == "observation_degraded"
    assert events[0]["code"] == "missing_predicate_keys"


def test_observe_passes_through_blocked_marker_object():
    bucket = _FakeObj("trash_can_116")
    marker = {"status": "BLOCKED", "error": "object_not_in_scope", "object": "can_of_soda_113"}
    robot = _FakeRobot()
    obs = observe(robot, {"can_of_soda_113": marker}, bucket, _pose_fn, "Inside", "ContactBodies")
    assert obs["objects"]["can_of_soda_113"]["status"] == "BLOCKED"
    assert obs["objects"]["can_of_soda_113"]["error"] == "object_not_in_scope"


def test_empty_hand_reads_no_false_grasp():
    bucket = _FakeObj("trash_can_116")
    empty = _FakeObj("can_of_soda_115", inside=False)
    robot = _make_robot({"can_of_soda_115": empty}, bucket)
    obs = observe(robot, {"can_of_soda_115": empty}, bucket, _pose_fn, "Inside", "ContactBodies")
    for arm in ("left", "right"):
        assert obs["arms"][arm]["object_in_hand"] is None
        assert obs["arms"][arm]["constraint_present"] is False


def test_none_bucket_is_skipped_not_read_as_object():
    empty = _FakeObj("can_of_soda_115", inside=False)
    robot = _make_robot({"can_of_soda_115": empty}, None)  # type: ignore[arg-type]
    obs = observe(robot, {"can_of_soda_115": empty}, None, _pose_fn, "Inside", "ContactBodies")
    assert "trash_can_116" not in obs["objects"]
    assert list(obs["objects"].keys()) == ["can_of_soda_115"]


def test_observer_absent_freeze_field_stays_unknown():
    bucket = _FakeObj("trash_can_116")
    empty = _FakeObj("can_of_soda_115", inside=False)
    robot = _make_robot({"can_of_soda_115": empty}, bucket)
    del robot._ag_freeze_gripper
    obs = observe(robot, {"can_of_soda_115": empty}, bucket, _pose_fn, "Inside", "ContactBodies")
    for arm in ("left", "right"):
        freeze = obs["arms"][arm]["freeze_gripper"]
        assert isinstance(freeze, dict) and freeze.get("status") == "UNKNOWN"
