from __future__ import annotations

import sys
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np

# Ensure the repo root is on the path so imports resolve when running tests
# standalone (e.g.  python -m pytest ...)
sys.path.insert(
    0,
    "/mnt/bn/behavior-data-hl/chenjunting/repo/BEHAVIOR-1K/OmniGibson",
)

from omnigibson.learning.utils.skill_completion import (
    check_skill_completed,
    eval_skill_metric,
    get_skill_object_bindings,
    check_skill_completed_rollout,
    _apply_object_bindings,
)
from omnigibson.learning.utils.segment_predicate_eval import SegmentPredicate


# ---------------------------------------------------------------------------
# Mock environment helpers
# ---------------------------------------------------------------------------


def _make_mock_env(
    *,
    objects: Optional[Dict[str, Any]] = None,
    robot_grasping: Optional[Dict[str, bool]] = None,
) -> MagicMock:
    """Build a minimal mock OmniGibson environment.

    Args:
        objects: Dict mapping object names to mock objects.  Each mock object
            should expose ``states[StateClass].get_value(...)``.
        robot_grasping: Dict ``{arm_name: bool}`` returned by
            ``robot.is_grasping(...)``.
    """
    env = MagicMock()
    env.scene = MagicMock()
    env.robots = [MagicMock()]
    env.task = MagicMock()

    # Object registry lookup
    def _object_registry(_by, name):
        return (objects or {}).get(name)

    env.scene.object_registry = _object_registry
    env.scene.get_task_metadata.return_value = {}

    # Robot grasping
    if robot_grasping is not None:
        env.robots[0].arm_names = list(robot_grasping.keys())
        env.robots[0].is_grasping.side_effect = lambda *, arm, candidate_obj: robot_grasping.get(arm, False)
    else:
        env.robots[0].arm_names = ["right"]
        env.robots[0].is_grasping.return_value = False

    # Robot pose (used by geometry metrics)
    env.robots[0].get_position_orientation.return_value = (
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 1.0]),
    )

    return env


class _MockStatesDict:
    """A dict-like object that supports class-key lookup for OmniGibson state classes."""

    def __init__(self, state_values: Optional[Dict[str, Any]] = None):
        self._state_values = state_values or {}

    def __getitem__(self, key):
        class_name = key.__name__ if hasattr(key, "__name__") else str(key)
        val = self._state_values.get(class_name, False)

        class _FakeState:
            def __init__(self, value):
                self._value = value

            def get_value(self, other=None):
                if callable(self._value):
                    return self._value(other)
                return self._value

        return _FakeState(val)

    def __setitem__(self, key, value):
        class_name = key.__name__ if hasattr(key, "__name__") else str(key)
        self._state_values[class_name] = value


def _make_mock_object(name: str, state_values: Optional[Dict[str, Any]] = None) -> MagicMock:
    """Create a mock scene object with OmniGibson-like state accessors."""
    obj = MagicMock()
    obj.name = name
    obj.get_position_orientation.return_value = (
        np.array([1.0, 1.0, 1.0]),
        np.array([0.0, 0.0, 0.0, 1.0]),
    )
    obj.states = _MockStatesDict(state_values)
    return obj


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGetSkillObjectBindings(unittest.TestCase):
    def test_known_skill_place_in(self):
        annotation = {
            "object_id": [["apple", "bowl"]],
            "manipulating_object_id": "apple",
        }
        env_objects = {"apple": "apple_001", "bowl": "bowl_001"}
        bindings = get_skill_object_bindings("place in", annotation, env_objects)
        self.assertIn("obj", bindings)
        self.assertIn("dst_or_target", bindings)
        self.assertEqual(bindings["obj"], "apple_001")
        self.assertEqual(bindings["dst_or_target"], "bowl_001")

    def test_unknown_skill_returns_empty(self):
        bindings = get_skill_object_bindings("nonexistent skill", {}, {})
        self.assertEqual(bindings, {})

    def test_pick_up_from_bindings(self):
        annotation = {
            "object_id": [["apple", "table"]],
            "manipulating_object_id": "apple",
        }
        env_objects = {"apple": "apple_001", "table": "table_001"}
        bindings = get_skill_object_bindings("pick up from", annotation, env_objects)
        self.assertIn("obj", bindings)
        self.assertIn("src_or_target", bindings)
        self.assertEqual(bindings["obj"], "apple_001")
        self.assertEqual(bindings["src_or_target"], "table_001")

    def test_unary_target_skill(self):
        annotation = {
            "object_id": [["microwave"]],
        }
        env_objects = {"microwave": "microwave_001"}
        bindings = get_skill_object_bindings("open door", annotation, env_objects)
        self.assertIn("unary_target", bindings)
        self.assertEqual(bindings["unary_target"], "microwave_001")


class TestCheckSkillCompleted(unittest.TestCase):
    def test_unknown_skill(self):
        env = _make_mock_env()
        result = check_skill_completed(env, "fly to the moon")
        self.assertFalse(result["completed"])
        self.assertEqual(result["result_type"], "unknown_skill")

    def test_empty_skill_desc(self):
        env = _make_mock_env()
        result = check_skill_completed(env, "")
        self.assertFalse(result["completed"])
        self.assertEqual(result["result_type"], "unknown_skill")

    def test_pick_up_from_satisfied(self):
        apple = _make_mock_object("apple", state_values={"OnTop": lambda other: False, "Inside": lambda other: False})
        table = _make_mock_object("table")
        env = _make_mock_env(
            objects={"apple": apple, "table": table},
            robot_grasping={"right": True},
        )
        result = check_skill_completed(
            env,
            "pick up from",
            object_bindings={"obj": "apple", "src_or_target": "table"},
        )
        self.assertTrue(result["completed"])
        self.assertEqual(result["result_type"], "predicate_satisfied")
        self.assertIn("grasped(agent,apple)", result["metrics"]["truth_map"])

    def test_pick_up_from_unsatisfied(self):
        apple = _make_mock_object("apple", state_values={"OnTop": lambda other: True})
        table = _make_mock_object("table")
        env = _make_mock_env(
            objects={"apple": apple, "table": table},
            robot_grasping={"right": False},
        )
        result = check_skill_completed(
            env,
            "pick up from",
            object_bindings={"obj": "apple", "src_or_target": "table"},
        )
        self.assertFalse(result["completed"])
        self.assertEqual(result["result_type"], "predicate_unsatisfied")

    def test_place_in_satisfied(self):
        apple = _make_mock_object("apple", state_values={"Inside": lambda other: True})
        env = _make_mock_env(
            objects={"apple": apple, "bowl": MagicMock()},
            robot_grasping={"right": False},
        )
        result = check_skill_completed(
            env,
            "place in",
            object_bindings={"obj": "apple", "dst_or_target": "bowl"},
        )
        self.assertTrue(result["completed"])
        self.assertEqual(result["result_type"], "predicate_satisfied")

    def test_missing_object(self):
        env = _make_mock_env(objects={})
        result = check_skill_completed(
            env,
            "pick up from",
            object_bindings={"obj": "nonexistent", "src_or_target": "table"},
        )
        self.assertFalse(result["completed"])
        self.assertEqual(result["result_type"], "missing_object")
        self.assertIn("nonexistent", result["metrics"]["missing_objects"])

    def test_open_door_satisfied(self):
        microwave = _make_mock_object("microwave", state_values={"Open": True})
        env = _make_mock_env(objects={"microwave": microwave})
        result = check_skill_completed(
            env,
            "open door",
            object_bindings={"unary_target": "microwave"},
        )
        self.assertTrue(result["completed"])
        self.assertEqual(result["result_type"], "predicate_satisfied")

    def test_turn_on_switch_satisfied(self):
        switch = _make_mock_object("switch", state_values={"ToggledOn": True})
        env = _make_mock_env(objects={"switch": switch})
        result = check_skill_completed(
            env,
            "turn on switch",
            object_bindings={"unary_target": "switch"},
        )
        self.assertTrue(result["completed"])
        self.assertEqual(result["result_type"], "predicate_satisfied")

    def test_any_of_combine_mode(self):
        # "push to" uses combine_mode="any_of": nextto OR object_pose_match
        box = _make_mock_object("box", state_values={"NextTo": lambda other: True})
        target = _make_mock_object("target")
        env = _make_mock_env(objects={"box": box, "target": target})
        result = check_skill_completed(
            env,
            "push to",
            object_bindings={"obj": "box", "target_or_dst": "target"},
        )
        self.assertTrue(result["completed"])
        self.assertEqual(result["result_type"], "predicate_satisfied")
        self.assertEqual(result["metrics"]["combine_mode"], "any_of")

    def test_demo_annotations_used(self):
        """Annotations should be forwarded into the segment for build_template_predicates."""
        apple = _make_mock_object("apple", state_values={"OnTop": lambda other: False, "Inside": lambda other: False})
        table = _make_mock_object("table")
        env = _make_mock_env(
            objects={"apple": apple, "table": table},
            robot_grasping={"right": True},
        )
        annotation = {
            "object_id": [["apple", "table"]],
            "manipulating_object_id": "apple",
        }
        result = check_skill_completed(
            env,
            "pick up from",
            demo_annotations=annotation,
        )
        # Because we didn't pass explicit bindings, build_template_predicates
        # will try to resolve from the annotation + env.  The mock env has the
        # object registered by name, so it should still work.
        self.assertTrue(result["completed"])
        self.assertEqual(result["result_type"], "predicate_satisfied")


class TestEvalSkillMetric(unittest.TestCase):
    def test_eval_skill_metric_pick_up(self):
        apple = _make_mock_object("apple", state_values={"OnTop": lambda other: False})
        table = _make_mock_object("table")
        env = _make_mock_env(
            objects={"apple": apple, "table": table},
            robot_grasping={"right": True},
        )
        metric_entry = {
            "metric_family": "grasp_relation",
            "object_roles": ["obj", "src_or_target"],
            "success_rule": "object is grasped and no longer ontop/inside original source",
            "metrics": [
                {"type": "predicate", "name": "grasped", "args": ["agent", "obj"], "desired": True},
                {"type": "predicate", "name": "ontop", "args": ["obj", "src_or_target"], "desired": False},
            ],
            "require_unsatisfied_at_start": True,
        }
        result = eval_skill_metric(
            env,
            metric_entry,
            object_bindings={"obj": "apple", "src_or_target": "table"},
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["result_type"], "predicate_satisfied")
        self.assertEqual(len(result["trace"]), 2)

    def test_eval_skill_metric_missing_bindings(self):
        env = _make_mock_env()
        metric_entry = {
            "metric_family": "grasp_relation",
            "metrics": [
                {"type": "predicate", "name": "grasped", "args": ["agent", "obj"], "desired": True},
            ],
        }
        result = eval_skill_metric(env, metric_entry, object_bindings={})
        self.assertFalse(result["success"])
        self.assertEqual(result["result_type"], "no_predicates")

    def test_eval_skill_metric_geometry(self):
        """Geometry metrics should be evaluated when bindings are complete."""
        target = _make_mock_object("target")
        env = _make_mock_env(objects={"target": target})
        metric_entry = {
            "metric_family": "geometry_base_target",
            "object_roles": ["target_or_obj"],
            "metrics": [
                {"type": "base_to_object", "role": "target_or_obj", "margin": 0.10, "min_threshold": 0.30},
            ],
        }
        result = eval_skill_metric(
            env,
            metric_entry,
            object_bindings={"target_or_obj": "target"},
        )
        # The robot is at (0,0,0) and target at (1,1,1); distance_xy = sqrt(2) ~ 1.414
        # threshold = max(0.30, 1.414 + 0.10) = 1.514, so it should be satisfied.
        self.assertTrue(result["success"])
        self.assertEqual(result["result_type"], "predicate_satisfied")


class TestCheckSkillCompletedRollout(unittest.TestCase):
    def test_empty_history_timeout(self):
        env = _make_mock_env()
        result = check_skill_completed_rollout(
            env,
            "pick up from",
            trace_history=[],
        )
        self.assertFalse(result["completed"])
        self.assertEqual(result["result_type"], "timeout")

    def test_window_satisfied_anytime(self):
        env = _make_mock_env()
        # Simulate two steps where all predicates are satisfied
        step_trace = [
            {"predicate": "grasped(agent,apple)", "satisfied": True, "metric_type": "predicate"},
            {"predicate": "ontop(apple,table)", "satisfied": True, "metric_type": "predicate"},
        ]
        result = check_skill_completed_rollout(
            env,
            "pick up from",
            trace_history=[step_trace, step_trace],
            window_mode="anytime",
        )
        self.assertTrue(result["completed"])
        self.assertEqual(result["result_type"], "predicate_satisfied")

    def test_window_unsatisfied(self):
        env = _make_mock_env()
        step_trace = [
            {"predicate": "grasped(agent,apple)", "satisfied": False, "metric_type": "predicate"},
        ]
        result = check_skill_completed_rollout(
            env,
            "pick up from",
            trace_history=[step_trace, step_trace],
            window_mode="anytime",
        )
        self.assertFalse(result["completed"])
        self.assertEqual(result["result_type"], "timeout")

    def test_env_truncated(self):
        env = _make_mock_env()
        step_trace = [
            {"predicate": "grasped(agent,apple)", "satisfied": True, "metric_type": "predicate"},
        ]
        result = check_skill_completed_rollout(
            env,
            "pick up from",
            trace_history=[step_trace],
            env_truncated=True,
        )
        self.assertFalse(result["completed"])
        self.assertEqual(result["result_type"], "truncated")

    def test_env_terminated(self):
        env = _make_mock_env()
        step_trace = [
            {"predicate": "grasped(agent,apple)", "satisfied": True, "metric_type": "predicate"},
        ]
        result = check_skill_completed_rollout(
            env,
            "pick up from",
            trace_history=[step_trace],
            env_terminated=True,
        )
        self.assertFalse(result["completed"])
        self.assertEqual(result["result_type"], "env_terminated")

    def test_missing_object_in_history(self):
        env = _make_mock_env()
        step_trace = [
            {
                "predicate": "grasped(agent,apple)",
                "satisfied": False,
                "metric_type": "predicate",
                "diagnostics": {"missing_object": "apple"},
            },
        ]
        result = check_skill_completed_rollout(
            env,
            "pick up from",
            trace_history=[step_trace],
        )
        self.assertFalse(result["completed"])
        self.assertEqual(result["result_type"], "missing_object")

    def test_short_proxy_success_blocked(self):
        """Very early success should be reclassified when min_rollout_steps_for_success is set."""
        env = _make_mock_env()
        step_trace = [
            {"predicate": "grasped(agent,apple)", "satisfied": True, "metric_type": "predicate"},
        ]
        result = check_skill_completed_rollout(
            env,
            "hand over",  # has min_rollout_steps_for_success=30
            trace_history=[step_trace],
            max_steps=300,
            final_step=5,
            min_rollout_steps_for_success=0,  # let the registry value apply
        )
        # hand over has min_rollout_steps_for_success=30, so step 5 is too early
        self.assertFalse(result["completed"])
        # result_type may be short_proxy_success or likely_proxy_false_positive
        self.assertIn(result["result_type"], {"short_proxy_success", "likely_proxy_false_positive"})


class TestApplyObjectBindings(unittest.TestCase):
    def test_override_predicate_args(self):
        specs = [
            SegmentPredicate(
                metric_type="predicate",
                name="grasped",
                args=["agent", "obj"],
                desired=True,
                source="registry",
                params={},
            ),
            SegmentPredicate(
                metric_type="predicate",
                name="ontop",
                args=["obj", "src_or_target"],
                desired=False,
                source="registry",
                params={},
            ),
        ]
        bindings = {"obj": "apple_001", "src_or_target": "table_001"}
        new_specs = _apply_object_bindings(specs, bindings)
        self.assertEqual(new_specs[0].args, ["agent", "apple_001"])
        self.assertEqual(new_specs[1].args, ["apple_001", "table_001"])

    def test_override_geometry_args(self):
        specs = [
            SegmentPredicate(
                metric_type="base_to_object",
                name="base_to_object",
                args=["target_or_obj"],
                desired=True,
                source="registry",
                params={"role": "target_or_obj"},
            ),
        ]
        bindings = {"target_or_obj": "chair_001"}
        new_specs = _apply_object_bindings(specs, bindings)
        self.assertEqual(new_specs[0].args, ["chair_001"])

    def test_no_change_for_unknown_roles(self):
        specs = [
            SegmentPredicate(
                metric_type="predicate",
                name="grasped",
                args=["agent", "obj"],
                desired=True,
                source="registry",
                params={},
            ),
        ]
        bindings = {"other_role": "value"}
        new_specs = _apply_object_bindings(specs, bindings)
        self.assertEqual(new_specs[0].args, ["agent", "obj"])


if __name__ == "__main__":
    unittest.main()
