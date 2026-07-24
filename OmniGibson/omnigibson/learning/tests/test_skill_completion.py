from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np

# Ensure the repo root is on the path so imports resolve when running tests
# standalone (e.g.  python -m pytest ...)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from omnigibson.learning.utils.skill_completion import (
    check_skill_completed,
    eval_skill_metric,
    get_skill_object_bindings,
    check_skill_completed_rollout,
    _apply_object_bindings,
)
from omnigibson.learning.utils.segment_predicate_eval import (
    SegmentPredicate,
    build_template_predicates,
)
from omnigibson.learning.utils.segment_skill_metric_registry import get_skill_metric_entry


# ---------------------------------------------------------------------------
# Mock environment helpers
# ---------------------------------------------------------------------------


def _make_mock_env(
    *,
    objects: Optional[Dict[str, Any]] = None,
    systems: Optional[Dict[str, Any]] = None,
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
    env.scene.get_system.side_effect = lambda name, force_init=False: (systems or {}).get(name)
    env.task.activity_name = ""
    env.task.ground_goal_state_options = []

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


class _FakeGoalHead:
    def __init__(self, body: List[str]):
        self.body = body


def _negative_goal(body: List[str]) -> _FakeGoalHead:
    # Real ground_goal_state_options wrap negative atoms in a HEAD whose body is
    # ["not", [predicate, arg, ...]].
    return _FakeGoalHead(["not", body])


def _set_task_goals(env, activity_name: str, *goals: Any) -> None:
    env.task.activity_name = activity_name
    env.task.ground_goal_state_options = [list(goals)]


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


class TestPlaceMetricRegistry(unittest.TestCase):
    PLACE_SKILLS = (
        "place in",
        "place on",
        "insert",
        "place under",
        "place on next to",
        "place in next to",
    )

    def test_place_release_is_diagnostic_not_primary(self):
        for skill in self.PLACE_SKILLS:
            with self.subTest(skill=skill):
                entry = get_skill_metric_entry(skill)
                self.assertIsNotNone(entry)
                self.assertNotIn("grasped", [metric.get("name") for metric in entry["metrics"]])
                self.assertEqual(
                    [metric.get("name") for metric in entry.get("diagnostic_metrics", [])],
                    ["grasped"],
                )

    def test_compound_goal_atoms_are_bound_and_labeled(self):
        env = _make_mock_env()
        _set_task_goals(
            env,
            "compound_place_task",
            _FakeGoalHead(["ontop", "apple", "table"]),
            _FakeGoalHead(["nextto", "apple", "vase"]),
        )
        segment = {
            "skill_description": ["place on next to"],
            "object_id": [["apple", "table", "vase"]],
            "manipulating_object_id": "apple",
        }
        specs, debug = build_template_predicates("skill", segment, env)
        self.assertEqual([spec.args for spec in specs], [["apple", "table"], ["apple", "vase"]])
        self.assertEqual(
            [spec.params["semantic_role"] for spec in specs],
            ["task_goal_ontop", "task_goal_nextto"],
        )
        self.assertEqual(debug["resolved_object_roles"]["support_target"], "table")
        self.assertEqual(debug["resolved_object_roles"]["neighbor_target"], "vase")
        self.assertEqual(specs[0].diagnostic_specs[0].params["semantic_role"], "release_state")

    def test_compound_missing_neighbor_does_not_degrade_when_goal_requires_nextto(self):
        env = _make_mock_env()
        _set_task_goals(
            env,
            "compound_place_task",
            _FakeGoalHead(["ontop", "apple", "table"]),
            _FakeGoalHead(["nextto", "apple", "vase"]),
        )
        segment = {
            "skill_description": ["place on next to"],
            "object_id": [["apple", "table"]],
            "manipulating_object_id": "apple",
        }
        specs, debug = build_template_predicates("skill", segment, env)
        self.assertEqual(specs, [])
        self.assertIsNone(debug["resolved_object_roles"]["neighbor_target"])
        self.assertEqual(debug["missing_template_roles"][0]["role"], "neighbor_target")

    def test_place_in_next_to_uses_inside_only_when_goal_has_no_nextto(self):
        env = _make_mock_env()
        _set_task_goals(
            env,
            "rearranging_kitchen_furniture",
            _FakeGoalHead(["inside", "food_processor", "cabinet"]),
        )
        segment = {
            "skill_description": ["place in next to"],
            "object_id": [["food_processor", "cabinet", "toaster"]],
            "manipulating_object_id": "food_processor",
        }

        specs, _ = build_template_predicates("skill", segment, env)

        self.assertEqual([(spec.name, spec.args) for spec in specs], [("inside", ["food_processor", "cabinet"])])

    def test_place_on_next_to_uses_nextto_only_when_ontop_is_not_a_goal(self):
        env = _make_mock_env()
        _set_task_goals(
            env,
            "putting_up_Christmas_decorations_inside",
            _FakeGoalHead(["nextto", "gift_box", "christmas_tree"]),
        )
        segment = {
            "skill_description": ["place on next to"],
            "object_id": [["gift_box", "floor", "christmas_tree"]],
            "manipulating_object_id": "gift_box",
        }

        specs, _ = build_template_predicates("skill", segment, env)

        self.assertEqual([(spec.name, spec.args) for spec in specs], [("nextto", ["gift_box", "christmas_tree"])])

    def test_place_on_next_to_uses_ontop_only_when_nextto_is_not_a_goal(self):
        env = _make_mock_env()
        _set_task_goals(
            env,
            "heterogeneous_place_task",
            _FakeGoalHead(["ontop", "book", "desk"]),
        )
        segment = {
            "skill_description": ["place on next to"],
            "object_id": [["book", "desk"]],
            "manipulating_object_id": "book",
        }

        specs, debug = build_template_predicates("skill", segment, env)

        self.assertEqual([(spec.name, spec.args) for spec in specs], [("ontop", ["book", "desk"])])
        self.assertNotIn("missing_template_roles", debug)

    def test_task_goal_metric_without_direct_goal_is_explicitly_invalid(self):
        env = _make_mock_env()
        _set_task_goals(
            env,
            "unmatched_place_task",
            _FakeGoalHead(["inside", "book", "cabinet"]),
        )
        segment = {
            "skill_description": ["place on next to"],
            "object_id": [["book", "desk", "lamp"]],
            "manipulating_object_id": "book",
        }

        specs, debug = build_template_predicates("skill", segment, env)

        self.assertEqual(specs, [])
        self.assertEqual(
            debug["invalid_task_goal_bindings"][0]["reason"],
            "no_direct_task_goal_match",
        )
        self.assertEqual(
            debug["missing_template_roles"][0]["reason"],
            "no_direct_task_goal_match",
        )

    def test_preserved_candidate_metrics_are_unchanged(self):
        expected_primary = {
            "chop": [("touching", True)],
            "place in": [("inside", True)],
            "place on": [("ontop", True)],
            "open door": [("open", True)],
            "close drawer": [("open", False)],
        }

        for skill, expected in expected_primary.items():
            with self.subTest(skill=skill):
                entry = get_skill_metric_entry(skill)
                actual = [(metric["name"], metric["desired"]) for metric in entry["metrics"]]
                self.assertEqual(actual, expected)


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

    def test_place_in_satisfied_while_grasped_is_diagnostic(self):
        apple = _make_mock_object("apple", state_values={"Inside": lambda other: True})
        env = _make_mock_env(
            objects={"apple": apple, "bowl": MagicMock()},
            robot_grasping={"right": True},
        )
        result = check_skill_completed(
            env,
            "place in",
            object_bindings={"obj": "apple", "dst_or_target": "bowl"},
        )
        self.assertTrue(result["completed"])
        self.assertEqual(result["result_type"], "predicate_satisfied")
        trace = result["metrics"]["trace"]
        self.assertEqual([item["predicate"] for item in trace], ["inside(apple,bowl)"])
        auxiliary = result["metrics"]["trace_summary"]["auxiliary_diagnostics"]
        self.assertEqual(len(auxiliary), 1)
        self.assertEqual(auxiliary[0]["predicate"], "grasped(agent,apple)")
        self.assertFalse(auxiliary[0]["contributes_to_success"])
        self.assertFalse(auxiliary[0]["satisfied"])

    def test_place_in_fails_on_primary_relation_even_when_released(self):
        apple = _make_mock_object("apple", state_values={"Inside": lambda other: False})
        env = _make_mock_env(
            objects={"apple": apple, "bowl": MagicMock()},
            robot_grasping={"right": False},
        )
        result = check_skill_completed(
            env,
            "place in",
            object_bindings={"obj": "apple", "dst_or_target": "bowl"},
        )
        self.assertFalse(result["completed"])
        summary = result["metrics"]["trace_summary"]
        self.assertEqual(summary["failed_primary_predicates"], ["inside(apple,bowl)"])
        self.assertEqual(summary["unsatisfied_auxiliary_predicates"], [])

    def test_compound_place_separates_support_neighbor_and_release(self):
        apple = _make_mock_object(
            "apple",
            state_values={
                "OnTop": lambda other: True,
                "NextTo": lambda other: False,
            },
        )
        env = _make_mock_env(
            objects={"apple": apple, "table": MagicMock(), "vase": MagicMock()},
            robot_grasping={"right": True},
        )
        _set_task_goals(
            env,
            "compound_place_task",
            _FakeGoalHead(["ontop", "apple", "table"]),
            _FakeGoalHead(["nextto", "apple", "vase"]),
        )
        result = check_skill_completed(
            env,
            "place on next to",
            object_bindings={
                "obj": "apple",
                "support_target": "table",
                "neighbor_target": "vase",
            },
        )
        self.assertFalse(result["completed"])
        trace = result["metrics"]["trace"]
        self.assertEqual(
            [(item["semantic_role"], item["satisfied"]) for item in trace],
            [("task_goal_ontop", True), ("task_goal_nextto", False)],
        )
        summary = result["metrics"]["trace_summary"]
        self.assertEqual(summary["failed_primary_predicates"], ["nextto(apple,vase)"])
        self.assertEqual(summary["unsatisfied_auxiliary_predicates"], ["grasped(agent,apple)"])

    def test_compound_place_accepts_ontop_subset_without_neighbor_binding(self):
        apple = _make_mock_object("apple", state_values={"OnTop": lambda other: True})
        env = _make_mock_env(objects={"apple": apple, "table": MagicMock()})
        _set_task_goals(
            env,
            "heterogeneous_place_task",
            _FakeGoalHead(["ontop", "apple", "table"]),
        )

        result = check_skill_completed(
            env,
            "place on next to",
            object_bindings={"obj": "apple", "support_target": "table"},
        )

        self.assertTrue(result["completed"])
        self.assertEqual([item["predicate"] for item in result["metrics"]["trace"]], ["ontop(apple,table)"])

    def test_compound_place_accepts_nextto_subset_without_support_binding(self):
        gift = _make_mock_object("gift", state_values={"NextTo": lambda other: True})
        env = _make_mock_env(objects={"gift": gift, "tree": MagicMock()})
        _set_task_goals(
            env,
            "heterogeneous_place_task",
            _FakeGoalHead(["nextto", "gift", "tree"]),
        )

        result = check_skill_completed(
            env,
            "place on next to",
            object_bindings={"obj": "gift", "neighbor_target": "tree"},
        )

        self.assertTrue(result["completed"])
        self.assertEqual([item["predicate"] for item in result["metrics"]["trace"]], ["nextto(gift,tree)"])

    def test_compound_place_rejects_missing_neighbor_when_goal_requires_it(self):
        env = _make_mock_env()
        _set_task_goals(
            env,
            "compound_place_task",
            _FakeGoalHead(["ontop", "apple", "table"]),
            _FakeGoalHead(["nextto", "apple", "vase"]),
        )

        result = check_skill_completed(
            env,
            "place on next to",
            object_bindings={"obj": "apple", "support_target": "table"},
        )

        self.assertFalse(result["completed"])
        self.assertEqual(result["result_type"], "invalid_object_bindings")
        invalid = result["metrics"]["invalid_role_bindings"]
        self.assertEqual(invalid[0]["reason"], "task_goal_match_role_missing")

    def test_attach_and_hang_use_bddl_relation_not_release_state(self):
        attached_obj = _make_mock_object("item", state_values={"AttachedTo": lambda other: True})
        env = _make_mock_env(
            objects={"item": attached_obj, "fixture": MagicMock()},
            robot_grasping={"right": True},
        )

        for skill in ("attach", "hang"):
            with self.subTest(skill=skill):
                result = check_skill_completed(
                    env,
                    skill,
                    object_bindings={"obj": "item", "dst_or_target": "fixture"},
                )
                self.assertTrue(result["completed"])
                self.assertEqual([item["predicate"] for item in result["metrics"]["trace"]], ["attached(item,fixture)"])
                auxiliary = result["metrics"]["trace_summary"]["auxiliary_diagnostics"]
                self.assertEqual([item["predicate"] for item in auxiliary], ["grasped(agent,item)"])
                self.assertFalse(auxiliary[0]["contributes_to_success"])

    def test_attach_task_release_uses_attached_goal_instead_of_grasp_state(self):
        camera = _make_mock_object("camera", state_values={"AttachedTo": lambda other: True})
        env = _make_mock_env(
            objects={"camera": camera, "tripod": MagicMock()},
            robot_grasping={"right": True},
        )
        _set_task_goals(
            env,
            "attach_a_camera_to_a_tripod",
            _FakeGoalHead(["attached", "camera", "tripod"]),
        )

        result = check_skill_completed(env, "release", object_bindings={"obj": "tripod"})

        self.assertTrue(result["completed"])
        self.assertEqual([item["predicate"] for item in result["metrics"]["trace"]], ["attached(camera,tripod)"])

    def test_non_attach_release_still_checks_release_state(self):
        broom = _make_mock_object("broom")
        env = _make_mock_env(objects={"broom": broom}, robot_grasping={"right": True})
        env.task.activity_name = "clean_a_patio"

        result = check_skill_completed(env, "release", object_bindings={"obj": "broom"})

        self.assertFalse(result["completed"])
        self.assertEqual([item["predicate"] for item in result["metrics"]["trace"]], ["grasped(agent,broom)"])

    def test_spray_uses_bound_covered_goal_not_tool_contact(self):
        pesticide = MagicMock(name="pesticide")
        pesticide.name = "pesticide"
        tree = _make_mock_object("tree", state_values={"Covered": lambda system: system is pesticide})
        atomizer = _make_mock_object("atomizer", state_values={"Touching": lambda other: False})
        env = _make_mock_env(
            objects={"tree": tree, "atomizer": atomizer},
            systems={"pesticide": pesticide},
        )
        env.scene.get_task_metadata.return_value = {
            "tree.n.01_1": "tree",
            "pesticide.n.01_1": "pesticide",
        }
        _set_task_goals(
            env,
            "spraying_fruit_trees",
            _FakeGoalHead(["covered", "tree.n.01_1", "pesticide.n.01_1"]),
        )

        result = check_skill_completed(
            env,
            "spray",
            object_bindings={"obj": "atomizer", "target_obj": "tree"},
        )

        self.assertTrue(result["completed"])
        self.assertEqual(
            [item["predicate"] for item in result["metrics"]["trace"]],
            ["covered(tree.n.01_1,pesticide.n.01_1)"],
        )

    def test_cleaning_effects_use_negative_covered_goal_not_contact(self):
        dirt = MagicMock(name="dirt")
        dirt.name = "dirt"
        target = _make_mock_object("target", state_values={"Covered": lambda system: False})
        tool = _make_mock_object("tool", state_values={"Touching": lambda other: False})
        env = _make_mock_env(objects={"target": target, "tool": tool}, systems={"dirt": dirt})
        _set_task_goals(env, "clean_task", _negative_goal(["covered", "target", "dirt"]))

        for skill, target_role in (("wipe hard", "target_obj"), ("sweep surface", "target_obj_or_surface")):
            with self.subTest(skill=skill):
                result = check_skill_completed(
                    env,
                    skill,
                    object_bindings={"obj": "tool", target_role: "target"},
                )
                self.assertTrue(result["completed"])
                trace = result["metrics"]["trace"]
                self.assertEqual([item["predicate"] for item in trace], ["covered(target,dirt)"])
                self.assertFalse(trace[0]["desired"])

    def test_effect_goal_no_goal_unmatched_target_and_malformed_atom_are_invalid(self):
        cases = {
            "no_goal": [],
            "unmatched_target": [_FakeGoalHead(["covered", "other_tree", "pesticide"])],
            "malformed_atom": [_FakeGoalHead(["covered", "tree"])],
        }

        for case, goals in cases.items():
            with self.subTest(case=case):
                env = _make_mock_env()
                _set_task_goals(env, "spraying_fruit_trees", *goals)
                result = check_skill_completed(
                    env,
                    "spray",
                    object_bindings={"obj": "atomizer", "target_obj": "tree"},
                )

                self.assertFalse(result["completed"])
                self.assertEqual(result["result_type"], "invalid_object_bindings")
                invalid = result["metrics"]["invalid_role_bindings"]
                self.assertEqual(invalid[0]["reason"], "no_direct_task_goal_match")
                self.assertEqual(result["metrics"]["trace"], [])

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
