from __future__ import annotations

from typing import Any, Dict, List, Optional


def _pred(name: str, args: List[str], desired: bool, **kwargs: Any) -> Dict[str, Any]:
    spec = {"type": "predicate", "name": name, "args": args, "desired": desired}
    spec.update(kwargs)
    return spec


def _geom(metric_type: str, role: str, **kwargs: Any) -> Dict[str, Any]:
    spec = {"type": metric_type, "role": role}
    spec.update(kwargs)
    return spec


SKILL_METRIC_REGISTRY: Dict[str, Dict[str, Any]] = {
    "move to": {
        "metric_family": "geometry_base_target",
        "object_roles": ["target_or_obj"],
        "success_rule": "robot base stays within a target-relative proximity threshold captured from demo end state",
        "metrics": [_geom("base_to_object", "target_or_obj", margin=0.10, min_threshold=0.30)],
        "require_unsatisfied_at_start": True,
    },
    "pick up from": {
        "metric_family": "grasp_relation",
        "object_roles": ["obj", "src_or_target"],
        "success_rule": "object is grasped and no longer ontop/inside original source",
        "metrics": [
            _pred("grasped", ["agent", "obj"], True),
            _pred("ontop", ["obj", "src_or_target"], False),
            _pred("inside", ["obj", "src_or_target"], False),
        ],
        "require_unsatisfied_at_start": True,
    },
    "place in": {
        "metric_family": "relation_place_inside",
        "object_roles": ["obj", "dst_or_target"],
        "success_rule": "object satisfies the full-task BDDL inside relation",
        "metrics": [
            _pred("inside", ["obj", "dst_or_target"], True, semantic_role="spatial_inside"),
        ],
        # Full-task BDDL placement goals encode the spatial relation, not robot release.
        # Preserve grasp state as boundary / restore telemetry without making it a success gate.
        "diagnostic_metrics": [
            _pred("grasped", ["agent", "obj"], False, semantic_role="release_state"),
        ],
    },
    "place on": {
        "metric_family": "relation_place_ontop",
        "object_roles": ["obj", "dst_or_target"],
        "success_rule": "object satisfies the full-task BDDL ontop relation",
        "metrics": [
            _pred("ontop", ["obj", "dst_or_target"], True, semantic_role="spatial_ontop"),
        ],
        "diagnostic_metrics": [
            _pred("grasped", ["agent", "obj"], False, semantic_role="release_state"),
        ],
    },
    "push to": {
        "metric_family": "geometry_object_target",
        "object_roles": ["obj", "target_or_dst"],
        "success_rule": "object reaches demo-end target pose or becomes nextto target",
        "metrics": [
            _pred("nextto", ["obj", "target_or_dst"], True),
            _geom("object_pose_match", "obj", xy_threshold=0.12, z_threshold=0.10),
        ],
        "combine_mode": "any_of",
        "require_unsatisfied_at_start": True,
    },
    "chop": {
        "metric_family": "contact_effect_proxy",
        "object_roles": ["obj", "target_obj"],
        "success_rule": "tool contacts target object during chopping window",
        "metrics": [_pred("touching", ["obj", "target_obj"], True)],
    },
    "open door": {
        "metric_family": "articulation_open",
        "object_roles": ["unary_target"],
        "success_rule": "target articulated object is open",
        "metrics": [_pred("open", ["unary_target"], True)],
    },
    "place on next to": {
        "metric_family": "relation_place_ontop_nextto",
        "object_roles": ["obj", "support_target", "neighbor_target"],
        "success_rule": (
            "object satisfies the bound full-task BDDL ontop and/or nextto subpredicates; "
            "conjoin only the relations present for this annotation binding"
        ),
        # This label is heterogeneous in the challenge annotations. Some bound goals require
        # both relations, some only one, and some contain no matching placement goal at all.
        # Extract only directly matching goal atoms instead of imposing a global conjunction.
        "metrics": [],
        "task_goal_predicates": ["ontop", "nextto"],
        "task_goal_match_roles": {
            "ontop": {0: ["obj"], 1: ["support_target"]},
            "nextto": {0: ["obj"], 1: ["neighbor_target"]},
        },
        "task_goal_replace_primary": True,
        "diagnostic_metrics": [
            _pred("grasped", ["agent", "obj"], False, semantic_role="release_state"),
        ],
    },
    "close door": {
        "metric_family": "articulation_close",
        "object_roles": ["unary_target"],
        "success_rule": "target articulated object is closed and remains closed for a short trailing window",
        "metrics": [_pred("open", ["unary_target"], False)],
        "success_min_consecutive": 3,
    },
    "sweep surface": {
        "metric_family": "effect_goal_covered",
        "object_roles": ["obj", "target_obj_or_surface"],
        "success_rule": "target surface satisfies its bound full-task BDDL covered goal subpredicate",
        "metrics": [],
        "task_goal_predicates": ["covered"],
        "task_goal_match_roles": {
            "covered": {0: ["target_obj_or_surface"]},
        },
        "task_goal_replace_primary": True,
        "require_unsatisfied_at_start": True,
    },
    "pour": {
        "metric_family": "relation_transfer_proxy",
        "object_roles": ["payload_or_obj", "dst_or_target", "obj"],
        "success_rule": "payload reaches target/support; container end-pose proxy alone is not sufficient",
        "metrics": [
            _pred("ontop", ["payload_or_obj", "dst_or_target"], True),
            _pred("inside", ["payload_or_obj", "dst_or_target"], True),
        ],
        "combine_mode": "any_of",
        "require_unsatisfied_at_start": True,
    },
    "turn on switch": {
        "metric_family": "toggle_on",
        "object_roles": ["unary_target"],
        "success_rule": "target is toggled on",
        "metrics": [_pred("toggled_on", ["unary_target"], True)],
    },
    "close lid": {
        "metric_family": "articulation_close",
        "object_roles": ["unary_target"],
        "success_rule": "lid/container is closed and remains closed for a short trailing window",
        "metrics": [_pred("open", ["unary_target"], False)],
        "success_min_consecutive": 5,
    },
    "turn to": {
        "metric_family": "geometry_base_facing",
        "object_roles": ["face_target"],
        "success_rule": "robot base yaw faces target object within threshold from demo end state",
        "metrics": [_geom("face_object", "face_target", yaw_margin=0.12, min_threshold=0.25)],
        "min_yaw_error_improvement": 0.02,
        "short_success_caution_steps": 30,
        "short_success_caution_step_fraction": 0.10,
        "require_unsatisfied_at_start": True,
    },
    "turn off switch": {
        "metric_family": "toggle_off",
        "object_roles": ["unary_target"],
        "success_rule": "target is toggled off",
        "metrics": [_pred("toggled_on", ["unary_target"], False)],
    },
    "hand over": {
        "metric_family": "transfer_pose_proxy",
        "object_roles": ["obj", "dst_or_target"],
        "success_rule": (
            "object reaches demo-end handover pose while remaining grasped and, when a handover target "
            "is available, is next to that target"
        ),
        "metrics": [
            _geom("object_pose_match", "obj", xy_threshold=0.10, z_threshold=0.10),
            _pred("grasped", ["agent", "obj"], True),
            _pred("nextto", ["obj", "dst_or_target"], True, optional=True),
        ],
        "min_rollout_steps_for_success": 30,
        "min_success_step_fraction": 0.10,
        "likely_false_positive_on_short_success": True,
    },
    "spray": {
        "metric_family": "effect_goal_covered",
        "object_roles": ["obj", "target_obj"],
        "success_rule": "spray target satisfies its bound full-task BDDL covered goal subpredicate",
        "metrics": [],
        "task_goal_predicates": ["covered"],
        "task_goal_match_roles": {
            "covered": {0: ["target_obj"]},
        },
        "task_goal_replace_primary": True,
        "require_unsatisfied_at_start": True,
    },
    "open lid": {
        "metric_family": "articulation_open",
        "object_roles": ["unary_target"],
        "success_rule": "lid/container is open and remains open for a short trailing window",
        "metrics": [_pred("open", ["unary_target"], True)],
        "success_min_consecutive": 5,
    },
    "hold": {
        "metric_family": "grasp_hold",
        "object_roles": ["obj"],
        "success_rule": "object is grasped",
        "metrics": [_pred("grasped", ["agent", "obj"], True)],
    },
    "release": {
        "metric_family": "grasp_release",
        "object_roles": ["obj"],
        "success_rule": "object is no longer grasped, unless a bound attach-task final relation supersedes release state",
        "metrics": [_pred("grasped", ["agent", "obj"], False)],
        "task_aware_final_relation_predicates": ["attached"],
        "task_aware_final_relation_task_prefixes": ["attach_"],
        # The attach task terminates as soon as its BDDL attached goal is true, before
        # the trailing release annotation. Do not conjoin a non-goal grasp predicate.
        "task_goal_replace_primary": True,
    },
    "tip over": {
        "metric_family": "orientation_proxy",
        "object_roles": ["obj"],
        "success_rule": "object orientation matches tipped end-state proxy",
        "metrics": [_geom("object_orientation_match", "obj", angle_threshold=0.35)],
    },
    "insert": {
        "metric_family": "relation_place_inside",
        "object_roles": ["obj", "dst_or_target"],
        "success_rule": "object satisfies the full-task BDDL inside relation",
        "metrics": [
            _pred("inside", ["obj", "dst_or_target"], True, semantic_role="spatial_inside"),
        ],
        "diagnostic_metrics": [
            _pred("grasped", ["agent", "obj"], False, semantic_role="release_state"),
        ],
    },
    "sweep off": {
        "metric_family": "relation_detach_surface",
        "object_roles": ["obj", "src_or_target"],
        "success_rule": "object is no longer ontop the original surface",
        "metrics": [_pred("ontop", ["obj", "src_or_target"], False)],
    },
    "open drawer": {
        "metric_family": "articulation_open",
        "object_roles": ["unary_target"],
        "success_rule": "drawer/openable target is open",
        "metrics": [_pred("open", ["unary_target"], True)],
    },
    "close drawer": {
        "metric_family": "articulation_close",
        "object_roles": ["unary_target"],
        "success_rule": "drawer/openable target is closed and remains closed for a short trailing window",
        "metrics": [_pred("open", ["unary_target"], False)],
        "success_min_consecutive": 3,
    },
    "place in next to": {
        "metric_family": "relation_place_inside_nextto",
        "object_roles": ["obj", "support_target", "neighbor_target"],
        "success_rule": "object satisfies the directly bound full-task BDDL inside subpredicate",
        # All audited exact rows bind this annotation label to an inside-only task goal.
        # The label's trailing "next to" text is not evidence for an extra success gate.
        "metrics": [],
        "task_goal_predicates": ["inside"],
        "task_goal_match_roles": {
            "inside": {0: ["obj"], 1: ["support_target"]},
        },
        "task_goal_replace_primary": True,
        "diagnostic_metrics": [
            _pred("grasped", ["agent", "obj"], False, semantic_role="release_state"),
        ],
    },
    "place under": {
        "metric_family": "relation_under",
        "object_roles": ["obj", "dst_or_target"],
        "success_rule": "object satisfies the full-task BDDL under relation",
        "metrics": [
            _pred("under", ["obj", "dst_or_target"], True, semantic_role="spatial_under"),
        ],
        "diagnostic_metrics": [
            _pred("grasped", ["agent", "obj"], False, semantic_role="release_state"),
        ],
    },
    "pull tray": {
        "metric_family": "articulation_open_proxy",
        "object_roles": ["unary_target"],
        "success_rule": "tray-bearing object is open (pulled out) and remains open for a short trailing window",
        "metrics": [_pred("open", ["unary_target"], True)],
        "success_min_consecutive": 5,
    },
    "press": {
        "metric_family": "toggle_on",
        "object_roles": ["unary_target"],
        "success_rule": "pressed target is toggled on",
        "metrics": [_pred("toggled_on", ["unary_target"], True)],
    },
    "ignite": {
        "metric_family": "effect_on_fire",
        "object_roles": ["target_obj"],
        "success_rule": "target object is on fire",
        "metrics": [_pred("on_fire", ["target_obj"], True)],
    },
    "hang": {
        "metric_family": "relation_attach",
        "object_roles": ["obj", "dst_or_target"],
        "success_rule": "object satisfies the full-task BDDL attached relation",
        "metrics": [
            _pred("attached", ["obj", "dst_or_target"], True, semantic_role="attachment_relation"),
        ],
        "diagnostic_metrics": [
            _pred("grasped", ["agent", "obj"], False, semantic_role="release_state"),
        ],
    },
    "attach": {
        "metric_family": "relation_attach",
        "object_roles": ["obj", "dst_or_target"],
        "success_rule": "object satisfies the full-task BDDL attached relation",
        "metrics": [
            _pred("attached", ["obj", "dst_or_target"], True, semantic_role="attachment_relation"),
        ],
        "diagnostic_metrics": [
            _pred("grasped", ["agent", "obj"], False, semantic_role="release_state"),
        ],
    },
    "wipe hard": {
        "metric_family": "effect_goal_covered",
        "object_roles": ["obj", "target_obj"],
        "success_rule": "wipe target satisfies its bound full-task BDDL covered goal subpredicate",
        "metrics": [],
        "task_goal_predicates": ["covered"],
        "task_goal_match_roles": {
            "covered": {0: ["target_obj"]},
        },
        "task_goal_replace_primary": True,
    },
    "push tray": {
        "metric_family": "articulation_close_proxy",
        "object_roles": ["unary_target"],
        "success_rule": "tray-bearing object is closed (pushed in)",
        "metrics": [_pred("open", ["unary_target"], False)],
    },
}


def get_skill_metric_entry(skill_desc: Optional[str]) -> Optional[Dict[str, Any]]:
    if skill_desc is None:
        return None
    return SKILL_METRIC_REGISTRY.get(skill_desc.strip().lower(), None)
