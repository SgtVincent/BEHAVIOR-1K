from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from omnigibson.object_states import (
    AttachedTo,
    Covered,
    Inside,
    NextTo,
    OnFire,
    OnTop,
    Open,
    ToggledOn,
    Touching,
    Under,
)
from omnigibson.learning.utils.segment_skill_metric_registry import get_skill_metric_entry


logger = logging.getLogger("segment_predicate_eval")
logger.setLevel(logging.INFO)

MISSING_OBJECT_RESULT_TYPE = "metric_invalid_missing_object"

_NON_OBJECT_ROLE_TOKENS = {"robot", "right", "left", "face", "low_level", ""}
_TASK_GOAL_PREDICATE_ARITIES = {
    "attached": 2,
    "covered": 2,
    "inside": 2,
    "nextto": 2,
    "ontop": 2,
}


def _flatten(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        out = []
        for y in x:
            out.extend(_flatten(y))
        return out
    return [x]


def _first_text(x: Any) -> Optional[str]:
    for item in _flatten(x):
        if isinstance(item, str):
            s = item.strip()
            if s:
                return s
    return None


def _safe_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    s = str(name).strip()
    return s or None


def _is_object_like_name(name: Optional[str]) -> bool:
    if name is None:
        return False
    s = str(name).strip().lower()
    return bool(s) and s not in _NON_OBJECT_ROLE_TOKENS


def extract_main_target(segment: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    obj = segment.get("object_id", None)
    if isinstance(obj, list) and obj:
        g = obj[0]
        if isinstance(g, list):
            if len(g) >= 2:
                return _safe_name(_first_text(g[0])), _safe_name(_first_text(g[-1]))
            if len(g) == 1:
                return _safe_name(_first_text(g[0])), None
    main = _safe_name(_first_text(obj))
    if main is None:
        return None, None
    flat = [_safe_name(x) for x in _flatten(obj) if _safe_name(x) is not None]
    tgt = flat[-1] if len(flat) >= 2 else None
    return main, tgt


def parse_object_slots(segment: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    obj = segment.get("object_id", None)
    if isinstance(obj, list) and obj and isinstance(obj[0], list):
        g = obj[0]
        if isinstance(g, list):
            vals = [_safe_name(_first_text(x)) for x in g[:3]]
            while len(vals) < 3:
                vals.append(None)
            return vals[0], vals[1], vals[2]
    return None, None, None


def extract_segment_args(segment_level: str, segment: Dict[str, Any]) -> Dict[str, Optional[str]]:
    verb = _first_text(segment.get(f"{segment_level}_description", None))
    main, target = extract_main_target(segment)
    parsed_obj, parsed_a, parsed_b = parse_object_slots(segment)
    manipulating_obj = _safe_name(_first_text(segment.get("manipulating_object_id", None)))
    obj = manipulating_obj or parsed_obj or main
    src = None
    dst = None

    if verb is not None:
        verb = verb.strip().lower()
    if verb in {"pick up from", "take out of", "push to"}:
        src = parsed_a or target
    elif verb in {"place on", "place in", "insert", "place under"}:
        dst = parsed_a or target
    elif verb in {"place on next to", "place in next to"}:
        dst = parsed_a or target

    return {
        "verb": verb,
        "obj": obj,
        "src": src,
        "dst": dst,
        "target": target,
        "parsed_obj": parsed_obj,
        "parsed_a": parsed_a,
        "parsed_b": parsed_b,
        "manipulating_object_id": manipulating_obj,
    }


def _object_from_name(env, name: Optional[str]):
    if not name:
        return None
    obj = env.scene.object_registry("name", name)
    if obj is not None:
        return obj
    for alias in _generated_object_name_aliases(name):
        obj = env.scene.object_registry("name", alias)
        if obj is not None:
            return obj
    inst_to_name = env.scene.get_task_metadata("inst_to_name")
    if isinstance(inst_to_name, dict):
        for inst, obj_name in inst_to_name.items():
            if inst == name:
                maybe = env.scene.object_registry("name", obj_name)
                if maybe is not None:
                    return maybe
            if obj_name == name:
                maybe = env.scene.object_registry("name", obj_name)
                if maybe is not None:
                    return maybe
            if inst in _generated_object_name_aliases(name) or obj_name in _generated_object_name_aliases(name):
                maybe = env.scene.object_registry("name", obj_name)
                if maybe is not None:
                    return maybe
    return None


def _generated_object_name_aliases(name: Optional[str]) -> List[str]:
    """Return scene-object aliases for generated slice / portion names."""

    if name is None:
        return []
    s = str(name).strip()
    if not s:
        return []
    aliases: List[str] = []
    prefixes = (
        "half_",
        "quarter_",
        "slice_",
        "sliced_",
        "piece_",
        "pieces_",
        "diced_",
        "chopped_",
        "cut_",
    )
    without_prefix = s
    for prefix in prefixes:
        if without_prefix.startswith(prefix):
            without_prefix = without_prefix[len(prefix) :]
            aliases.append(without_prefix)
            break
    # Generated products often append an extra product index after the source
    # object instance id, e.g. bell_pepper_213_0 -> bell_pepper_213.
    for base in [s, without_prefix]:
        stripped = re.sub(r"_(\d+)$", "", base)
        if stripped != base:
            aliases.append(stripped)
    out: List[str] = []
    seen = {s}
    for alias in aliases:
        if alias and alias not in seen:
            out.append(alias)
            seen.add(alias)
    return out


def _task_entity_aliases(env, name: Optional[str]) -> List[str]:
    if not name:
        return []
    aliases = [str(name), *_generated_object_name_aliases(name)]
    inst_to_name = env.scene.get_task_metadata("inst_to_name")
    if isinstance(inst_to_name, dict):
        for inst, scene_name in inst_to_name.items():
            if str(name) in {str(inst), str(scene_name)}:
                aliases.extend([str(inst), str(scene_name)])
    return list(dict.fromkeys(alias for alias in aliases if alias))


def _same_task_entity(env, left: Optional[str], right: Optional[str]) -> bool:
    if not left or not right:
        return False
    left_obj = _object_from_name(env, left)
    right_obj = _object_from_name(env, right)
    if left_obj is not None and right_obj is not None:
        return left_obj is right_obj or getattr(left_obj, "name", None) == getattr(right_obj, "name", None)
    return bool(set(_task_entity_aliases(env, left)) & set(_task_entity_aliases(env, right)))


def _system_from_name(env, name: Optional[str]):
    if not name:
        return None
    for candidate in _task_entity_aliases(env, name):
        try:
            system = env.scene.get_system(candidate, force_init=False)
        except (AssertionError, KeyError, TypeError, ValueError):
            continue
        if system is not None:
            return system
    return None


def _task_activity_name(env) -> str:
    return str(getattr(getattr(env, "task", None), "activity_name", "") or "")


def _goal_body_arg_to_name(arg: Any) -> Optional[str]:
    if isinstance(arg, (dict, list, tuple)):
        return None
    return _safe_name(str(arg).lstrip("?"))


def _goal_atom(head: Any) -> Tuple[Optional[Sequence[Any]], bool]:
    """Return an atomic goal body and polarity from a grounded BDDL goal node."""

    body = getattr(head, "body", None)
    if body:
        # Grounded BDDL options contain HEAD nodes. A negative atom is encoded as
        # HEAD.body == ["not", [predicate, arg, ...]], not as a Negation node.
        if str(body[0]).lower() == "not":
            if len(body) == 2 and isinstance(body[1], (list, tuple)) and body[1]:
                return body[1], False
            return None, False
        return body, True
    if head.__class__.__name__.lower() == "negation":
        children = getattr(head, "children", None) or []
        if len(children) == 1:
            child = children[0]
            child_body = getattr(child, "body", None)
            state_name = getattr(child, "STATE_NAME", None)
            if child_body and state_name:
                return [state_name, *child_body], False
            if child_body:
                return child_body, False
    return None, True


def _no_direct_task_goal_match(predicate_names: Sequence[str]) -> Dict[str, Any]:
    return {
        "metric_type": "task_goal_binding",
        "metric_name": "|".join(str(name) for name in predicate_names),
        "role": "task_goal",
        "reason": "no_direct_task_goal_match",
        "candidate_predicates": [str(name) for name in predicate_names],
    }


def _task_goal_predicates(
    env,
    predicate_names: Sequence[str],
    *,
    metric_family: str = "task_goal_relation",
    success_rule: str = "bound full-task BDDL goal subpredicate is satisfied",
) -> List[SegmentPredicate]:
    allowed = set(predicate_names)
    specs: List[SegmentPredicate] = []
    seen = set()
    for option in getattr(getattr(env, "task", None), "ground_goal_state_options", []) or []:
        for head in option or []:
            body, desired = _goal_atom(head)
            if not body or len(body) < 2 or str(body[0]) not in allowed:
                continue
            predicate_name = str(body[0])
            expected_arity = _TASK_GOAL_PREDICATE_ARITIES.get(predicate_name)
            if expected_arity is not None and len(body) != expected_arity + 1:
                continue
            args = [_goal_body_arg_to_name(arg) for arg in body[1:]]
            if any(arg is None for arg in args):
                continue
            key = (str(body[0]), tuple(args), desired)
            if key in seen:
                continue
            seen.add(key)
            specs.append(
                SegmentPredicate(
                    metric_type="predicate",
                    name=str(body[0]),
                    args=[str(arg) for arg in args],
                    desired=desired,
                    source="task_goal_subpredicate",
                    params={
                        "metric_family": metric_family,
                        "success_rule": success_rule,
                        "semantic_role": f"task_goal_{body[0]}",
                        "contributes_to_success": True,
                    },
                )
            )
    return specs


def _og_eval_predicate_detailed(env, name: str, args: Sequence[str]) -> Tuple[bool, Dict[str, Any]]:
    diagnostics: Dict[str, Any] = {
        "predicate_name": name,
        "predicate_args": list(args),
    }
    # Evaluate a compact predicate subset directly on OmniGibson runtime state.
    if name == "grasped":
        obj_name = args[1]
        obj = _object_from_name(env, obj_name)
        if obj is None:
            diagnostics["missing_object"] = obj_name
            return False, diagnostics
        robot = env.robots[0]
        arm_states = {
            arm: bool(robot.is_grasping(arm=arm, candidate_obj=obj))
            for arm in robot.arm_names
        }
        diagnostics["arm_grasp_states"] = arm_states
        return any(arm_states.values()), diagnostics
    if name == "ontop":
        obj = _object_from_name(env, args[0])
        other = _object_from_name(env, args[1])
        if obj is None or other is None:
            diagnostics["missing_object"] = args[0] if obj is None else args[1]
            return False, diagnostics
        return bool(obj.states[OnTop].get_value(other)), diagnostics
    if name == "inside":
        obj = _object_from_name(env, args[0])
        other = _object_from_name(env, args[1])
        if obj is None or other is None:
            diagnostics["missing_object"] = args[0] if obj is None else args[1]
            return False, diagnostics
        return bool(obj.states[Inside].get_value(other)), diagnostics
    if name == "touching":
        obj = _object_from_name(env, args[0])
        other = _object_from_name(env, args[1])
        if obj is None or other is None:
            diagnostics["missing_object"] = args[0] if obj is None else args[1]
            return False, diagnostics
        return bool(obj.states[Touching].get_value(other)), diagnostics
    if name == "covered":
        obj = _object_from_name(env, args[0])
        system = _system_from_name(env, args[1])
        if obj is None or system is None:
            missing = args[0] if obj is None else args[1]
            diagnostics["missing_object"] = missing
            if system is None:
                diagnostics["missing_system"] = args[1]
            return False, diagnostics
        diagnostics["system_name"] = getattr(system, "name", args[1])
        return bool(obj.states[Covered].get_value(system)), diagnostics
    if name == "nextto":
        obj = _object_from_name(env, args[0])
        other = _object_from_name(env, args[1])
        if obj is None or other is None:
            diagnostics["missing_object"] = args[0] if obj is None else args[1]
            return False, diagnostics
        return bool(obj.states[NextTo].get_value(other)), diagnostics
    if name == "attached":
        obj = _object_from_name(env, args[0])
        other = _object_from_name(env, args[1])
        if obj is None or other is None:
            diagnostics["missing_object"] = args[0] if obj is None else args[1]
            return False, diagnostics
        return bool(obj.states[AttachedTo].get_value(other)), diagnostics
    if name == "under":
        obj = _object_from_name(env, args[0])
        other = _object_from_name(env, args[1])
        if obj is None or other is None:
            diagnostics["missing_object"] = args[0] if obj is None else args[1]
            return False, diagnostics
        return bool(obj.states[Under].get_value(other)), diagnostics
    if name == "open":
        obj = _object_from_name(env, args[0])
        if obj is None:
            diagnostics["missing_object"] = args[0]
            return False, diagnostics
        return bool(obj.states[Open].get_value()), diagnostics
    if name == "toggled_on":
        obj = _object_from_name(env, args[0])
        if obj is None:
            diagnostics["missing_object"] = args[0]
            return False, diagnostics
        return bool(obj.states[ToggledOn].get_value()), diagnostics
    if name == "on_fire":
        obj = _object_from_name(env, args[0])
        if obj is None:
            diagnostics["missing_object"] = args[0]
            return False, diagnostics
        return bool(obj.states[OnFire].get_value()), diagnostics
    raise NotImplementedError(f"Unsupported predicate for segment_predicates: {name}")


@dataclass
class SegmentPredicate:
    metric_type: str
    name: str
    args: List[str]
    desired: Optional[bool]
    source: str
    params: Dict[str, Any] = field(default_factory=dict)
    # Auxiliary predicates are evaluated and emitted as nested diagnostics, but
    # never participate in the primary success reduction.
    diagnostic_specs: List["SegmentPredicate"] = field(default_factory=list)


def _resolve_role_name(info: Dict[str, Optional[str]], role: str) -> Optional[str]:
    parsed_a = info.get("parsed_a") if _is_object_like_name(info.get("parsed_a")) else None
    parsed_b = info.get("parsed_b") if _is_object_like_name(info.get("parsed_b")) else None
    target = info.get("target") if _is_object_like_name(info.get("target")) else None
    dst = info.get("dst") if _is_object_like_name(info.get("dst")) else None
    src = info.get("src") if _is_object_like_name(info.get("src")) else None
    obj = info.get("obj") if _is_object_like_name(info.get("obj")) else None
    parsed_obj = info.get("parsed_obj") if _is_object_like_name(info.get("parsed_obj")) else None
    support_target = parsed_a or dst or target
    # A two-slot compound annotation does not identify a distinct neighbor. Do
    # not silently reuse the support as the nextto target; surface the missing
    # role instead so the metric is invalid rather than semantically wrong.
    neighbor_target = parsed_b or (target if target != support_target else None)
    mapping = {
        "obj": obj,
        "src": src,
        "dst": dst,
        "target": target,
        "parsed_obj": parsed_obj,
        "parsed_a": parsed_a,
        "parsed_b": parsed_b,
        "target_or_obj": target or obj,
        "target_or_dst": target or dst or obj,
        "src_or_target": src or target,
        "dst_or_target": dst or target,
        "support_target": support_target,
        "neighbor_target": neighbor_target,
        "payload_or_obj": parsed_obj,
        "target_obj": parsed_a or target,
        "target_obj_or_surface": parsed_a or target,
        "unary_target": dst or target or obj,
        "face_target": parsed_a or target or obj,
    }
    return _safe_name(mapping.get(role))


def _filter_task_goal_predicates(
    env,
    specs: Sequence[SegmentPredicate],
    info: Dict[str, Optional[str]],
    match_roles: Dict[str, Dict[int, Sequence[str]]],
    *,
    role_bindings: Optional[Dict[str, Optional[str]]] = None,
) -> Tuple[List[SegmentPredicate], List[Dict[str, Any]]]:
    """Keep only goal atoms whose configured arguments match annotation roles."""

    matched: List[SegmentPredicate] = []
    missing_roles: List[Dict[str, Any]] = []
    for spec in specs:
        predicate_rules = match_roles.get(spec.name, {})
        if not predicate_rules:
            matched.append(spec)
            continue
        checks: List[Optional[bool]] = []
        unresolved: List[Tuple[int, List[str]]] = []
        for raw_index, raw_roles in predicate_rules.items():
            arg_index = int(raw_index)
            roles = [raw_roles] if isinstance(raw_roles, str) else list(raw_roles)
            resolved_names = []
            for role in roles:
                resolved = (role_bindings or {}).get(role)
                if resolved is None:
                    resolved = _resolve_role_name(info, role)
                if resolved is not None:
                    resolved_names.append(resolved)
            if not resolved_names:
                checks.append(None)
                unresolved.append((arg_index, roles))
                continue
            checks.append(
                arg_index < len(spec.args)
                and any(_same_task_entity(env, spec.args[arg_index], resolved) for resolved in resolved_names)
            )
        concrete_checks = [check for check in checks if check is not None]
        if unresolved:
            # Report a missing role only when all other configured arguments (normally
            # the goal subject) match this annotation. Unrelated task-goal atoms must
            # not invalidate the segment.
            if concrete_checks and all(concrete_checks):
                for arg_index, roles in unresolved:
                    missing_roles.append(
                        {
                            "metric_type": "predicate",
                            "metric_name": spec.name,
                            "goal_arg_index": arg_index,
                            "role": "|".join(roles),
                            "reason": "task_goal_match_role_missing",
                        }
                    )
            continue
        if checks and all(checks):
            matched.append(spec)
    return matched, missing_roles


def _capture_object_pose(env, obj_name: str) -> Optional[Dict[str, Any]]:
    obj = _object_from_name(env, obj_name)
    if obj is None:
        return None
    pos, quat = obj.get_position_orientation()
    return {"name": obj_name, "position": np.asarray(pos, dtype=float).tolist(), "quat": np.asarray(quat, dtype=float).tolist()}


def _capture_robot_base(env) -> Dict[str, Any]:
    pos, quat = env.robots[0].get_position_orientation()
    quat = np.asarray(quat, dtype=float)
    x, y, z, w = quat
    yaw = float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    return {"position": np.asarray(pos, dtype=float).tolist(), "quat": quat.tolist(), "yaw": yaw}


def _quat_angle_diff(q1: Sequence[float], q2: Sequence[float]) -> float:
    a = np.asarray(q1, dtype=float)
    b = np.asarray(q2, dtype=float)
    a = a / max(np.linalg.norm(a), 1e-8)
    b = b / max(np.linalg.norm(b), 1e-8)
    dot = float(np.clip(np.abs(np.dot(a, b)), -1.0, 1.0))
    return 2.0 * float(np.arccos(dot))


def build_template_predicates(
    segment_level: str,
    segment: Dict[str, Any],
    env,
    info: Optional[Dict[str, Optional[str]]] = None,
) -> Tuple[List[SegmentPredicate], Dict[str, Any]]:
    info = info or extract_segment_args(segment_level, segment)
    entry = get_skill_metric_entry(info.get("verb"))
    if entry is None:
        return [], {"registry_missing": True}

    resolved_object_roles = {
        role: _resolve_role_name(info, role) for role in entry.get("object_roles", [])
    }
    invalid_role_bindings: List[Dict[str, Any]] = []
    for role_group in entry.get("required_distinct_roles", []):
        bindings = {role: resolved_object_roles.get(role) for role in role_group}
        bound_values = [value for value in bindings.values() if value is not None]
        if len(bound_values) == len(role_group) and len(set(bound_values)) != len(bound_values):
            invalid_role_bindings.append(
                {
                    "reason": "required_roles_resolve_to_same_object",
                    "roles": list(role_group),
                    "bindings": bindings,
                }
            )

    specs: List[SegmentPredicate] = []
    missing_template_roles: List[Dict[str, Any]] = []
    missing_diagnostic_roles: List[Dict[str, Any]] = []
    for metric in entry.get("metrics", []):
        metric_type = metric.get("type")
        if metric_type == "predicate":
            args = []
            valid = True
            for role in metric.get("args", []):
                if role == "agent":
                    args.append("agent")
                    continue
                resolved = _resolve_role_name(info, role)
                if resolved is None:
                    valid = False
                    if not bool(metric.get("optional", False)):
                        missing_template_roles.append(
                            {
                                "metric_type": metric_type,
                                "metric_name": metric.get("name"),
                                "role": role,
                            }
                        )
                    break
                args.append(resolved)
            if not valid:
                continue
            params = {
                "metric_family": entry["metric_family"],
                "success_rule": entry["success_rule"],
                "semantic_role": metric.get("semantic_role", metric["name"]),
                "contributes_to_success": True,
            }
            if bool(metric.get("optional", False)):
                params["optional"] = True
            specs.append(
                SegmentPredicate(
                    metric_type="predicate",
                    name=metric["name"],
                    args=args,
                    desired=bool(metric["desired"]),
                    source="registry",
                    params=params,
                )
            )
        elif metric_type in {"base_to_object", "face_object", "object_pose_match", "object_orientation_match"}:
            resolved = _resolve_role_name(info, metric["role"])
            if resolved is None:
                if not bool(metric.get("optional", False)):
                    missing_template_roles.append(
                        {
                            "metric_type": metric_type,
                            "metric_name": metric.get("name", metric_type),
                            "role": metric.get("role"),
                        }
                    )
                continue
            params = dict(metric)
            params["resolved_role_name"] = resolved
            params["semantic_role"] = metric.get("semantic_role", metric_type)
            params["contributes_to_success"] = True
            if metric_type in {"object_pose_match", "object_orientation_match"}:
                captured = _capture_object_pose(env, resolved)
                if captured is None:
                    continue
                params.update(captured)
            elif metric_type == "base_to_object":
                target_obj = _object_from_name(env, resolved)
                if target_obj is None:
                    continue
                robot_base = _capture_robot_base(env)
                target_pos, _ = target_obj.get_position_orientation()
                dist = float(np.linalg.norm(np.asarray(robot_base["position"][:2]) - np.asarray(target_pos[:2])))
                params["threshold"] = max(float(metric.get("min_threshold", 0.9)), dist + float(metric.get("margin", 0.35)))
            elif metric_type == "face_object":
                target_obj = _object_from_name(env, resolved)
                if target_obj is None:
                    continue
                robot_base = _capture_robot_base(env)
                target_pos, _ = target_obj.get_position_orientation()
                vec = np.asarray(target_pos[:2]) - np.asarray(robot_base["position"][:2])
                target_yaw = float(np.arctan2(vec[1], vec[0]))
                end_err = abs(np.arctan2(np.sin(robot_base["yaw"] - target_yaw), np.cos(robot_base["yaw"] - target_yaw)))
                params["threshold"] = max(float(metric.get("min_threshold", 0.4)), end_err + float(metric.get("yaw_margin", 0.2)))
            specs.append(
                SegmentPredicate(
                    metric_type=metric_type,
                    name=metric_type,
                    args=[resolved],
                    desired=True,
                    source="registry",
                    params=params,
                )
            )

    diagnostic_specs: List[SegmentPredicate] = []
    for metric in entry.get("diagnostic_metrics", []):
        metric_type = metric.get("type")
        if metric_type != "predicate":
            missing_diagnostic_roles.append(
                {
                    "metric_type": metric_type,
                    "metric_name": metric.get("name", metric_type),
                    "reason": "unsupported_diagnostic_metric_type",
                }
            )
            continue
        args = []
        valid = True
        for role in metric.get("args", []):
            if role == "agent":
                args.append("agent")
                continue
            resolved = _resolve_role_name(info, role)
            if resolved is None:
                valid = False
                missing_diagnostic_roles.append(
                    {
                        "metric_type": metric_type,
                        "metric_name": metric.get("name"),
                        "role": role,
                    }
                )
                break
            args.append(resolved)
        if not valid:
            continue
        diagnostic_specs.append(
            SegmentPredicate(
                metric_type="predicate",
                name=metric["name"],
                args=args,
                desired=bool(metric["desired"]),
                source="registry_diagnostic",
                params={
                    "metric_family": entry["metric_family"],
                    "semantic_role": metric.get("semantic_role", metric["name"]),
                    "contributes_to_success": False,
                },
            )
        )

    task_prefixes = entry.get("task_aware_final_relation_task_prefixes") or []
    relation_predicate_names = (
        entry.get("task_goal_predicates") or entry.get("task_aware_final_relation_predicates") or []
    )
    task_scope_matches = not task_prefixes or any(
        _task_activity_name(env).startswith(str(prefix)) for prefix in task_prefixes
    )
    task_goal_binding_errors: List[Dict[str, Any]] = []
    if relation_predicate_names and task_scope_matches:
        task_goal_specs = _task_goal_predicates(
            env,
            relation_predicate_names,
            metric_family=entry["metric_family"],
            success_rule=entry["success_rule"],
        )
        task_goal_specs, task_goal_binding_errors = _filter_task_goal_predicates(
            env,
            task_goal_specs,
            info,
            entry.get("task_goal_match_roles") or {},
            role_bindings=resolved_object_roles,
        )
        if entry.get("task_goal_replace_primary", False) and not task_goal_specs and not task_goal_binding_errors:
            task_goal_binding_errors.append(_no_direct_task_goal_match(relation_predicate_names))
        missing_template_roles.extend(task_goal_binding_errors)
        if entry.get("task_goal_replace_primary", False):
            specs = task_goal_specs
        else:
            existing = {(spec.name, tuple(spec.args), spec.desired) for spec in specs if spec.metric_type == "predicate"}
            for spec in task_goal_specs:
                key = (spec.name, tuple(spec.args), spec.desired)
                if key not in existing:
                    specs.append(spec)
                    existing.add(key)

    if invalid_role_bindings:
        for invalid in invalid_role_bindings:
            missing_template_roles.append(
                {
                    "metric_type": "binding",
                    "metric_name": "required_distinct_roles",
                    "role": "|".join(invalid["roles"]),
                    **invalid,
                }
            )
    if (
        invalid_role_bindings
        or task_goal_binding_errors
        or (entry.get("required_distinct_roles") and missing_template_roles)
    ):
        # A malformed or unmatched task-goal binding must fail closed instead of
        # degrading to a partial relation or the auto-mined fallback.
        specs = []
    elif specs and diagnostic_specs:
        # Keep auxiliary state nested under the first primary trace item. This
        # preserves compatibility with callers that reduce the top-level trace.
        specs[0].diagnostic_specs.extend(diagnostic_specs)

    debug = {
        "metric_family": entry["metric_family"],
        "success_rule": entry["success_rule"],
        "combine_mode": entry.get("combine_mode", "all_of"),
        "require_unsatisfied_at_start": bool(entry.get("require_unsatisfied_at_start", True)),
        "resolved_object_roles": resolved_object_roles,
        "primary_metric_specs": [
            {
                "metric_type": spec.metric_type,
                "name": spec.name,
                "args": list(spec.args),
                "desired": spec.desired,
                "semantic_role": spec.params.get("semantic_role", spec.name),
            }
            for spec in specs
        ],
        "diagnostic_metric_specs": [
            {
                "metric_type": spec.metric_type,
                "name": spec.name,
                "args": list(spec.args),
                "desired": spec.desired,
                "semantic_role": spec.params.get("semantic_role", spec.name),
                "contributes_to_success": False,
            }
            for spec in diagnostic_specs
        ],
    }
    if missing_template_roles:
        debug["missing_template_roles"] = missing_template_roles
    if missing_diagnostic_roles:
        debug["missing_diagnostic_roles"] = missing_diagnostic_roles
    if invalid_role_bindings:
        debug["invalid_role_bindings"] = invalid_role_bindings
    if task_goal_binding_errors:
        debug["invalid_task_goal_bindings"] = task_goal_binding_errors
    debug.update(
        {
            key: value
            for key, value in entry.items()
            if key
            not in {
                "metric_family",
                "success_rule",
                "combine_mode",
                "require_unsatisfied_at_start",
                "metrics",
                "diagnostic_metrics",
                "object_roles",
            }
        }
    )
    return specs, debug


def build_auto_mined_predicates(
    segment_level: str,
    segment: Dict[str, Any],
    env,
    start_truth: Optional[Dict[str, bool]] = None,
    end_truth: Optional[Dict[str, bool]] = None,
) -> List[SegmentPredicate]:
    info = extract_segment_args(segment_level, segment)
    names = {
        _resolve_role_name(info, "obj"),
        _resolve_role_name(info, "src_or_target"),
        _resolve_role_name(info, "dst_or_target"),
        _resolve_role_name(info, "target_or_obj"),
        _resolve_role_name(info, "target_obj"),
        _resolve_role_name(info, "support_target"),
        _resolve_role_name(info, "neighbor_target"),
    }
    names = {x for x in names if x}
    candidates: List[Tuple[str, List[str]]] = []
    for obj in list(names):
        candidates.append(("grasped", ["agent", obj]))
        candidates.append(("toggled_on", [obj]))
        candidates.append(("open", [obj]))
        candidates.append(("on_fire", [obj]))
    ordered_names = sorted(names)
    for a in ordered_names:
        for b in ordered_names:
            if a == b:
                continue
            candidates.append(("ontop", [a, b]))
            candidates.append(("inside", [a, b]))
            candidates.append(("touching", [a, b]))
            candidates.append(("nextto", [a, b]))
            candidates.append(("attached", [a, b]))
            candidates.append(("under", [a, b]))

    deduped = []
    seen = set()
    for name, args in candidates:
        key = (name, tuple(args))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((name, args))

    if start_truth is None or end_truth is None:
        return [SegmentPredicate("predicate", name, args, True, "auto_mine_candidate") for name, args in deduped]

    mined: List[SegmentPredicate] = []
    for name, args in deduped:
        key = f"{name}({','.join(args)})"
        s = start_truth.get(key)
        e = end_truth.get(key)
        if s is None or e is None:
            continue
        if (not s) and e:
            mined.append(SegmentPredicate("predicate", name, args, True, "auto_mine_delta"))
        elif s and (not e):
            mined.append(SegmentPredicate("predicate", name, args, False, "auto_mine_delta"))
    return mined


def _eval_geometry_metric_detailed(env, spec: SegmentPredicate) -> Tuple[bool, Dict[str, Any]]:
    diagnostics: Dict[str, Any] = {
        "metric_name": spec.name,
        "metric_type": spec.metric_type,
        "metric_args": list(spec.args),
    }
    if spec.metric_type == "base_to_object":
        target = _object_from_name(env, spec.args[0])
        if target is None:
            diagnostics["missing_object"] = spec.args[0]
            return False, diagnostics
        robot_pos, _ = env.robots[0].get_position_orientation()
        target_pos, _ = target.get_position_orientation()
        dist = float(np.linalg.norm(np.asarray(robot_pos[:2]) - np.asarray(target_pos[:2])))
        threshold = float(spec.params["threshold"])
        diagnostics.update({"distance_xy": dist, "threshold": threshold, "target_name": spec.args[0]})
        return dist <= threshold, diagnostics
    if spec.metric_type == "face_object":
        target = _object_from_name(env, spec.args[0])
        if target is None:
            diagnostics["missing_object"] = spec.args[0]
            return False, diagnostics
        robot_pos, robot_quat = env.robots[0].get_position_orientation()
        q = np.asarray(robot_quat, dtype=float)
        x, y, z, w = q
        robot_yaw = float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
        target_pos, _ = target.get_position_orientation()
        vec = np.asarray(target_pos[:2]) - np.asarray(robot_pos[:2])
        target_yaw = float(np.arctan2(vec[1], vec[0]))
        err = abs(np.arctan2(np.sin(robot_yaw - target_yaw), np.cos(robot_yaw - target_yaw)))
        threshold = float(spec.params["threshold"])
        diagnostics.update({"yaw_error": err, "threshold": threshold, "target_name": spec.args[0]})
        return err <= threshold, diagnostics
    if spec.metric_type == "object_pose_match":
        obj = _object_from_name(env, spec.args[0])
        if obj is None:
            diagnostics["missing_object"] = spec.args[0]
            return False, diagnostics
        pos, _ = obj.get_position_orientation()
        tgt = np.asarray(spec.params["position"], dtype=float)
        pos = np.asarray(pos, dtype=float)
        xy_err = float(np.linalg.norm(pos[:2] - tgt[:2]))
        z_err = abs(float(pos[2] - tgt[2]))
        xy_threshold = float(spec.params.get("xy_threshold", 0.2))
        z_threshold = float(spec.params.get("z_threshold", 0.2))
        diagnostics.update(
            {
                "xy_error": xy_err,
                "z_error": z_err,
                "xy_threshold": xy_threshold,
                "z_threshold": z_threshold,
                "target_name": spec.args[0],
            }
        )
        return xy_err <= xy_threshold and z_err <= z_threshold, diagnostics
    if spec.metric_type == "object_orientation_match":
        obj = _object_from_name(env, spec.args[0])
        if obj is None:
            diagnostics["missing_object"] = spec.args[0]
            return False, diagnostics
        _, quat = obj.get_position_orientation()
        err = _quat_angle_diff(quat, spec.params["quat"])
        threshold = float(spec.params.get("angle_threshold", 0.55))
        diagnostics.update({"angle_error": err, "angle_threshold": threshold, "target_name": spec.args[0]})
        return err <= threshold, diagnostics
    raise NotImplementedError(f"Unknown geometry metric: {spec.metric_type}")


def eval_segment_predicates(
    env,
    predicate_specs: Sequence[SegmentPredicate],
) -> Tuple[Dict[str, bool], List[Dict[str, Any]]]:
    if not predicate_specs:
        return {}, []

    def evaluate_spec(spec: SegmentPredicate, *, contributes_to_success: bool) -> Tuple[str, bool, Dict[str, Any]]:
        semantic_role = spec.params.get("semantic_role", spec.name)
        try:
            if spec.metric_type == "predicate":
                value, diagnostics = _og_eval_predicate_detailed(env, spec.name, spec.args)
                value = bool(value)
                key = f"{spec.name}({','.join(spec.args)})"
            else:
                value, diagnostics = _eval_geometry_metric_detailed(env, spec)
                value = bool(value)
                key = f"{spec.metric_type}({','.join(spec.args)})"
            item = {
                "predicate": key,
                "metric_type": spec.metric_type,
                "semantic_role": semantic_role,
                "contributes_to_success": contributes_to_success,
                "desired": spec.desired,
                "value": value,
                "satisfied": value == bool(spec.desired),
                "source": spec.source,
                "params": spec.params,
                "diagnostics": diagnostics,
            }
        except Exception as error:
            key = f"{spec.name}({','.join(spec.args)})"
            value = False
            item = {
                "predicate": key,
                "metric_type": spec.metric_type,
                "semantic_role": semantic_role,
                "contributes_to_success": contributes_to_success,
                "desired": spec.desired,
                "value": False,
                "satisfied": False,
                "source": spec.source,
                "params": spec.params,
                "error": str(error),
            }
        return key, value, item

    trace: List[Dict[str, Any]] = []
    truth: Dict[str, bool] = {}
    for spec in predicate_specs:
        key, value, item = evaluate_spec(spec, contributes_to_success=True)
        truth[key] = value
        auxiliary_diagnostics: List[Dict[str, Any]] = []
        for diagnostic_spec in spec.diagnostic_specs:
            diagnostic_key, diagnostic_value, diagnostic_item = evaluate_spec(
                diagnostic_spec,
                contributes_to_success=False,
            )
            truth[diagnostic_key] = diagnostic_value
            auxiliary_diagnostics.append(diagnostic_item)
        if auxiliary_diagnostics:
            item["auxiliary_diagnostics"] = auxiliary_diagnostics
        trace.append(item)
    return truth, trace


def trace_missing_objects(trace: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    missing_objects: List[Dict[str, Any]] = []
    for item in trace or []:
        diagnostics = item.get("diagnostics") or {}
        missing_object = diagnostics.get("missing_object")
        if missing_object is None:
            continue
        missing_objects.append(
            {
                "predicate": item.get("predicate"),
                "metric_type": item.get("metric_type"),
                "desired": item.get("desired"),
                "value": item.get("value"),
                "satisfied": item.get("satisfied"),
                "missing_object": missing_object,
            }
        )
    return missing_objects


def trace_has_missing_object(trace: Sequence[Dict[str, Any]]) -> bool:
    return bool(trace_missing_objects(trace))


def trace_auxiliary_diagnostics(trace: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten non-gating predicate diagnostics nested in a primary trace."""

    return [
        diagnostic
        for item in trace or []
        for diagnostic in item.get("auxiliary_diagnostics", []) or []
    ]


def predicate_trace_satisfied(trace: Sequence[Dict[str, Any]], combine_mode: str = "all_of") -> bool:
    """Reduce primary trace items only; nested diagnostics never gate success."""

    if not trace or trace_has_missing_object(trace):
        return False
    if combine_mode == "any_of":
        return any(item.get("satisfied", False) for item in trace)
    return all(item.get("satisfied", False) for item in trace)


def summarize_predicate_trace(
    trace: Sequence[Dict[str, Any]],
    combine_mode: str = "all_of",
) -> Dict[str, Any]:
    """Separate BDDL-aligned primary failures from auxiliary state telemetry."""

    auxiliary = trace_auxiliary_diagnostics(trace)
    return {
        "primary_satisfied": predicate_trace_satisfied(trace, combine_mode),
        "failed_primary_predicates": [
            item.get("predicate") for item in trace or [] if not item.get("satisfied", False)
        ],
        "primary_semantic_roles": [item.get("semantic_role") for item in trace or []],
        "auxiliary_diagnostics": auxiliary,
        "unsatisfied_auxiliary_predicates": [
            item.get("predicate") for item in auxiliary if not item.get("satisfied", False)
        ],
    }


def predicate_window_satisfied(
    history: Sequence[List[Dict[str, Any]]],
    mode: str = "anytime",
    last_k: int = 20,
    min_consecutive: int = 1,
    combine_mode: str = "all_of",
    min_history_index: int = 0,
) -> bool:
    if not history:
        return False
    min_history_index = max(int(min_history_index), 0)
    if min_history_index:
        history = history[min_history_index:]
    if not history:
        return False

    sat_flags = [predicate_trace_satisfied(step_trace, combine_mode) for step_trace in history]
    if mode == "last_k":
        window = sat_flags[-max(last_k, 1):]
        return any(window)
    if mode == "consecutive":
        run = 0
        for flag in reversed(sat_flags):
            run = run + 1 if flag else 0
            if not flag:
                break
            if run >= max(min_consecutive, 1):
                return True
        return run >= max(min_consecutive, 1)
    return any(sat_flags)
