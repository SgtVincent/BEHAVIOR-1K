"""Read-only PnP observations and three-can predicate transitions.

Ported from the a28 frozen baseline ``trash_observer.py``. Unknown data never
becomes a success boolean: a failed read returns ``{"status": "UNKNOWN",
"error": ...}`` rather than a False/zero.

Re-bind boundary (gap G3): the robot/object accessor surface
(``is_grasping``, ``arm_names``, assisted-grasping counters/constraints,
object states ``Inside``/``ContactBodies``) is the OmniGibson 3.7.2 surface and
must be re-verified against post1 ``bd049de3``. This module performs no SDK
import itself; it consumes whatever read surface the sim-adapter provides.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List


def clean(value: Any) -> Any:
    if hasattr(value, "detach"):
        return clean(value.detach().cpu().numpy())
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"status": "UNKNOWN", "error": "nonfinite_value"}
    return value


def read_value(fn: Callable[[], Any]) -> Any:
    try:
        return clean(fn())
    except Exception as exc:  # noqa: BLE001 - unknown read must stay UNKNOWN, not raise
        return {"status": "UNKNOWN", "error": type(exc).__name__ + ": " + str(exc)}


def grasp_state(robot: Any, arm: str, obj: Any) -> Dict[str, Any]:
    state = robot.is_grasping(arm=arm, candidate_obj=obj)
    # IsGraspingState.FALSE=-1 is truthy. Never convert this enum with bool().
    return {"name": state.name, "value": int(state)}


def _label(obj: Any, fallback: str) -> str:
    if isinstance(obj, dict):
        return str(obj.get("name", fallback))
    return read_value(lambda: obj.name) if hasattr(obj, "name") else fallback


def observe(robot: Any, cans: Dict[str, Any], bucket: Any,
             pose_fn: Callable[[Any], Any], inside_type: Any,
             contact_type: Any) -> Dict[str, Any]:
    grasping_mode = read_value(lambda: robot.grasping_mode)
    arm_names = read_value(lambda: list(robot.arm_names))
    if not isinstance(arm_names, list):
        arm_names = []
    result: Dict[str, Any] = {"grasping_mode": grasping_mode, "objects": {}, "arms": {}}

    objects = dict(cans)
    bucket_resolved = bucket is not None and not (
        isinstance(bucket, dict) and bucket.get("status") in ("UNKNOWN", "BLOCKED"))
    if bucket_resolved:
        objects[_label(bucket, "bucket")] = bucket
    for name, obj in objects.items():
        if isinstance(obj, dict) and obj.get("status") in ("UNKNOWN", "BLOCKED"):
            result["objects"][name] = dict(obj)
            continue
        entry: Dict[str, Any] = {
            "pose_world": read_value(lambda obj=obj: pose_fn(obj)),
            "linear_velocity_world": read_value(obj.get_linear_velocity),
            "angular_velocity_world": read_value(obj.get_angular_velocity),
            "contact_bodies": read_value(
                lambda obj=obj: sorted(link.prim_path for link in obj.states[contact_type].get_value())
            ),
        }
        if name in cans:
            entry["inside_bucket"] = read_value(lambda obj=obj: obj.states[inside_type].get_value(bucket))
            entry["grasp_by_arm"] = {
                arm: read_value(lambda arm=arm, obj=obj: grasp_state(robot, arm, obj))
                for arm in arm_names
            }
        result["objects"][name] = entry
    for arm in arm_names:
        def ag(arm=arm) -> Dict[str, Any]:
            held = robot._ag_obj_in_hand[arm]
            return {
                "object_in_hand": None if held is None else held.name,
                "release_counter": clean(robot._ag_release_counter[arm]),
                "grasp_counter": clean(robot._ag_grasp_counter[arm]),
                "constraint_present": robot._ag_obj_constraints[arm] is not None,
                "constraint_params": clean(robot._ag_obj_constraint_params[arm]),
                "freeze_gripper": (clean(robot._ag_freeze_gripper[arm])
                                   if hasattr(robot, "_ag_freeze_gripper")
                                   else {"status": "UNKNOWN", "error": "field_absent_post1"}),
            }
        result["arms"][arm] = read_value(ag)
    return result


def transitions(previous: Dict[str, Any], current: Dict[str, Any],
                can_names: List[str]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    prev_objects = previous.get("objects", {}) if isinstance(previous, dict) else {}
    curr_objects = current.get("objects", {}) if isinstance(current, dict) else {}
    for name in can_names:
        before = prev_objects.get(name)
        after = curr_objects.get(name)
        degraded = (
            not isinstance(before, dict) or not isinstance(after, dict)
            or "inside_bucket" not in before or "inside_bucket" not in after
        )
        if degraded:
            # Missing predicate keys stay UNKNOWN/BLOCKED; do not synthesize
            # False/zero observations or raise a bare KeyError.
            events.append({"event": "observation_degraded", "object": name,
                           "code": "missing_predicate_keys"})
            continue
        if before["inside_bucket"] is False and after["inside_bucket"] is True:
            events.append({"event": "inside_entered", "object": name})
        if before["inside_bucket"] is True and after["inside_bucket"] is False:
            events.append({"event": "inside_exited", "object": name})
        old_grasps = before.get("grasp_by_arm") or {}
        new_grasps = after.get("grasp_by_arm") or {}
        for arm, state in new_grasps.items():
            old = old_grasps.get(arm) or {}
            if old.get("name") == "FALSE" and state.get("name") == "TRUE":
                events.append({"event": "sdk_grasp_started", "object": name, "arm": arm})
            if old.get("name") == "TRUE" and state.get("name") == "FALSE":
                events.append({"event": "sdk_grasp_released", "object": name, "arm": arm})
    return events
