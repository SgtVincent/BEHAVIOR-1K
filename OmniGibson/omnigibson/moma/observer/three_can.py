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


def observe(robot: Any, cans: Dict[str, Any], bucket: Any,
             pose_fn: Callable[[Any], Any], inside_type: Any,
             contact_type: Any) -> Dict[str, Any]:
    result: Dict[str, Any] = {"grasping_mode": robot.grasping_mode, "objects": {}, "arms": {}}
    for name, obj in {**cans, bucket.name: bucket}.items():
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
                for arm in robot.arm_names
            }
        result["objects"][name] = entry
    for arm in robot.arm_names:
        def ag(arm=arm) -> Dict[str, Any]:
            held = robot._ag_obj_in_hand[arm]
            return {
                "object_in_hand": None if held is None else held.name,
                "release_counter": clean(robot._ag_release_counter[arm]),
                "grasp_counter": clean(robot._ag_grasp_counter[arm]),
                "constraint_present": robot._ag_obj_constraints[arm] is not None,
                "constraint_params": clean(robot._ag_obj_constraint_params[arm]),
                "freeze_gripper": clean(robot._ag_freeze_gripper[arm]),
            }
        result["arms"][arm] = read_value(ag)
    return result


def transitions(previous: Dict[str, Any], current: Dict[str, Any],
                can_names: List[str]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for name in can_names:
        before = previous["objects"][name]
        after = current["objects"][name]
        if before["inside_bucket"] is False and after["inside_bucket"] is True:
            events.append({"event": "inside_entered", "object": name})
        if before["inside_bucket"] is True and after["inside_bucket"] is False:
            events.append({"event": "inside_exited", "object": name})
        for arm, state in after["grasp_by_arm"].items():
            old = before["grasp_by_arm"][arm]
            if old.get("name") == "FALSE" and state.get("name") == "TRUE":
                events.append({"event": "sdk_grasp_started", "object": name, "arm": arm})
            if old.get("name") == "TRUE" and state.get("name") == "FALSE":
                events.append({"event": "sdk_grasp_released", "object": name, "arm": arm})
    return events
