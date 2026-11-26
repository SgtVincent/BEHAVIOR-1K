"""Source-compatible target bootstrap config (CPU, sim-free).

Builds the post1 managed-entry environment config from the sealed a28 source
episode config. This layer only selects and validates parameters against the
original SDK-supported schema; it does **not** launch a simulator, does not
import omnigibson/torch, and never mutates the source hdf5.

Evidence base (read-only against post1 ``bd049de3…60fa20c108``):

- ``eval/utils/eval_utils.py`` ``generate_basic_environment_config``: env
  frequencies ``action_frequency=30`` / ``rendering_frequency=30`` /
  ``physics_frequency=120``; scene ``InteractiveTraversableScene``; task
  ``BehaviorTask``; ``online_object_sampling=False``.
- ``envs/env_base.py`` L275–305: the legacy robot ``type`` key is migrated to
  lowercase ``model`` and dropped; both must not coexist; ``model`` must be a
  registered robot id.
- ``eval/evaluator.py`` ``_build_robot_config``: canonical config requires
  ``model`` and ``name``; rejects legacy ``type``; pops the ``eval`` block.
- ``eval/r1pro.yaml``: controller order
  ``arm_left, arm_right, gripper_left, gripper_right, base, trunk, camera``;
  reset vector layout base 6 + trunk 4 + left arm 7 + right arm 7 + gripper
  fingers 4 = 28; 15 proprioception keys.

Field provenance:

- ``from_source``: value read from the sealed source episode config.
- ``post1_schema``: value fixed by the post1 SDK/eval schema evidence above.
- ``unknown``: no evidence; a required field in this state blocks the bootstrap
  instead of being guessed.

A blocked required field fails the **entire** bootstrap. A partial config is
never reported as a successful initial-state recipe.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .restore_compat import compute_uuid

# -- provenance labels --------------------------------------------------------

PROV_SOURCE = "from_source"
PROV_SCHEMA = "post1_schema"
PROV_UNKNOWN = "unknown"

# -- post1 schema snapshot (read-only evidence) -------------------------------

TARGET_FREQ: Dict[str, int] = {
    "action_frequency": 30,
    "rendering_frequency": 30,
    "physics_frequency": 120,
}

TARGET_SCENE_TYPE = "InteractiveTraversableScene"
TARGET_TASK_TYPE = "BehaviorTask"
TARGET_ROBOT_MODEL = "r1pro"

TARGET_CONTROLLER_ORDER: Tuple[str, ...] = (
    "arm_left",
    "arm_right",
    "gripper_left",
    "gripper_right",
    "base",
    "trunk",
    "camera",
)

# Expected controller signature per post1 eval/r1pro.yaml.
TARGET_CONTROLLERS: Dict[str, Dict[str, Any]] = {
    "arm_left": {"name": "JointController", "motor_type": "position",
                 "use_impedances": False, "use_delta_commands": False},
    "arm_right": {"name": "JointController", "motor_type": "position",
                  "use_impedances": False, "use_delta_commands": False},
    "trunk": {"name": "JointController", "motor_type": "position",
              "use_impedances": False, "use_delta_commands": False},
    "gripper_left": {"name": "MultiFingerGripperController", "mode": "smooth"},
    "gripper_right": {"name": "MultiFingerGripperController", "mode": "smooth"},
    "base": {"name": "HolonomicBaseJointController", "motor_type": "velocity"},
    "camera": {"name": "NullJointController"},
}

# Reset vector layout: 6 base + 4 trunk + 7 left + 7 right + 4 fingers.
TARGET_RESET_LAYOUT: Tuple[Tuple[str, int], ...] = (
    ("base", 6),
    ("trunk", 4),
    ("left_arm", 7),
    ("right_arm", 7),
    ("gripper_fingers", 4),
)
TARGET_RESET_COUNT = sum(n for _, n in TARGET_RESET_LAYOUT)

TARGET_PROPRIO_KEYS: Tuple[str, ...] = (
    "base_qvel",
    "arm_left_qpos",
    "arm_left_qvel",
    "eef_left_pos",
    "eef_left_quat",
    "gripper_left_qpos",
    "gripper_left_qvel",
    "arm_right_qpos",
    "arm_right_qvel",
    "eef_right_pos",
    "eef_right_quat",
    "gripper_right_qpos",
    "gripper_right_qvel",
    "trunk_qpos",
    "trunk_qvel",
)

# Robot config keys accepted by the original Robot constructor / env config.
# Anything else in the source record is not silently carried into the target.
TARGET_ROBOT_KEYS = frozenset({
    "model", "name", "position", "orientation", "controller_config",
    "reset_joint_pos", "obs_modalities", "proprio_obs", "grasping_mode",
    "self_collisions", "action_normalize", "sensor_config",
})

# -- block reasons ------------------------------------------------------------

REASON_FREQ_MISMATCH = "frequency_mismatch"
REASON_SCENE_TYPE_MISMATCH = "scene_type_mismatch"
REASON_SCENE_MODEL_MISSING = "scene_model_missing"
REASON_TASK_TYPE_MISMATCH = "task_type_mismatch"
REASON_TASK_NAME_MISSING = "task_name_missing"
REASON_ROBOT_COUNT = "robot_count_invalid"
REASON_MODEL_MISMATCH = "robot_model_mismatch"
REASON_NAME_MISSING = "robot_name_missing"
REASON_CONTROLLER_UNKNOWN = "controller_unknown"
REASON_CONTROLLER_SIGNATURE = "controller_signature_mismatch"
REASON_RESET_COUNT = "reset_joint_count_mismatch"
REASON_RESET_NONFINITE = "reset_joint_nonfinite"
REASON_UNSUPPORTED_ROBOT_FIELD = "unsupported_robot_field"
REASON_POSE_INVALID = "robot_pose_invalid"
REASON_REQUIRED_UNKNOWN = "required_field_unknown"


def _clean(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def _finite_vec(value: Any, length: int) -> Optional[List[float]]:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (length,):
        return None
    if not np.isfinite(arr).all():
        return None
    return [float(x) for x in arr]


@dataclass
class FieldMapping:
    """One source->target field migration with evidence provenance."""

    source_key: Optional[str]
    target_key: str
    provenance: str
    value: Any = None
    note: str = ""

    def record(self) -> Dict[str, Any]:
        return {
            "source_key": self.source_key or "UNKNOWN",
            "target_key": self.target_key,
            "provenance": self.provenance,
            "value": _clean(self.value) if self.value is not None else "UNKNOWN",
            "note": self.note,
        }


@dataclass
class BootstrapConfig:
    """Fail-closed source-compat bootstrap result."""

    env: Dict[str, Any] = field(default_factory=dict)
    scene: Dict[str, Any] = field(default_factory=dict)
    task: Dict[str, Any] = field(default_factory=dict)
    robot: Dict[str, Any] = field(default_factory=dict)
    mappings: List[Dict[str, Any]] = field(default_factory=list)
    blocked: List[Dict[str, Any]] = field(default_factory=list)
    dropped: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return not self.blocked and bool(self.robot)

    def record(self) -> Dict[str, Any]:
        return {
            "ready": self.ready,
            "env": _clean(self.env),
            "scene": _clean(self.scene),
            "task": _clean(self.task),
            "robot": _clean(self.robot),
            "mappings": [m.record() if isinstance(m, FieldMapping) else m
                          for m in self.mappings],
            "blocked": _clean(self.blocked),
            "dropped": _clean(self.dropped),
        }


class CompatConfigBuilder:
    """Builds the post1 bootstrap config from a sealed source episode config.

    ``source_cfg`` is the plain dict loaded from the hdf5 ``data.attrs.config``;
    ``source_robot`` is its ``robots[0]`` record. Both are read-only inputs.
    """

    def __init__(self, source_cfg: Dict[str, Any], source_robot: Dict[str, Any],
                 *, playback: bool = True):
        self.source_cfg = dict(source_cfg or {})
        self.source_robot = dict(source_robot or {})
        self.playback = playback
        self.result = BootstrapConfig()
        self._mappings: List[FieldMapping] = []

    # -- internal helpers -------------------------------------------------

    def _block(self, reason: str, detail: Any) -> None:
        self.result.blocked.append({"reason": reason, "detail": _clean(detail)})

    def _map(self, source_key: Optional[str], target_key: str,
             provenance: str, value: Any, note: str = "") -> None:
        self._mappings.append(FieldMapping(
            source_key=source_key, target_key=target_key,
            provenance=provenance, value=value, note=note,
        ))

    # -- env / scene / task -----------------------------------------------

    def _build_env(self) -> None:
        src_env = self.source_cfg.get("env") or {}
        env: Dict[str, Any] = {}
        for key, expected in TARGET_FREQ.items():
            value = src_env.get(key)
            if value is None:
                self._block(REASON_REQUIRED_UNKNOWN, {"field": f"env.{key}"})
                continue
            if int(value) != expected:
                self._block(REASON_FREQ_MISMATCH,
                            {"field": key, "source": value, "expected": expected})
            env[key] = int(value)
            self._map(key, f"env.{key}", PROV_SOURCE, value)

        # Sealed create_env overrides (DataPlaybackWrapper playback mode).
        env["automatic_reset"] = False
        env["flatten_obs_space"] = True
        self._map(None, "env.automatic_reset", PROV_SCHEMA, False,
                  "sealed create_env: automatic_reset=False")
        self._map(None, "env.flatten_obs_space", PROV_SCHEMA, True,
                  "sealed create_env: flatten_obs_space=True")
        self.result.dropped["env"] = sorted(set(src_env) - set(env))
        self.result.env = env

    def _build_scene(self) -> None:
        src_scene = self.source_cfg.get("scene") or {}
        scene: Dict[str, Any] = {}

        scene_type = src_scene.get("type")
        if scene_type != TARGET_SCENE_TYPE:
            self._block(REASON_SCENE_TYPE_MISMATCH,
                        {"source": scene_type, "expected": TARGET_SCENE_TYPE})
        scene["type"] = scene_type or "UNKNOWN"
        self._map("type", "scene.type", PROV_SOURCE, scene_type)

        scene_model = src_scene.get("scene_model")
        if not scene_model:
            self._block(REASON_SCENE_MODEL_MISSING, "scene.scene_model")
        scene["scene_model"] = scene_model or "UNKNOWN"
        self._map("scene_model", "scene.scene_model", PROV_SOURCE, scene_model)

        scene["load_room_types"] = list(src_scene.get("load_room_types") or [])
        self._map("load_room_types", "scene.load_room_types", PROV_SOURCE,
                  scene["load_room_types"])
        scene["scene_instance"] = src_scene.get("scene_instance") or "UNKNOWN"
        self._map("scene_instance", "scene.scene_instance", PROV_SOURCE,
                  scene["scene_instance"])
        scene["include_robots"] = False
        self._map(None, "scene.include_robots", PROV_SCHEMA, False)
        # scene_file is bound to the playback wrapper from the scene_file attr
        # at wrapper construction; it is not invented here.
        self.result.dropped["scene"] = sorted(set(src_scene) - set(scene))
        self.result.scene = scene

    def _build_task(self) -> None:
        src_task = self.source_cfg.get("task") or {}
        task: Dict[str, Any] = {}

        task_type = src_task.get("type")
        if task_type != TARGET_TASK_TYPE:
            self._block(REASON_TASK_TYPE_MISMATCH,
                        {"source": task_type, "expected": TARGET_TASK_TYPE})
        task["type"] = task_type or "UNKNOWN"
        self._map("type", "task.type", PROV_SOURCE, task_type)

        activity = src_task.get("activity_name")
        if not activity:
            self._block(REASON_TASK_NAME_MISSING, "task.activity_name")
        task["activity_name"] = activity or "UNKNOWN"
        self._map("activity_name", "task.activity_name", PROV_SOURCE, activity)

        task["online_object_sampling"] = bool(
            src_task.get("online_object_sampling", False))
        self._map("online_object_sampling", "task.online_object_sampling",
                  PROV_SOURCE, task["online_object_sampling"])

        if self.playback:
            task["include_obs"] = True
            self._map(None, "task.include_obs", PROV_SCHEMA, True,
                      "sealed create_env: include_obs=True for playback")
        else:
            task["include_obs"] = bool(src_task.get("include_obs", False))

        max_steps = src_task.get("termination_config", {}).get("max_steps")
        task["termination_config"] = {"max_steps": int(max_steps) if max_steps else "UNKNOWN"}
        self._map("termination_config.max_steps", "task.termination_config.max_steps",
                  PROV_SOURCE, max_steps)
        self.result.dropped["task"] = sorted(set(src_task) - set(task))
        self.result.task = task

    # -- robot ------------------------------------------------------------

    def _build_robot(self) -> None:
        robots = self.source_cfg.get("robots") or []
        if len(robots) != 1:
            self._block(REASON_ROBOT_COUNT, {"n_robots": len(robots)})
            return

        src = self.source_robot
        robot: Dict[str, Any] = {}

        # Canonical migration: legacy `type` -> lowercase `model`, drop `type`.
        legacy_type = src.get("type")
        model = src.get("model")
        if model is None and legacy_type is not None:
            model = str(legacy_type).lower()
            self._map("type", "robot.model", PROV_SCHEMA, model,
                      "legacy type migrated to lowercase model; type dropped")
        else:
            self._map("model", "robot.model", PROV_SOURCE, model)
        if model != TARGET_ROBOT_MODEL:
            self._block(REASON_MODEL_MISMATCH,
                        {"source_model": model, "expected": TARGET_ROBOT_MODEL})
        robot["model"] = model

        name = src.get("name")
        if not name:
            self._block(REASON_NAME_MISSING, "robot.name")
        robot["name"] = name or "UNKNOWN"
        self._map("name", "robot.name", PROV_SOURCE, name)
        if name:
            robot["_source_uuid"] = compute_uuid(str(name))

        position = _finite_vec(src.get("position"), 3)
        orientation = _finite_vec(src.get("orientation"), 4)
        if position is None or orientation is None:
            self._block(REASON_POSE_INVALID,
                        {"position": src.get("position"),
                         "orientation": src.get("orientation")})
        else:
            robot["position"] = position
            robot["orientation"] = orientation
            self._map("position", "robot.position", PROV_SOURCE, position)
            self._map("orientation", "robot.orientation", PROV_SOURCE, orientation)

        self._build_controllers(src, robot)
        self._build_reset(src, robot)

        robot["obs_modalities"] = ["proprio", "rgb"]
        self._map(None, "robot.obs_modalities", PROV_SCHEMA,
                  robot["obs_modalities"], "sealed create_env modalities")
        robot["proprio_obs"] = list(TARGET_PROPRIO_KEYS)
        self._map(None, "robot.proprio_obs", PROV_SCHEMA,
                  robot["proprio_obs"], "post1 eval r1pro.yaml proprio keys")

        for key in ("grasping_mode", "self_collisions", "action_normalize",
                    "sensor_config"):
            if key in src:
                robot[key] = _clean(src[key])
                self._map(key, f"robot.{key}", PROV_SOURCE, src[key])

        # Reject any source robot field not in the SDK-supported target set.
        allowed = TARGET_ROBOT_KEYS | {"type", "_source_uuid"}
        unsupported = sorted(set(src.keys()) - allowed)
        if unsupported:
            self._block(REASON_UNSUPPORTED_ROBOT_FIELD, {"fields": unsupported})

        # Strip internal marker before the consumer-facing record.
        self.result.robot = {k: v for k, v in robot.items() if k != "_source_uuid"}
        self.result.robot["_source_uuid"] = robot.get("_source_uuid", "UNKNOWN")

    def _build_controllers(self, src: Dict[str, Any], robot: Dict[str, Any]) -> None:
        cc = src.get("controller_config") or {}
        order = list(cc.keys())
        unknown = sorted(set(order) - set(TARGET_CONTROLLERS))
        missing = sorted(set(TARGET_CONTROLLER_ORDER) - set(order))
        if unknown or missing:
            self._block(REASON_CONTROLLER_UNKNOWN,
                        {"unknown": unknown, "missing": missing})

        # The emitted config follows the target schema order; source insertion
        # order is recorded as evidence but does not have to match it.
        self._map("controller_config", "robot.controller_config.source_order",
                  PROV_SOURCE, order,
                  "target emitted order follows post1 eval/r1pro.yaml")

        mapped: Dict[str, Any] = {}
        for ctl in TARGET_CONTROLLER_ORDER:
            if ctl not in cc:
                continue
            spec = dict(cc[ctl])
            expected = TARGET_CONTROLLERS[ctl]
            for field_name, exp_value in expected.items():
                if spec.get(field_name) != exp_value:
                    self._block(REASON_CONTROLLER_SIGNATURE,
                                {"controller": ctl, "field": field_name,
                                 "source": spec.get(field_name),
                                 "expected": exp_value})
            mapped[ctl] = _clean(spec)
            self._map(f"controller_config.{ctl}", f"robot.controller_config.{ctl}",
                      PROV_SOURCE, spec)
        robot["controller_config"] = mapped

    def _build_reset(self, src: Dict[str, Any], robot: Dict[str, Any]) -> None:
        reset = src.get("reset_joint_pos")
        if reset is None:
            self._block(REASON_RESET_COUNT, {"source": None})
            return
        arr = np.asarray(reset, dtype=np.float64)
        if arr.size != TARGET_RESET_COUNT:
            self._block(REASON_RESET_COUNT,
                        {"source_count": int(arr.size),
                         "expected": TARGET_RESET_COUNT})
        elif not np.isfinite(arr).all():
            self._block(REASON_RESET_NONFINITE, {"detail": "non-finite entries"})
        robot["reset_joint_pos"] = [float(x) for x in arr.tolist()]
        self._map("reset_joint_pos", "robot.reset_joint_pos", PROV_SOURCE,
                  robot["reset_joint_pos"],
                  f"layout {dict(TARGET_RESET_LAYOUT)}")

    # -- public entry -----------------------------------------------------

    def build(self) -> BootstrapConfig:
        self._build_env()
        self._build_scene()
        self._build_task()
        self._build_robot()
        self.result.mappings = list(self._mappings)
        return self.result


def validate_effective(built: BootstrapConfig,
                       effective: Dict[str, Any]) -> Dict[str, Any]:
    """Read-back check against an actual module-produced effective config.

    Compares only the schema-selected surface. ``effective`` is duck-typed: a
    dict with ``env`` / ``scene`` / ``task`` / ``robots`` keys, as produced by
    the managed entry. Missing or mismatched fields are reported; nothing is
    mutated.
    """
    matches: List[str] = []
    mismatches: List[Dict[str, Any]] = []

    eff_env = effective.get("env") or {}
    for key, expected in TARGET_FREQ.items():
        if int(eff_env.get(key, -1)) != expected:
            mismatches.append({"surface": f"env.{key}", "expected": expected,
                               "actual": eff_env.get(key)})
        else:
            matches.append(f"env.{key}")

    eff_scene = effective.get("scene") or {}
    if eff_scene.get("type") != TARGET_SCENE_TYPE:
        mismatches.append({"surface": "scene.type", "expected": TARGET_SCENE_TYPE,
                           "actual": eff_scene.get("type")})
    else:
        matches.append("scene.type")
    if eff_scene.get("scene_model") != built.scene.get("scene_model"):
        mismatches.append({"surface": "scene.scene_model",
                           "expected": built.scene.get("scene_model"),
                           "actual": eff_scene.get("scene_model")})
    else:
        matches.append("scene.scene_model")

    eff_task = effective.get("task") or {}
    if eff_task.get("activity_name") != built.task.get("activity_name"):
        mismatches.append({"surface": "task.activity_name",
                           "expected": built.task.get("activity_name"),
                           "actual": eff_task.get("activity_name")})
    else:
        matches.append("task.activity_name")

    eff_robots = effective.get("robots") or []
    if len(eff_robots) != 1:
        mismatches.append({"surface": "robots", "expected": 1,
                           "actual": len(eff_robots)})
    else:
        eff_robot = eff_robots[0]
        if "type" in eff_robot:
            mismatches.append({"surface": "robot.type", "expected": "absent",
                               "actual": "legacy type present"})
        if eff_robot.get("model") != built.robot.get("model"):
            mismatches.append({"surface": "robot.model",
                               "expected": built.robot.get("model"),
                               "actual": eff_robot.get("model")})
        else:
            matches.append("robot.model")
        if eff_robot.get("name") != built.robot.get("name"):
            mismatches.append({"surface": "robot.name",
                               "expected": built.robot.get("name"),
                               "actual": eff_robot.get("name")})
        else:
            matches.append("robot.name")
        eff_cc = list((eff_robot.get("controller_config") or {}).keys())
        if eff_cc != list(TARGET_CONTROLLER_ORDER):
            mismatches.append({"surface": "robot.controller_config.order",
                               "expected": list(TARGET_CONTROLLER_ORDER),
                               "actual": eff_cc})
        else:
            matches.append("robot.controller_config.order")
        eff_reset = eff_robot.get("reset_joint_pos")
        if eff_reset is None or len(eff_reset) != TARGET_RESET_COUNT:
            mismatches.append({"surface": "robot.reset_joint_pos",
                               "expected": TARGET_RESET_COUNT,
                               "actual": None if eff_reset is None else len(eff_reset)})
        else:
            matches.append("robot.reset_joint_pos")

    return {"matches": matches, "mismatches": mismatches,
            "ready": not mismatches and built.ready}
