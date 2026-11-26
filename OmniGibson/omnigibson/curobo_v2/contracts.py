# Copyright (c) 2026 B1K2026 integration line
# SPDX-License-Identifier: Apache-2.0
"""Frozen 23-d native action slices and R1Pro planner profile for the cuRobo v2 lane.

Sim-free: standard library only. This module must not import torch, curobo,
omnigibson, warp or any simulator dependency. Boundary rules:

- ``native23_joint`` order: base3 velocity | trunk4 position | armL7 | gripL1 |
  armR7 | gripR1. The three 23-d schemas share width but not semantics; a bare
  23-vector without its label is rejected.
- The planner owns trunk4 + armL7 + armR7 (18 active joints). Base holonomic
  velocity and continuous gripper width are supplied by the navigation / grasp
  seams outside the planner.
- Tool frames, gripper mapping and control dt are sealed here. cuRobo v2 types
  never cross this boundary: only plain dicts/lists/floats leave the adapter.

The R1Pro joint/link inventory below was read from ``r1pro_source.urdf``
(sha256 ``e923509408b69e8ca4f54633902d270bd0353b48a61beedf50d5e1b59bceee46``)
on 2026-11-26. Franka example fixtures are never substituted for R1Pro frames.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

ACTION_DIM = 23

NATIVE23_JOINT = "native23_joint"
EXECUTION23_SE3_DELTA = "execution23_se3_delta"
BRIDGE_DECODED23_IK_POSE = "bridge_decoded23_ik_pose"
ALLOWED_LABELS = frozenset(
    {NATIVE23_JOINT, EXECUTION23_SE3_DELTA, BRIDGE_DECODED23_IK_POSE}
)

#: native23 group order (verified against moma.schema / brvla_bridge.contracts).
NATIVE23_GROUPS: Tuple[Tuple[str, int], ...] = (
    ("base", 3),
    ("trunk", 4),
    ("arm_left", 7),
    ("gripper_left", 1),
    ("arm_right", 7),
    ("gripper_right", 1),
)


def _group_slices() -> Dict[str, slice]:
    out: Dict[str, slice] = {}
    start = 0
    for name, width in NATIVE23_GROUPS:
        out[name] = slice(start, start + width)
        start += width
    return out


NATIVE23_SLICES: Dict[str, slice] = _group_slices()

#: R1Pro active planning joints, URDF order trunk4 -> armL7 -> armR7.
R1PRO_PLANNER_JOINTS: Tuple[str, ...] = (
    "torso_joint1",
    "torso_joint2",
    "torso_joint3",
    "torso_joint4",
    "left_arm_joint1",
    "left_arm_joint2",
    "left_arm_joint3",
    "left_arm_joint4",
    "left_arm_joint5",
    "left_arm_joint6",
    "left_arm_joint7",
    "right_arm_joint1",
    "right_arm_joint2",
    "right_arm_joint3",
    "right_arm_joint4",
    "right_arm_joint5",
    "right_arm_joint6",
    "right_arm_joint7",
)

#: Joints locked fixed at planning time (holonomic wheels/steer, finger prisms).
R1PRO_LOCKED_JOINTS: Tuple[str, ...] = (
    "steer_motor_joint1",
    "wheel_motor_joint1",
    "steer_motor_joint2",
    "wheel_motor_joint2",
    "steer_motor_joint3",
    "wheel_motor_joint3",
    "left_gripper_finger_joint1",
    "left_gripper_finger_joint2",
    "right_gripper_finger_joint1",
    "right_gripper_finger_joint2",
)

#: Planner tool frames mapped onto URDF gripper links (eef_offset applied by adapter).
R1PRO_TOOL_FRAMES: Dict[str, str] = {
    "left": "left_gripper_link",
    "right": "right_gripper_link",
}

#: Sealed planner interpolated dt (s), recorded from the v2 franka gate receipt
#: 68c. Tracking thresholds (P95 <= 0.02 m, max <= 0.1 m) are not relaxed here.
PLANNER_DT_S = 0.025


class ContractViolation(ValueError):
    """A target/action payload violates the frozen boundary contract."""


def _coerce_finite(values: Sequence[float]) -> Tuple[float, ...]:
    try:
        coerced = tuple(float(v) for v in values)
    except (TypeError, ValueError) as exc:
        raise ContractViolation("action values must be numeric") from exc
    if len(coerced) != ACTION_DIM:
        raise ContractViolation(
            f"action23_invalid_shape: expected ({ACTION_DIM},), got {len(coerced)}"
        )
    if any(not math.isfinite(v) for v in coerced):
        raise ContractViolation("action23_nonfinite")
    return coerced


@dataclass(frozen=True)
class TaggedAction:
    """A 23-d command bound to its schema label (duck-compatible with moma TaggedAction)."""

    label: str
    vector: Tuple[float, ...]
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.label not in ALLOWED_LABELS:
            raise ContractViolation(f"unknown_action_schema: {self.label!r}")
        object.__setattr__(self, "vector", _coerce_finite(self.vector))

    def assert_compatible(self, other_label: str) -> None:
        if other_label not in ALLOWED_LABELS:
            raise ContractViolation(f"unknown_action_schema: {other_label!r}")
        if other_label != self.label:
            raise ContractViolation(
                f"schema_mismatch: {self.label} cannot be bridged to {other_label}"
            )

    def to_list(self) -> List[float]:
        return [float(x) for x in self.vector]


def native23_action(vector: Sequence[float], meta: Optional[Dict[str, Any]] = None) -> TaggedAction:
    return TaggedAction(NATIVE23_JOINT, list(vector), dict(meta or {}))


def assemble_native23(
    *,
    base_vel: Sequence[float],
    trunk: Sequence[float],
    arm_left: Sequence[float],
    gripper_left: float,
    arm_right: Sequence[float],
    gripper_right: float,
) -> Tuple[float, ...]:
    """Assemble the sealed native23 vector from planner + seam slices.

    Raises :class:`ContractViolation` on wrong width or non-finite values; the
    planner never emits a zero-filled placeholder action.
    """
    groups: Dict[str, Sequence[float]] = {
        "base": base_vel,
        "trunk": trunk,
        "arm_left": arm_left,
        "gripper_left": [gripper_left],
        "arm_right": arm_right,
        "gripper_right": [gripper_right],
    }
    out: List[float] = [0.0] * ACTION_DIM
    for name, width in NATIVE23_GROUPS:
        vals = list(groups[name])
        if len(vals) != width:
            raise ContractViolation(f"slice {name} width {len(vals)} != {width}")
        if any(not math.isfinite(float(v)) for v in vals):
            raise ContractViolation(f"slice {name} non-finite")
        out[NATIVE23_SLICES[name]] = [float(v) for v in vals]
    return tuple(out)


def validate_trajectory_shape(
    shape: Sequence[int], joint_count: int
) -> Dict[str, Any]:
    """Validate an interpolated planner trajectory tensor shape, fail closed on
    batch / horizon > 1.

    Accepted layouts (joint and time order preserved; no cross-batch or
    cross-horizon concatenation):

    - 2D ``[T, D]``
    - 3D ``[B, T, D]`` with ``B == 1``
    - 4D ``[B, H, T, D]`` with ``B == 1`` and ``H == 1``

    Returns a dict with ``batch``, ``horizon``, ``waypoints``, ``joints`` and a
    ``selector`` tuple used to index the tensor down to a ``[T, D]`` matrix.
    Raises :class:`ContractViolation` on any unsupported, multi-batch,
    multi-horizon, or joint-dim mismatched plan.
    """
    nd = len(shape)
    if nd == 2:
        batch, horizon, rows, dim = 1, 1, shape[0], shape[1]
        selector: Tuple[Any, ...] = ()
    elif nd == 3:
        batch, rows, dim = shape
        horizon = 1
        selector = (0,)
    elif nd == 4:
        batch, horizon, rows, dim = shape
        selector = (0, 0)
    else:
        raise ContractViolation(f"trajectory_shape_unsupported_ndim:{nd}")
    if batch != 1:
        raise ContractViolation("trajectory_batch_gt1")
    if horizon != 1:
        raise ContractViolation("trajectory_horizon_gt1")
    if dim != joint_count:
        raise ContractViolation(
            f"trajectory_joint_dim_mismatch:{dim}!={joint_count}"
        )
    if rows < 1:
        raise ContractViolation("trajectory_no_waypoints")
    return {
        "batch": batch,
        "horizon": horizon,
        "waypoints": rows,
        "joints": dim,
        "selector": selector,
    }


def _pose8x4_to_pos_quat(rows: Any) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    """Convert one 4x4 homogeneous pose (provider target row) to pos3 + quat4 (x,y,z,w)."""
    try:
        mat = [[float(v) for v in row] for row in rows]
    except (TypeError, ValueError) as exc:
        raise ContractViolation("invalid_pose_shape_or_finite") from exc
    if len(mat) != 4 or any(len(r) != 4 for r in mat):
        raise ContractViolation("invalid_pose_shape_or_finite")
    if not all(math.isfinite(v) for row in mat for v in row):
        raise ContractViolation("invalid_pose_shape_or_finite")
    if any(abs(mat[3][c] - expected) > 1e-7 for c, expected in enumerate([0, 0, 0, 1])):
        raise ContractViolation("invalid_homogeneous_row")
    pos = (mat[0][3], mat[1][3], mat[2][3])
    r = [mat[i][:3] for i in range(3)]
    r00, r01, r02 = r[0]
    r10, r11, r12 = r[1]
    r20, r21, r22 = r[2]
    tr = float(r00 + r11 + r22)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (r21 - r12) / s
        y = (r02 - r20) / s
        z = (r10 - r01) / s
    elif r00 > r11 and r00 > r22:
        s = math.sqrt(1.0 + r00 - r11 - r22) * 2
        w = (r21 - r12) / s
        x = 0.25 * s
        y = (r10 + r01) / s
        z = (r02 + r20) / s
    elif r11 > r22:
        s = math.sqrt(1.0 + r11 - r00 - r22) * 2
        w = (r02 - r20) / s
        x = (r10 + r01) / s
        y = 0.25 * s
        z = (r21 + r12) / s
    else:
        s = math.sqrt(1.0 + r22 - r00 - r11) * 2
        w = (r21 - r12) / s
        x = (r02 + r20) / s
        y = (r21 + r12) / s
        z = 0.25 * s
    quat = (x, y, z, w)
    if not all(math.isfinite(v) for v in pos + quat):
        raise ContractViolation("pose_quat_nonfinite")
    return pos, quat  # type: ignore[return-value]


@dataclass(frozen=True)
class PlannerGoal:
    """Normalized object-centric goal handed to the planner backend.

    ``source`` records whether it came from the MoMaGen-style object provider
    (live-aligned SE3 rows) or from an S2 primitive args dict (object id only;
    the live pose is resolved by the observer seam, not fabricated here).
    """

    source: str
    arm: str
    object_id: Optional[str] = None
    primitive: Optional[str] = None
    left_pose: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]] = None
    right_pose: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]] = None
    grippers: Optional[Tuple[float, float]] = None
    base_vel: Optional[Tuple[float, float, float]] = None
    live_object_pose: Optional[Any] = None


def parse_target(target: Dict[str, Any]) -> PlannerGoal:
    """Normalize a caller target dict. Fail closed on unknown/malformed shapes."""
    if not isinstance(target, dict):
        raise ContractViolation("target_must_be_dict")

    if "target_pose8x4_world" in target:
        rows = target["target_pose8x4_world"]
        try:
            left = _pose8x4_to_pos_quat(rows[0])
        except (IndexError, TypeError, ValueError) as exc:
            raise ContractViolation(f"provider_target_invalid: {exc}") from exc
        right = None
        if len(rows) > 1 and rows[1] is not None:
            try:
                right = _pose8x4_to_pos_quat(rows[1])
            except (TypeError, ValueError) as exc:
                raise ContractViolation(f"provider_target_invalid: {exc}") from exc
        arm = "both" if right is not None else "left"
        grips = target.get("grippers")
        grippers = None
        if grips is not None:
            g0, g1 = float(grips[0]), float(grips[1])
            if not (math.isfinite(g0) and math.isfinite(g1)):
                raise ContractViolation("gripper_nonfinite")
            grippers = (g0, g1)
        base_raw = target.get("base_vel")
        base_vel = None
        if base_raw is not None:
            if len(base_raw) != 3:
                raise ContractViolation("base_vel_width_not_3")
            b0, b1, b2 = (float(v) for v in base_raw)
            if not all(math.isfinite(v) for v in (b0, b1, b2)):
                raise ContractViolation("base_vel_nonfinite")
            base_vel = (b0, b1, b2)
        return PlannerGoal(
            source="object_provider",
            arm=arm,
            object_id=target.get("target_object") or target.get("object_id"),
            grippers=grippers,
            base_vel=base_vel,
            left_pose=left,
            right_pose=right,
            live_object_pose=target.get("live_object_pose_world"),
        )

    if "primitive" in target:
        return PlannerGoal(
            source="s2_args",
            arm="right",
            object_id=target.get("target_object"),
            primitive=target["primitive"],
        )

    raise ContractViolation("target_shape_unrecognized")
