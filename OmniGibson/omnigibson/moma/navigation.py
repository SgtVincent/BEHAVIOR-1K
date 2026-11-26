"""PnP navigation waypoint feedback; pure arithmetic, no simulator or state writes.

Ported from the a28 frozen baseline ``navigation_feedback.py`` (reuse_archive,
numpy-only). A source waypoint may take several real ticks. Velocity is
recomputed toward the same phase-transformed base pose on every tick, not
replayed and not forced zero.

Gap G2: controller input/output/velocity limits are bound to the 2026 R1Pro
controller definition at the executor seam; the values passed in here are not
hardcoded from the 2025 baseline.
"""
from __future__ import annotations

import numpy as np


def navigation_feedback_enabled(source_index: int, navigation_start: int) -> bool:
    if isinstance(source_index, bool) or isinstance(navigation_start, bool):
        raise ValueError("invalid_navigation_boundary_index")
    i = int(source_index)
    start = int(navigation_start)
    if i < 0 or start < 0:
        raise ValueError("invalid_navigation_boundary_index")
    return i >= start


def _pose(position, quaternion) -> np.ndarray:
    p = np.asarray(position, dtype=float)
    q = np.asarray(quaternion, dtype=float)
    if p.shape != (3,) or q.shape != (4,) or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("invalid_live_base_pose")
    n = np.linalg.norm(q)
    if n <= 1e-12:
        raise ValueError("zero_live_base_quaternion")
    x, y, z, w = q / n
    m = np.eye(4)
    m[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    m[:3, 3] = p
    return m


def navigation_command(control, target_world, input_limits, output_limits,
                       controller_velocity_limits, *, dt: float = 1 / 30,
                       linear_limit: float = 0.12, angular_limit: float = 0.50,
                       points_world=()):
    """Return native base3 and its predicted SDK/world velocity (not measured).

    Reuses the existing SDK input scale and canonical XY/local Z semantics.
    The base is additionally slowed to the already-existing TCP speed bounds;
    the coupled arm QP must constrain TOTAL TCP velocity, including this offset.
    """
    live = _pose(control["root_pos"], control["root_quat"])
    canonical = _pose(control["canonical_pos"], control["canonical_quat"])
    target = np.asarray(target_world, dtype=float)
    if target.shape != (4, 4) or not np.isfinite(target).all():
        raise ValueError("invalid_navigation_target")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("invalid_navigation_dt")
    cur = np.linalg.solve(canonical, live)
    tgt = np.linalg.solve(canonical, target)
    velocity_xy = (tgt[:2, 3] - cur[:2, 3]) / dt
    local_xy = np.linalg.solve(cur[:2, :2], velocity_xy)
    rot = cur[:3, :3].T @ tgt[:3, :3]
    y = rot[1, 0] - rot[0, 1]
    x = rot[0, 0] + rot[1, 1]
    if abs(x) + abs(y) <= 1e-12:
        raise ValueError("ambiguous_base_yaw_projection")
    desired_local = np.r_[local_xy, np.arctan2(y, x) / dt]
    il, ih = np.asarray(input_limits, float)
    ol, oh = np.asarray(output_limits, float)
    vl, vh = np.asarray(controller_velocity_limits, float)
    if any(a.shape != (3,) or not np.isfinite(a).all() for a in (il, ih, ol, oh, vl, vh)):
        raise ValueError("invalid_base_controller_limits")
    if np.any(ih <= il) or np.any(oh <= ol) or np.any(vh < vl):
        raise ValueError("invalid_base_controller_range")
    raw = (desired_local - ol) / (oh - ol) * (ih - il) + il
    clipped_raw = np.clip(raw, il, ih)
    local = (clipped_raw - il) / (ih - il) * (oh - ol) + ol
    canonical_velocity = np.clip(np.r_[cur[:2, :2] @ local[:2], local[2]], vl, vh)
    speed = np.linalg.norm(canonical[:3, :3] @ np.r_[canonical_velocity[:2], 0.0])
    angular_speed = abs(canonical_velocity[2])
    base_linear = canonical[:3, :3] @ np.r_[canonical_velocity[:2], 0.0]
    base_angular = live[:3, 2] * canonical_velocity[2]
    for point in points_world:
        point = np.asarray(point, float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("invalid_base_tcp_point")
        speed = max(speed, float(np.linalg.norm(base_linear + np.cross(base_angular, point - live[:3, 3]))))
    scale = min(
        1.0,
        linear_limit * (1 - 1e-6) / speed if speed > 0 else 1.0,
        angular_limit * (1 - 1e-6) / angular_speed if angular_speed > 0 else 1.0,
    )
    canonical_velocity *= scale
    slowed_local = np.r_[np.linalg.solve(cur[:2, :2], canonical_velocity[:2]), canonical_velocity[2]]
    command = ((slowed_local - ol) / (oh - ol) * (ih - il) + il).astype(np.float32)
    actual_raw = np.clip(command.astype(float), il, ih)
    actual_local = (actual_raw - il) / (ih - il) * (oh - ol) + ol
    predicted = np.clip(np.r_[cur[:2, :2] @ actual_local[:2], actual_local[2]], vl, vh)
    world_linear = canonical[:3, :3] @ np.r_[predicted[:2], 0.0]
    world_angular = live[:3, 2] * predicted[2]
    return command, {
        "method": "phase_base_pose_feedback_total_tcp_compensation",
        "live_base_pose_world": live,
        "target_base_pose_world": target,
        "predicted_sdk_controller_goal": predicted,
        "predicted_base_linear_world_mps": world_linear,
        "predicted_base_angular_world_radps": world_angular,
        "source_waypoint_position_error_m": float(np.linalg.norm(target[:3, 3] - live[:3, 3])),
        "unclipped_raw_base": raw,
        "base_slowdown_scale": scale,
        "pre_coupled_base_command_native": command.copy(),
        "pre_coupled_predicted_sdk_controller_goal": predicted.copy(),
        "pre_coupled_base_linear_world_mps": world_linear.copy(),
        "pre_coupled_base_angular_world_radps": world_angular.copy(),
        "command_input_limits": np.stack([il, ih]),
        "command_output_limits": np.stack([ol, oh]),
        "current_canonical_xy_rotation": cur[:2, :2].copy(),
        "canonical_rotation_world": canonical[:3, :3].copy(),
        "live_base_z_axis_world": live[:3, 2].copy(),
        "physical_validation": False,
    }


def apply_coupled_base_scale(record, scale):
    """Map the QP-selected fraction back through the exact SDK native scale.

    The fraction is constrained to [0,1], so this can only slow the previously
    bounded navigation command. The returned predicted goal/twist include the
    final float32 native command rounding used by env.step.
    """
    s = float(scale)
    if not np.isfinite(s) or not -1e-9 <= s <= 1 + 1e-9:
        raise ValueError("invalid_coupled_base_scale")
    s = float(np.clip(s, 0.0, 1.0))
    required = [
        "pre_coupled_base_command_native", "pre_coupled_predicted_sdk_controller_goal",
        "command_input_limits", "command_output_limits", "current_canonical_xy_rotation",
        "canonical_rotation_world", "live_base_z_axis_world",
    ]
    if any(k not in record for k in required):
        raise ValueError("incomplete_navigation_scale_record")
    il, ih = np.asarray(record["command_input_limits"], float)
    ol, oh = np.asarray(record["command_output_limits"], float)
    cur = np.asarray(record["current_canonical_xy_rotation"], float)
    desired = s * np.asarray(record["pre_coupled_predicted_sdk_controller_goal"], float)
    local = np.r_[np.linalg.solve(cur, desired[:2]), desired[2]]
    command = (((local - ol) / (oh - ol)) * (ih - il) + il).astype(np.float32)
    actual_raw = np.clip(command.astype(float), il, ih)
    actual_local = (actual_raw - il) / (ih - il) * (oh - ol) + ol
    predicted = np.r_[cur @ actual_local[:2], actual_local[2]]
    canonical_R = np.asarray(record["canonical_rotation_world"], float)
    live_z = np.asarray(record["live_base_z_axis_world"], float)
    world_linear = canonical_R @ np.r_[predicted[:2], 0.0]
    world_angular = live_z * predicted[2]
    record.update(
        coupled_base_scale=s,
        issued_base_command_native=command,
        predicted_sdk_controller_goal=predicted,
        predicted_base_linear_world_mps=world_linear,
        predicted_base_angular_world_radps=world_angular,
        coupled_scale_float32_goal_error_max=float(
            np.max(np.abs(predicted - s * np.asarray(record["pre_coupled_predicted_sdk_controller_goal"], float)))
        ),
    )
    return command, record


def base_point_twist(record, point_world):
    """Known commanded base contribution at a world TCP. No dynamics guarantee."""
    linear = np.asarray(record["predicted_base_linear_world_mps"], float)
    angular = np.asarray(record["predicted_base_angular_world_radps"], float)
    origin = np.asarray(record["live_base_pose_world"], float)[:3, 3]
    point = np.asarray(point_world, float)
    return np.r_[linear + np.cross(angular, point - origin), angular]
