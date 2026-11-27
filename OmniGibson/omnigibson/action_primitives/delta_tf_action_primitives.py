"""Delta-TF action primitives on the upstream cuRobo / controller stack.

No IK, trajectory optimizer, collision exporter or physics loop is implemented
here. The decoder accepts the BRVLA execution23 layout, NOT native joint23.
Saved pairwise deltas and model chunk-cumulative deltas are distinct inputs.
"""
from enum import IntEnum

import gymnasium as gym
import torch as th

import omnigibson.utils.transform_utils as T
from omnigibson.action_primitives.action_primitive_set_base import (
    ActionPrimitiveError,
    BaseActionPrimitiveSet,
)
from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection, CuRoboMotionGenerator
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
from omnigibson.controllers import JointController
from omnigibson.controllers.controller_view import ControllerView


class DeltaTFAction(IntEnum):
    MOVE = 0


class DeltaTFDecoder:
    """Fixed-anchor execution23 -> absolute link poses, body displacement, widths.

    Layout: left pose6, right pose6, two gripper widths, base dx/dy/dyaw,
    chest pose6. Pose6 rotation is extrinsic xyz (Rz @ Ry @ Rx), metres/radians.
    The decoder owns no simulator or controller state. Failed consumers must not
    reuse a partially consumed decoder; reset/rebind an entire chunk explicitly.
    """

    def __init__(self, anchors, representation):
        if representation not in ("pairwise_delta", "chunk_cumulative"):
            raise ValueError("Specify pairwise_delta or chunk_cumulative explicitly")
        self.representation = representation
        self.anchors = {k: self._matrix(v) for k, v in anchors.items()}
        if set(self.anchors) != {"left", "right", "chest"}:
            raise ValueError("Exactly left, right and chest anchors are required")
        self.relative = {k: th.eye(4, dtype=th.float32) for k in self.anchors}
        self.previous_move = th.zeros(3, dtype=th.float32)

    @staticmethod
    def _matrix(pose):
        mat = th.as_tensor(pose, dtype=th.float32).detach().cpu().clone()
        if mat.shape != (4, 4) or not th.isfinite(mat).all():
            raise ValueError("Anchor must be a finite 4x4 SE3 matrix")
        r = mat[:3, :3]
        if not th.allclose(mat[3], th.tensor([0., 0., 0., 1.]), atol=1e-6):
            raise ValueError("Invalid homogeneous row")
        if not th.allclose(r.T @ r, th.eye(3), atol=1e-5) or not th.isclose(th.linalg.det(r), th.tensor(1.), atol=1e-5):
            raise ValueError("Invalid rotation matrix")
        return mat

    def decode(self, row):
        row = th.as_tensor(row, dtype=th.float32).detach().cpu()
        if row.shape != (23,) or not th.isfinite(row).all():
            raise ValueError("A finite execution23 row is required, not native joint actions")
        if th.any(row[12:14] < 0):
            raise ValueError("Gripper widths must be nonnegative metres")
        poses = {}
        for name, sl in (("left", slice(0, 6)), ("right", slice(6, 12)), ("chest", slice(17, 23))):
            p = row[sl]
            delta = T.pose2mat((p[:3], T.euler2quat(p[3:])))
            relative = self.relative[name] @ delta if self.representation == "pairwise_delta" else delta
            self.relative[name] = relative
            poses[name] = self.anchors[name] @ relative
        current = row[14:17].clone()
        if self.representation == "pairwise_delta":
            displacement = current
        else:
            diff = current - self.previous_move
            previous_rotation = T.euler2mat(th.tensor([0., 0., self.previous_move[2]]))
            translation = previous_rotation[:2, :2].T @ diff[:2]
            displacement = th.cat((translation, diff[2:3]))
            self.previous_move = current
        return {"poses": poses, "base_displacement": displacement, "gripper_widths": row[12:14].clone()}


class DeltaTFActionPrimitives(StarterSemanticActionPrimitives):
    """Parallel primitive set; reuse upstream planner and feedback execution.

    This is a waypoint-tracking adapter, not a time-faithful replay certificate.
    Timing / tracking of converted demonstrations remains a separate GPU gate.
    All requested tool links must exist in the upstream planner profile. Do not
    substitute source joints or silently drop unsupported chest/arm constraints.
    """

    def __init__(self, env, robot, tool_links, representation="pairwise_delta", batch_size=2):
        # Do not call Starter.__init__: it changes the global grasp window.
        BaseActionPrimitiveSet.__init__(self, env, robot)
        if robot.model != "r1pro" or set(tool_links) != {"left", "right", "chest"}:
            raise ValueError("R1Pro and explicit left/right/chest tool links are required")
        for name, (key, _) in robot.controllers.items():
            if not ControllerView.is_controller_type(key, JointController) or ControllerView.get_use_delta_commands(key):
                raise ValueError(f"{name} is incompatible with upstream q_to_action; use an explicit joint profile")
        self.tool_links = dict(tool_links)
        if len(set(self.tool_links.values())) != 3:
            raise ValueError("Tool links must be distinct")
        if any(link not in robot.links for link in self.tool_links.values()):
            raise ValueError("All tool links must be present in the live robot")
        self._enable_head_tracking = False
        self._motion_generator = CuRoboMotionGenerator(robot, batch_size=batch_size)
        self._emb = CuRoboEmbodimentSelection.DEFAULT
        if self._emb not in self._motion_generator.mg:
            available = sorted(str(k) for k in self._motion_generator.mg.keys())
            raise ValueError(
                f"Upstream cuRobo profile has no DEFAULT embodiment; available={available}. "
                "Bind a matching robot curobo profile; do not silently fall back to another embodiment."
            )
        supported = set(self._motion_generator.additional_links[self._emb]) | {self._motion_generator.ee_link[self._emb]}
        if not set(tool_links.values()) <= supported:
            raise ValueError("Upstream cuRobo profile lacks requested tool links; no silent constraint drop")
        self.representation = representation
        self.decoder = None

    def get_action_space(self):
        return gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(23,), dtype="float32")

    def bind_anchors(self):
        base = T.pose2mat(self.robot.links[self.robot.base_footprint_link_name].get_position_orientation())
        anchors = {
            name: T.pose_inv(base) @ T.pose2mat(self.robot.links[link].get_position_orientation())
            for name, link in self.tool_links.items()
        }
        self.decoder = DeltaTFDecoder(anchors, self.representation)

    def apply(self, action):
        return self.apply_ref(DeltaTFAction.MOVE, action)

    def apply_ref(self, primitive, row):
        if primitive != DeltaTFAction.MOVE:
            raise ValueError("Unknown delta-TF primitive")
        if self.decoder is None:
            raise ValueError("Bind run/chunk anchors explicitly after the initial state is restored")
        decoded = self.decoder.decode(row)
        # Reject unsupported motion before spending time constructing a plan.
        # These channels must participate in collision planning, not be patched
        # onto a trajectory afterwards. This first slice intentionally handles
        # fixed-base, unchanged-gripper targets only.
        if th.any(decoded["base_displacement"].abs() > 1e-7):
            raise NotImplementedError("Moving-base delta TF requires an upstream multi-link base target binding")
        for i, arm in enumerate(("left", "right")):
            measured = self.robot.get_joint_positions()[self.robot.gripper_control_idx[arm]]
            if not th.allclose(measured, decoded["gripper_widths"][i].expand_as(measured), atol=1e-4):
                raise NotImplementedError("Changing gripper width requires explicit collision-aware grasp-phase binding")
        base = T.pose2mat(self.robot.links[self.robot.base_footprint_link_name].get_position_orientation())
        targets = {self.tool_links[name]: T.mat2pose(base @ pose) for name, pose in decoded["poses"].items()}
        pos = {k: v[0].unsqueeze(0) for k, v in targets.items()}
        quat = {k: v[1].unsqueeze(0) for k, v in targets.items()}
        success, paths = self._motion_generator.compute_trajectories(
            target_pos=pos, target_quat=quat, is_local=False, emb_sel=self._emb,
            ik_only=False, ik_world_collision_check=True, skip_obstacle_update=False,
        )
        ok = th.where(success)[0]
        if len(ok) == 0:
            raise ActionPrimitiveError(ActionPrimitiveError.Reason.PLANNING_ERROR, "Official cuRobo rejected the target")
        q = self._motion_generator.path_to_joint_trajectory(paths[int(ok[0])], get_full_js=True, emb_sel=self._emb).cpu()
        if q.ndim != 2 or len(q) == 0 or not th.isfinite(q).all():
            raise ValueError("Official planner returned an invalid trajectory")
        current_q = self.robot.get_joint_positions().cpu()
        fixed_idx = th.cat((self.robot.base_control_idx, *[self.robot.gripper_control_idx[a] for a in ("left", "right")])).long()
        if not th.allclose(q[:, fixed_idx], current_q[fixed_idx].expand(len(q), -1), atol=1e-4):
            raise ValueError("Planner moved fixed base/gripper joints; this slice cannot execute that plan")
        yield from self._execute_motion_plan(q, ignore_physics=False, ignore_failure=False)
