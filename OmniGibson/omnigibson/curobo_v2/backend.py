# Copyright (c) 2026 B1K2026 integration line
# SPDX-License-Identifier: Apache-2.0
"""Lazy cuRobo v2 planner backend binding.

Heavy imports (torch, curobo, warp) land only in :meth:`_bind`, which runs in
the GPU-bound stage. Static :meth:`profile` and :meth:`can_serve` never import
the planner, so the MoMa/BR doctor can classify the plugin without a Kit or
GPU boot.

Responsibilities sealed in this adapter:

- robot profile (R1Pro active joints, tool frames, locked joints, dt);
- object-centric goal -> v2 IK / MotionPlanner call shapes (sourced from the
  pinned v2 examples, not the legacy v1 ``action_primitives.curobo``);
- planner trajectory -> native23 slices;
- certificate digest (backend name, resolved version, device UUID, timing,
  pose error). R1Pro frames/spheres are never verified against the franka
  example fixture.

Until an R1Pro robot configuration (URDF-derived curobo yaml, collision sphere
map, self-collision ignore) and its canary evidence are supplied, binding
raises :class:`RobotProfileUnbound`; the executor fails closed instead of
substituting a franka plan or a zero action.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from .contracts import (
    PLANNER_DT_S,
    R1PRO_LOCKED_JOINTS,
    R1PRO_PLANNER_JOINTS,
    R1PRO_TOOL_FRAMES,
    PlannerGoal,
    assemble_native23,
    validate_trajectory_shape,
)
from .errors import BackendUnavailable, BackendImportFailed, RobotProfileUnbound


class CuroboV2Backend:
    """Owns the live v2 planner objects for one robot profile.

    Parameters
    ----------
    robot_profile:
        ``"r1pro"`` for the 2026 dual-arm robot. ``"franka_example"`` exists
        only as the verified GPU0 gate fixture and must not be registered as an
        R1Pro executor backend.
    device_uuid:
        Physical GPU UUID the planner process is pinned to. Recorded in every
        certificate; cross-lane cards are never touched.
    robot_config_yaml:
        Path to the verified curobo robot yaml for the profile. Required for
        R1Pro binding.
    urdf_path / asset_path:
        URDF and mesh asset root, used only when building the config through
        the v2 ``RobotBuilder``. Private paths live in the external manifest,
        never hard-coded in git.
    canary_evidence:
        Provenance digest for the bound config (URDF sha, sphere-map source,
        canary receipt). Absence keeps the backend uncertified.
    """

    def __init__(
        self,
        *,
        robot_profile: str = "r1pro",
        device_uuid: Optional[str] = None,
        robot_config_yaml: Optional[str] = None,
        urdf_path: Optional[str] = None,
        asset_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        canary_evidence: Optional[Dict[str, Any]] = None,
        private_manifest: Optional[Dict[str, Any]] = None,
        arm_profile: str = "left",
    ) -> None:
        if arm_profile not in ("left", "right"):
            raise ValueError(f"unsupported arm_profile: {arm_profile!r}")
        self._profile_name = robot_profile
        self._arm_profile = arm_profile
        self._device_uuid = device_uuid
        self._robot_config_yaml = robot_config_yaml
        self._urdf_path = urdf_path
        self._asset_path = asset_path
        self._cache_dir = cache_dir
        self._canary_evidence = dict(canary_evidence or {})
        self._private_manifest = dict(private_manifest or {})
        self._bound = False
        self._objects: Dict[str, Any] = {}
        self._backend_name: Optional[str] = None
        self._version: Optional[str] = None
        self._bound_arm: Optional[str] = None
        self._primary_tool: Optional[str] = None

    # -- static surface (no heavy import) ---------------------------------

    def profile(self) -> Dict[str, Any]:
        return {
            "robot_profile": self._profile_name,
            "device_uuid": self._device_uuid,
            "active_joints": list(R1PRO_PLANNER_JOINTS),
            "locked_joints": list(R1PRO_LOCKED_JOINTS),
            "tool_frames": dict(R1PRO_TOOL_FRAMES),
            "arm_profile": self._arm_profile,
            "bound_arm": self._bound_arm,
            "primary_tool": self._primary_tool,
            "dt_s": PLANNER_DT_S,
            "bound": self._bound,
            "certified": self._is_certified(),
            "robot_config_yaml": self._robot_config_yaml,
            "canary_evidence": dict(self._canary_evidence),
        }

    def _is_certified(self) -> bool:
        evidence = self._canary_evidence
        return bool(evidence.get("canary_pass")) and bool(evidence.get("receipt_sha256"))

    def can_serve(self) -> bool:
        return self._bound and self._is_certified()

    # -- lazy GPU-bound binding -------------------------------------------

    def _bind(self) -> None:
        if self._bound:
            return
        if self._profile_name != "r1pro":
            raise RobotProfileUnbound(
                f"profile {self._profile_name!r} is not a registered 2026 executor backend"
            )
        if not self._robot_config_yaml:
            raise RobotProfileUnbound(
                "R1Pro curobo robot config yaml not supplied; URDF/sphere-map binding unverified"
            )
        if not self._is_certified():
            raise RobotProfileUnbound("R1Pro canary evidence missing or incomplete")

        try:
            import curobo  # noqa: F401  (lazy, GPU-bound stage only)
            import torch  # noqa: F401
            from curobo._src.curobolib.backends import get_backend_name
        except ImportError as exc:
            raise BackendImportFailed(f"{exc}") from exc

        try:
            from pathlib import Path

            import yaml

            from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

            robot_dict = yaml.safe_load(Path(self._robot_config_yaml).read_text())
            if not isinstance(robot_dict, dict):
                raise RobotProfileUnbound("robot config yaml did not parse to a dict")
            # Per-arm unit gate: the internal IK solver expects one tool frame per
            # planner. Each arm profile binds its own tool frame; the opposite arm
            # is an unproven unit on this planner and fails closed rather than
            # sharing a dual-frame solver.
            primary = R1PRO_TOOL_FRAMES[self._arm_profile]
            robot_dict.setdefault("kinematics", {})["tool_frames"] = [primary]
            cfg = MotionPlannerCfg.create(robot=robot_dict, self_collision_check=True)
            planner = MotionPlanner(cfg)
            planner.warmup(enable_graph=True, num_warmup_iterations=5)
            self._objects["planner"] = planner
            self._bound_arm = self._arm_profile
            self._primary_tool = primary
            self._backend_name = get_backend_name()
            self._version = getattr(curobo, "__version__", "unknown")
        except RobotProfileUnbound:
            raise
        except Exception as exc:  # bind failure surfaces as classified blocker
            raise BackendImportFailed(f"planner construct: {exc}") from exc

        self._bound = True

    # -- planning calls ---------------------------------------------------

    def plan_goal(self, goal: PlannerGoal) -> Dict[str, Any]:
        """Resolve one object-centric goal to a native23 command + certificate.

        Returns a plain dict. Raises :class:`BackendUnavailable` before binding;
        a rejected plan is returned with ``success=False`` (the executor then
        fails closed rather than emitting a placeholder).
        """
        if not self._bound:
            self._bind()
        if goal.source == "s2_args" and goal.left_pose is None and goal.right_pose is None:
            # S2 primitive args carry an object id only; the live SE3 pose must
            # arrive through the observer/provider seam. Do not fabricate one.
            return {
                "success": False,
                "native23": None,
                "trajectory": None,
                "certificate": {
                    "status": "blocked",
                    "reason": "goal_pose_unresolved",
                    "backend": self._backend_name,
                    "version": self._version,
                    "device_uuid": self._device_uuid,
                },
                "error": "S2 args target has no live SE3 pose; observer seam must resolve it",
            }

        routed = self._route_arm(goal)
        if routed is not None:
            return routed

        planner = self._objects.get("planner")
        if planner is None:
            raise BackendUnavailable("planner object missing after bind")

        # v2 call shapes follow the pinned examples: GoalToolPose -> plan_pose;
        # interpolated plan -> finite joint trajectory. curobo tensors stay in
        # this method and are converted to plain lists before returning.
        import torch

        from curobo.types import JointState, GoalToolPose

        pose = goal.right_pose if self._arm_profile == "right" else goal.left_pose
        if pose is None:
            return self._blocked("goal_pose_missing")

        pos, quat = pose
        tool_frame = self._primary_tool or R1PRO_TOOL_FRAMES[self._arm_profile]
        q_start = JointState.from_position(
            planner.default_joint_state.position.unsqueeze(0),
            joint_names=planner.joint_names,
        )
        goal_pose = GoalToolPose(
            tool_frames=[tool_frame],
            position=torch.tensor(list(pos), device="cuda", dtype=torch.float32).view(1, 1, 1, 1, 3),
            quaternion=torch.tensor(list(quat), device="cuda", dtype=torch.float32).view(1, 1, 1, 1, 4),
        )
        result = planner.plan_pose(goal_pose, q_start)
        if result is None or not bool(result.success.any().item()):
            return self._blocked("planner_no_success", result=result)

        interp = result.get_interpolated_plan()
        joint_pos = interp.position.detach().cpu()
        if not bool(torch.isfinite(joint_pos).all().item()):
            return self._blocked("trajectory_nonfinite", result=result)

        # Validate B/H semantics explicitly. Multi-batch or multi-horizon plans
        # are rejected rather than flattened: cross-batch/horizon rows must
        # never be concatenated into one time-ordered trajectory.
        # The interpolated plan carries the full sealed state width (active 18
        # + locked 10); the planner's reported joint_names list covers the
        # active subset only. Validate against the sealed full-state width.
        full_state_width = len(R1PRO_PLANNER_JOINTS) + len(R1PRO_LOCKED_JOINTS)
        spec = validate_trajectory_shape(
            tuple(joint_pos.shape), full_state_width
        )
        rows_tensor = joint_pos[spec["selector"]] if spec["selector"] else joint_pos
        waypoints = rows_tensor.shape[0]
        last = rows_tensor[-1].tolist()
        native23 = self._assemble(planner.joint_names, last, goal)
        # Per-waypoint native23 commands let the consumer step the trajectory
        # at dt; the final vector above is the goal configuration, not a
        # single-step velocity, and must not be applied as an endpoint jump.
        waypoint_native23 = [
            list(self._assemble(planner.joint_names, row.tolist(), goal))
            for row in rows_tensor
        ]
        return {
            "success": True,
            "native23": list(native23),
            "trajectory": {
                "joints": rows_tensor.tolist(),
                "native23_waypoints": waypoint_native23,
                "dt": PLANNER_DT_S,
                "waypoints": waypoints,
                "batch": spec["batch"],
                "horizon": spec["horizon"],
                "seam_bindings": self._seam_bindings(goal),
                "action_semantics": "final_configuration_not_single_step_velocity; "
                "consumer must step native23_waypoints sequentially at dt",
            },
            "certificate": {
                "status": "optimal",
                "backend": self._backend_name,
                "version": self._version,
                "device_uuid": self._device_uuid,
                "planning_time_s": float(getattr(result, "total_time", 0.0) or 0.0),
                "position_error_mm": float(result.position_error.max().item()) * 1000.0,
                "robot_profile": self._profile_name,
                **self._scope_info(),
                **self._provenance(),
            },
            "error": None,
        }

    def _assemble(
        self,
        joint_names: List[str],
        last_joints: List[float],
        goal: PlannerGoal,
    ) -> Tuple[float, ...]:
        index = {name: i for i, name in enumerate(joint_names)}
        try:
            trunk = [last_joints[index[n]] for n in R1PRO_PLANNER_JOINTS[:4]]
            arm_left = [last_joints[index[n]] for n in R1PRO_PLANNER_JOINTS[4:11]]
            arm_right = [last_joints[index[n]] for n in R1PRO_PLANNER_JOINTS[11:18]]
        except KeyError as exc:
            raise RobotProfileUnbound(f"planner joint missing in config: {exc}") from exc
        grip_l, grip_r = 0.0, 0.0
        if goal.grippers is not None:
            grip_l, grip_r = goal.grippers
        base_vel = list(goal.base_vel) if goal.base_vel is not None else [0.0, 0.0, 0.0]
        return assemble_native23(
            base_vel=base_vel,
            trunk=trunk,
            arm_left=arm_left,
            gripper_left=grip_l,
            arm_right=arm_right,
            gripper_right=grip_r,
        )

    def _provenance(self) -> Dict[str, Any]:
        """Static receipt/config back-link for certificates (spec §8).

        Read straight from the sealed canary evidence; no planner object is
        touched, so this is safe to assert on the CPU test surface.
        """
        return {
            "receipt": self._canary_evidence.get("receipt_sha256"),
            "config_yml_sha256": self._canary_evidence.get("config_yml_sha256"),
        }

    def _route_arm(self, goal: PlannerGoal) -> Optional[Dict[str, Any]]:
        """Fail-closed arm routing for the single-arm bound profile.

        A dual-arm goal (both poses) is rejected as ``dual_arm_unproven``. On a
        left-bound profile a right-only goal is ``right_planner_unproven``; on a
        right-bound profile a left-only goal is ``left_planner_unproven``. The
        bound profile must never silently drop the opposite pose and report an
        optimal plan as a coordinated-arm result.
        """
        if self._arm_profile == "right":
            if goal.arm == "both" or goal.left_pose is not None:
                if goal.arm == "left" or goal.right_pose is None:
                    return self._blocked("left_planner_unproven")
                return self._blocked("dual_arm_unproven")
            return None
        if goal.arm == "both" or goal.right_pose is not None:
            if goal.arm == "right" or goal.left_pose is None:
                return self._blocked("right_planner_unproven")
            return self._blocked("dual_arm_unproven")
        return None

    def _scope_info(self) -> Dict[str, Any]:
        """Explicit planner-scope record attached to every success certificate."""
        right = self._arm_profile == "right"
        return {
            "bound_arm": self._bound_arm or self._arm_profile,
            "primary_tool": self._primary_tool or R1PRO_TOOL_FRAMES[self._arm_profile],
            "certified_scope": f"{self._arm_profile}_arm_only; self_collision; scene_model_null",
            "right_arm_planned": right,
            "device_uuid_source": "manifest_argument_not_live_probe",
        }

    def _seam_bindings(self, goal: PlannerGoal) -> Dict[str, Any]:
        """Record which native23 slots are owned by external seams.

        Base holonomic velocity and continuous gripper width are not produced
        by the planner. When the seam does not supply them, the slot is held at
        zero for shape compatibility but flagged ``supplied=false``; the zero
        must not be read as a navigation or grasp command.
        """
        grip_l, grip_r = (goal.grippers if goal.grippers is not None else (None, None))
        return {
            "base_vel": {
                "owner": "navigation_seam",
                "supplied": goal.base_vel is not None,
                "value": list(goal.base_vel) if goal.base_vel is not None else None,
            },
            "gripper_left": {"owner": "grasp_seam", "supplied": grip_l is not None, "value": grip_l},
            "gripper_right": {"owner": "grasp_seam", "supplied": grip_r is not None, "value": grip_r},
        }

    def _blocked(self, reason: str, result: Any = None) -> Dict[str, Any]:
        return {
            "success": False,
            "native23": None,
            "trajectory": None,
            "certificate": {
                "status": "blocked",
                "reason": reason,
                "backend": self._backend_name,
                "version": self._version,
                "device_uuid": self._device_uuid,
            },
            "error": reason,
        }

    def close(self) -> None:
        planner = self._objects.get("planner")
        if planner is not None:
            try:
                planner.destroy()
            except Exception:
                pass
        self._objects.clear()
        self._bound = False
