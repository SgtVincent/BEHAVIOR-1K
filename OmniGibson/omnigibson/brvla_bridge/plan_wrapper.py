"""Managed replay wrapper: decoded fixed-anchor targets -> planner seam -> grid.

This is the CPU composition layer between the frozen replay delta-TF targets,
the certified cuRobo v2 planner executor seam, and the sealed 30 Hz native23
controller vector. It does not re-implement a control loop:

- :class:`~brvla_bridge.delta_tf.FixedAnchorTargets` decodes the saved
  execution23 row into fixed-anchor SE3 targets, body velocity and gripper
  widths.
- The executor seam (duck-typed ``plan(goal)`` callable, registered through the
  backend registry) returns a certified optimal plan with native23
  waypoints; the wrapper never imports curobo directly.
- :class:`~brvla_bridge.planner_grid.PlannerTrajectoryGrid` resamples the
  certified trajectory onto physics timestamps.
- :mod:`~brvla_bridge.native_action` assembles the 23-d vector with explicit
  per-group provenance; unproven groups fail closed.

Scope verification happens before any planning: a left-only certificate can
only serve a left-only consumer. Base velocity and gripper widths come from the
replay row; planner zero values are never substituted for a missing seam.

Stdlib-only at import.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from . import contracts, native_action, planner_grid, scope
from .delta_tf import FixedAnchorTargets
from .errors import BackendNotDeclaredError, ContractViolation
from .native_action import (
    SOURCE_CURRENT_Q,
    SOURCE_PLANNER,
    SOURCE_REPLAY,
    assemble_native23,
    extract_group,
)


class ReplayWrapperViolation(ContractViolation):
    """Wrapper composition input or certified-plan payload violates the seam."""


class ManagedReplayWrapper:
    """Compose replay targets, executor seam, and time grid for one episode.

    Args:
        targets: decoded fixed-anchor replay targets for this episode.
        scope: verified :class:`~brvla_bridge.scope.ScopeReport` for the
            planner certificate this wrapper may consume.
        executor: optional duck-typed planner executor with
            ``plan(goal) -> {"action23", "certificate", "trajectory"|"meta"}``.
            When ``None`` the first planning call raises
            :class:`~brvla_bridge.errors.BackendNotDeclaredError`.
        planner_dt: fallback planner sample period when the plan meta omits it.
        physics_dt: evaluator control step (sealed 1/30).
    """

    def __init__(
        self,
        *,
        targets: FixedAnchorTargets,
        scope_report: scope.ScopeReport,
        executor: Optional[Any] = None,
        planner_dt: float = planner_grid.PLANNER_DT_S,
        physics_dt: float = contracts.CONTROL_DT,
        current_q: Optional[Dict[str, Dict[str, Any]]] = None,
        max_step_delta: Optional[float] = None,
    ) -> None:
        if not isinstance(targets, FixedAnchorTargets):
            raise ReplayWrapperViolation("targets must be a FixedAnchorTargets")
        scope_report.require()
        self._targets = targets
        self._scope = scope_report
        self._executor = executor
        self._planner_dt = float(planner_dt)
        self._physics_dt = float(physics_dt)
        self._max_step_delta = max_step_delta
        # Inactive-arm current-q bindings: group name -> bind entry from
        # :func:`native_action.bind_current_q`. An unbound inactive group
        # still fails closed; no zero default is substituted.
        self._current_q: Dict[str, Dict[str, Any]] = dict(current_q or {})

    def bind_current_q(
        self,
        arm: str,
        qpos: Sequence[Any],
        *,
        source_sha256: str,
        source_kind: str = "live_sdk_qpos",
    ) -> None:
        """Bind an inactive arm's live current qpos to hold under collision."""
        entry = native_action.bind_current_q(
            arm, qpos, source_sha256=source_sha256, source_kind=source_kind)
        self._current_q[entry["group"]] = entry

    # -- read-only identity -------------------------------------------------

    @property
    def episode_length(self) -> int:
        return self._targets.length

    def targets_at(self, index: int) -> Dict[str, Any]:
        return self._targets.target(index)

    # -- goal construction --------------------------------------------------

    def planner_goal(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """Build the plain goal dict handed to the executor seam.

        Rows are 4x4 homogeneous SE3 matrices in the physical base frame, in
        the staged executor's ``target_pose8x4_world`` layout. The right row is
        included only when the certified scope covers the right arm; a
        left-only certificate never sends a right goal it did not plan.
        """
        matrices = target.get("pose_matrices")
        if not isinstance(matrices, dict):
            raise ReplayWrapperViolation("target missing pose_matrices")
        left = matrices.get("left")
        if left is None:
            raise ReplayWrapperViolation("target missing left pose matrix")
        rows: list = [left]
        evidence = self._scope.evidence
        if evidence.get("right_arm_planned") is True:
            right = matrices.get("right")
            if right is not None:
                rows.append(right)
        goal: Dict[str, Any] = {
            "target_pose8x4_world": rows,
            "grippers": list(target["continuous_gripper"]),
            "base_vel": list(target["base_velocity_body"]),
        }
        return goal

    # -- plan / grid composition --------------------------------------------

    def _get_executor(self) -> Any:
        if self._executor is None:
            raise BackendNotDeclaredError("no planner executor seam registered")
        return self._executor  # duck-typed executor object with .plan(goal)

    def plan_row(self, index: int) -> Dict[str, Any]:
        """Request one certified plan for replay row ``index`` and build its grid.

        Returns the grid, certificate, and plan meta. The wrapper does not cache
        or hold a plan across steps; the caller owns chunking and budget.
        """
        target = self._targets.target(index)
        goal = self.planner_goal(target)
        executor = self._get_executor()
        if executor is None:
            raise BackendNotDeclaredError("no planner executor seam registered")
        result = executor.plan(goal)
        if not isinstance(result, dict):
            raise ReplayWrapperViolation("executor.plan must return a dict")

        meta = result.get("trajectory") or result.get("meta")
        if not isinstance(meta, dict):
            raise ReplayWrapperViolation("plan payload missing trajectory/meta")
        waypoints = meta.get("native23_waypoints")
        if waypoints is None:
            raise ReplayWrapperViolation("plan trajectory missing native23_waypoints")
        planner_dt = meta.get("dt") or self._planner_dt

        grid = planner_grid.PlannerTrajectoryGrid(
            waypoints=waypoints,
            planner_dt=planner_dt,
            physics_dt=self._physics_dt,
            max_step_delta=self._max_step_delta,
        )
        return {
            "grid": grid,
            "certificate": result.get("certificate") or meta.get("certificate") or {},
            "meta": meta,
            "target": target,
        }

    # -- native23 assembly ---------------------------------------------------

    def assemble_step(self, plan: Dict[str, Any], physics_step: int) -> Dict[str, Any]:
        """Assemble the native23 command for one physics step of a certified plan.

        Planner-sourced groups: trunk4, arm_left7, and (when certified)
        arm_right7. Replay-sourced groups: base velocity3, gripper_left1,
        gripper_right1. Any group without a verified source raises; no zero
        or planner-default placeholder is emitted.
        """
        grid: planner_grid.PlannerTrajectoryGrid = plan["grid"]
        target: Dict[str, Any] = plan["target"]
        command = grid.command_at(physics_step)
        planner_row = list(command["native23"])

        evidence = self._scope.evidence
        bound_arm = evidence.get("bound_arm")
        left_planned = bound_arm in (scope.ARM_LEFT, scope.ARM_BOTH)
        right_certified = evidence.get("right_arm_planned") is True

        def _arm(group: str, planned: bool) -> tuple:
            if planned:
                return extract_group(planner_row, group), SOURCE_PLANNER, None
            entry = self._current_q.get(group)
            if entry is not None:
                return list(entry["values"]), SOURCE_CURRENT_Q, entry
            return None, "missing", None

        arm_left, src_left, q_left = _arm("arm_left", left_planned)
        arm_right, src_right, q_right = _arm("arm_right", right_certified)

        groups = {
            "trunk": extract_group(planner_row, "trunk"),
            "arm_left": arm_left,
            "gripper_left": [float(target["continuous_gripper"][0])],
            "gripper_right": [float(target["continuous_gripper"][1])],
            "base": list(target["base_velocity_body"]),
            "arm_right": arm_right,
        }
        sources = {
            "trunk": SOURCE_PLANNER,
            "arm_left": src_left,
            "gripper_left": SOURCE_REPLAY,
            "gripper_right": SOURCE_REPLAY,
            "base": SOURCE_REPLAY,
            "arm_right": src_right,
        }

        assembled = assemble_native23(
            base_vel=groups["base"],
            trunk=groups["trunk"],
            arm_left=groups["arm_left"],
            gripper_left=groups["gripper_left"],
            arm_right=groups["arm_right"],
            gripper_right=groups["gripper_right"],
            sources=sources,
        )
        held_q = {
            g: {"source": e["source"], "source_kind": e["source_kind"],
                "source_sha256": e["source_sha256"]}
            for g, e in (("arm_left", q_left), ("arm_right", q_right)) if e is not None
        }
        assembled.update({
            "physics_step": physics_step,
            "planner_segment": command["planner_segment"],
            "fraction": command["fraction"],
            "time_s": command["time_s"],
            "target_row": target["source_row"],
            "held_current_q": held_q,
        })
        return assembled
