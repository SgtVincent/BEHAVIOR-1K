"""Explicit time-grid mapping from planner waypoints to physics control steps.

The cuRobo v2 executor returns a joint trajectory sampled at its own planning
dt (``0.025 s`` on the sealed R1Pro receipt); the 2026 evaluator steps the
robot on the sealed 30 Hz control grid (``1/30 s``). This seam resamples the
certified trajectory onto the physics timestamps instead of applying the final
configuration as an instant jump.

Semantics taken from the v2 adapter handoff:

- ``native23`` is a final goal configuration, not a single-step velocity; the
  trajectory must be consumed waypoint by waypoint.
- Batch / horizon are 1; waypoints are ordered in time; no cross-batch or
  cross-horizon concatenation.
- After the plan duration the seam exhausts: no last-waypoint hold, no repeat,
  no zero-filled fallback.

Stdlib only at import. The grid does not solve IK, call the planner, or read
live robot state; it only time-aligns an already certified plain trajectory.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

from . import contracts
from ._arr import is_finite, to_plain
from .errors import ContractViolation, Port2026Error

PLANNER_DT_S = 0.025


class TrajectoryExhausted(Port2026Error):
    """The certified plan covers no further physics step; no hold/repeat."""


class TrajectoryGridViolation(ContractViolation):
    """Waypoint payload, dt, or per-step bounded-motion gate failed."""


def _validate_waypoints(
    waypoints: Sequence[Sequence[float]], action_dim: int
) -> List[List[float]]:
    plain = [list(to_plain(row)) for row in waypoints]
    if not plain:
        raise TrajectoryGridViolation("trajectory has no waypoints")
    for row in plain:
        if len(row) != action_dim:
            raise TrajectoryGridViolation(
                f"waypoint width {len(row)} != {action_dim}"
            )
        if not is_finite(row):
            raise TrajectoryGridViolation("waypoint non-finite")
    return [[float(v) for v in row] for row in plain]


class PlannerTrajectoryGrid:
    """Resample a certified ``[T, D]`` planner trajectory onto physics steps.

    Args:
        waypoints: ``T`` certified native-action vectors, ordered by time.
        planner_dt: planner trajectory sample period in seconds (receipt value).
        physics_dt: evaluator control step in seconds (sealed 30 Hz default).
        action_dim: sealed action width (23).
        max_step_delta: optional closed per-coordinate bound between successive
            physics commands. ``None`` disables the motion gate; a finite
            non-negative scalar raises :class:`TrajectoryGridViolation` when a
            resampled step moves beyond it.
    """

    def __init__(
        self,
        waypoints: Sequence[Sequence[float]],
        planner_dt: float = PLANNER_DT_S,
        physics_dt: float = contracts.CONTROL_DT,
        action_dim: int = contracts.ACTION23,
        max_step_delta: Optional[float] = None,
    ) -> None:
        self._waypoints = _validate_waypoints(waypoints, action_dim)
        self._planner_dt = self._positive_dt(planner_dt, "planner_dt")
        self._physics_dt = self._positive_dt(physics_dt, "physics_dt")
        if max_step_delta is not None:
            bound = float(max_step_delta)
            if not math.isfinite(bound) or bound < 0.0:
                raise TrajectoryGridViolation("max_step_delta must be finite >= 0")
            self._max_step_delta: Optional[float] = bound
        else:
            self._max_step_delta = None
        self._previous: Optional[List[float]] = None

    @staticmethod
    def _positive_dt(value: float, label: str) -> float:
        dt = float(value)
        if not math.isfinite(dt) or dt <= 0.0:
            raise TrajectoryGridViolation(f"{label} must be finite and positive")
        return dt

    # -- read-only meta -----------------------------------------------------

    @property
    def waypoint_count(self) -> int:
        return len(self._waypoints)

    @property
    def planner_dt(self) -> float:
        return self._planner_dt

    @property
    def physics_dt(self) -> float:
        return self._physics_dt

    @property
    def duration_s(self) -> float:
        return (len(self._waypoints) - 1) * self._planner_dt

    def interval_start(self, physics_step: int) -> float:
        """Start time of the physics control interval ``[k*dt, (k+1)*dt]``."""
        return int(physics_step) * self._physics_dt

    def command_time(self, physics_step: int) -> float:
        """Trajectory timestamp this step's command targets: interval end."""
        return (int(physics_step) + 1) * self._physics_dt

    def plan_covers_steps(self) -> int:
        """Number of full physics step intervals covered by the plan."""
        if self.duration_s <= 0.0:
            return 0
        return int(math.floor(self.duration_s / self._physics_dt + 1e-9))

    def reset(self) -> None:
        """Clear the per-step motion gate reference; the trajectory is unchanged."""
        self._previous = None

    # -- time-aligned command -----------------------------------------------

    def command_at(self, physics_step: int) -> Dict[str, Any]:
        """Return the native action for one physics step on its explicit time grid.

        The physics step ``k`` maps to timestamp ``t = k * physics_dt`` inside
        the planner trajectory. Adjacent waypoints are interpolated by the
        elapsed fraction of the planner segment; the command never jumps to
        the final waypoint ahead of time.
        """
        if type(physics_step) is not int or physics_step < 0:
            raise TrajectoryGridViolation("physics_step must be a non-negative int")
        t = self.command_time(physics_step)
        if t > self.duration_s + 1e-9:
            raise TrajectoryExhausted(
                f"plan duration {self.duration_s:.6f}s covers step {physics_step} "
                f"(command t={t:.6f}s); no hold or last-waypoint repeat"
            )

        count = len(self._waypoints)
        if t >= self.duration_s - 1e-9:
            segment = count - 2
            fraction = 1.0
            command = list(self._waypoints[-1])
        else:
            segment = min(int(t / self._planner_dt), count - 2)
            t0 = segment * self._planner_dt
            fraction = min(1.0, max(0.0, (t - t0) / self._planner_dt))
            left = self._waypoints[segment]
            right = self._waypoints[segment + 1]
            command = [
                left[k] + (right[k] - left[k]) * fraction
                for k in range(len(left))
            ]

        if not all(math.isfinite(v) for v in command):
            raise TrajectoryGridViolation("resampled command non-finite")

        if self._previous is not None:
            worst = max(abs(command[k] - self._previous[k]) for k in range(len(command)))
            if self._max_step_delta is not None and worst > self._max_step_delta:
                raise TrajectoryGridViolation(
                    f"per-step bounded motion gate exceeded: {worst:.6g} > "
                    f"{self._max_step_delta:.6g}"
                )
        self._previous = list(command)

        return {
            "native23": command,
            "physics_step": physics_step,
            "interval_start_s": self.interval_start(physics_step),
            "time_s": t,
            "planner_segment": segment,
            "fraction": fraction,
            "planner_dt": self._planner_dt,
            "physics_dt": self._physics_dt,
        }

    def receipt(self) -> Dict[str, Any]:
        return {
            "waypoints": len(self._waypoints),
            "planner_dt": self._planner_dt,
            "physics_dt": self._physics_dt,
            "duration_s": self.duration_s,
            "plan_covers_steps": self.plan_covers_steps(),
            "max_step_delta": self._max_step_delta,
        }
