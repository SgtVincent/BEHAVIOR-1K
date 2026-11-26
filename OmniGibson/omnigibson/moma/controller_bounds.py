"""Pure NumPy hard-limit / preferred-margin QP velocity bounds.

Ported from the a28 frozen baseline ``controller_bounds.py``. The limit tables
themselves are bound to the 2026 R1Pro controller profile at the executor seam
(gap G2); this module only implements the one-step safety and soft-margin math.
"""
from __future__ import annotations

import numpy as np


def recoverable_velocity_bounds(
    q, hard_lower, hard_upper, velocity_lower, velocity_upper, *, dt: float,
    margin: float = 0.02, speed_limit: float = 1.2,
):
    q = np.asarray(q, dtype=np.float64)
    hard_lower = np.asarray(hard_lower, dtype=np.float64)
    hard_upper = np.asarray(hard_upper, dtype=np.float64)
    lower = np.maximum(np.asarray(velocity_lower, dtype=np.float64), -float(speed_limit))
    upper = np.minimum(np.asarray(velocity_upper, dtype=np.float64), float(speed_limit))
    if not (q.shape == hard_lower.shape == hard_upper.shape == lower.shape == upper.shape):
        raise ValueError("QP bound arrays must have identical shapes")
    if not all(np.isfinite(x).all() for x in (q, hard_lower, hard_upper, lower, upper)):
        raise ValueError("QP bound inputs must be finite")
    if dt <= 0.0 or margin < 0.0 or speed_limit <= 0.0:
        raise ValueError("dt, margin, and speed_limit must be valid positive values")
    hard_valid = hard_lower < hard_upper
    # A loaded physical state can begin just outside a hard limit. Preserve q
    # unchanged and require the NEXT 30Hz command to satisfy the original hard
    # bounds via the one-step (hard_lower-q)/dt, (hard_upper-q)/dt intervals
    # below; an unreachable return remains infeasible.

    # One-step hard-limit safety is mandatory. Preferred margins are soft:
    # when already beyond one, allow only inward / non-worsening motion instead
    # of demanding an impossible return to the margin in a single physics step.
    lower = np.maximum(lower, np.where(hard_valid, (hard_lower - q) / dt, -np.inf))
    upper = np.minimum(upper, np.where(hard_valid, (hard_upper - q) / dt, np.inf))
    preferred_lower = hard_lower + margin
    preferred_upper = hard_upper - margin
    below_margin = hard_valid & (q < preferred_lower)
    above_margin = hard_valid & (q > preferred_upper)
    lower = np.where(below_margin, np.maximum(lower, 0.0), lower)
    upper = np.where(above_margin, np.minimum(upper, 0.0), upper)
    # Inside the preferred interval, only hard-limit one-step safety applies.
    # The margin is directional recovery guidance, not a hard one-step box.

    feasible = lower <= upper + 1e-12
    diagnostics = {
        "below_preferred_margin": below_margin,
        "above_preferred_margin": above_margin,
        "hard_limit_valid": hard_valid,
        "feasible": feasible,
        "lower": lower,
        "upper": upper,
        "preferred_lower": preferred_lower,
        "preferred_upper": preferred_upper,
    }
    if not np.all(feasible):
        bad = np.where(~feasible)[0].tolist()
        raise ValueError(f"empty QP velocity interval at dofs {bad}")
    return lower, upper, diagnostics
