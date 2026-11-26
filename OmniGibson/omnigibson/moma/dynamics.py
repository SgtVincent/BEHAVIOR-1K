"""Notch-EMA quasi-static load bias plus unsettled velocity response model.

Ported from the a28 frozen baseline ``dynamics_response.py``. The R1Pro SDK
arm/trunk controllers in the source episode use position mode with
use_impedances=False, forwarding absolute position setpoints to PhysX joint
drives over four 120Hz substeps per 30Hz action.

WARNING (gap G3): the constants below are data-derived from the same 2025
environment. They must be revalidated against the 2026 controller response in
the canary before being used for a real policy; this module only keeps the
filter state math sim-free and unit-testable.

Parameters:
  c_term  = terminal-velocity displacement ratio over the 4-substep interval
  beta    = first-substep velocity breakthrough gain on excess velocity
  alpha   = quasi-static EMA retention
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TERMINAL_VELOCITY_RATIO = 0.19
EXCESS_VELOCITY_GAIN = 0.20
QUASI_STATIC_EMA_ALPHA = 0.25
COMMAND_SLEW_TRUST_RAD = 0.006
COMMAND_SLEW_TIEBREAKER = 0.5e-4


@dataclass(frozen=True)
class DynamicsParams:
    terminal_velocity_ratio: float = TERMINAL_VELOCITY_RATIO
    excess_velocity_gain: float = EXCESS_VELOCITY_GAIN
    ema_alpha: float = QUASI_STATIC_EMA_ALPHA


def hard_stop_clip(vec, q, position_limits):
    vec = np.asarray(vec, float)
    q = np.asarray(q, float)
    lo, hi = (np.asarray(v, float) for v in position_limits)
    if vec.shape != q.shape or lo.shape != q.shape or hi.shape != q.shape:
        raise ValueError("invalid_hard_stop_clip_input")
    if not all(np.isfinite(x).all() for x in (vec, q, lo, hi)):
        raise ValueError("invalid_hard_stop_clip_input")
    return np.clip(vec, lo - q, hi - q)


def advance_response_state(*, arm, q, qvel, position_limits, raw_residual,
                           previous_goal_delta, state, dt: float = 1 / 30,
                           has_prior_step: bool,
                           params: DynamicsParams | None = None):
    p = params or DynamicsParams()
    q = np.asarray(q, float)
    qvel = np.asarray(qvel, float)
    raw_residual = np.asarray(raw_residual, float)
    previous_goal_delta = np.asarray(previous_goal_delta, float)
    if any(v.shape != q.shape or not np.isfinite(v).all()
           for v in (q, qvel, raw_residual, previous_goal_delta)):
        raise ValueError(f"invalid_response_state_input_{arm}")
    clipped_residual = hard_stop_clip(raw_residual, q, position_limits)
    excess_velocity_delta = qvel * float(dt) - p.terminal_velocity_ratio * previous_goal_delta
    if has_prior_step:
        prev_excess = np.asarray(state["excess_velocity_delta"], float)
        prev_obs = np.asarray(state["bias_observation"], float)
        prev_ema = np.asarray(state["quasi_static_bias"], float)
        if any(v.shape != q.shape or not np.isfinite(v).all()
               for v in (prev_excess, prev_obs, prev_ema)):
            raise ValueError(f"invalid_prior_response_filter_state_{arm}")
        bias_obs = clipped_residual - p.excess_velocity_gain * prev_excess
        notched = 0.5 * (bias_obs + prev_obs)
        bias_ema = p.ema_alpha * notched + (1.0 - p.ema_alpha) * prev_ema
    else:
        bias_obs = clipped_residual.copy()
        notched = bias_obs.copy()
        bias_ema = bias_obs.copy()
    unclipped_offset = bias_ema + p.excess_velocity_gain * excess_velocity_delta
    offset = hard_stop_clip(unclipped_offset, q, position_limits)
    saturated = np.flatnonzero(np.abs(offset - unclipped_offset) > 1e-12).tolist()
    next_state = {
        "bias_observation": bias_obs.copy(),
        "notched_bias": notched.copy(),
        "quasi_static_bias": bias_ema.copy(),
        "excess_velocity_delta": excess_velocity_delta.copy(),
        "clipped_residual": clipped_residual.copy(),
        "predicted_response_offset": offset.copy(),
        "hard_stop_saturated_dofs": saturated,
    }
    return offset, next_state
