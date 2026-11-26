"""Snapshot freshness and decision-identity binding gate (a28 port).

The a28 owner boundary binds every S2 request to the live observation it was
issued against: ``run_id``, ``observation_id`` and a bounded snapshot-version /
sim-time window. A late-returned tool plan targeting an expired or mismatched
snapshot is rejected before any MP plan or env.step; the current live state
must not be pseudochanged to fit a stale plan.

Constants are the frozen a28 values (``s2_max_snapshot_age_steps=1800``,
``s2_max_snapshot_age_sim_s=60.0``) carried by the runner config. This module
does not retune dt, windows or allowed delays, and does not import the
simulator, torch or cuRobo.

Rejection codes mirror the frozen baseline ``submit_primitive_request`` gate:

- ``run_id_mismatch``
- ``missing_observation_id``
- ``stale_observation_id`` / ``invalid_observation_id``
- ``invalid_snapshot_version``
- ``stale_or_invalid_snapshot_version``
- ``stale_observation_snapshot_expired``
- ``invalid_observed_sim_time_s``
- ``stale_or_invalid_snapshot_sim_time``
"""
from __future__ import annotations

from typing import Any, Dict, FrozenSet, Optional, Set


class DecisionFreshnessRejected(RuntimeError):
    """The returned decision targets a stale, mismatched or expired snapshot.

    Raised at the owner boundary before the executor plan and env.step. The
    rejected plan must not be executed and must never fall back to a zero
    action.
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


# Field names kept identical to the a28 observation envelope / request record.
ENVELOPE_KEYS: FrozenSet[str] = frozenset({
    "run_id",
    "observation_id",
    "snapshot_version",
    "decision_entry_snapshot_version",
    "decision_seq",
    "sim_time_s",
})

_BINDING_KEYS = frozenset({
    "run_id",
    "observation_id",
    "snapshot_version",
    "observed_sim_time_s",
})


def build_envelope(
    *,
    run_id: str,
    observation_id: str,
    snapshot_version: int,
    decision_entry_snapshot_version: int,
    decision_seq: int,
    sim_time_s: Optional[float],
    max_age_steps: int,
    max_age_sim_s: float,
) -> Dict[str, Any]:
    """Attach the freshness envelope to a fresh boundary observation."""
    return {
        "run_id": run_id,
        "observation_id": observation_id,
        "snapshot_version": int(snapshot_version),
        "decision_entry_snapshot_version": int(decision_entry_snapshot_version),
        "decision_seq": int(decision_seq),
        "sim_time_s": (None if sim_time_s is None else float(sim_time_s)),
        "freshness_policy": {
            "binding": "observation_id_plus_bounded_snapshot_version_window",
            "max_snapshot_age_steps": int(max_age_steps),
            "max_snapshot_age_sim_s": float(max_age_sim_s),
        },
    }


def binding_from_envelope(envelope: Dict[str, Any]) -> Dict[str, Any]:
    """Owner-side identity snapshot captured when the decision request leaves.

    The binding is copied from the envelope the owner built; an external
    model cannot mint or rewrite this identity by filling the tool arguments.
    """
    return {
        "run_id": envelope.get("run_id"),
        "observation_id": envelope.get("observation_id"),
        "snapshot_version": envelope.get("snapshot_version"),
        "observed_sim_time_s": envelope.get("sim_time_s"),
    }


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def validate_binding(
    *,
    binding: Dict[str, Any],
    envelope: Dict[str, Any],
    historical_observation_ids: Set[str],
    run_id: str,
    max_age_steps: int,
    max_age_sim_s: float,
) -> None:
    """Reject a decision whose binding does not match the current snapshot.

    ``binding`` is the identity snapshot attached to the returned decision;
    ``envelope`` is the current live boundary observation. ``historical`` ids
    are observations already consumed by a completed decision.
    """
    if not isinstance(binding, dict):
        raise DecisionFreshnessRejected("binding_not_dict", f"got {type(binding).__name__}")
    if not isinstance(envelope, dict):
        raise DecisionFreshnessRejected("envelope_not_dict", f"got {type(envelope).__name__}")

    # -- run identity -----------------------------------------------------
    req_run = binding.get("run_id")
    if req_run != run_id or req_run != envelope.get("run_id"):
        raise DecisionFreshnessRejected(
            "run_id_mismatch", f"expected {run_id!r}, got {req_run!r}"
        )

    # -- observation identity --------------------------------------------
    req_obs = binding.get("observation_id")
    current_obs = envelope.get("observation_id")
    if not isinstance(req_obs, str) or not req_obs:
        raise DecisionFreshnessRejected("missing_observation_id", "observation_id is required")
    if req_obs in historical_observation_ids and req_obs != current_obs:
        raise DecisionFreshnessRejected(
            "stale_observation_id", f"{req_obs} consumed; current is {current_obs}"
        )
    if req_obs != current_obs:
        code = "stale_observation_id" if req_obs in historical_observation_ids else "invalid_observation_id"
        raise DecisionFreshnessRejected(
            code, f"{req_obs} does not match current {current_obs}"
        )

    # -- snapshot version window and age ----------------------------------
    entry_version = envelope.get("decision_entry_snapshot_version")
    current_version = envelope.get("snapshot_version")
    req_version = binding.get("snapshot_version")

    if req_version is not None:
        if _is_bool(req_version) or not isinstance(req_version, int):
            raise DecisionFreshnessRejected("invalid_snapshot_version", "snapshot_version must be an integer")
        if not isinstance(entry_version, int) or not isinstance(current_version, int):
            raise DecisionFreshnessRejected(
                "stale_or_invalid_snapshot_version", "current envelope lacks snapshot window"
            )
        if req_version < entry_version or req_version > current_version:
            raise DecisionFreshnessRejected(
                "stale_or_invalid_snapshot_version",
                f"snapshot_version {req_version} outside window [{entry_version}, {current_version}]",
            )
        age_steps = current_version - req_version
        if age_steps > int(max_age_steps):
            raise DecisionFreshnessRejected(
                "stale_observation_snapshot_expired",
                f"snapshot age {age_steps} steps exceeds max {max_age_steps}",
            )

    # -- sim time window and age ------------------------------------------
    req_sim_t = binding.get("observed_sim_time_s")
    current_sim_t = envelope.get("sim_time_s")
    entry_sim_t = envelope.get("decision_entry_sim_time_s")
    if req_sim_t is not None:
        if _is_bool(req_sim_t) or not isinstance(req_sim_t, (int, float)):
            raise DecisionFreshnessRejected(
                "invalid_observed_sim_time_s", "observed_sim_time_s must be a number"
            )
        if current_sim_t is None:
            raise DecisionFreshnessRejected(
                "stale_or_invalid_snapshot_sim_time", "current envelope has no sim_time_s"
            )
        entry_t = float(entry_sim_t) if entry_sim_t is not None else float(current_sim_t)
        if float(req_sim_t) < entry_t - 1e-6 or float(req_sim_t) > float(current_sim_t) + 1e-6:
            raise DecisionFreshnessRejected(
                "stale_or_invalid_snapshot_sim_time",
                f"observed_sim_time_s {req_sim_t} outside [{entry_t}, {current_sim_t}]",
            )
        age_sim = float(current_sim_t) - float(req_sim_t)
        if age_sim > float(max_age_sim_s) + 1e-6:
            raise DecisionFreshnessRejected(
                "stale_observation_snapshot_expired",
                f"sim time age {age_sim:.3f}s exceeds max {max_age_sim_s}s",
            )
