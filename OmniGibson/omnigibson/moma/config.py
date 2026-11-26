"""Effective runner configuration frozen from the a28 successful baseline.

The a28 reference run is ``trash_agent_api_a28_fake_normal_20261115``
(condition ``normal_unperturbed``, policy mode ``external_s2``, assisted grasp,
fake rule S2 service). All numeric/object/capacity fields below are copied from
its frozen runtime/harness manifests; they are not re-tuned in this line.

Sim-free: only stdlib. No simulator, torch or cuRobo import.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .s2.contract import digest

_RUN_ID = re.compile(r"^[A-Za-z0-9_.-]{1,160}$")

# Frozen a28 object identities (BDDL instance names in the 2026 scene scope).
FROZEN_CAN_NAMES = ["can_of_soda_114", "can_of_soda_115", "can_of_soda_113"]
FROZEN_BUCKET_NAME = "trash_can_116"
FROZEN_TASK_NAME = "picking_up_trash"
FROZEN_ROBOT_NAME = "r1pro"

FROZEN_ALLOWED_PRIMITIVES = [
    "continue_nominal_tracking",
    "stabilize_and_brake",
    "open_right_gripper",
    "retract_to_pregrasp",
    "reanchor_to_live_object",
    "approach_object",
    "close_and_acquire_grasp",
]

FROZEN_PRIMITIVE_CAPS: Dict[str, int] = {
    "continue_nominal_tracking": 12000,
    "stabilize_and_brake": 120,
    "open_right_gripper": 120,
    "retract_to_pregrasp": 240,
    "reanchor_to_live_object": 240,
    "approach_object": 800,
    "close_and_acquire_grasp": 60,
}


@dataclass
class RunnerConfig:
    """Effective config for one single-env three-can episode."""

    run_id: str
    task_name: str = FROZEN_TASK_NAME
    robot_name: str = FROZEN_ROBOT_NAME
    can_names: List[str] = field(default_factory=lambda: list(FROZEN_CAN_NAMES))
    bucket_name: str = FROZEN_BUCKET_NAME
    condition: str = "normal_unperturbed"
    policy_mode: str = "external_s2"
    assisted_grasping: bool = True

    step_hz: int = 30
    physics_hz: int = 120
    max_steps: int = 7000
    experiment_timeout_s: float = 5400.0
    poll_s: float = 0.5
    startup_wait_s: float = 2400.0
    max_decisions: int = 24
    history_turns: int = 8
    snapshot_max_age_steps: int = 1800
    snapshot_max_age_sim_s: float = 60.0
    s2_wait_max_hold_steps: int = 1800

    allowed_primitives: List[str] = field(default_factory=lambda: list(FROZEN_ALLOWED_PRIMITIVES))
    primitive_caps: Dict[str, int] = field(default_factory=lambda: dict(FROZEN_PRIMITIVE_CAPS))

    model_kind: str = "fake_rule_service_not_llm"
    model_name: str = "fake-embodied-s2-native-v2"
    model_base_url: str = "http://127.0.0.1:10823/v1"
    model_temperature: float = 0.0
    model_timeout_s: float = 15.0
    model_max_response_bytes: int = 1_048_576
    model_api_key_env: Optional[str] = None

    # Same-source initial restore source: restored at most once before the
    # first action. In the 2026 stack these resolve through the local USD/state
    # subset cache manifest; NAS stays a persistent source only.
    initial_state_npz: Optional[str] = None
    initial_state_hdf5: Optional[str] = None
    asset_manifest_path: Optional[str] = None

    output_dir: Optional[str] = None
    write_video: bool = True

    def validate(self) -> List[str]:
        """Return a list of human-readable config problems; empty means valid."""
        problems: List[str] = []
        if not _RUN_ID.fullmatch(self.run_id) or ".." in self.run_id:
            problems.append("invalid_run_id")
        if not self.can_names:
            problems.append("can_names_empty")
        if self.bucket_name in self.can_names:
            problems.append("bucket_is_also_can")
        if self.step_hz <= 0 or self.physics_hz <= 0:
            problems.append("hz_must_be_positive")
        if self.max_steps <= 0:
            problems.append("max_steps_must_be_positive")
        if self.max_decisions <= 0:
            problems.append("max_decisions_must_be_positive")
        if self.policy_mode not in ("internal", "external_s2"):
            problems.append("policy_mode_unknown")
        unknown = [p for p in self.allowed_primitives if p not in FROZEN_ALLOWED_PRIMITIVES]
        if unknown:
            problems.append(f"unknown_primitives: {','.join(unknown)}")
        for name, cap in self.primitive_caps.items():
            if name not in self.allowed_primitives:
                problems.append(f"cap_without_primitive: {name}")
            if not isinstance(cap, int) or cap <= 0:
                problems.append(f"cap_invalid: {name}")
        if self.model_api_key_env is not None and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.model_api_key_env):
            problems.append("api_key_env_name_invalid")
        return problems

    def s2_cfg(self) -> Dict[str, Any]:
        """Contract view consumed by the S2 validator / terminal gate."""
        return {
            "can_names": list(self.can_names),
            "bucket_name": self.bucket_name,
            "allowed_primitives": list(self.allowed_primitives),
            "primitive_caps": dict(self.primitive_caps),
        }

    def effective_digest(self) -> str:
        """Stable sha256 over the effective (frozen-tuned) config."""
        return digest(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_name": self.task_name,
            "robot_name": self.robot_name,
            "can_names": list(self.can_names),
            "bucket_name": self.bucket_name,
            "condition": self.condition,
            "policy_mode": self.policy_mode,
            "assisted_grasping": self.assisted_grasping,
            "step_hz": self.step_hz,
            "physics_hz": self.physics_hz,
            "max_steps": self.max_steps,
            "experiment_timeout_s": self.experiment_timeout_s,
            "poll_s": self.poll_s,
            "startup_wait_s": self.startup_wait_s,
            "max_decisions": self.max_decisions,
            "history_turns": self.history_turns,
            "snapshot_max_age_steps": self.snapshot_max_age_steps,
            "snapshot_max_age_sim_s": self.snapshot_max_age_sim_s,
            "s2_wait_max_hold_steps": self.s2_wait_max_hold_steps,
            "allowed_primitives": list(self.allowed_primitives),
            "primitive_caps": dict(self.primitive_caps),
            "model_kind": self.model_kind,
            "model_name": self.model_name,
            "model_temperature": self.model_temperature,
            "model_timeout_s": self.model_timeout_s,
            "model_max_response_bytes": self.model_max_response_bytes,
            "model_api_key_env": self.model_api_key_env,
            "initial_state_npz": self.initial_state_npz,
            "initial_state_hdf5": self.initial_state_hdf5,
            "asset_manifest_path": self.asset_manifest_path,
            "write_video": self.write_video,
        }

    @classmethod
    def from_frozen_baseline(cls, run_id: str, **overrides: Any) -> "RunnerConfig":
        """Build the a28 effective config; only output/asset paths may override."""
        cfg = cls(run_id=run_id, **overrides)
        return cfg
