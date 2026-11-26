"""Read-only robot accessor bound to the live 2026 OmniGibson SDK.

The accessor forwards only post1-verified robot surface calls (see probe
evidence ``42_moma_caller_surface`` / ``43_moma_caller_detail``). It performs no
simulator import itself: the live robot object is duck-typed, which keeps this
layer CPU-testable while every SDK coupling stays behind the binding call.

Verified surface (post1 ``bd049de3``):

- ``arm_names``, ``action_dim``, ``controller_order``, ``controller_action_idx``
- ``_get_proprioception_dict()`` -> ``arm_<arm>_qpos/qvel``, ``eef_<arm>_pos/quat``,
  ``gripper_<arm>_qpos/qvel``, ``base_qvel``, ``trunk_qpos/qvel``
- ``get_relative_eef_pose(arm=...)``
- ``is_grasping(arm=..., candidate_obj=...)`` assisted-grasping readout
- assisted-grasping bookkeeping: ``_ag_obj_in_hand``, ``_ag_release_counter``,
  ``_ag_grasp_counter``, ``_ag_obj_constraints``, ``_ag_obj_constraint_params``.
  ``_ag_freeze_gripper`` does not exist on post1 ``bd049de3``; the record reports
  an UNKNOWN field rather than raising (canary attempt3 evidence).
- ``release_grasp_immediately(arm=...)``

Gap G2/G3: DOF order, frames and controller mode are re-verified against the
live R1Pro at the GPU canary; the accessor never invents joint slices.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..observer.three_can import clean, read_value


class AccessorUnboundError(RuntimeError):
    """Raised when an accessor method is called before a live robot is bound."""


@dataclass(frozen=True)
class GraspReadout:
    """SDK grasp state; ``FALSE`` maps to -1 and must never be cast with bool()."""

    name: str
    value: int


class SdkRobotAccessor:
    """Thin, read-mostly adapter over one live SDK robot."""

    def __init__(self, robot: Any = None) -> None:
        self._robot: Any = robot

    def bind(self, robot: Any) -> None:
        self._robot = robot

    @property
    def bound(self) -> bool:
        return self._robot is not None

    @property
    def robot(self) -> Any:
        if self._robot is None:
            raise AccessorUnboundError("robot accessor not bound")
        return self._robot

    # -- identity / controller layout -------------------------------------

    def arm_names(self) -> List[str]:
        return list(self.robot.arm_names)

    def action_dim(self) -> int:
        return int(self.robot.action_dim)

    def controller_order(self) -> List[str]:
        return list(self.robot.controller_order)

    def controller_action_idx(self) -> Dict[str, Any]:
        return {k: clean(v) for k, v in self.robot.controller_action_idx.items()}

    # -- proprioception ----------------------------------------------------

    def proprio_dict(self) -> Dict[str, Any]:
        return self.robot._get_proprioception_dict()

    def arm_qpos(self, arm: str) -> np.ndarray:
        return np.asarray(self.proprio_dict()[f"arm_{arm}_qpos"], dtype=float)

    def arm_qvel(self, arm: str) -> np.ndarray:
        return np.asarray(self.proprio_dict()[f"arm_{arm}_qvel"], dtype=float)

    def gripper_qpos(self, arm: str) -> np.ndarray:
        return np.asarray(self.proprio_dict()[f"gripper_{arm}_qpos"], dtype=float)

    def gripper_qvel(self, arm: str) -> np.ndarray:
        return np.asarray(self.proprio_dict()[f"gripper_{arm}_qvel"], dtype=float)

    def eef_pose(self, arm: str) -> Tuple[np.ndarray, np.ndarray]:
        pos, quat = self.robot.get_relative_eef_pose(arm=arm)
        return np.asarray(pos, dtype=float), np.asarray(quat, dtype=float)

    def base_qvel(self) -> np.ndarray:
        return np.asarray(self.proprio_dict().get("base_qvel"), dtype=float)

    def trunk_qpos(self) -> np.ndarray:
        return np.asarray(self.proprio_dict().get("trunk_qpos"), dtype=float)

    def trunk_qvel(self) -> np.ndarray:
        return np.asarray(self.proprio_dict().get("trunk_qvel"), dtype=float)

    def joint_qpos(self) -> np.ndarray:
        """Full robot joint position vector via the post1 method surface.

        Gap G3 (canary attempt3): post1 ``Robot`` exposes
        ``get_joint_positions()``; there is no ``.joint_positions`` property.
        This method reads the method, never a property guess.
        """
        return np.asarray(self.robot.get_joint_positions(), dtype=float)

    # -- grasp / assisted-grasping bookkeeping ----------------------------

    def grasp(self, arm: str, obj: Any = None) -> GraspReadout:
        state = self.robot.is_grasping(arm=arm, candidate_obj=obj)
        return GraspReadout(name=state.name, value=int(state))

    def object_in_hand(self, arm: str) -> Optional[str]:
        held = self.robot._ag_obj_in_hand[arm]
        return None if held is None else held.name

    def release_counter(self, arm: str) -> Optional[int]:
        return clean(self.robot._ag_release_counter[arm])

    def grasp_counter(self, arm: str) -> Optional[int]:
        return clean(self.robot._ag_grasp_counter[arm])

    def hand_record(self, arm: str) -> Dict[str, Any]:
        """Real consumer read of assisted-grasping state for one arm.

        ``valid_sdk_hold`` is a derived boolean: something is held AND the SDK
        grasp readout is TRUE. The raw counters remain untouched so release
        windows can be checked by the observer/terminal gate.
        """
        held_name = self.object_in_hand(arm)
        held_obj = self.robot._ag_obj_in_hand[arm]
        grasp_state = self.grasp(arm, held_obj) if held_obj is not None else self.grasp(arm)
        record: Dict[str, Any] = {
            "object_in_hand": held_name,
            "release_counter": self.release_counter(arm),
            "grasp_counter": self.grasp_counter(arm),
            "constraint_present": self.robot._ag_obj_constraints[arm] is not None,
            "constraint_params": clean(self.robot._ag_obj_constraint_params[arm]),
            "freeze_gripper": (clean(self.robot._ag_freeze_gripper[arm])
                               if hasattr(self.robot, "_ag_freeze_gripper")
                               else {"status": "UNKNOWN", "error": "field_absent_post1"}),
            "sdk_grasp_state": grasp_state.name,
            "valid_sdk_hold": held_name is not None and grasp_state.name == "TRUE",
        }
        return record

    def release(self, arm: str) -> None:
        self.robot.release_grasp_immediately(arm=arm)

    # -- failure-isolated reads -------------------------------------------

    def safe_hand_record(self, arm: str) -> Dict[str, Any]:
        return read_value(lambda arm=arm: self.hand_record(arm))
