"""
SubTask Evaluator for BEHAVIOR-1K

This module provides subtask-level (primitive-level) evaluation for policies,
following the BRS paper evaluation protocol where:
1. If a subtask fails, reset to the start of the next subtask and continue evaluation
2. Report both subtask (ST) success rates and end-to-end (ET) success rates

This enables more granular evaluation of policy performance on long-horizon tasks.

Usage:
    python subtask_eval.py policy=websocket task.name=turning_on_radio \
        demo_data_path=/path/to/2025-challenge-demos \
        log_path=./eval_logs/subtask_eval

    # NOTE: `demo_data_path` must point to the `2025-challenge-demos` folder (it should contain `annotations` and `data` subfolders).

"""

import csv
import cv2
import h5py
import hydra
import json
import logging
import math
import numpy as np
import omnigibson as og
import omnigibson.utils.transform_utils as T
import os
import pandas as pd
import re
import sys
import torch as th
import traceback
from inspect import getsourcefile
from omegaconf import DictConfig, OmegaConf
from omnigibson.controllers import IsGraspingState
from omnigibson.learning.eval import Evaluator, m
from omnigibson.learning.utils.config_utils import register_omegaconf_resolvers
from omnigibson.learning.utils.eval_utils import (
    ROBOT_CAMERA_NAMES,
    PROPRIOCEPTION_INDICES,
    PROPRIO_QPOS_INDICES,
    TASK_NAMES_TO_INDICES,
    flatten_obs_dict,
)
from omnigibson.learning.utils.obs_utils import create_video_writer, write_video
from omnigibson.learning.transition_state_restore import (
    TransitionManifestError,
    build_restore_component_validity,
    materialize_transition_events,
    normalize_transition_manifest,
    require_exact_cache_artifacts,
    restore_method_is_exact,
    restore_recorded_scene_base,
    transition_events_before_state_frame,
    validate_serialized_state_consumption,
)
from omnigibson.macros import gm
from omnigibson.utils.python_utils import (
    create_object_from_init_info,
    get_uuid,
    h5py_group_to_torch,
    recursively_convert_to_torch,
)
from pathlib import Path
from signal import signal, SIGINT
from typing import Any, Dict, List, Mapping, Optional, Tuple


# create module logger
logger = logging.getLogger("subtask_evaluator")
logger.setLevel(20)  # info


# Raw HDF5 demonstrations record the source scene snapshot, typed initialization
# metadata, and exact frame-keyed object/system lifecycle metadata. Serialized state
# vectors contain UUIDs and values but not this schema contract, so the recorded scene
# and cumulative transitions must be restored before a mid-episode load. Support is
# artifact-driven rather than task-name-driven; incomplete metadata fails closed.


class SubTaskEvaluator(Evaluator):
    """
    SubTask Evaluator class that extends the base Evaluator for primitive-level evaluation.
    
    This evaluator supports:
    1. Loading primitive annotations from the dataset
    2. Evaluating policy on each primitive/subtask
    3. Resetting robot state to the beginning of the next primitive if current one fails
    4. Tracking both per-primitive success rates (ST) and end-to-end success rates (ET)
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        
        # SubTask evaluation specific attributes
        self.demo_data_path = cfg.get("demo_data_path", None)
        # Expect `demo_data_path` to point directly to the `2025-challenge-demos` folder.
        # The provided path will be used as the dataset root without searching parent directories.
        self.demo_data_root = None
        if self.demo_data_path is not None:
            self.demo_data_root = os.path.expanduser(self.demo_data_path)
            logger.info(f"Using demo data root: {self.demo_data_root}")
        else:
            logger.debug("No demo_data_path provided in config; demo annotations will not be available.")
        self.primitive_timeout_multiplier = cfg.get("primitive_timeout_multiplier", 3.0)
        self.use_primitive_eval = cfg.get("use_primitive_eval", True)

        # State restoration behavior
        # - reset_on_primitive_failure (existing behavior): only restore when previous primitive failed
        # - restore_at_each_primitive_start: restore at the start of every primitive (useful for debugging)
        self.restore_at_each_primitive_start = cfg.get("restore_at_each_primitive_start", False)

        # Optional cache of per-primitive full simulation states (serialized), extracted from raw rawdata.
        # If present, we can restore full world state without requiring the full rawdata HDF5 at eval-time.
        self.primitive_state_cache_dir = cfg.get("primitive_state_cache_dir", None)
        if self.primitive_state_cache_dir is None and self.demo_data_path is not None:
            base = getattr(self, "demo_data_root", self.demo_data_path)
            self.primitive_state_cache_dir = os.path.join(base, "meta", "primitive_states")
        self.current_primitive_state_cache = None
        
        # Track primitive-level metrics
        self.n_primitive_trials = 0
        self.n_primitive_successes = 0
        self.n_endtoend_trials = 0
        self.n_endtoend_successes = 0
        
        # Current episode tracking
        self.current_primitive_annotations = []
        self.current_primitive_idx = 0
        self.current_segment_metadata: Optional[Dict[str, Any]] = None
        self.current_demo_data = None
        self.current_demo_id = None
        self._last_rawdata_restore_result: Dict[str, Any] = {"attempted": False}
        self._last_cache_restore_result: Dict[str, Any] = {"attempted": False}
        self._last_restore_debug: Dict[str, Any] = {
            "selected_method": "none",
            "exact_world_state": False,
            "component_validity": build_restore_component_validity(
                restored=False,
                rigid_state_valid=False,
                assisted_grasp_state_valid=False,
                particle_state_valid=False,
                contact_state_valid=False,
            ),
        }

        # If a standalone segment starts while the demo is already holding a manipulating
        # object (e.g. a `place in` segment immediately after `pick up`), avoid calling
        # robot.keep_still() on gripper joints during segment-load stabilization. In
        # physical grasping mode, keep_still() clears joint effort and can drop the
        # restored object before replay starts.
        self.preserve_gripper_on_segment_start_grasp = cfg.get("preserve_gripper_on_segment_start_grasp", True)

        # Video annotation state
        self._video_primitive_desc = None
        self._video_primitive_progress = None
        self._video_primitive_idx = None
        self._video_n_primitives = None

        # Debug / reporting for primitive-level success criteria
        self._last_primitive_state_errors: Optional[Dict[str, float]] = None
        self._last_primitive_success_reason: Optional[str] = None
        
        logger.info("SubTaskEvaluator initialized with primitive evaluation support")

    def _record_rollout_trace(self, step_idx: int) -> None:
        """Record robot state for segment-level replay diagnostics."""
        if not hasattr(self, "rollout_trace"):
            self.rollout_trace = []
        base_pos, base_quat = self.robot.get_position_orientation()
        joint_qpos = self.robot.get_joint_positions()
        proprio = self.obs.get("robot_r1::proprio") if isinstance(self.obs, dict) else None
        if proprio is None:
            proprio = th.empty(0, dtype=th.float32)
        self.rollout_trace.append(
            {
                "step": int(step_idx),
                "base_pos": th.as_tensor(base_pos, dtype=th.float32).detach().cpu().numpy(),
                "base_quat": th.as_tensor(base_quat, dtype=th.float32).detach().cpu().numpy(),
                "joint_qpos": th.as_tensor(joint_qpos, dtype=th.float32).detach().cpu().numpy(),
                "proprio": th.as_tensor(proprio, dtype=th.float32).detach().cpu().numpy(),
            }
        )

    def save_rollout_trace(self, path: str) -> None:
        """Persist collected segment rollout robot-state trace as an npz file."""
        if not self.cfg.get("save_rollout", False):
            return
        if not getattr(self, "rollout_trace", None):
            logger.warning("No rollout trace was collected; skip saving rollout npz.")
            return
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            step=np.asarray([x["step"] for x in self.rollout_trace], dtype=np.int64),
            base_pos=np.asarray([x["base_pos"] for x in self.rollout_trace], dtype=np.float32),
            base_quat=np.asarray([x["base_quat"] for x in self.rollout_trace], dtype=np.float32),
            joint_qpos=np.asarray([x["joint_qpos"] for x in self.rollout_trace], dtype=np.float32),
            proprio=np.asarray([x["proprio"] for x in self.rollout_trace], dtype=np.float32),
        )
        logger.info(f"Saved rollout trace to {path}")

    @staticmethod
    def _parse_restore_exception_details(exc: Exception) -> Dict[str, Any]:
        details: Dict[str, Any] = {
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        }
        match = re.search(r"hash_key\s+(?P<hash_key>\w+):\s+(?P<value>\d+)", str(exc))
        if match:
            details["missing_hash_key"] = match.group("hash_key")
            details["missing_hash_value"] = int(match.group("value"))
        return details

    def _set_last_rawdata_restore_result(
        self,
        *,
        attempted: bool,
        restored: bool,
        frame_idx: int,
        hdf5_path: Optional[str],
        stage: Optional[str] = None,
        reason: Optional[str] = None,
        exception: Optional[Exception] = None,
        selected_key: Optional[str] = None,
        available_groups: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "attempted": attempted,
            "restored": restored,
            "frame_idx": int(frame_idx),
            "hdf5_path": hdf5_path,
        }
        if stage is not None:
            payload["stage"] = stage
        if reason is not None:
            payload["reason"] = reason
        if selected_key is not None:
            payload["selected_demo_group"] = selected_key
        if available_groups is not None:
            payload["available_demo_groups"] = list(available_groups)
        if exception is not None:
            payload.update(self._parse_restore_exception_details(exception))
        if extra is not None:
            payload.update(extra)
        self._last_rawdata_restore_result = payload

    #region debug-point restore-grasp-state
    def _debug_log_grasp_state(self, tag: str) -> None:
        try:
            robot: Any = self.robot
            all_qpos = robot.get_joint_positions()
            all_qvel = robot.get_joint_velocities()
            for arm in getattr(robot, "arm_names", []):
                controller = robot.controllers.get(f"gripper_{arm}", None)
                qpos = all_qpos[robot.gripper_control_idx[arm]]
                qvel = all_qvel[robot.gripper_control_idx[arm]]
                logger.info(
                    "[debug][%s] grasp_mode=%s arm=%s is_grasping=%s qpos=%s qvel=%s goal=%s control=%s",
                    tag,
                    getattr(robot, "grasping_mode", None),
                    arm,
                    str(robot.is_grasping(arm=arm)),
                    np.asarray(qpos).tolist(),
                    np.asarray(qvel).tolist(),
                    getattr(controller, "_goal", None),
                    getattr(controller, "_control", None),
                )
        except Exception as e:
            logger.info("[debug][%s] failed to inspect grasp state: %s", tag, e)
    #endregion debug-point restore-grasp-state

    @staticmethod
    def _flatten_annotation_strings(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        if isinstance(value, (list, tuple)):
            out: List[str] = []
            for item in value:
                out.extend(SubTaskEvaluator._flatten_annotation_strings(item))
            return out
        text = str(value).strip()
        return [text] if text else []

    def _segment_start_grasp_metadata(self, frame_idx: Optional[int]) -> Tuple[bool, List[str], str]:
        """Return whether current segment metadata implies a held object at segment start.

        The BEHAVIOR skill metadata does not store arm-level physical contacts, but it
        does identify the object being manipulated by each segment. For release-style
        segments (place / put / pour / insert), a non-empty manipulating_object_id means
        the segment starts after the object has already been picked up, so segment-load
        stabilization should preserve gripper commands even if physical is_grasping()
        is briefly UNKNOWN/FALSE immediately after deserializing raw state.
        """
        if not bool(getattr(self, "preserve_gripper_on_segment_start_grasp", True)):
            return False, [], "disabled"

        segment = getattr(self, "current_segment_metadata", None)
        if not isinstance(segment, dict):
            return False, [], "no_current_segment_metadata"

        frame_duration = segment.get("frame_duration") or []
        if frame_idx is not None:
            try:
                start_frame = int(frame_duration[0])
            except Exception:
                return False, [], "invalid_frame_duration"
            if int(frame_idx) != start_frame:
                return False, [], "not_segment_start"

        manipulating_objects = self._flatten_annotation_strings(segment.get("manipulating_object_id"))
        if not manipulating_objects:
            return False, [], "no_manipulating_object_id"

        desc_values: List[str] = []
        for key in ("skill_description", "primitive_description", "segment_description", "description"):
            desc_values.extend(self._flatten_annotation_strings(segment.get(key)))
        desc = " ".join(desc_values).strip().lower()
        if desc.startswith(("pick", "pick up", "grasp", "grab")):
            return False, manipulating_objects, f"pickup_segment:{desc}"
        if desc.startswith(("place", "put", "pour", "insert", "drop", "release")):
            return True, manipulating_objects, f"release_segment:{desc}"

        # Conservative fallback: only trust manipulating_object_id as a start-grasp
        # signal for non-navigation / non-pickup segments.
        skill_types = {x.lower() for x in self._flatten_annotation_strings(segment.get("skill_type"))}
        if "navigation" not in skill_types:
            return True, manipulating_objects, f"manipulating_non_navigation:{desc or sorted(skill_types)}"
        return False, manipulating_objects, f"navigation_segment:{desc}"

    def _stabilize_robot_post_restore(self, frame_idx: Optional[int] = None) -> None:
        """
        Stabilize the robot after rawdata restore without clearing gripper holding force.

        For physical grasping, calling robot.keep_still() will zero all joint efforts, which can
        immediately drop an object that is already being held at the restored frame. In that case,
        only freeze the base and non-gripper joints.
        """
        try:
            robot: Any = self.robot
            grasping_mode = getattr(robot, "grasping_mode", None)
            grasping_arms = [
                arm for arm in getattr(robot, "arm_names", []) if robot.is_grasping(arm=arm) == IsGraspingState.TRUE
            ]
            segment_start_grasp, segment_objects, segment_reason = self._segment_start_grasp_metadata(frame_idx)
        except Exception:
            grasping_mode = None
            grasping_arms = []
            segment_start_grasp = False
            segment_objects = []
            segment_reason = "exception"

        if grasping_mode == "physical" and (grasping_arms or segment_start_grasp):
            try:
                robot.set_linear_velocity(th.zeros(3))
                robot.set_angular_velocity(th.zeros(3))
                joint_vel = robot.get_joint_velocities().clone()
                all_qpos = robot.get_joint_positions()
                protected_arms = list(grasping_arms)
                if not protected_arms and segment_start_grasp:
                    for arm in getattr(robot, "arm_names", []):
                        try:
                            qpos = all_qpos[robot.gripper_control_idx[arm]]
                            if bool(float(th.min(qpos).item()) < 0.04):
                                protected_arms.append(arm)
                        except Exception:
                            continue
                if not protected_arms and segment_start_grasp:
                    protected_arms = list(getattr(robot, "arm_names", []))
                protected_idx = set()
                for arm in protected_arms:
                    protected_idx.update(robot.gripper_control_idx[arm].tolist())
                non_gripper_idx = [i for i in range(joint_vel.shape[0]) if i not in protected_idx]
                if len(non_gripper_idx) > 0:
                    joint_vel[non_gripper_idx] = 0.0
                    robot.set_joint_velocities(joint_vel, drive=False)
                logger.info(
                    "[debug][post_restore_stabilize] preserving physical grasp on arms=%s objects=%s reason=%s by skipping keep_still on gripper joints",
                    protected_arms,
                    segment_objects,
                    segment_reason if segment_start_grasp else "runtime_is_grasping",
                )
                return
            except Exception as e:
                logger.info("[debug][post_restore_stabilize] selective stabilize failed, fallback to keep_still: %s", e)

        self.robot.keep_still()

    def _maybe_recover_assisted_grasp_post_restore(self, tag: str) -> None:
        robot: Any = self.robot
        if getattr(robot, "grasping_mode", None) == "physical":
            return

        recovered = []
        for arm in getattr(robot, "arm_names", []):
            try:
                if robot.is_grasping(arm=arm) != IsGraspingState.FALSE:
                    continue
            except Exception:
                continue

            try:
                contact_data, _ = robot._find_gripper_contacts(arm=arm, return_contact_positions=True)
                raycast_paths = sorted(robot._find_gripper_raycast_collisions(arm=arm))
                if getattr(robot, "grasping_mode", None) == "assisted":
                    ag_data = robot._calculate_in_hand_object_rigid(arm=arm, restore_fallback=True)
                else:
                    ag_data = robot._calculate_in_hand_object(arm=arm)
                contact_paths = sorted({prim_path for prim_path, _ in contact_data})
                logger.info(
                    "[debug][%s] assisted probe arm=%s contact_paths=%s raycast_paths=%s rigid_candidate=%s",
                    tag,
                    arm,
                    contact_paths,
                    raycast_paths,
                    None if ag_data is None else getattr(ag_data[1], "prim_path", None),
                )
            except Exception as e:
                logger.info("[debug][%s] assisted regrasp probe failed for arm=%s: %s", tag, arm, e)
                continue

            if ag_data is None and getattr(robot, "grasping_mode", None) == "assisted":
                try:
                    qpos = robot.get_joint_positions()[robot.gripper_control_idx[arm]]
                    gripper_closed = bool(float(th.min(qpos).item()) < 0.04)
                except Exception:
                    gripper_closed = False

                if gripper_closed and len(raycast_paths) == 1:
                    candidate_path = raycast_paths[0]
                    obj = robot.scene.object_registry("prim_path", "/".join(candidate_path.split("/")[:-1]))
                    if obj is not None:
                        link_name = candidate_path.split("/")[-1]
                        link = obj.links.get(link_name, None)
                        if link is not None:
                            ag_data = (obj, link)
                            logger.info(
                                "[debug][%s] restore-only raycast fallback arm=%s candidate=%s qpos=%s",
                                tag,
                                arm,
                                candidate_path,
                                np.asarray(qpos).tolist(),
                            )

            if ag_data is None:
                continue

            try:
                contact_pos = None
                obj, link = ag_data
                target_prim_path = link.prim_path
                target_points = [point for prim_path, point in contact_data if prim_path == target_prim_path]
                if len(target_points) > 0:
                    stacked = th.stack(target_points)
                    contact_pos = stacked.mean(dim=0)
                else:
                    contact_pos = link.get_position_orientation()[0]
                    logger.info(
                        "[debug][%s] assisted regrasp using fallback contact_pos=link_center for arm=%s link=%s",
                        tag,
                        arm,
                        target_prim_path,
                    )
                robot._establish_grasp(arm=arm, ag_data=ag_data, contact_pos=contact_pos)
                recovered.append(arm)
            except Exception as e:
                logger.info("[debug][%s] assisted regrasp establish failed for arm=%s: %s", tag, arm, e)

        if recovered:
            logger.info("[debug][%s] recovered assisted grasp on arms=%s", tag, recovered)

    def _get_cfg_float(self, key: str, default: float) -> float:
        val = self.cfg.get(key, default)
        if val is None:
            return float(default)
        try:
            return float(val)
        except Exception:
            return float(default)

    def get_primitive_success_thresholds(self) -> Dict[str, float]:
        """Return thresholds used by the primitive-level state-match success check."""
        return {
            "base_pos": self._get_cfg_float("primitive_success_base_pos_threshold", 0.15),
            "yaw": self._get_cfg_float("primitive_success_yaw_threshold", 0.35),
            "eef_pos": self._get_cfg_float("primitive_success_eef_pos_threshold", 0.12),
            "gripper_qpos": self._get_cfg_float("primitive_success_gripper_qpos_threshold", 0.03),
            "std_joint_qpos_rmse": self._get_cfg_float("primitive_success_std_joint_qpos_rmse_threshold", 0.25),
            # Fallback when EEF slices are missing / not comparable
            "joint_qpos_rmse": self._get_cfg_float("primitive_success_joint_qpos_rmse_threshold", 0.25),
        }

    @staticmethod
    def _wrap_to_pi(x: float) -> float:
        # Map angle to [-pi, pi]
        return (x + math.pi) % (2.0 * math.pi) - math.pi

    def _get_demo_state_at_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        if self.current_demo_data is None:
            return None
        if frame_idx < 0 or frame_idx >= len(self.current_demo_data):
            return None
        try:
            state = self.current_demo_data.iloc[frame_idx]["observation.state"]
        except Exception:
            return None
        try:
            arr = np.asarray(state, dtype=np.float32).reshape(-1)
        except Exception:
            return None
        return arr

    def _get_current_proprio_state(self) -> Optional[np.ndarray]:
        """Get current robot proprio state vector (should match demo `observation.state` layout)."""
        if getattr(self, "obs", None) is None:
            return None
        proprio = self.obs.get("robot_r1::proprio", None)
        if proprio is None:
            return None
        try:
            # torch tensor
            if hasattr(proprio, "detach"):
                proprio = proprio.detach().cpu().numpy()
            arr = np.asarray(proprio, dtype=np.float32).reshape(-1)
        except Exception:
            return None
        return arr

    def compute_primitive_state_errors(self, primitive: Dict) -> Optional[Dict[str, float]]:
        """Compute state-match errors between current state and the demo end state of this primitive."""
        if self.current_demo_data is None:
            return None
        cur = self._get_current_proprio_state()
        if cur is None:
            return None

        try:
            _, end_frame = primitive["frame_duration"]
        except Exception:
            return None

        end_frame = int(end_frame)
        end_frame = max(0, min(end_frame, len(self.current_demo_data) - 1))
        tgt = self._get_demo_state_at_frame(end_frame)
        if tgt is None or tgt.shape != cur.shape:
            return None

        # Core pose / arm / gripper errors (task-agnostic, but aligned to demo boundaries)
        robot_pos_slice = PROPRIOCEPTION_INDICES["R1Pro"].get("robot_pos", None)
        robot_yaw_slice = PROPRIOCEPTION_INDICES["R1Pro"].get("robot_2d_ori", None)
        eef_left_slice = PROPRIOCEPTION_INDICES["R1Pro"].get("eef_left_pos", None)
        eef_right_slice = PROPRIOCEPTION_INDICES["R1Pro"].get("eef_right_pos", None)
        grip_left_slice = PROPRIOCEPTION_INDICES["R1Pro"].get("gripper_left_qpos", None)
        grip_right_slice = PROPRIOCEPTION_INDICES["R1Pro"].get("gripper_right_qpos", None)
        joint_qpos_slice = PROPRIOCEPTION_INDICES["R1Pro"].get("joint_qpos", None)

        errors: Dict[str, float] = {}

        if robot_pos_slice is not None:
            errors["base_pos_err"] = float(np.linalg.norm(cur[robot_pos_slice] - tgt[robot_pos_slice]))
        if robot_yaw_slice is not None:
            dyaw = float(cur[robot_yaw_slice][0] - tgt[robot_yaw_slice][0])
            errors["yaw_err"] = abs(self._wrap_to_pi(dyaw))

        if eef_left_slice is not None:
            errors["eef_left_pos_err"] = float(np.linalg.norm(cur[eef_left_slice] - tgt[eef_left_slice]))
        if eef_right_slice is not None:
            errors["eef_right_pos_err"] = float(np.linalg.norm(cur[eef_right_slice] - tgt[eef_right_slice]))
        if grip_left_slice is not None:
            errors["gripper_left_qpos_err"] = float(np.linalg.norm(cur[grip_left_slice] - tgt[grip_left_slice]))
        if grip_right_slice is not None:
            errors["gripper_right_qpos_err"] = float(np.linalg.norm(cur[grip_right_slice] - tgt[grip_right_slice]))

        if joint_qpos_slice is not None:
            dq = cur[joint_qpos_slice] - tgt[joint_qpos_slice]
            errors["joint_qpos_rmse"] = float(np.sqrt(np.mean(np.square(dq))))

            # "Standard-track" subset of qpos (excludes privileged / global base pose).
            # These indices are defined relative to the `joint_qpos` vector layout.
            try:
                cur_qpos = cur[joint_qpos_slice]
                tgt_qpos = tgt[joint_qpos_slice]
                std_parts_cur = []
                std_parts_tgt = []
                for sl in PROPRIO_QPOS_INDICES.get("R1Pro", {}).values():
                    std_parts_cur.append(cur_qpos[sl])
                    std_parts_tgt.append(tgt_qpos[sl])
                if std_parts_cur and std_parts_tgt:
                    std_cur = np.concatenate([np.asarray(x, dtype=np.float32).reshape(-1) for x in std_parts_cur])
                    std_tgt = np.concatenate([np.asarray(x, dtype=np.float32).reshape(-1) for x in std_parts_tgt])
                    if std_cur.shape == std_tgt.shape and std_cur.size > 0:
                        dstd = std_cur - std_tgt
                        errors["std_joint_qpos_rmse"] = float(np.sqrt(np.mean(np.square(dstd))))
            except Exception:
                pass

        return errors

    def _unwrap_env(self):
        """Return the innermost OmniGibson Environment, unwrapping any EnvironmentWrapper layers."""
        env = self.env
        # Unwrap a handful of times in case wrappers are stacked
        for _ in range(8):
            inner = getattr(env, "env", None)
            if inner is None:
                break
            env = inner
        return env

    def _get_obs_for_policy(self) -> dict:
        """Get a raw observation dict suitable for policy consumption.

        Important: EnvironmentWrapper subclasses may only inject extra fields (e.g. WBVIMAWrapper adds task obs)
        inside reset()/step(). Calling env.get_obs() bypasses those hooks, so we replicate the injection here.
        """
        raw = self.env.get_obs()
        obs = raw[0] if isinstance(raw, (tuple, list)) else raw
        if not isinstance(obs, dict):
            raise TypeError(f"Expected dict obs from env.get_obs(), got {type(obs)}")

        # If task info isn't present (common when calling get_obs directly), inject it from the base env.
        if "task" not in obs:
            base_env = self._unwrap_env()
            task = getattr(base_env, "task", None)
            if task is not None:
                try:
                    task_obs = task.get_obs(base_env)
                except Exception:
                    task_obs = None
                if isinstance(task_obs, dict) and len(task_obs) > 0:
                    obs = dict(obs)
                    obs["task"] = task_obs
        return obs

    def load_rawdata_hdf5(self, demo_id: str) -> Optional[h5py.File]:
        """
        Load raw HDF5 data file for a specific demo if available.
        
        The raw HDF5 files contain complete simulation states that can be used
        for exact state restoration.
        
        Args:
            demo_id: Demo ID string (e.g., "00000010")
            
        Returns:
            h5py.File object if found, None otherwise
        """
        rawdata_path = self.cfg.get("rawdata_path", None)
        if rawdata_path is None:
            return None
            
        task_idx = TASK_NAMES_TO_INDICES[self.cfg.task.name]
        task_name = f"task-{task_idx:04d}"

        episode_names = self._get_episode_name_candidates(demo_id)

        # Try common raw data file patterns, including zero-padded episode ids.
        possible_paths = [
            os.path.join(rawdata_path, task_name, f"{episode_name}.hdf5")
            for episode_name in episode_names
        ] + [
            os.path.join(rawdata_path, f"{episode_name}.hdf5")
            for episode_name in episode_names
        ] + [
            os.path.join(rawdata_path, task_name, f"{demo_id}.hdf5"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found raw HDF5 data at {path}")
                return h5py.File(path, "r")
                
        logger.debug(f"Raw HDF5 data not found for demo {demo_id}")
        return None

    def _restore_serialized_state_with_transition_manifest(
        self,
        *,
        state_tensor: th.Tensor,
        raw_manifest: Any,
        recorded_scene_file: Any,
        init_metadata: Dict[str, th.Tensor],
        state_frame: int,
        source: str,
    ) -> Dict[str, Any]:
        """Restore the recorded clean scene, reconstruct its lifecycle, then load state.

        Segment evaluation can request frames in non-monotonic order (start, end, then
        rollout start). Every request therefore starts from the HDF5-recorded scene
        snapshot and per-object initialization metadata used by official playback,
        never the evaluator's current ``scene._initial_file``.
        """

        manifest_present = raw_manifest is not None
        manifest = normalize_transition_manifest(raw_manifest)
        events = transition_events_before_state_frame(manifest, state_frame)
        base_env = self._unwrap_env()
        scene = base_env.scene

        with gm.unlocked():
            transition_rules_were_enabled = bool(gm.ENABLE_TRANSITION_RULES)
            # Official DataPlaybackWrapper requires transition rules to be disabled
            # because lifecycle events are replayed explicitly. Temporarily do the same
            # so the initialization step cannot independently fire a duplicate rule.
            gm.ENABLE_TRANSITION_RULES = False

        operation_succeeded = False
        summary: Dict[str, Any] = {}
        try:
            summary = restore_recorded_scene_base(
                scene=scene,
                raw_scene_file=recorded_scene_file,
                init_metadata=init_metadata,
                stopped_context=og.sim.stopped,
            )
            # Scene.restore's existing object/system callbacks keep BehaviorTask BDDL
            # entities bound as names are removed and re-added. Same-name entities retain
            # their live object identity, matching official DataPlaybackWrapper.
            summary.update(
                materialize_transition_events(
                    scene=scene,
                    events=events,
                    create_object_from_init_info=create_object_from_init_info,
                    uuid_from_name=lambda name: get_uuid(name, deterministic=True),
                    park_object=lambda obj, index: obj.set_position(th.ones(3) * (100.0 + 5.0 * index)),
                    initialize_event=og.sim.step,
                )
            )
            # Inspect source component presence with the same current-runtime schema used
            # by official policy evaluation. This is telemetry only; current asset MD5
            # identity is deliberately not an official-eval compatibility gate.
            source_state, source_consumed = og.sim.deserialize(state_tensor)
            source_consumption = validate_serialized_state_consumption(
                consumed=source_consumed,
                total=state_tensor.numel(),
            )
            source_scene_state = source_state.get(scene.idx)
            if source_scene_state is None and len(source_state) == 1:
                source_scene_state = next(iter(source_state.values()))
            if not isinstance(source_scene_state, Mapping):
                raise TransitionManifestError("serialized_source_scene_state_missing", scene_idx=scene.idx)
            source_registry = source_scene_state.get("registry", {})
            if not isinstance(source_registry, Mapping):
                raise TransitionManifestError("serialized_source_registry_missing", scene_idx=scene.idx)
            source_object_registry = source_registry.get("object_registry", {})
            source_system_registry = source_registry.get("system_registry", {})
            if not isinstance(source_object_registry, Mapping) or not isinstance(
                source_system_registry, Mapping
            ):
                raise TransitionManifestError("serialized_source_component_registry_missing")
            source_robot_state = source_object_registry.get(self.robot.name, {})
            source_has_assisted_grasp = isinstance(source_robot_state, Mapping) and (
                "ag_obj_constraint_params" in source_robot_state
            )
            assisted_grasp_valid = bool(
                getattr(self.robot, "grasping_mode", None) == "physical" or source_has_assisted_grasp
            )
            source_system_count = len(source_system_registry)

            # Serialized deserialization is intentionally last: every UUID referenced
            # by the vector must already resolve through the reconstructed registries.
            og.sim.load_state(state_tensor, serialized=True)
            component_validity = build_restore_component_validity(
                restored=True,
                rigid_state_valid=True,
                assisted_grasp_state_valid=assisted_grasp_valid,
                # Phase 1 is conservative: serialized system payloads require a future
                # round-trip GPU gate before particle-dependent segment metrics are valid.
                particle_state_valid=source_system_count == 0,
                contact_state_valid=False,
                historical_asset_identity_verified=False,
                source_assisted_grasp_state_present=source_has_assisted_grasp,
                source_system_count=source_system_count,
            )
            summary.update(
                {
                    "source_deserialize_consumed_length": int(source_consumed),
                    "source_deserialization": source_consumption,
                    "component_validity": component_validity,
                    "official_eval_asset_compatible": True,
                    "exact_world_state": component_validity["historical_world_state_exact"],
                }
            )
            operation_succeeded = True
        finally:
            with gm.unlocked():
                gm.ENABLE_TRANSITION_RULES = transition_rules_were_enabled
            if transition_rules_were_enabled and scene.transition_rule_api is not None:
                try:
                    scene.transition_rule_api.refresh_all_rules()
                except Exception:
                    if operation_succeeded:
                        raise
                    logger.exception("Failed to refresh transition rules after failed manifest restore")

        summary.update(
            {
                "transition_manifest_present": manifest_present,
                "transition_manifest_event_count": len(manifest),
                "transition_manifest_applied_event_count": len(events),
                "transition_state_frame": int(state_frame),
                "transition_source": source,
            }
        )
        return summary

    def restore_full_state_from_rawdata(self, hdf5_file: h5py.File, frame_idx: int) -> bool:
        """
        Restore full simulation state from raw HDF5 data.
        
        This provides more accurate state restoration by loading the complete
        simulation state (all object poses, robot state, etc.) from the raw data.
        
        Args:
            hdf5_file: Open h5py.File containing raw demo data
            frame_idx: Frame index to restore state from
            
        Returns:
            True if state was successfully restored, False otherwise
        """
        hdf5_path = getattr(hdf5_file, "filename", None)
        self._set_last_rawdata_restore_result(
            attempted=True,
            restored=False,
            frame_idx=frame_idx,
            hdf5_path=hdf5_path,
            stage="begin",
            reason="started",
        )
        manifest_present = False
        transition_summary: Dict[str, Any] = {}
        try:
            # Find the demo group.
            # Most files contain a single demo group, but some rawdata files include
            # a short bootstrap / warmup group (e.g. demo_0) plus the full episode
            # in a later group (e.g. demo_1). Selecting demo_0 unconditionally can
            # make valid frame indices look out-of-range and force an unnecessary
            # robot-only fallback.
            data_grp = hdf5_file["data"]
            demo_keys = [k for k in data_grp.keys() if k.startswith("demo_")]
            if not demo_keys:
                logger.warning("No demo groups found in HDF5 file")
                self._set_last_rawdata_restore_result(
                    attempted=True,
                    restored=False,
                    frame_idx=frame_idx,
                    hdf5_path=hdf5_path,
                    stage="select_demo_group",
                    reason="no_demo_keys",
                )
                return False

            demo_candidates = []
            target_len = len(self.current_demo_data) if getattr(self, "current_demo_data", None) is not None else None
            for key in demo_keys:
                grp = data_grp[key]
                if "state" not in grp or "state_size" not in grp:
                    continue
                state_len = len(grp["state"])
                fits_frame = frame_idx < state_len
                len_gap = abs(state_len - target_len) if target_len is not None else state_len
                demo_candidates.append((not fits_frame, len_gap, -state_len, key, grp, state_len))

            if not demo_candidates:
                logger.warning("No state data found in HDF5 file")
                self._set_last_rawdata_restore_result(
                    attempted=True,
                    restored=False,
                    frame_idx=frame_idx,
                    hdf5_path=hdf5_path,
                    stage="select_demo_group",
                    reason="no_state_data_groups",
                    available_groups=demo_keys,
                )
                return False

            _, _, _, selected_key, demo_grp, selected_len = min(demo_candidates)
            if selected_key != demo_keys[0]:
                logger.info(
                    "Selected rawdata demo group %s for frame %s (available groups=%s, selected_len=%s)",
                    selected_key,
                    frame_idx,
                    demo_keys,
                    selected_len,
                )
            
            # Check if state data is available
            if "state" not in demo_grp:
                logger.warning("No state data found in HDF5 file")
                self._set_last_rawdata_restore_result(
                    attempted=True,
                    restored=False,
                    frame_idx=frame_idx,
                    hdf5_path=hdf5_path,
                    stage="load_state",
                    reason="missing_state_dataset",
                    selected_key=selected_key,
                    available_groups=demo_keys,
                )
                return False
                
            state = demo_grp["state"]
            state_size = demo_grp["state_size"]
            
            if frame_idx >= len(state):
                logger.warning(f"Frame {frame_idx} exceeds state data length {len(state)}")
                self._set_last_rawdata_restore_result(
                    attempted=True,
                    restored=False,
                    frame_idx=frame_idx,
                    hdf5_path=hdf5_path,
                    stage="load_state",
                    reason="frame_exceeds_state_length",
                    selected_key=selected_key,
                    available_groups=demo_keys,
                )
                return False
                
            # Reconstruct the exact registry lifecycle before serialized UUID lookup.
            state_data = th.tensor(state[frame_idx, :int(state_size[frame_idx])])
            manifest_present = "transitions" in demo_grp.attrs
            if not manifest_present:
                raise TransitionManifestError("transition_manifest_missing")
            raw_manifest = demo_grp.attrs["transitions"]
            recorded_scene_file = data_grp.attrs.get("scene_file")
            if "init_metadata" not in demo_grp:
                raise TransitionManifestError("recorded_init_metadata_missing")
            init_metadata = h5py_group_to_torch(demo_grp["init_metadata"])
            logger.info(
                "Restoring FULL world state from rawdata frame %s "
                "(serialized, recorded_scene=%s, transition_manifest=%s)",
                frame_idx,
                recorded_scene_file is not None,
                manifest_present,
            )
            transition_summary = self._restore_serialized_state_with_transition_manifest(
                state_tensor=state_data,
                raw_manifest=raw_manifest,
                recorded_scene_file=recorded_scene_file,
                init_metadata=init_metadata,
                state_frame=frame_idx,
                source=f"rawdata:{hdf5_path}:{selected_key}",
            )
            self._debug_log_grasp_state(f"rawdata_restore_frame_{frame_idx}_after_load_state")
            self._maybe_recover_assisted_grasp_post_restore(f"rawdata_restore_frame_{frame_idx}_after_load_state")
            
            # Step physics to stabilize
            for _ in range(5):
                og.sim.step_physics()
                try:
                    self._stabilize_robot_post_restore(frame_idx=frame_idx)
                except Exception:
                    pass
            self._maybe_recover_assisted_grasp_post_restore(f"rawdata_restore_frame_{frame_idx}_after_keep_still")
            self._debug_log_grasp_state(f"rawdata_restore_frame_{frame_idx}_after_keep_still")

            # Render to refresh sensors / video
            for _ in range(2):
                og.sim.render()
            self._maybe_recover_assisted_grasp_post_restore(f"rawdata_restore_frame_{frame_idx}_after_render")
            self._debug_log_grasp_state(f"rawdata_restore_frame_{frame_idx}_after_render")
                
            logger.info(f"Restored full simulation state from frame {frame_idx}")
            self._set_last_rawdata_restore_result(
                attempted=True,
                restored=True,
                frame_idx=frame_idx,
                hdf5_path=hdf5_path,
                stage="restored",
                reason="ok",
                selected_key=selected_key,
                available_groups=demo_keys,
                extra=transition_summary,
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore full state from rawdata: {e}")
            traceback.print_exc()
            if isinstance(e, TransitionManifestError):
                if e.reason.startswith("recorded_"):
                    stage = "prepare_recorded_scene"
                elif e.reason.startswith("exact_cache_"):
                    stage = "precheck"
                elif e.reason.startswith("serialized_source_"):
                    stage = "deserialize_precheck"
                else:
                    stage = "materialize_transitions"
                reason = e.reason
                extra = {
                    "exact_world_state": False,
                    "transition_manifest_present": manifest_present,
                    "transition_manifest_error_details": e.details,
                }
            else:
                stage = "deserialize"
                missing_entity = "Could not find object" in str(e) or "Could not find" in str(e)
                reason = (
                    "transition_manifest_missing_for_serialized_state"
                    if missing_entity and not manifest_present
                    else "exception"
                )
                extra = {
                    "exact_world_state": False,
                    "transition_manifest_present": manifest_present,
                    **transition_summary,
                }
            self._set_last_rawdata_restore_result(
                attempted=True,
                restored=False,
                frame_idx=frame_idx,
                hdf5_path=hdf5_path,
                stage=stage,
                reason=reason,
                exception=e,
                extra=extra,
            )
            return False

    def _try_restore_to_frame(self, frame_idx: int) -> Tuple[bool, str]:
        """Try to restore state for the given demo frame.

        Restore order:
          1) rawdata full state (preferred)
          2) primitive-state cache full state
          3) robot-only proprioception

        Returns:
            (restored, method) where method is one of: rawdata | cache | robot | none
        """
        unavailable_component_validity = build_restore_component_validity(
            restored=False,
            rigid_state_valid=False,
            assisted_grasp_state_valid=False,
            particle_state_valid=False,
            contact_state_valid=False,
        )
        debug: Dict[str, Any] = {
            "frame_idx": int(frame_idx),
            "selected_method": "none",
            "fallback_used": False,
            "exact_world_state": False,
            "component_validity": unavailable_component_validity,
            "rawdata": {"attempted": False, "restored": False},
            "cache": {"attempted": False, "restored": False},
            "robot": {"attempted": False, "restored": False, "exact_world_state": False},
        }

        # 1) rawdata. Support is determined by the artifact's exact transition
        # manifest, never by a task-name denylist.
        if getattr(self, "current_rawdata_hdf5", None) is not None:
            debug["rawdata"]["attempted"] = True
            if self.restore_full_state_from_rawdata(self.current_rawdata_hdf5, frame_idx):
                debug["rawdata"] = dict(getattr(self, "_last_rawdata_restore_result", {}) or {})
                debug["selected_method"] = "rawdata"
                debug["exact_world_state"] = restore_method_is_exact("rawdata", debug["rawdata"])
                debug["component_validity"] = debug["rawdata"].get(
                    "component_validity", unavailable_component_validity
                )
                self._last_restore_debug = debug
                return True, "rawdata"
            debug["rawdata"] = dict(getattr(self, "_last_rawdata_restore_result", {}) or {})
            logger.warning("Rawdata state restore failed; falling back without exact-world semantics")

        # 2) cache. Only schema v3 carries the recorded source-scene contract;
        # legacy caches are diagnostic artifacts and cannot be exact restores.
        if getattr(self, "current_primitive_state_cache", None) is not None:
            debug["cache"]["attempted"] = True
            if self.restore_full_state_from_cache(frame_idx):
                debug["cache"] = dict(getattr(self, "_last_cache_restore_result", {}) or {})
                debug["selected_method"] = "cache"
                debug["fallback_used"] = bool(debug["rawdata"].get("attempted"))
                debug["exact_world_state"] = restore_method_is_exact("cache", debug["cache"])
                debug["component_validity"] = debug["cache"].get(
                    "component_validity", unavailable_component_validity
                )
                self._last_restore_debug = debug
                return True, "cache"
            debug["cache"] = dict(getattr(self, "_last_cache_restore_result", {}) or {})

        # 3) robot-only remains a diagnostic fallback and is explicitly non-exact.
        debug["robot"]["attempted"] = True
        if self.restore_robot_state_from_frame(frame_idx):
            debug["robot"]["restored"] = True
            debug["selected_method"] = "robot"
            debug["fallback_used"] = bool(debug["rawdata"].get("attempted") or debug["cache"].get("attempted"))
            debug["exact_world_state"] = restore_method_is_exact("robot", debug["robot"])
            self._last_restore_debug = debug
            return True, "robot"

        self._last_restore_debug = debug
        return False, "none"

    @staticmethod
    def _draw_text_with_background(
        img,
        text: str,
        position,
        font_scale: float = 0.8,
        thickness: int = 2,
        text_color=(255, 255, 255),
        bg_color=(0, 0, 0),
    ):
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x, y = position
        cv2.rectangle(img, (x, y - text_height - baseline), (x + text_width, y + baseline), bg_color, -1)
        cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)

    @staticmethod
    def _draw_progress_bar(
        img,
        progress: float,
        position,
        width: int,
        height: int,
        color=(0, 255, 0),
        bg_color=(50, 50, 50),
    ):
        x, y = position
        progress = float(np.clip(progress, 0.0, 1.0))
        cv2.rectangle(img, (x, y), (x + width, y + height), bg_color, -1)
        fill_width = int(width * progress)
        cv2.rectangle(img, (x, y), (x + fill_width, y + height), color, -1)

    def _write_video(self) -> None:
        """Write the current robot observations to video, with primitive overlay."""
        if getattr(self, "_video_writer", None) is None:
            return
        if ROBOT_CAMERA_NAMES["R1Pro"]["head"] + "::rgb" not in self.obs:
            return

        left_wrist_rgb = cv2.resize(
            self.obs[ROBOT_CAMERA_NAMES["R1Pro"]["left_wrist"] + "::rgb"].numpy(),
            (224, 224),
        )
        right_wrist_rgb = cv2.resize(
            self.obs[ROBOT_CAMERA_NAMES["R1Pro"]["right_wrist"] + "::rgb"].numpy(),
            (224, 224),
        )
        head_rgb = cv2.resize(
            self.obs[ROBOT_CAMERA_NAMES["R1Pro"]["head"] + "::rgb"].numpy(),
            (448, 448),
        )

        composite = np.hstack([np.vstack([left_wrist_rgb, right_wrist_rgb]), head_rgb]).copy()

        # Overlay primitive info (similar style to tutorials/annotate_video.py)
        if self._video_primitive_desc is not None and self._video_primitive_progress is not None:
            idx_txt = ""
            if self._video_primitive_idx is not None and self._video_n_primitives is not None:
                idx_txt = f" ({self._video_primitive_idx}/{self._video_n_primitives})"
            self._draw_text_with_background(
                composite,
                f"Primitive{idx_txt}: {self._video_primitive_desc}",
                (20, 40),
                bg_color=(0, 100, 0),
            )
            self._draw_progress_bar(
                composite,
                float(self._video_primitive_progress),
                (20, 50),
                300,
                10,
                color=(50, 205, 50),
            )

        write_video(
            np.expand_dims(composite, 0),
            video_writer=self.video_writer,
            batch_size=1,
            mode="rgb",
        )

    def _primitive_state_cache_paths(self, demo_id: str) -> List[str]:
        if self.primitive_state_cache_dir is None:
            return []
        task_idx = TASK_NAMES_TO_INDICES[self.cfg.task.name]
        task_name = f"task-{task_idx:04d}"
        return [
            os.path.join(self.primitive_state_cache_dir, task_name, f"{episode_name}.npz")
            for episode_name in self._get_episode_name_candidates(demo_id)
        ]

    def load_primitive_state_cache(self, demo_id: str) -> Optional[dict]:
        """Load per-primitive serialized states extracted from raw data.

        Expected file format is the one produced by
        `OmniGibson/scripts/learning/extract_primitive_world_states.py`.
        """
        cache_candidates = self._primitive_state_cache_paths(demo_id)
        cache_path = next((p for p in cache_candidates if os.path.exists(p)), None)
        if cache_path is None:
            return None
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                source = "primitive_state_cache"
                if "source" in data.files:
                    try:
                        src_val = data["source"]
                        source = str(src_val.item() if hasattr(src_val, "item") else src_val)
                    except Exception:
                        source = "primitive_state_cache"
                schema_version = int(data["schema_version"].item()) if "schema_version" in data.files else 1
                transition_manifest_present = "transition_manifest_json" in data.files
                transition_manifest_json = None
                if transition_manifest_present:
                    manifest_value = data["transition_manifest_json"]
                    transition_manifest_json = str(
                        manifest_value.item() if hasattr(manifest_value, "item") else manifest_value
                    )
                recorded_scene_file_json = None
                if "recorded_scene_file_json" in data.files:
                    scene_value = data["recorded_scene_file_json"]
                    recorded_scene_file_json = str(
                        scene_value.item() if hasattr(scene_value, "item") else scene_value
                    )
                init_metadata: Dict[str, th.Tensor] = {}
                if "init_metadata_keys_json" in data.files:
                    keys_value = data["init_metadata_keys_json"]
                    metadata_keys = json.loads(
                        str(keys_value.item() if hasattr(keys_value, "item") else keys_value)
                    )
                    for index, key in enumerate(metadata_keys):
                        array_key = f"init_metadata_{index:04d}"
                        if array_key not in data.files:
                            raise ValueError(f"Cache missing {array_key} for init metadata {key}")
                        init_metadata[key] = th.from_numpy(np.asarray(data[array_key]).copy())
                cache = {
                    "frame_indices": data["frame_indices"].astype(np.int64),
                    "state_sizes": data["state_sizes"].astype(np.int64),
                    "state_offsets": data["state_offsets"].astype(np.int64),
                    "state_flat": np.asarray(data["state_flat"]),
                    "source": source,
                    "cache_path": cache_path,
                    "schema_version": schema_version,
                    "transition_manifest_present": transition_manifest_present,
                    "transition_manifest_json": transition_manifest_json,
                    "recorded_scene_file_json": recorded_scene_file_json,
                    "init_metadata": init_metadata,
                }
            logger.info(
                "Loaded primitive state cache from %s (%s snapshots, schema=%s, transition_manifest=%s)",
                cache_path,
                len(cache["frame_indices"]),
                cache["schema_version"],
                cache["transition_manifest_present"],
            )
            return cache
        except Exception as e:
            logger.warning(f"Failed to load primitive state cache at {cache_path}: {e}")
            return None

    def restore_full_state_from_cache(self, frame_idx: int, max_frame_delta: int = 0) -> bool:
        """Restore a serialized frame from a transition-aware cache.

        Nearest-frame substitution is disabled by default. Schema-v3 caches carry the
        recorded scene snapshot, typed initialization metadata, and transition manifest
        required for component-valid restoration. Strict historical exactness is reported
        separately and can remain false when AG, particle, or asset provenance is absent.
        Legacy caches fail closed.
        """
        cache = getattr(self, "current_primitive_state_cache", None)
        if cache is None:
            self._last_cache_restore_result = {"attempted": False, "restored": False}
            return False

        self._last_cache_restore_result = {
            "attempted": True,
            "restored": False,
            "frame_idx": int(frame_idx),
            "cache_path": cache.get("cache_path"),
            "cache_schema_version": int(cache.get("schema_version", 1)),
            "transition_manifest_present": bool(cache.get("transition_manifest_present", False)),
            "exact_world_state": False,
        }
        frame_indices = cache["frame_indices"]
        pos = int(np.searchsorted(frame_indices, frame_idx))
        candidates = []
        if pos < len(frame_indices):
            candidates.append(pos)
        if pos - 1 >= 0:
            candidates.append(pos - 1)

        best = None
        best_delta = 1_000_000_000
        for idx in candidates:
            delta = abs(int(frame_indices[idx]) - int(frame_idx))
            if best is None or delta < best_delta:
                best = idx
                best_delta = delta

        if best is None or best_delta > max_frame_delta:
            self._last_cache_restore_result.update(
                {
                    "stage": "select_frame",
                    "reason": "exact_cache_frame_missing",
                    "nearest_frame_delta": int(best_delta),
                }
            )
            logger.debug("Primitive state cache exact-frame miss for frame %s", frame_idx)
            return False

        cached_frame = int(frame_indices[best])
        transition_summary: Dict[str, Any] = {}
        try:
            require_exact_cache_artifacts(cache)
            offset = int(cache["state_offsets"][best])
            size = int(cache["state_sizes"][best])
            state_vec = np.asarray(cache["state_flat"][offset : offset + size])
            state_tensor = th.from_numpy(state_vec)
            transition_summary = self._restore_serialized_state_with_transition_manifest(
                state_tensor=state_tensor,
                raw_manifest=cache.get("transition_manifest_json"),
                recorded_scene_file=cache.get("recorded_scene_file_json"),
                init_metadata=cache.get("init_metadata", {}),
                state_frame=cached_frame,
                source=str(cache.get("source", "primitive_state_cache")),
            )
            for _ in range(5):
                og.sim.step_physics()
                try:
                    self._stabilize_robot_post_restore(frame_idx=cached_frame)
                except Exception:
                    pass

            for _ in range(2):
                og.sim.render()
            self._last_cache_restore_result.update(
                {
                    "restored": True,
                    "stage": "restored",
                    "reason": "ok",
                    "cached_frame": cached_frame,
                    "nearest_frame_delta": int(best_delta),
                    **transition_summary,
                }
            )
            logger.info("Restored component-classified simulation state from cache frame %s", cached_frame)
            return True
        except Exception as e:
            logger.error(f"Failed to restore full state from primitive cache: {e}")
            traceback.print_exc()
            if isinstance(e, TransitionManifestError):
                if e.reason.startswith("recorded_"):
                    stage = "prepare_recorded_scene"
                elif e.reason.startswith("exact_cache_"):
                    stage = "precheck"
                elif e.reason.startswith("serialized_source_"):
                    stage = "deserialize_precheck"
                else:
                    stage = "materialize_transitions"
                reason = e.reason
                error_details = e.details
            else:
                stage = "deserialize"
                missing_entity = "Could not find object" in str(e) or "Could not find" in str(e)
                reason = (
                    "transition_manifest_missing_for_serialized_state"
                    if missing_entity and not cache.get("transition_manifest_present", False)
                    else "exception"
                )
                error_details = {}
            self._last_cache_restore_result.update(
                {
                    "stage": stage,
                    "reason": reason,
                    "cached_frame": cached_frame,
                    "transition_manifest_error_details": error_details,
                    **self._parse_restore_exception_details(e),
                    **transition_summary,
                }
            )
            return False

    def load_demo_annotations(self, demo_id: str) -> Optional[Dict]:
        """
        Load primitive annotations for a specific demo.
        
        Args:
            demo_id: Demo ID string (e.g., "00000010")
            
        Returns:
            Dict containing primitive_annotation and skill_annotation, or None if not found
        """
        if self.demo_data_path is None:
            logger.warning("demo_data_path not set, cannot load annotations")
            return None
            
        task_idx = TASK_NAMES_TO_INDICES[self.cfg.task.name]
        task_name = f"task-{task_idx:04d}"

        base = getattr(self, "demo_data_root", self.demo_data_path)

        episode_names = self._get_episode_name_candidates(demo_id)

        annotation_candidates = [
            os.path.join(base, "annotations", task_name, f"{episode_name}.json")
            for episode_name in episode_names
        ]

        annotation_path = next((p for p in annotation_candidates if os.path.exists(p)), None)
        if annotation_path is None:
            logger.warning(
                "Annotation file not found. Tried: %s",
                annotation_candidates,
            )
            return None
            
        with open(annotation_path, "r") as f:
            annotations = json.load(f)
            
        logger.info(f"Loaded annotations from {annotation_path}")
        logger.info(f"  - {len(annotations.get('skill_annotation', []))} skills")
        logger.info(f"  - {len(annotations.get('primitive_annotation', []))} primitives")
        
        return annotations

    @staticmethod
    def _get_episode_name_candidates(demo_id: str) -> List[str]:
        demo_id_int = None
        try:
            demo_id_int = int(demo_id)
        except (TypeError, ValueError):
            pass

        episode_names = [f"episode_{demo_id}"]
        if demo_id_int is not None:
            padded_episode_name = f"episode_{demo_id_int:08d}"
            if padded_episode_name not in episode_names:
                episode_names.append(padded_episode_name)
        return episode_names

    def load_demo_lowdim_data(self, demo_id: str) -> Optional[pd.DataFrame]:
        """
        Load low-dimensional data (observation.state, actions) from parquet file.
        
        Args:
            demo_id: Demo ID string (e.g., "00000010")
            
        Returns:
            DataFrame containing the demo data, or None if not found
        """
        if self.demo_data_path is None:
            logger.warning("demo_data_path not set, cannot load demo data")
            return None
            
        task_idx = TASK_NAMES_TO_INDICES[self.cfg.task.name]
        task_name = f"task-{task_idx:04d}"
        
        base = getattr(self, "demo_data_root", self.demo_data_path)
        parquet_candidates = [
            os.path.join(base, "data", task_name, f"{episode_name}.parquet")
            for episode_name in self._get_episode_name_candidates(demo_id)
        ]
        parquet_path = next((p for p in parquet_candidates if os.path.exists(p)), None)

        if parquet_path is None:
            logger.warning("Parquet file not found. Tried: %s", parquet_candidates)
            return None
            
        df = pd.read_parquet(parquet_path)
        logger.info(f"Loaded demo data from {parquet_path}, {len(df)} frames")
        
        return df

    def restore_robot_state_from_frame(self, frame_idx: int) -> bool:
        """
        Restore robot state from a specific frame in the demo data.
        
        This sets the robot's joint positions based on the proprioception data
        stored in observation.state at the given frame.
        
        For R1Pro robot, the observation.state has 256 dimensions containing:
        - joint_qpos [0:28]: Full robot joint positions (first 6 are base virtual joints)
        - robot_pos [140:143]: Global position
        - robot_2d_ori [149:150]: 2D global orientation (yaw)
        
        Args:
            frame_idx: Frame index to restore state from
            
        Returns:
            True if state was successfully restored, False otherwise
        """
        if self.current_demo_data is None:
            logger.warning("No demo data loaded, cannot restore state")
            return False
            
        if frame_idx < 0 or frame_idx >= len(self.current_demo_data):
            logger.warning(f"Frame index {frame_idx} out of range [0, {len(self.current_demo_data)})")
            return False
            
        try:
            # Get the observation state at the target frame
            obs_state = np.array(self.current_demo_data.iloc[frame_idx]["observation.state"])
            
            # Extract joint positions from the observation state
            # Based on PROPRIOCEPTION_INDICES for R1Pro:
            # joint_qpos is at indices 0:28 (full robot joint positions)
            # First 6 are base virtual joints (x, y, z, rx, ry, rz)
            joint_qpos_slice = PROPRIOCEPTION_INDICES["R1Pro"]["joint_qpos"]
            joint_qpos = th.tensor(obs_state[joint_qpos_slice], dtype=th.float32)
            
            # Extract robot position from the observation state
            # robot_pos is at indices 140:143
            robot_pos_slice = PROPRIOCEPTION_INDICES["R1Pro"]["robot_pos"]
            robot_pos = th.tensor(obs_state[robot_pos_slice], dtype=th.float32)
            
            # Get 2D orientation (yaw) from indices 149:150
            robot_2d_ori_slice = PROPRIOCEPTION_INDICES["R1Pro"]["robot_2d_ori"]
            robot_2d_ori = obs_state[robot_2d_ori_slice][0]  # yaw angle
            
            # Convert 2D orientation to quaternion (rotation around z-axis)
            # Quaternion format is (x, y, z, w)
            half_yaw = robot_2d_ori / 2.0
            robot_quat = th.tensor([0, 0, np.sin(half_yaw), np.cos(half_yaw)], dtype=th.float32)
            
            # First, set the robot base pose (this handles the virtual base joints internally)
            self.robot.set_position_orientation(robot_pos, robot_quat)
            
            # Then set all joint positions
            # The HolonomicBaseRobot's set_position_orientation handles base_idx joints,
            # but we still need to set the remaining joints (torso, arms, grippers)
            # Set all joints at once - the robot will handle base joints internally
            self.robot.set_joint_positions(joint_qpos, drive=False)
            
            # Keep robot still to avoid drifting
            self._stabilize_robot_post_restore(frame_idx=frame_idx)
            
            # Step physics a few times to stabilize
            for _ in range(10):
                og.sim.step_physics()
                self._stabilize_robot_post_restore(frame_idx=frame_idx)
                
            # Render to update visual state
            for _ in range(3):
                og.sim.render()
                
            logger.info(f"Restored robot state from frame {frame_idx}")
            logger.debug(f"  Position: {robot_pos}")
            logger.debug(f"  Yaw: {robot_2d_ori:.4f} rad")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore robot state: {e}")
            traceback.print_exc()
            return False

    def get_primitive_timeout(self, primitive: Dict) -> int:
        """
        Calculate timeout steps for a primitive based on its duration in demo.
        
        Args:
            primitive: Primitive annotation dict with 'frame_duration'
            
        Returns:
            Number of steps before timeout (primitive duration * multiplier)
        """
        start, end = primitive["frame_duration"]
        duration = end - start
        timeout = int(duration * self.primitive_timeout_multiplier)
        return max(timeout, 100)  # Minimum 100 steps

    def check_primitive_success(
        self,
        primitive: Dict,
        current_step: int,
        terminated: bool,
        timeout_steps: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """
        Check if the current primitive has been completed successfully.
        
        This is a simplified check - in practice you may want to add
        task-specific success conditions.
        
        Args:
            primitive: Current primitive annotation
            current_step: Current step within this primitive
            terminated: Whether the environment has terminated (task success)
            
        Returns:
            Tuple of (is_done, result) where result is "success", "timeout", or "failed"
        """
        timeout = int(timeout_steps) if timeout_steps is not None else self.get_primitive_timeout(primitive)

        # Reset debug state each call
        self._last_primitive_success_reason = None
        self._last_primitive_state_errors = None

        if terminated:
            # Environment termination corresponds to full task completion.
            self._last_primitive_success_reason = "env_terminated"
            return True, "success_env"

        # Primitive-level success: state-match against demo end frame.
        # This is intentionally generic / task-agnostic, and can be refined per-task.
        if bool(self.cfg.get("primitive_success_use_state_match", True)):
            errors = self.compute_primitive_state_errors(primitive)
            self._last_primitive_state_errors = errors
            if errors is not None:
                thr = self.get_primitive_success_thresholds()
                # Primary success check: match the standard-track joint subset.
                std_rmse = errors.get("std_joint_qpos_rmse", float("inf"))
                if np.isfinite(std_rmse) and std_rmse <= thr["std_joint_qpos_rmse"]:
                    self._last_primitive_success_reason = "state_match_std_joint_qpos"
                    return True, "success_state"

                # Secondary: EEF/gripper-based matching if available.
                eef_errs = [
                    errors.get("eef_left_pos_err", float("inf")),
                    errors.get("eef_right_pos_err", float("inf")),
                ]
                grip_errs = [
                    errors.get("gripper_left_qpos_err", float("inf")),
                    errors.get("gripper_right_qpos_err", float("inf")),
                ]
                has_eef = any(np.isfinite(x) for x in eef_errs)
                has_grip = any(np.isfinite(x) for x in grip_errs)
                if has_eef and has_grip:
                    eef_ok = min(eef_errs) <= thr["eef_pos"]
                    grip_ok = min(grip_errs) <= thr["gripper_qpos"]
                    if eef_ok and grip_ok:
                        self._last_primitive_success_reason = "state_match_eef_gripper"
                        return True, "success_state"

                # Last-resort fallback: full joint RMSE.
                jq = errors.get("joint_qpos_rmse", float("inf"))
                if np.isfinite(jq) and jq <= thr["joint_qpos_rmse"]:
                    self._last_primitive_success_reason = "state_match_joint_rmse"
                    return True, "success_state"

        if current_step >= timeout:
            return True, "timeout"

        return False, "in_progress"

    def step_primitive(self, primitive: Dict, max_steps: Optional[int] = None) -> Tuple[bool, str]:
        """
        Execute steps for a single primitive until success or timeout.
        
        Args:
            primitive: Primitive annotation dict
            max_steps: Optional max steps override
            
        Returns:
            Tuple of (success, result_type) where result_type is "success", "timeout", or "error"
        """
        if max_steps is None:
            max_steps = self.get_primitive_timeout(primitive)
            
        primitive_desc = primitive.get("primitive_description", ["unknown"])[0]
        logger.info(f"Starting primitive: {primitive_desc} (timeout: {max_steps} steps)")

        # Video overlay state
        self._video_primitive_desc = primitive_desc
        self._video_primitive_progress = 0.0
        
        for step in range(max_steps):
            terminated, truncated = self.step()

            self._video_primitive_progress = (step + 1) / float(max_steps)

            if self.cfg.write_video:
                self._write_video()

            if truncated:
                # Environment truncated (shouldn't happen within primitive)
                logger.warning(f"Environment truncated during primitive '{primitive_desc}'")
                return False, "truncated"

            done, result_type = self.check_primitive_success(
                primitive=primitive,
                current_step=step + 1,
                terminated=terminated,
                timeout_steps=max_steps,
            )
            if done:
                success = str(result_type).startswith("success")
                if success:
                    logger.info(f"Primitive '{primitive_desc}' succeeded at step {step}")
                else:
                    logger.info(f"Primitive '{primitive_desc}' finished with result={result_type} at step {step}")
                return success, str(result_type)
                
        logger.info(f"Primitive '{primitive_desc}' timed out after {max_steps} steps")
        return False, "timeout"

    def run_primitive_evaluation(
        self,
        demo_id: str,
        reset_on_failure: bool = True
    ) -> Dict[str, Any]:
        """
        Run primitive-level evaluation on a demo trajectory.
        
        Following the BRS paper protocol:
        - Evaluate each primitive in sequence
        - If a primitive fails, optionally reset to start of next primitive
        - Track both per-primitive and end-to-end success rates
        
        Args:
            demo_id: Demo ID to use for annotations and state restoration
            reset_on_failure: If True, reset to next primitive start on failure
            
        Returns:
            Dict with evaluation results including:
            - primitive_results: List of (primitive_desc, success, result_type)
            - n_primitives: Total number of primitives
            - n_primitive_successes: Number of successful primitives  
            - endtoend_success: Whether all primitives succeeded
        """
        # Load annotations and demo data
        annotations = self.load_demo_annotations(demo_id)
        if annotations is None:
            logger.error(f"Could not load annotations for demo {demo_id}")
            return {"error": "no_annotations"}
            
        self.current_demo_data = self.load_demo_lowdim_data(demo_id)
        if self.current_demo_data is None:
            logger.error(f"Could not load demo data for demo {demo_id}")
            return {"error": "no_demo_data"}
            
        self.current_demo_id = demo_id
        
        # Try to load raw HDF5 data for more accurate state restoration
        self.current_rawdata_hdf5 = self.load_rawdata_hdf5(demo_id)
        use_rawdata = self.current_rawdata_hdf5 is not None
        if use_rawdata:
            logger.info("Using raw HDF5 data for state restoration (more accurate)")
        else:
            # Try primitive-state cache next (full world state without rawdata dependency)
            self.current_primitive_state_cache = self.load_primitive_state_cache(demo_id)
            if self.current_primitive_state_cache is not None:
                logger.info("Using primitive state cache for state restoration (full world state)")
            else:
                logger.info("Using proprioception data for state restoration (robot only)")
        
        # Get primitive annotations
        primitives = annotations.get("primitive_annotation", [])
        if not primitives:
            logger.warning("No primitive annotations found, falling back to episodic evaluation")
            return {"error": "no_primitives"}
            
        # Sort primitives by start frame
        primitives = sorted(primitives, key=lambda x: x["frame_duration"][0])

        # Store counts for video overlay
        self._video_n_primitives = len(primitives)
        
        results = {
            "primitive_results": [],
            "n_primitives": len(primitives),
            "n_primitive_successes": 0,
            "endtoend_success": True,
        }
        
        logger.info(f"Starting primitive evaluation with {len(primitives)} primitives")
        
        for i, primitive in enumerate(primitives):
            start_frame, end_frame = primitive["frame_duration"]
            primitive_desc = primitive.get("primitive_description", ["unknown"])[0]
            self.current_segment_metadata = dict(primitive)

            self._video_primitive_idx = i + 1
            
            logger.info(f"\n{'='*50}")
            logger.info(f"Primitive {i+1}/{len(primitives)}: {primitive_desc}")
            logger.info(f"Frames: {start_frame} - {end_frame}")
            logger.info(f"{'='*50}")
            
            # Decide whether to restore at the start of this primitive
            should_restore = False
            reason = None
            if self.restore_at_each_primitive_start:
                should_restore = True
                reason = "restore_at_each_primitive_start=true"
            elif i > 0 and reset_on_failure and not results["primitive_results"][-1][1]:
                should_restore = True
                reason = "previous_primitive_failed"

            # Ensure initial state restoration (non-fatal) is attempted for the first primitive
            # so that the very first observation sent to the agent reflects the saved demo state.
            if i == 0 and not should_restore:
                logger.info(f"Attempting initial state restore to primitive start frame {start_frame} (non-fatal)")
                restored_init, method_init = self._try_restore_to_frame(start_frame)
                logger.info(f"Initial state restore result: restored={restored_init} method={method_init}")
                if not restored_init:
                    logger.info("Initial state restore not available; proceeding with current world state")

            if should_restore:
                logger.info(f"Restoring state to primitive start frame {start_frame} ({reason})")
                restored, method = self._try_restore_to_frame(start_frame)
                logger.info(f"State restore result: restored={restored} method={method}")
                if not restored:
                    logger.error("Failed to restore state")
                    results["primitive_results"].append((primitive_desc, False, "restore_failed"))
                    results["endtoend_success"] = False
                    continue
            else:
                # If we attempted an initial non-fatal restore above, that was logged;
                logger.info("No state restoration at this primitive start (continuing from current world state)")
            
            # Reset policy for new primitive
            self.policy.reset()
            
            # Get fresh observation after potential state restoration
            self.obs = self._preprocess_obs(self._get_obs_for_policy())
            
            # Execute primitive
            success, result_type = self.step_primitive(primitive)
            
            # Record results
            results["primitive_results"].append((primitive_desc, success, result_type))
            
            if success:
                results["n_primitive_successes"] += 1
            else:
                results["endtoend_success"] = False
                
            self.n_primitive_trials += 1
            if success:
                self.n_primitive_successes += 1
                
        # Update end-to-end metrics
        self.n_endtoend_trials += 1
        if results["endtoend_success"]:
            self.n_endtoend_successes += 1
        
        # Cleanup raw data file handle
        if hasattr(self, 'current_rawdata_hdf5') and self.current_rawdata_hdf5 is not None:
            self.current_rawdata_hdf5.close()
            self.current_rawdata_hdf5 = None

        self.current_primitive_state_cache = None
        self.current_segment_metadata = None
            
        return results

    def __exit__(self, exc_type, exc_value, exc_tb):
        logger.info("")
        logger.info("=" * 60)
        logger.info("SUBTASK EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Primitive (ST) Results:")
        logger.info(f"  - Total primitives: {self.n_primitive_trials}")
        logger.info(f"  - Successful primitives: {self.n_primitive_successes}")
        if self.n_primitive_trials > 0:
            logger.info(f"  - ST Success rate: {self.n_primitive_successes / self.n_primitive_trials:.2%}")
        logger.info(f"End-to-End (ET) Results:")
        logger.info(f"  - Total episodes: {self.n_endtoend_trials}")
        logger.info(f"  - Successful episodes: {self.n_endtoend_successes}")
        if self.n_endtoend_trials > 0:
            logger.info(f"  - ET Success rate: {self.n_endtoend_successes / self.n_endtoend_trials:.2%}")
        logger.info("=" * 60)
        logger.info("")
        
        # Also print regular episodic metrics from parent
        super().__exit__(exc_type, exc_value, exc_tb)


def get_demo_ids_for_task(demo_data_path: str, task_name: str, limit: Optional[int] = None) -> List[str]:
    """
    Get list of available demo IDs for a task.
    
    Args:
        demo_data_path: Path to demo data folder
        task_name: Name of the task
        limit: Optional limit on number of demos to return
        
    Returns:
        List of demo ID strings
    """
    task_idx = TASK_NAMES_TO_INDICES[task_name]
    task_folder = f"task-{task_idx:04d}"
    
    # Expect `demo_data_path` to point directly to the `2025-challenge-demos` folder.
    annotations_path = os.path.join(demo_data_path, "annotations", task_folder)
    if not os.path.exists(annotations_path):
        logger.warning(f"Annotations folder not found: {annotations_path}")
        return []
        
    demo_ids = []
    for fname in sorted(os.listdir(annotations_path)):
        if fname.endswith(".json"):
            # Extract demo ID from filename (e.g., "episode_00000010.json" -> "00000010")
            demo_id = fname.replace("episode_", "").replace(".json", "")
            demo_ids.append(demo_id)
            
    if limit is not None:
        demo_ids = demo_ids[:limit]
        
    return demo_ids


if __name__ == "__main__":
    register_omegaconf_resolvers()
    
    # Load config
    src = getsourcefile(lambda: 0) or __file__
    with hydra.initialize_config_dir(f"{Path(src).parents[0]}/configs", version_base="1.1"):
        config = hydra.compose("eval_subtask_reset_config.yaml", overrides=sys.argv[1:])
    OmegaConf.resolve(config)
    
    # Set headless mode
    with gm.unlocked():
        gm.HEADLESS = config.headless
    
    # Set video path
    if config.write_video:
        video_path = Path(config.log_path).expanduser() / "videos"
        video_path.mkdir(parents=True, exist_ok=True)
        
    # Get demo IDs to evaluate
    demo_data_path = config.get("demo_data_path", None)
    if demo_data_path is None:
        logger.error("demo_data_path must be specified for subtask evaluation")
        sys.exit(1)
        
    explicit_demo_ids = config.get("demo_ids", None)
    if explicit_demo_ids is not None:
        if isinstance(explicit_demo_ids, (str, int)):
            demo_ids = [explicit_demo_ids]
        else:
            demo_ids = [str(demo_id) for demo_id in explicit_demo_ids]
    else:
        demo_ids = get_demo_ids_for_task(
            demo_data_path=demo_data_path,
            task_name=config.task.name,
            limit=config.get("num_demos", None)
        )
    
    if not demo_ids:
        logger.error(f"No demos found for task {config.task.name}")
        sys.exit(1)
        
    logger.info(f"Found {len(demo_ids)} demos for task {config.task.name}")
    
    # Metrics storage
    all_results = []
    metrics_path = Path(config.log_path).expanduser() / "metrics"
    metrics_path.mkdir(parents=True, exist_ok=True)
    
    with SubTaskEvaluator(config) as evaluator:
        logger.info("Starting subtask evaluation...")
        
        for demo_idx, demo_id in enumerate(demo_ids):
            logger.info(f"\n{'#'*60}")
            logger.info(f"Evaluating demo {demo_idx + 1}/{len(demo_ids)}: {demo_id}")
            logger.info(f"{'#'*60}")
            
            # Reset environment and load task instance
            evaluator.reset()
            
            # For subtask evaluation, we need to use the demo's original task instance
            # The demo_id encodes the instance (demo_id // 10 % 1000)
            instance_id = int(demo_id) // 10 % 1000
            evaluator.load_task_instance(instance_id, test_hidden=config.test_hidden)
            
            # Setup video writer
            video_name = None
            if config.write_video:
                video_name = str(video_path) + f"/{config.task.name}_demo{demo_id}.mp4"
                evaluator.video_writer = create_video_writer(
                    fpath=video_name,
                    resolution=(448, 672),
                )
            
            # Reset after loading instance
            evaluator.reset()
            
            # Run primitive evaluation
            results = evaluator.run_primitive_evaluation(
                demo_id=demo_id,
                reset_on_failure=config.get("reset_on_primitive_failure", True)
            )
            
            results["demo_id"] = demo_id
            results["instance_id"] = instance_id
            all_results.append(results)
            
            # Save intermediate results
            with open(metrics_path / f"subtask_eval_{config.task.name}_{demo_id}.json", "w") as f:
                json.dump(results, f, indent=2)
                
            # Reset video writer
            if config.write_video and video_name is not None:
                evaluator.video_writer = None  # type: ignore
                logger.info(f"Saved video to {video_name}")
                
        # Save aggregate results
        aggregate = {
            "task_name": config.task.name,
            "n_demos": len(demo_ids),
            "primitive_success_rate": evaluator.n_primitive_successes / max(1, evaluator.n_primitive_trials),
            "endtoend_success_rate": evaluator.n_endtoend_successes / max(1, evaluator.n_endtoend_trials),
            "total_primitives": evaluator.n_primitive_trials,
            "successful_primitives": evaluator.n_primitive_successes,
            "total_episodes": evaluator.n_endtoend_trials,
            "successful_episodes": evaluator.n_endtoend_successes,
            "per_demo_results": all_results,
        }
        
        with open(metrics_path / f"subtask_eval_{config.task.name}_aggregate.json", "w") as f:
            json.dump(aggregate, f, indent=2)
            
        logger.info(f"Aggregate results saved to {metrics_path}")
