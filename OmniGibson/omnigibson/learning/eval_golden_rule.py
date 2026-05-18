"""
Golden Rule Evaluator for BEHAVIOR-1K

This module provides skill-level evaluation for policies that follow a ground-truth
skill plan ("golden rule"). The evaluator:
1. Loads a ground-truth skill plan from demo annotations
2. Wraps the underlying policy with a GoldenRulePolicyWrapper that advances through
the plan step-by-step
3. Evaluates each skill until completion or timeout
4. Reports skill-level success rates and end-to-end success rates

Usage:
    python eval_golden_rule.py task.name=turning_on_radio \
        demo_data_path=/path/to/2025-challenge-demos \
        demo_id=00000010 \
        log_path=./eval_logs/golden_rule

    # NOTE: `demo_data_path` must point to the `2025-challenge-demos` folder
    # (it should contain `annotations` and `data` subfolders).
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
import sys
import torch as th
import traceback
from inspect import getsourcefile
from omegaconf import DictConfig, OmegaConf
from omnigibson.controllers import IsGraspingState
from omnigibson.learning.eval_subtask_reset import SubTaskEvaluator
from omnigibson.learning.utils.config_utils import register_omegaconf_resolvers
from omnigibson.learning.utils.eval_utils import (
    ROBOT_CAMERA_NAMES,
    PROPRIOCEPTION_INDICES,
    PROPRIO_QPOS_INDICES,
    TASK_NAMES_TO_INDICES,
    flatten_obs_dict,
)
from omnigibson.learning.utils.obs_utils import create_video_writer, write_video
from omnigibson.macros import gm
from omnigibson.utils.python_utils import recursively_convert_to_torch
from pathlib import Path
from signal import signal, SIGINT
from typing import Any, Dict, List, Optional, Tuple

# create module logger
logger = logging.getLogger("golden_rule_evaluator")
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Mock / fallback policy wrapper (used when openpi-comet is not available)
# ---------------------------------------------------------------------------

class _MockGoldenRulePolicyWrapper:
    """Minimal mock of GoldenRulePolicyWrapper for framework integration."""

    def __init__(self, base_policy: Any, gt_plan_loader: Any, cfg: DictConfig) -> None:
        self.base_policy = base_policy
        self.gt_plan_loader = gt_plan_loader
        self.cfg = cfg
        self.current_skill_idx = 0
        self.skills: List[Dict[str, Any]] = []

    def set_plan(self, skills: List[Dict[str, Any]]) -> None:
        self.skills = skills
        self.current_skill_idx = 0
        logger.info(f"MockGoldenRulePolicyWrapper: loaded plan with {len(skills)} skills")

    def advance_skill(self) -> None:
        self.current_skill_idx += 1
        logger.info(f"MockGoldenRulePolicyWrapper: advanced to skill {self.current_skill_idx}/{len(self.skills)}")

    def current_skill(self) -> Optional[Dict[str, Any]]:
        if 0 <= self.current_skill_idx < len(self.skills):
            return self.skills[self.current_skill_idx]
        return None

    def forward(self, obs: dict, *args, **kwargs) -> th.Tensor:
        # Delegate to base policy; in the real wrapper this would inject skill context.
        if hasattr(self.base_policy, "forward"):
            return self.base_policy.forward(obs, *args, **kwargs)
        if hasattr(self.base_policy, "act"):
            return self.base_policy.act(obs)
        raise RuntimeError("Base policy has no forward/act method")

    def reset(self) -> None:
        self.current_skill_idx = 0
        if hasattr(self.base_policy, "reset"):
            self.base_policy.reset()


# ---------------------------------------------------------------------------
# GTPlanLoader (local implementation until Worker A lands the real one)
# ---------------------------------------------------------------------------

class GTPlanLoader:
    """Load a ground-truth skill plan from demo annotations."""

    def __init__(self, demo_data_path: str) -> None:
        self.demo_data_path = os.path.expanduser(demo_data_path)

    def load_plan(self, task_name: str, demo_id: str) -> List[Dict[str, Any]]:
        task_idx = TASK_NAMES_TO_INDICES[task_name]
        task_name_fmt = f"task-{task_idx:04d}"
        episode_name = f"episode_{demo_id}"
        annotation_path = os.path.join(
            self.demo_data_path, "annotations", task_name_fmt, f"{episode_name}.json"
        )
        if not os.path.exists(annotation_path):
            logger.warning(f"Annotation file not found: {annotation_path}")
            return []

        with open(annotation_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)

        skills = annotations.get("skill_annotation", []) or []
        plan = []
        for i, skill in enumerate(skills):
            if not isinstance(skill, dict):
                continue
            desc = skill.get("skill_description", ["unknown"])
            if isinstance(desc, list):
                desc = desc[0] if desc else "unknown"
            plan.append(
                {
                    "index": i,
                    "description": str(desc),
                    "frame_duration": skill.get("frame_duration"),
                    "object_id": skill.get("object_id"),
                    "manipulating_object_id": skill.get("manipulating_object_id"),
                }
            )
        logger.info(f"GTPlanLoader: loaded {len(plan)} skills from {annotation_path}")
        return plan


# ---------------------------------------------------------------------------
# GoldenRuleEvaluator
# ---------------------------------------------------------------------------

class GoldenRuleEvaluator(SubTaskEvaluator):
    """
    Evaluator for policies that follow a ground-truth skill plan (golden rule).

    Key behaviours:
    1. Loads a GT skill plan for the target demo.
    2. Wraps the base policy with ``GoldenRulePolicyWrapper`` (or a mock when
       openpi-comet is unavailable) so that the policy is driven by the plan.
    3. Steps the environment, checks skill-level completion, and advances the
       plan when a skill is done.
    4. Collects per-skill and end-to-end metrics.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        # Golden-rule specific attributes
        self.use_gt_plan: bool = cfg.get("use_gt_plan", True)
        self.skill_timeout_steps: int = int(cfg.get("skill_timeout_steps", 300))
        self.skill_max_steps_multiplier: float = float(cfg.get("skill_max_steps_multiplier", 2.0))
        self.control_mode: str = str(cfg.get("control_mode", "receeding_horizon"))
        self.max_len: int = int(cfg.get("max_len", 32))
        self.fine_grained_level: int = int(cfg.get("fine_grained_level", 2))

        # Skill-level tracking
        self.n_skill_trials = 0
        self.n_skill_successes = 0
        self.n_endtoend_trials = 0
        self.n_endtoend_successes = 0

        # Current episode state
        self.current_skill_plan: List[Dict[str, Any]] = []
        self.current_skill_idx = 0
        self.current_skill_step = 0
        self.current_demo_id: Optional[str] = None

        # Internal caches
        self._gt_plan_loader: Optional[GTPlanLoader] = None
        self._golden_rule_policy: Optional[Any] = None

        logger.info("GoldenRuleEvaluator initialized")
        logger.info(f"  use_gt_plan={self.use_gt_plan}")
        logger.info(f"  skill_timeout_steps={self.skill_timeout_steps}")
        logger.info(f"  skill_max_steps_multiplier={self.skill_max_steps_multiplier}")

    # ------------------------------------------------------------------
    # Policy loading
    # ------------------------------------------------------------------

    def load_policy(self) -> Any:
        """
        Load the underlying policy and wrap it with ``GoldenRulePolicyWrapper``.

        The wrapper is instantiated from openpi-comet when available; otherwise a
        local mock is used so that the integration framework can still be exercised.
        """
        # Load base policy exactly as the parent Evaluator does
        base_policy = super().load_policy()

        # Attempt to import the real wrapper from openpi-comet
        GoldenRulePolicyWrapper = None  # type: ignore
        try:
            from openpi_comet.policy.golden_rule_wrapper import GoldenRulePolicyWrapper  # type: ignore
            logger.info("Imported GoldenRulePolicyWrapper from openpi-comet")
        except Exception as exc:
            logger.warning(
                f"Could not import GoldenRulePolicyWrapper from openpi-comet ({exc}); "
                "falling back to mock wrapper."
            )
            GoldenRulePolicyWrapper = _MockGoldenRulePolicyWrapper  # type: ignore

        # Create GTPlanLoader (local implementation until Worker A lands the real one)
        demo_data_path = self.cfg.get("demo_data_path", None)
        if demo_data_path is not None:
            self._gt_plan_loader = GTPlanLoader(demo_data_path)
        else:
            logger.warning("demo_data_path not set; GTPlanLoader will be unavailable")

        # Wrap the base policy
        self._golden_rule_policy = GoldenRulePolicyWrapper(
            base_policy=base_policy,
            gt_plan_loader=self._gt_plan_loader,
            cfg=self.cfg,
        )
        logger.info("Policy wrapped with GoldenRulePolicyWrapper")
        return self._golden_rule_policy

    # ------------------------------------------------------------------
    # Episode / plan setup
    # ------------------------------------------------------------------

    def setup_episode(self, demo_id: str) -> bool:
        """
        Load the GT skill plan for *demo_id* and prepare the evaluator.

        Returns:
            True if a plan was successfully loaded and the episode is ready.
        """
        self.current_demo_id = demo_id
        self.current_skill_idx = 0
        self.current_skill_step = 0

        # Load skill plan
        if self.use_gt_plan and self._gt_plan_loader is not None:
            self.current_skill_plan = self._gt_plan_loader.load_plan(
                task_name=self.cfg.task.name,
                demo_id=demo_id,
            )
        else:
            self.current_skill_plan = []

        if not self.current_skill_plan:
            logger.error(f"No skill plan available for demo {demo_id}")
            return False

        # Push the plan into the policy wrapper
        if hasattr(self.policy, "set_plan"):
            self.policy.set_plan(self.current_skill_plan)
        else:
            logger.warning("Policy wrapper does not expose set_plan(); skipping plan injection")

        # Load demo low-dim data for state-match checks
        self.current_demo_data = self.load_demo_lowdim_data(demo_id)
        if self.current_demo_data is None:
            logger.warning(f"Could not load demo low-dim data for {demo_id}")

        # Try to load rawdata / cache for state restoration
        self.current_rawdata_hdf5 = self.load_rawdata_hdf5(demo_id)
        if self.current_rawdata_hdf5 is not None:
            logger.info("Using raw HDF5 data for state restoration")
        else:
            self.current_primitive_state_cache = self.load_primitive_state_cache(demo_id)
            if self.current_primitive_state_cache is not None:
                logger.info("Using primitive state cache for state restoration")
            else:
                logger.info("Using proprioception data for state restoration (robot only)")

        logger.info(f"Episode setup complete: {len(self.current_skill_plan)} skills")
        return True

    # ------------------------------------------------------------------
    # Skill-level success check
    # ------------------------------------------------------------------

    def get_skill_timeout(self, skill: Dict[str, Any]) -> int:
        """Compute timeout for a skill based on its demo duration and config."""
        fd = skill.get("frame_duration")
        if fd is not None:
            try:
                start, end = fd
                duration = int(end) - int(start)
                timeout = int(duration * self.skill_max_steps_multiplier)
                return max(timeout, self.skill_timeout_steps)
            except Exception:
                pass
        return self.skill_timeout_steps

    def check_skill_success(
        self,
        skill: Dict[str, Any],
        current_step: int,
        terminated: bool,
        timeout_steps: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """
        Check whether the current skill has completed.

        Returns:
            (is_done, result) where result is one of:
            - "success_env"   : full environment termination
            - "success_state" : state-match success against demo end frame
            - "timeout"       : exceeded step budget
            - "in_progress"   : still running
        """
        timeout = int(timeout_steps) if timeout_steps is not None else self.get_skill_timeout(skill)

        self._last_primitive_success_reason = None
        self._last_primitive_state_errors = None

        if terminated:
            self._last_primitive_success_reason = "env_terminated"
            return True, "success_env"

        # Re-use the primitive-level state-match machinery from SubTaskEvaluator.
        # We treat the skill's frame_duration as the primitive boundary.
        if bool(self.cfg.get("primitive_success_use_state_match", True)):
            errors = self.compute_primitive_state_errors(skill)
            self._last_primitive_state_errors = errors
            if errors is not None:
                thr = self.get_primitive_success_thresholds()
                std_rmse = errors.get("std_joint_qpos_rmse", float("inf"))
                if np.isfinite(std_rmse) and std_rmse <= thr["std_joint_qpos_rmse"]:
                    self._last_primitive_success_reason = "state_match_std_joint_qpos"
                    return True, "success_state"

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

                jq = errors.get("joint_qpos_rmse", float("inf"))
                if np.isfinite(jq) and jq <= thr["joint_qpos_rmse"]:
                    self._last_primitive_success_reason = "state_match_joint_rmse"
                    return True, "success_state"

        if current_step >= timeout:
            return True, "timeout"

        return False, "in_progress"

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def step(self) -> Tuple[bool, bool, Dict[str, Any]]:
        """
        Execute one environment step and check skill completion.

        Returns:
            (terminated, truncated, info) where *info* contains:
            - "skill_done"   : whether the current skill finished this step
            - "skill_result" : result string (see check_skill_success)
            - "skill_idx"    : index of the skill that was active
        """
        # Delegate to parent step (policy forward + env step)
        terminated, truncated = super().step()
        self.current_skill_step += 1

        info: Dict[str, Any] = {
            "skill_done": False,
            "skill_result": "in_progress",
            "skill_idx": self.current_skill_idx,
        }

        # Determine active skill
        active_skill = None
        if 0 <= self.current_skill_idx < len(self.current_skill_plan):
            active_skill = self.current_skill_plan[self.current_skill_idx]

        if active_skill is not None:
            skill_done, skill_result = self.check_skill_success(
                primitive=active_skill,
                current_step=self.current_skill_step,
                terminated=terminated,
            )
            info["skill_done"] = skill_done
            info["skill_result"] = skill_result

            if skill_done:
                skill_desc = active_skill.get("description", "unknown")
                if str(skill_result).startswith("success"):
                    logger.info(
                        f"Skill {self.current_skill_idx + 1}/{len(self.current_skill_plan)} "
                        f"'{skill_desc}' succeeded ({skill_result}) at step {self.current_skill_step}"
                    )
                    self.n_skill_successes += 1
                else:
                    logger.info(
                        f"Skill {self.current_skill_idx + 1}/{len(self.current_skill_plan)} "
                        f"'{skill_desc}' finished with result={skill_result} at step {self.current_skill_step}"
                    )
                self.n_skill_trials += 1

                # Advance plan in the policy wrapper
                if hasattr(self.policy, "advance_skill"):
                    self.policy.advance_skill()

                self.current_skill_idx += 1
                self.current_skill_step = 0

                # Optionally restore state to the start of the next skill
                if not terminated and not truncated:
                    next_skill = None
                    if 0 <= self.current_skill_idx < len(self.current_skill_plan):
                        next_skill = self.current_skill_plan[self.current_skill_idx]
                    if next_skill is not None and self.cfg.get("reset_on_primitive_failure", True):
                        # Only restore if previous skill failed, or if configured to restore every time
                        prev_failed = not str(skill_result).startswith("success")
                        if prev_failed or self.cfg.get("restore_at_each_primitive_start", False):
                            fd = next_skill.get("frame_duration")
                            if fd is not None:
                                try:
                                    start_frame = int(fd[0])
                                    restored, method = self._try_restore_to_frame(start_frame)
                                    logger.info(
                                        f"Restored state to next skill start frame {start_frame}: "
                                        f"restored={restored} method={method}"
                                    )
                                except Exception as exc:
                                    logger.warning(f"State restore to next skill failed: {exc}")

        return terminated, truncated, info

    # ------------------------------------------------------------------
    # Episode execution
    # ------------------------------------------------------------------

    def run_episode(self, demo_id: str) -> Dict[str, Any]:
        """
        Run a full episode for *demo_id* following the golden-rule skill plan.

        Returns:
            Dict with keys:
            - "demo_id"            : the demo ID
            - "skill_results"      : list of (skill_desc, success, result_type)
            - "n_skills"           : total number of skills
            - "n_skill_successes"  : number of successful skills
            - "endtoend_success"   : whether all skills succeeded
            - "total_steps"        : total environment steps taken
            - "terminated"         : whether the env terminated (task success)
            - "truncated"          : whether the env truncated
        """
        if not self.setup_episode(demo_id):
            return {"error": "setup_failed", "demo_id": demo_id}

        skill_results: List[Tuple[str, bool, str]] = []
        total_steps = 0
        terminated = False
        truncated = False

        logger.info(f"Starting golden-rule episode for demo {demo_id}")
        logger.info(f"  Skills: {len(self.current_skill_plan)}")

        # Reset policy at episode start
        self.policy.reset()
        self.obs = self._preprocess_obs(self._get_obs_for_policy())

        # Run until env terminates/truncates or we exhaust the skill plan
        max_total_steps = self.cfg.get("max_steps", None)
        if max_total_steps is None:
            # Default: generous multiplier over average human demo length
            max_total_steps = int(getattr(self, "human_stats", {}).get("length", 5000) * 3)

        while total_steps < max_total_steps:
            terminated, truncated, info = self.step()
            total_steps += 1

            if self.cfg.write_video:
                self._write_video()

            # Record skill completion when it happens
            if info.get("skill_done"):
                skill_idx = info.get("skill_idx", 0)
                if 0 <= skill_idx < len(self.current_skill_plan):
                    skill = self.current_skill_plan[skill_idx]
                    skill_desc = skill.get("description", "unknown")
                    success = str(info.get("skill_result", "")).startswith("success")
                    skill_results.append((skill_desc, success, info["skill_result"]))

            if terminated or truncated:
                break

        # Ensure we record any trailing skill that finished exactly on termination
        if len(skill_results) < len(self.current_skill_plan):
            # The active skill may not have been recorded if termination happened first
            pass

        n_successes = sum(1 for _, s, _ in skill_results if s)
        endtoend_success = len(skill_results) == len(self.current_skill_plan) and all(
            s for _, s, _ in skill_results
        )

        # Update aggregate counters
        self.n_endtoend_trials += 1
        if endtoend_success:
            self.n_endtoend_successes += 1

        results: Dict[str, Any] = {
            "demo_id": demo_id,
            "skill_results": skill_results,
            "n_skills": len(self.current_skill_plan),
            "n_skill_successes": n_successes,
            "endtoend_success": endtoend_success,
            "total_steps": total_steps,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }

        logger.info(f"Episode complete: {n_successes}/{len(self.current_skill_plan)} skills succeeded")
        logger.info(f"  End-to-end success: {endtoend_success}")

        # Cleanup rawdata handle
        if hasattr(self, "current_rawdata_hdf5") and self.current_rawdata_hdf5 is not None:
            self.current_rawdata_hdf5.close()
            self.current_rawdata_hdf5 = None
        self.current_primitive_state_cache = None

        return results

    # ------------------------------------------------------------------
    # Context manager exit
    # ------------------------------------------------------------------

    def __exit__(self, exc_type, exc_value, exc_tb):
        logger.info("")
        logger.info("=" * 60)
        logger.info("GOLDEN RULE EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Skill (ST) Results:")
        logger.info(f"  - Total skills: {self.n_skill_trials}")
        logger.info(f"  - Successful skills: {self.n_skill_successes}")
        if self.n_skill_trials > 0:
            logger.info(f"  - ST Success rate: {self.n_skill_successes / self.n_skill_trials:.2%}")
        logger.info(f"End-to-End (ET) Results:")
        logger.info(f"  - Total episodes: {self.n_endtoend_trials}")
        logger.info(f"  - Successful episodes: {self.n_endtoend_successes}")
        if self.n_endtoend_trials > 0:
            logger.info(f"  - ET Success rate: {self.n_endtoend_successes / self.n_endtoend_trials:.2%}")
        logger.info("=" * 60)
        logger.info("")
        super().__exit__(exc_type, exc_value, exc_tb)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_demo_ids_for_task(demo_data_path: str, task_name: str, limit: Optional[int] = None) -> List[str]:
    """
    Get list of available demo IDs for a task.
    """
    task_idx = TASK_NAMES_TO_INDICES[task_name]
    task_folder = f"task-{task_idx:04d}"
    annotations_path = os.path.join(demo_data_path, "annotations", task_folder)
    if not os.path.exists(annotations_path):
        logger.warning(f"Annotations folder not found: {annotations_path}")
        return []

    demo_ids = []
    for fname in sorted(os.listdir(annotations_path)):
        if fname.endswith(".json"):
            demo_id = fname.replace("episode_", "").replace(".json", "")
            demo_ids.append(demo_id)

    if limit is not None:
        demo_ids = demo_ids[:limit]
    return demo_ids


# ---------------------------------------------------------------------------
# __main__ entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    register_omegaconf_resolvers()

    src = getsourcefile(lambda: 0) or __file__
    config_dir = f"{Path(src).parents[0]}/configs"
    with hydra.initialize_config_dir(config_dir, version_base="1.1"):
        config = hydra.compose("eval_golden_rule_config.yaml", overrides=sys.argv[1:])
    OmegaConf.resolve(config)

    gm.HEADLESS = config.headless

    if config.write_video:
        video_path = Path(config.log_path).expanduser() / "videos"
        video_path.mkdir(parents=True, exist_ok=True)

    demo_data_path = config.get("demo_data_path", None)
    if demo_data_path is None:
        logger.error("demo_data_path must be specified for golden-rule evaluation")
        sys.exit(1)

    demo_id = config.get("demo_id", None)
    if demo_id is not None:
        demo_ids = [str(demo_id)]
    else:
        demo_ids = get_demo_ids_for_task(
            demo_data_path=demo_data_path,
            task_name=config.task.name,
            limit=config.get("num_demos", None),
        )

    if not demo_ids:
        logger.error(f"No demos found for task {config.task.name}")
        sys.exit(1)

    logger.info(f"Evaluating {len(demo_ids)} demo(s) for task {config.task.name}")

    metrics_path = Path(config.log_path).expanduser() / "metrics"
    metrics_path.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict[str, Any]] = []

    with GoldenRuleEvaluator(config) as evaluator:
        for demo_idx, demo_id in enumerate(demo_ids):
            logger.info(f"\n{'#'*60}")
            logger.info(f"Evaluating demo {demo_idx + 1}/{len(demo_ids)}: {demo_id}")
            logger.info(f"{'#'*60}")

            evaluator.reset()
            instance_id = int(demo_id) // 10 % 1000
            evaluator.load_task_instance(instance_id, test_hidden=config.test_hidden)

            video_name = None
            if config.write_video:
                video_name = str(video_path) + f"/{config.task.name}_golden_rule_demo{demo_id}.mp4"
                evaluator.video_writer = create_video_writer(
                    fpath=video_name,
                    resolution=(448, 672),
                )

            evaluator.reset()
            results = evaluator.run_episode(demo_id=demo_id)
            results["instance_id"] = instance_id
            all_results.append(results)

            with open(metrics_path / f"golden_rule_{config.task.name}_{demo_id}.json", "w") as f:
                json.dump(results, f, indent=2)

            if config.write_video and video_name is not None:
                evaluator.video_writer = None  # type: ignore
                logger.info(f"Saved video to {video_name}")

        # Aggregate
        aggregate = {
            "task_name": config.task.name,
            "n_demos": len(demo_ids),
            "skill_success_rate": evaluator.n_skill_successes / max(1, evaluator.n_skill_trials),
            "endtoend_success_rate": evaluator.n_endtoend_successes / max(1, evaluator.n_endtoend_trials),
            "total_skills": evaluator.n_skill_trials,
            "successful_skills": evaluator.n_skill_successes,
            "total_episodes": evaluator.n_endtoend_trials,
            "successful_episodes": evaluator.n_endtoend_successes,
            "per_demo_results": all_results,
        }
        with open(metrics_path / f"golden_rule_{config.task.name}_aggregate.json", "w") as f:
            json.dump(aggregate, f, indent=2)

        logger.info(f"Aggregate results saved to {metrics_path}")
