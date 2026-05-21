"""Segment-level (primitive / skill) evaluator for BEHAVIOR-1K using BDDL predicate subgoal.

This script evaluates a single segment using BDDL predicate-based subgoal success criteria.
Unlike eval_primitive.py which uses state-match, this script uses predicate satisfaction
to determine segment success.

Key features:
- Supports both primitive and skill level segmentation
- Uses BDDL ground_goal_state_options to extract subgoal predicates
- Computes predicate truth at start/end frames and uses delta as subgoal
- Provides diagnostic output (grounding selection, predicate progress, q_score)

Typical usage:
  python OmniGibson/omnigibson/learning/eval_segment.py \
    policy=websocket task.name=turning_on_radio \
    env_wrapper._target_=omnigibson.learning.wrappers.wbvima_wrapper.WBVIMAWrapper \
    demo_data_path=/path/to/2025-challenge-demos \
    rawdata_path=/path/to/2025-challenge-rawdata \
    demo_id=00000010 segment_level=primitive segment_idx=0 \
    success_mode=predicate_subgoal \
    log_path=./eval_logs/segment_eval

Notes:
- `segment_idx` refers to the segment index *after sorting by start frame*.
- `segment_level` can be "primitive" or "skill".
- `success_mode` can be:
  - predicate_subgoal: segment succeeds if all subgoal predicates are satisfied
  - predicate_progress: only record progress, no binary success
  - state_match: fallback to state-match (like eval_primitive)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import copy
from inspect import getsourcefile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from omnigibson.learning.eval_subtask_reset import SubTaskEvaluator, get_demo_ids_for_task
from omnigibson.learning.utils.config_utils import register_omegaconf_resolvers
from omnigibson.learning.utils.eval_diagnostics import (
    ENV_TASK_SUCCESS_BEFORE_SEGMENT_SUCCESS,
    build_pre_satisfied_start_result,
    build_termination_summary,
    classify_short_proxy_success,
    classify_short_video_success,
    finalize_segment_predicate_result_type,
    update_segment_env_termination_telemetry,
)
from omnigibson.learning.utils.eval_utils import ROBOT_CAMERA_NAMES
from omnigibson.learning.utils.obs_utils import create_video_writer
from omnigibson.learning.utils.predicate_utils import (
    eval_ground_option,
    diff_subgoal,
    rank_groundings,
    select_best_grounding,
    compute_q_score_delta,
    compute_subgoal_progress,
    format_head_predicate,
    get_subgoal_predicates,
)
from omnigibson.learning.utils.segment_predicate_eval import (
    MISSING_OBJECT_RESULT_TYPE,
    build_auto_mined_predicates,
    build_template_predicates,
    eval_segment_predicates,
    predicate_window_satisfied,
    trace_missing_objects,
)
from omnigibson.macros import gm


logger = logging.getLogger("segment_evaluator")
logger.setLevel(logging.INFO)


def _sanitize_restore_debug(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _sanitize_restore_debug(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_restore_debug(v) for v in value]
    try:
        return copy.deepcopy(value)
    except Exception:
        return repr(value)


def _snapshot_restore_debug(evaluator: SubTaskEvaluator) -> Optional[Dict[str, Any]]:
    debug = getattr(evaluator, "_last_restore_debug", None)
    if debug is None:
        return None
    try:
        return copy.deepcopy(debug)
    except Exception as exc:
        logger.warning("Failed to deepcopy restore telemetry; sanitizing debug payload instead: %s", exc)
        sanitized = _sanitize_restore_debug(debug)
        return sanitized if isinstance(sanitized, dict) else {"value": sanitized}


def _as_demo_id(x: Any) -> str:
    s = str(x)
    if s.isdigit() and len(s) < 8:
        return s.zfill(8)
    return s


def _get_instance_id_from_demo_id(demo_id: str) -> int:
    return int(demo_id) // 10 % 1000


def _video_rgb_obs_ready(evaluator: SubTaskEvaluator) -> bool:
    obs = getattr(evaluator, "obs", None) or {}
    return all(
        ROBOT_CAMERA_NAMES["R1Pro"][camera_name] + "::rgb" in obs
        for camera_name in ("head", "left_wrist", "right_wrist")
    )


def _pad_review_video_tail(
    evaluator: SubTaskEvaluator,
    *,
    current_frames: int,
    min_frames: int,
) -> int:
    """Append duplicate final-state frames so very short review videos are inspectable.

    This does not step the simulator or affect rollout metrics; it only writes extra copies of
    the current observation to the active video writer.
    """
    missing_frames = max(int(min_frames) - max(int(current_frames), 0), 0)
    if missing_frames <= 0 or getattr(evaluator, "video_writer", None) is None:
        return 0
    if not _video_rgb_obs_ready(evaluator):
        try:
            evaluator.obs = evaluator._preprocess_obs(evaluator._get_obs_for_policy())
        except Exception as exc:
            logger.warning("Unable to refresh observation for review-video tail padding: %s", exc)
            return 0
    if not _video_rgb_obs_ready(evaluator):
        return 0

    evaluator._video_primitive_progress = 1.0
    for _ in range(missing_frames):
        evaluator._write_video()
    return missing_frames


def _flatten_numeric(x: Any) -> List[int]:
    out: List[int] = []
    if isinstance(x, (list, tuple)):
        for y in x:
            out.extend(_flatten_numeric(y))
    elif isinstance(x, (int, float)):
        out.append(int(x))
    return out


def _normalize_frame_duration(raw: Any) -> Tuple[int, int]:
    vals = _flatten_numeric(raw)
    if len(vals) < 2:
        raise ValueError(f"Invalid frame_duration: {raw}")
    return int(vals[0]), int(vals[-1])


def _to_numpy_image(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    arr = np.asarray(x)
    if arr.ndim < 3:
        return None
    arr = arr[..., :3]
    if np.issubdtype(arr.dtype, np.floating):
        scale = 255.0 if float(arr.max(initial=0.0)) <= 1.5 else 1.0
        arr = np.clip(arr * scale, 0.0, 255.0).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _find_obs_key(obs: Dict[str, Any], key: str) -> Optional[str]:
    if key in obs:
        return key
    suffix = f"{key.split('::')[0]}::rgb"
    for candidate in obs:
        if candidate.endswith(suffix):
            return candidate
    return None


def _build_review_composite(obs: Dict[str, Any]) -> Optional[np.ndarray]:
    """Build a single RGB image for quick visual review.

    Primary path: R1Pro (left/right wrist + head) composite.
    Fallback path: find any available `rgb` observations and build a stable 448x672 montage.
    """

    def _collect_rgb_candidates() -> List[Tuple[str, np.ndarray]]:
        candidates: List[Tuple[str, np.ndarray]] = []

        def visit(prefix: str, value: Any, depth: int) -> None:
            if depth <= 0:
                return
            # Recurse common containers
            if isinstance(value, dict):
                for k, v in value.items():
                    visit(f"{prefix}/{k}" if prefix else str(k), v, depth - 1)
                return
            if isinstance(value, (list, tuple)):
                for i, v in enumerate(value):
                    visit(f"{prefix}[{i}]", v, depth - 1)
                return

            path_l = prefix.lower()
            hinted = any(tok in path_l for tok in ("rgb", "image", "camera"))
            img = _to_numpy_image(value)
            if img is None:
                return
            h, w = int(img.shape[0]), int(img.shape[1])
            if h < 32 or w < 32:
                return
            if hinted or img.shape[-1] == 3:
                candidates.append((prefix, img))

        visit("", obs, depth=5)
        # Prefer larger images first (often the head / main camera)
        candidates.sort(key=lambda item: int(item[1].shape[0]) * int(item[1].shape[1]), reverse=True)
        return candidates

    # 1) Preferred R1Pro montage if keys exist.
    left_key = _find_obs_key(obs, ROBOT_CAMERA_NAMES["R1Pro"]["left_wrist"] + "::rgb")
    right_key = _find_obs_key(obs, ROBOT_CAMERA_NAMES["R1Pro"]["right_wrist"] + "::rgb")
    head_key = _find_obs_key(obs, ROBOT_CAMERA_NAMES["R1Pro"]["head"] + "::rgb")
    if left_key is not None and right_key is not None and head_key is not None:
        left = _to_numpy_image(obs.get(left_key))
        right = _to_numpy_image(obs.get(right_key))
        head = _to_numpy_image(obs.get(head_key))
        if left is not None and right is not None and head is not None:
            left = cv2.resize(left, (224, 224))
            right = cv2.resize(right, (224, 224))
            head = cv2.resize(head, (448, 448))
            return np.hstack([np.vstack([left, right]), head]).copy()

    # 2) Fallback: any rgb keys.
    candidates = _collect_rgb_candidates()
    if not candidates:
        return None

    main = cv2.resize(candidates[0][1], (448, 448))
    if len(candidates) >= 2:
        left_top = cv2.resize(candidates[1][1], (224, 224))
    else:
        left_top = cv2.resize(candidates[0][1], (224, 224))
    if len(candidates) >= 3:
        left_bottom = cv2.resize(candidates[2][1], (224, 224))
    else:
        left_bottom = left_top

    return np.hstack([np.vstack([left_top, left_bottom]), main]).copy()


def _capture_review_frame(evaluator: SubTaskEvaluator, out_path: Optional[Path]) -> Optional[str]:
    if out_path is None:
        return None
    try:
        obs = evaluator._get_obs_for_policy()
        composite = _build_review_composite(obs)
        if composite is None:
            all_keys = [str(k) for k in obs.keys()]
            rgb_keys = [k for k in all_keys if "rgb" in k.lower()]
            logger.warning(
                "Failed to build review composite for %s (found %d rgb-like keys). Keys (first 30): %s",
                str(out_path),
                len(rgb_keys),
                all_keys[:30],
            )
            return None
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if cv2.imwrite(str(out_path), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)):
            return str(out_path)
    except Exception as exc:
        logger.warning(f"Failed to capture review frame {out_path}: {exc}")
    return None


def load_segment_annotations(
    evaluator: SubTaskEvaluator,
    demo_id: str,
    segment_level: str,
) -> Optional[Dict]:
    annotations = evaluator.load_demo_annotations(demo_id)
    if annotations is None:
        return None
    key = f"{segment_level}_annotation"
    if key not in annotations:
        logger.warning(f"No {key} found in annotations")
        return None
    segments = annotations[key]
    if not segments:
        logger.warning(f"Empty {key}")
        return None
    normalized_segments = []
    for seg in segments:
        seg_copy = dict(seg)
        start_frame, end_frame = _normalize_frame_duration(seg_copy.get("frame_duration"))
        seg_copy["frame_duration"] = [start_frame, end_frame]
        normalized_segments.append(seg_copy)
    segments = sorted(normalized_segments, key=lambda x: x["frame_duration"][0])
    return {"segments": segments, "annotations": annotations}


def get_segment(
    evaluator: SubTaskEvaluator,
    demo_id: str,
    segment_level: str,
    segment_idx: int,
) -> Tuple[Optional[Dict], Optional[Dict]]:
    result = load_segment_annotations(evaluator, demo_id, segment_level)
    if result is None:
        return None, None

    segments = result["segments"]
    annotations = result["annotations"]

    if segment_idx < 0 or segment_idx >= len(segments):
        logger.error(f"segment_idx {segment_idx} out of range [0, {len(segments)-1}]")
        return None, None

    segment = segments[segment_idx]
    return segment, annotations


def restore_and_eval_predicates(
    evaluator: SubTaskEvaluator,
    frame_idx: int,
) -> Tuple[bool, str, Optional[List[List[bool]]]]:
    restored, method = evaluator._try_restore_to_frame(int(frame_idx))
    if not restored:
        return False, method, None

    ground_options = getattr(evaluator.env.task, "ground_goal_state_options", None)
    if ground_options is None:
        logger.warning("No ground_goal_state_options available")
        return True, method, None

    truth_vectors = []
    for option in ground_options:
        truth_vec = eval_ground_option(option)
        truth_vectors.append(truth_vec)

    return True, method, truth_vectors


def extract_subgoal_for_segment(
    evaluator: SubTaskEvaluator,
    segment: Dict,
    ground_options: List,
    s_start: List[List[bool]],
    s_end: List[List[bool]],
    topk: int = 3,
) -> Dict[str, Any]:
    ranked = rank_groundings(ground_options, s_start, s_end, topk=topk)

    chosen_idx = select_best_grounding(ground_options, s_start, s_end)
    chosen_option = ground_options[chosen_idx] if chosen_idx < len(ground_options) else None
    chosen_start = s_start[chosen_idx] if chosen_idx < len(s_start) else []
    chosen_end = s_end[chosen_idx] if chosen_idx < len(s_end) else []

    subgoal_indices = diff_subgoal(chosen_start, chosen_end)

    subgoal_predicates = []
    if chosen_option is not None:
        subgoal_predicates = get_subgoal_predicates(chosen_option, subgoal_indices)

    return {
        "chosen_option_idx": chosen_idx,
        "topk_candidates": ranked,
        "subgoal_indices": subgoal_indices,
        "subgoal_predicates": subgoal_predicates,
        "n_grounding_options": len(ground_options),
    }


def run_single_segment(
    evaluator: SubTaskEvaluator,
    demo_id: str,
    segment_level: str,
    segment_idx: int,
    success_mode: str = "predicate_subgoal",
    dry_run: bool = False,
    segment_max_steps: Optional[int] = None,
    review_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    segment, annotations = get_segment(evaluator, demo_id, segment_level, segment_idx)
    if segment is None:
        return {"error": "segment_not_found", "segment_idx": segment_idx}

    start_frame, end_frame = segment["frame_duration"]
    segment_desc = segment.get(f"{segment_level}_description", ["unknown"])[0]

    evaluator.current_demo_data = evaluator.load_demo_lowdim_data(demo_id)
    if evaluator.current_demo_data is None:
        return {"error": "no_demo_data"}

    evaluator.current_demo_id = demo_id
    evaluator.current_rawdata_hdf5 = evaluator.load_rawdata_hdf5(demo_id)
    if evaluator.current_rawdata_hdf5 is None:
        evaluator.current_primitive_state_cache = evaluator.load_primitive_state_cache(demo_id)

    logger.info("\n" + "=" * 60)
    logger.info(f"Evaluating segment level={segment_level} idx={segment_idx} / {segment_desc}")
    logger.info(f"Demo: {demo_id}")
    logger.info(f"Frames: {start_frame} - {end_frame}")
    logger.info(f"Success mode: {success_mode}")
    logger.info("=" * 60)

    ground_options = getattr(evaluator.env.task, "ground_goal_state_options", None)
    if ground_options is None or len(ground_options) == 0:
        return {"error": "no_ground_goal_state_options"}

    if success_mode == "segment_predicates":
        def _restore_entry(restored: bool, method: str, debug: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            entry: Dict[str, Any] = {"restored": bool(restored), "method": method}
            if debug is not None:
                entry["debug"] = debug
            return entry

        review_artifacts = {
            "start_restore_rgb": None,
            "end_restore_rgb": None,
            "final_rollout_rgb": None,
        }
        evaluator.current_demo_id = demo_id
        evaluator.current_rawdata_hdf5 = evaluator.load_rawdata_hdf5(demo_id)
        if evaluator.current_rawdata_hdf5 is None:
            evaluator.current_primitive_state_cache = evaluator.load_primitive_state_cache(demo_id)

        restored_start, method_start, _ = restore_and_eval_predicates(evaluator, start_frame)
        restore_debug_start = _snapshot_restore_debug(evaluator)
        if restored_start:
            review_artifacts["start_restore_rgb"] = _capture_review_frame(
                evaluator,
                review_dir / "start_restore.png" if review_dir is not None else None,
            )
        restored_end, method_end, _ = restore_and_eval_predicates(evaluator, end_frame)
        restore_debug_end = _snapshot_restore_debug(evaluator)
        if restored_end:
            review_artifacts["end_restore_rgb"] = _capture_review_frame(
                evaluator,
                review_dir / "end_restore.png" if review_dir is not None else None,
            )
        if not restored_start or not restored_end:
            return {
                "demo_id": demo_id,
                "segment_level": segment_level,
                "segment_idx": segment_idx,
                "segment_desc": segment_desc,
                "frame_duration": [int(start_frame), int(end_frame)],
                "restore": {
                    "start": _restore_entry(restored_start, method_start, restore_debug_start),
                    "end": _restore_entry(restored_end, method_end, restore_debug_end),
                },
                "success_mode": str(success_mode),
                "success": False,
                "result_type": "restore_failed",
                "review_artifacts": review_artifacts,
            }

        template_specs, metric_debug = build_template_predicates(segment_level, segment, evaluator.env)
        restored_start_for_compare, _, _ = restore_and_eval_predicates(evaluator, start_frame)
        restore_debug_start_for_compare = _snapshot_restore_debug(evaluator)
        if not restored_start_for_compare:
            return {
                "demo_id": demo_id,
                "segment_level": segment_level,
                "segment_idx": segment_idx,
                "segment_desc": segment_desc,
                "frame_duration": [int(start_frame), int(end_frame)],
                "restore": {
                    "start": _restore_entry(False, method_start, restore_debug_start_for_compare),
                    "end": _restore_entry(True, method_end, restore_debug_end),
                },
                "success_mode": str(success_mode),
                "success": False,
                "result_type": "restore_failed",
                "review_artifacts": review_artifacts,
            }
        start_truth_map, start_trace = eval_segment_predicates(evaluator.env, template_specs)
        restored_end, method_end, _ = restore_and_eval_predicates(evaluator, end_frame)
        restore_debug_end_for_compare = _snapshot_restore_debug(evaluator)
        end_truth_map, end_trace = eval_segment_predicates(evaluator.env, template_specs)
        auto_specs = build_auto_mined_predicates(
            segment_level,
            segment,
            evaluator.env,
            start_truth=start_truth_map,
            end_truth=end_truth_map,
        )
        missing_template_roles = metric_debug.get("missing_template_roles") or []
        if missing_template_roles:
            return {
                "demo_id": demo_id,
                "segment_level": segment_level,
                "segment_idx": segment_idx,
                "segment_desc": segment_desc,
                "frame_duration": [int(start_frame), int(end_frame)],
                "restore": {
                    "start": _restore_entry(True, method_start, restore_debug_start_for_compare),
                    "end": _restore_entry(True, method_end, restore_debug_end_for_compare),
                },
                "success_mode": str(success_mode),
                "effective_success_mode": "segment_predicates",
                "success": False,
                "result_type": MISSING_OBJECT_RESULT_TYPE,
                "predicate_debug": {
                    **metric_debug,
                    "template_trace_start": start_trace,
                    "template_trace_end": end_trace,
                    "missing_object_traces": [
                        {
                            "predicate": None,
                            "metric_type": item.get("metric_type"),
                            "desired": None,
                            "value": None,
                            "satisfied": False,
                            "missing_object": item.get("role"),
                            "missing_role": item.get("role"),
                            "metric_name": item.get("metric_name"),
                            "trace_stage": "template_resolution",
                        }
                        for item in missing_template_roles
                    ],
                },
                "rollout": {
                    "max_steps": 0,
                    "final_step": 0,
                    "rollout_attempted": False,
                    "termination_reason": MISSING_OBJECT_RESULT_TYPE,
                },
                "review_artifacts": review_artifacts,
            }
        predicate_specs = template_specs if template_specs else auto_specs
        if template_specs and auto_specs:
            existing = {(p.metric_type, p.name, tuple(p.args), p.desired) for p in template_specs}
            predicate_specs = list(template_specs) + [
                p for p in auto_specs if (p.metric_type, p.name, tuple(p.args), p.desired) not in existing
            ]

        if len(predicate_specs) == 0:
            return {
                "demo_id": demo_id,
                "segment_level": segment_level,
                "segment_idx": segment_idx,
                "segment_desc": segment_desc,
                "frame_duration": [int(start_frame), int(end_frame)],
                "restore": {
                    "start": _restore_entry(True, method_start, restore_debug_start_for_compare),
                    "end": _restore_entry(True, method_end, restore_debug_end_for_compare),
                },
                "success_mode": str(success_mode),
                "effective_success_mode": "segment_predicates",
                "success": False,
                "result_type": "no_predicates_generated",
                "review_artifacts": review_artifacts,
            }

        # Segment rollout must start from the segment start frame, not the end frame used for target metric capture.
        restored_rollout_start, _, _ = restore_and_eval_predicates(evaluator, start_frame)
        restore_debug_rollout_start = _snapshot_restore_debug(evaluator)
        if not restored_rollout_start:
            return {
                "demo_id": demo_id,
                "segment_level": segment_level,
                "segment_idx": segment_idx,
                "segment_desc": segment_desc,
                "frame_duration": [int(start_frame), int(end_frame)],
                "restore": {
                    "start": _restore_entry(False, method_start, restore_debug_rollout_start),
                    "end": _restore_entry(True, method_end, restore_debug_end_for_compare),
                    "rollout_start": _restore_entry(False, method_start, restore_debug_rollout_start),
                },
                "success_mode": str(success_mode),
                "effective_success_mode": "segment_predicates",
                "success": False,
                "result_type": "restore_failed_before_rollout",
                "review_artifacts": review_artifacts,
            }

        result = {
            "demo_id": demo_id,
            "segment_level": segment_level,
            "segment_idx": segment_idx,
            "segment_desc": segment_desc,
            "frame_duration": [int(start_frame), int(end_frame)],
            "restore": {
                "start": _restore_entry(True, method_start, restore_debug_start_for_compare),
                "end": _restore_entry(True, method_end, restore_debug_end_for_compare),
                "rollout_start": _restore_entry(True, method_start, restore_debug_rollout_start),
            },
            "success_mode": str(success_mode),
            "effective_success_mode": "segment_predicates",
            "predicate_spec": [
                {
                    "metric_type": p.metric_type,
                    "name": p.name,
                    "args": p.args,
                    "desired": p.desired,
                    "source": p.source,
                    "params": p.params,
                }
                for p in predicate_specs
            ],
            "predicate_debug": {
                **metric_debug,
                "template_trace_start": start_trace,
                "template_trace_end": end_trace,
            },
            "review_artifacts": review_artifacts,
        }

        if dry_run:
            result["success"] = None
            result["result_type"] = "dry_run"
            return result

        evaluator.policy.reset()
        evaluator.obs = evaluator._preprocess_obs(evaluator._get_obs_for_policy())

        meta = {
            f"{segment_level}_idx": int(segment_idx),
            f"{segment_level}_desc": segment_desc,
        }
        max_steps = max(
            int(segment_max_steps)
            if segment_max_steps is not None
            else int((end_frame - start_frame) * evaluator.primitive_timeout_multiplier),
            1,
        )
        trace_history: List[List[Dict[str, Any]]] = []
        activation_trace_history: List[List[Dict[str, Any]]] = []
        success = False
        result_type = "timeout"

        window_mode = str(evaluator.cfg.get("segment_predicate_window_mode", "anytime"))
        last_k = int(evaluator.cfg.get("segment_predicate_last_k", 20))
        min_consecutive = max(
            int(evaluator.cfg.get("segment_predicate_min_consecutive", 1)),
            int(metric_debug.get("success_min_consecutive", 1)),
        )
        effective_window_mode = "consecutive" if min_consecutive > 1 else window_mode
        min_success_steps_cfg = int(evaluator.cfg.get("segment_predicate_min_success_steps", 150))
        env_termination_is_terminal = bool(evaluator.cfg.get("segment_predicate_env_termination_is_terminal", False))
        combine_mode = str(metric_debug.get("combine_mode", "all_of"))
        min_success_steps = max(min_success_steps_cfg, 0)
        first_predicate_satisfied_step: Optional[int] = None
        early_predicate_satisfied_steps = 0
        missing_object_traces: List[Dict[str, Any]] = []
        for trace_stage, trace in (("template_start", start_trace), ("template_end", end_trace)):
            for item in trace_missing_objects(trace):
                item["trace_stage"] = trace_stage
                missing_object_traces.append(item)
        result["predicate_debug"]["missing_object_traces"] = missing_object_traces
        if missing_object_traces:
            env_terminal_debug = build_termination_summary(
                terminated=False,
                truncated=False,
                env_current_step=int(getattr(evaluator.env, "_current_step", 0)),
                done_info=None,
                prompt_debug=getattr(evaluator, "_last_prompt_debug", None),
                generated_subtask=getattr(evaluator, "_last_generated_subtask", None),
            )
            result["rollout"] = {
                "max_steps": max_steps,
                "final_step": 0,
                "predicate_window_mode": window_mode,
                "combine_mode": combine_mode,
                "rollout_attempted": False,
                "termination_reason": MISSING_OBJECT_RESULT_TYPE,
                "terminated": False,
                "truncated": False,
                "env_done_success": None,
                "env_terminal_debug": env_terminal_debug,
            }
            result["review_artifacts"]["final_rollout_rgb"] = _capture_review_frame(
                evaluator,
                review_dir / "final_rollout.png" if review_dir is not None else None,
            )
            if bool(evaluator.cfg.get("segment_predicate_dump_trace", True)):
                result["predicate_trace"] = trace_history
            result["success"] = False
            result["result_type"] = MISSING_OBJECT_RESULT_TYPE
            return result
        start_all_satisfied = (
            any(item.get("satisfied", False) for item in start_trace)
            if combine_mode == "any_of"
            else all(item.get("satisfied", False) for item in start_trace)
        ) if len(start_trace) > 0 else False
        require_unsatisfied_at_start = bool(metric_debug.get("require_unsatisfied_at_start", True))
        invalid_start_state = require_unsatisfied_at_start and start_all_satisfied
        result["predicate_debug"]["start_all_satisfied"] = start_all_satisfied
        result["predicate_debug"]["require_unsatisfied_at_start"] = require_unsatisfied_at_start

        if invalid_start_state:
            result["rollout"], result_type = build_pre_satisfied_start_result(
                max_steps=max_steps,
                window_mode=window_mode,
                combine_mode=combine_mode,
                start_all_satisfied=start_all_satisfied,
                require_unsatisfied_at_start=require_unsatisfied_at_start,
            )
            result["review_artifacts"]["final_rollout_rgb"] = _capture_review_frame(
                evaluator,
                review_dir / "final_rollout.png" if review_dir is not None else None,
            )
            if bool(evaluator.cfg.get("segment_predicate_dump_trace", True)):
                result["predicate_trace"] = trace_history
            result["success"] = False
            result["result_type"] = result_type
            return result

        min_yaw_error_improvement = metric_debug.get("min_yaw_error_improvement")
        start_yaw_errors = {
            item.get("predicate"): float(item.get("diagnostics", {}).get("yaw_error"))
            for item in start_trace
            if item.get("metric_type") == "face_object"
            and item.get("diagnostics", {}).get("yaw_error") is not None
        }

        def _yaw_improvement_satisfied(step_trace: List[Dict[str, Any]]) -> bool:
            if min_yaw_error_improvement is None or not start_yaw_errors:
                return True
            try:
                min_improvement = float(min_yaw_error_improvement)
            except (TypeError, ValueError):
                return True
            if min_improvement <= 0:
                return True
            for item in step_trace:
                if item.get("metric_type") != "face_object":
                    continue
                predicate_name = item.get("predicate")
                if predicate_name not in start_yaw_errors:
                    continue
                current_yaw_error = item.get("diagnostics", {}).get("yaw_error")
                if current_yaw_error is None:
                    return False
                if start_yaw_errors[predicate_name] - float(current_yaw_error) < min_improvement:
                    return False
            return True

        final_step = 0
        last_terminated = False
        last_truncated = False
        last_done_info: Dict[str, Any] = {}
        last_env_done_success = None
        env_termination_telemetry: Dict[str, Any] = {
            "env_termination_is_terminal": env_termination_is_terminal,
            "env_terminated_seen": False,
            "env_done_success_seen": False,
            "first_env_terminated_step": None,
            "first_env_done_success_step": None,
            "env_termination_count": 0,
        }
        for step in range(max_steps):
            evaluator.obs["_meta"] = meta
            terminated, truncated = evaluator.step()
            final_step = step + 1
            last_terminated = bool(terminated)
            last_truncated = bool(truncated)
            last_done_info = dict(getattr(evaluator, "_last_done_info", {}) or {})
            last_env_done_success = getattr(evaluator, "_last_env_done_success", None)
            update_segment_env_termination_telemetry(
                env_termination_telemetry,
                step=step + 1,
                terminated=last_terminated,
                env_done_success=last_env_done_success,
            )
            _, step_trace = eval_segment_predicates(evaluator.env, predicate_specs)
            trace_history.append(step_trace)
            activation_trace_history.append(step_trace)

            evaluator._video_primitive_progress = (step + 1) / float(max_steps)
            if evaluator.cfg.write_video:
                evaluator._write_video()

            step_missing_objects = trace_missing_objects(step_trace)
            if step_missing_objects:
                for item in step_missing_objects:
                    item["trace_stage"] = "rollout"
                    item["rollout_step"] = step
                result["predicate_debug"].setdefault("missing_object_traces", []).extend(step_missing_objects)
                success = False
                result_type = MISSING_OBJECT_RESULT_TYPE
                break

            raw_window_satisfied = predicate_window_satisfied(
                activation_trace_history,
                mode=effective_window_mode,
                last_k=last_k,
                min_consecutive=min_consecutive,
                combine_mode=combine_mode,
            )
            eligible_window_satisfied = predicate_window_satisfied(
                activation_trace_history,
                mode=effective_window_mode,
                last_k=last_k,
                min_consecutive=min_consecutive,
                combine_mode=combine_mode,
                min_history_index=max(min_success_steps - 1, 0),
            )
            if raw_window_satisfied:
                if first_predicate_satisfied_step is None:
                    first_predicate_satisfied_step = final_step
                if final_step < min_success_steps:
                    early_predicate_satisfied_steps += 1
                    continue
            if eligible_window_satisfied and _yaw_improvement_satisfied(step_trace):
                success = True
                result_type = "predicate_satisfied"
                break

            if truncated:
                success = False
                result_type = "truncated"
                break

            if terminated and env_termination_is_terminal:
                # Env termination does NOT necessarily imply this segment's predicates were satisfied.
                # The predicate_window_satisfied() check above is the only success condition.
                success = False
                result_type = "env_terminated"
                break

        result_type = finalize_segment_predicate_result_type(
            success=success,
            result_type=result_type,
            env_done_success_seen=bool(env_termination_telemetry["env_done_success_seen"]),
        )
        success, result_type, short_proxy_diagnostics = classify_short_proxy_success(
            success=success,
            result_type=result_type,
            final_step=final_step or max_steps,
            max_steps=max_steps,
            metric_debug=metric_debug,
        )
        result["predicate_debug"].update(short_proxy_diagnostics)
        success, result_type, short_video_diagnostics = classify_short_video_success(
            success=success,
            result_type=result_type,
            final_step=final_step or max_steps,
            max_steps=max_steps,
            min_rollout_steps_for_success=min_success_steps,
        )
        result["predicate_debug"].update(short_video_diagnostics)
        result["predicate_debug"].update(
            {
                "min_success_steps": min_success_steps,
                "min_consecutive": min_consecutive,
                "effective_predicate_window_mode": effective_window_mode,
                "first_predicate_satisfied_step": first_predicate_satisfied_step,
                "early_predicate_satisfied_steps": early_predicate_satisfied_steps,
            }
        )
        env_terminal_debug = build_termination_summary(
            terminated=last_terminated,
            truncated=last_truncated,
            env_current_step=int(getattr(evaluator.env, "_current_step", 0)),
            done_info=last_done_info,
            prompt_debug=getattr(evaluator, "_last_prompt_debug", None),
            generated_subtask=getattr(evaluator, "_last_generated_subtask", None),
        )
        result["rollout"] = {
            "max_steps": max_steps,
            "final_step": final_step or max_steps,
            "predicate_window_mode": window_mode,
            "predicate_min_success_steps": min_success_steps,
            "predicate_min_consecutive": min_consecutive,
            "combine_mode": combine_mode,
            "rollout_attempted": True,
            "termination_reason": result_type,
            "env_termination_reason": env_terminal_debug.get("termination_reason"),
            "terminated": bool(env_termination_telemetry["env_terminated_seen"]),
            "last_terminated": last_terminated,
            "truncated": last_truncated,
            "env_done_success": bool(env_termination_telemetry["env_done_success_seen"])
            if env_termination_telemetry["env_done_success_seen"] or last_env_done_success is not None
            else None,
            "last_env_done_success": None if last_env_done_success is None else bool(last_env_done_success),
            "env_terminated_seen": bool(env_termination_telemetry["env_terminated_seen"]),
            "env_done_success_seen": bool(env_termination_telemetry["env_done_success_seen"]),
            "first_env_terminated_step": env_termination_telemetry["first_env_terminated_step"],
            "first_env_done_success_step": env_termination_telemetry["first_env_done_success_step"],
            "env_termination_count": int(env_termination_telemetry["env_termination_count"]),
            "env_task_success_before_segment_success": result_type == ENV_TASK_SUCCESS_BEFORE_SEGMENT_SUCCESS,
            "env_terminal_debug": env_terminal_debug,
        }
        result["review_artifacts"]["final_rollout_rgb"] = _capture_review_frame(
            evaluator,
            review_dir / "final_rollout.png" if review_dir is not None else None,
        )
        if bool(evaluator.cfg.get("segment_predicate_dump_trace", True)):
            result["predicate_trace"] = trace_history
        result["success"] = success
        result["result_type"] = result_type
        return result

    restored_start, method_start, s_start = restore_and_eval_predicates(evaluator, start_frame)
    logger.info(f"Restore at start frame {start_frame}: restored={restored_start}, method={method_start}")

    if not restored_start:
        return {
            "demo_id": demo_id,
            "segment_level": segment_level,
            "segment_idx": segment_idx,
            "segment_desc": segment_desc,
            "frame_duration": [int(start_frame), int(end_frame)],
            "restore": {"restored": False, "method": method_start, "at": "start"},
            "success": False,
            "result_type": "restore_failed",
        }

    restored_end, method_end, s_end = restore_and_eval_predicates(evaluator, end_frame)
    logger.info(f"Restore at end frame {end_frame}: restored={restored_end}, method={method_end}")

    if not restored_end:
        return {
            "demo_id": demo_id,
            "segment_level": segment_level,
            "segment_idx": segment_idx,
            "segment_desc": segment_desc,
            "frame_duration": [int(start_frame), int(end_frame)],
            "restore": {"restored": False, "method": method_end, "at": "end"},
            "success": False,
            "result_type": "restore_failed",
        }

    subgoal_info = extract_subgoal_for_segment(
        evaluator, segment, ground_options, s_start, s_end,
        topk=evaluator.cfg.get("grounding_topk", 3)
    )

    q_score_start = 0.0
    q_score_end = 0.0
    if subgoal_info["chosen_option_idx"] < len(ground_options):
        chosen_opt = ground_options[subgoal_info["chosen_option_idx"]]
        cs = s_start[subgoal_info["chosen_option_idx"]] if subgoal_info["chosen_option_idx"] < len(s_start) else []
        ce = s_end[subgoal_info["chosen_option_idx"]] if subgoal_info["chosen_option_idx"] < len(s_end) else []
        q_score_start = compute_q_score_delta(chosen_opt, cs, cs)
        q_score_end = compute_q_score_delta(chosen_opt, cs, ce)

    result = {
        "demo_id": demo_id,
        "segment_level": segment_level,
        "segment_idx": segment_idx,
        "segment_desc": segment_desc,
        "frame_duration": [int(start_frame), int(end_frame)],
        "restore": {
            "start": {"restored": True, "method": method_start},
            "end": {"restored": True, "method": method_end},
        },
        "success_mode": str(success_mode),
        "grounding": {
            "chosen_option_idx": subgoal_info["chosen_option_idx"],
            "topk_candidates": subgoal_info["topk_candidates"],
            "n_options": subgoal_info["n_grounding_options"],
        },
        "subgoal": {
            "indices": subgoal_info["subgoal_indices"],
            "predicates": subgoal_info["subgoal_predicates"],
            "size": len(subgoal_info["subgoal_indices"]),
        },
        "q_score": {
            "start": q_score_start,
            "end": q_score_end,
            "delta": q_score_end - q_score_start,
        },
    }

    # If this segment does not induce any task-level predicate delta (common for intermediate skills like "pick up from"),
    # then predicate_subgoal can never terminate successfully. Fall back to state_match in that case.
    requested_success_mode = str(success_mode)
    effective_success_mode = requested_success_mode
    fallback_reason = None
    if requested_success_mode == "predicate_subgoal" and len(subgoal_info["subgoal_indices"]) == 0:
        effective_success_mode = "state_match"
        fallback_reason = "empty_subgoal_fallback_to_state_match"
        logger.warning(
            "Empty subgoal_indices for %s=%s (%s). Falling back from predicate_subgoal -> state_match",
            segment_level,
            segment_idx,
            segment_desc,
        )
    result["effective_success_mode"] = effective_success_mode
    result["success_fallback_reason"] = fallback_reason

    if dry_run:
        result["result_type"] = "dry_run"
        result["success"] = None
        logger.info("Dry run mode - skipping rollout")
        return result

    evaluator.policy.reset()
    evaluator.obs = evaluator._preprocess_obs(evaluator._get_obs_for_policy())

    meta = {
        f"{segment_level}_idx": int(segment_idx),
        f"{segment_level}_desc": segment_desc,
    }
    expected_skill = evaluator.cfg.get("expected_skill", None)
    if expected_skill is not None:
        meta["expected_skill"] = str(expected_skill)

    if segment_max_steps is None:
        segment_max_steps = int((end_frame - start_frame) * evaluator.primitive_timeout_multiplier)
    max_steps = max(int(segment_max_steps), 1)

    demo_annotations = evaluator.load_demo_annotations(demo_id) or {}
    evaluator._video_n_primitives = len(demo_annotations.get(f"{segment_level}_annotation", []))
    evaluator._video_primitive_idx = segment_idx + 1
    evaluator._video_primitive_desc = segment_desc
    evaluator._video_primitive_progress = 0.0

    subgoal_indices = subgoal_info["subgoal_indices"]
    success = False
    result_type = "timeout"
    best_progress = 0.0

    for step in range(max_steps):
        evaluator.obs["_meta"] = meta
        terminated, truncated = evaluator.step()

        current_truth = None
        if subgoal_info["chosen_option_idx"] < len(ground_options):
            chosen_opt = ground_options[subgoal_info["chosen_option_idx"]]
            current_truth = eval_ground_option(chosen_opt)

        current_progress = 0.0
        if current_truth is not None and subgoal_indices:
            current_progress = compute_subgoal_progress(current_truth, subgoal_indices)
            best_progress = max(best_progress, current_progress)

        evaluator._video_primitive_progress = (step + 1) / float(max_steps)
        if evaluator.cfg.write_video:
            evaluator._write_video()

        if truncated:
            success = False
            result_type = "truncated"
            break

        if terminated:
            success = True
            result_type = "env_terminated"
            break

        if effective_success_mode == "predicate_subgoal" and subgoal_indices and current_truth is not None:
            all_satisfied = all(current_truth[i] for i in subgoal_indices)
            if all_satisfied:
                success = True
                result_type = "predicate_satisfied"
                break

        if step >= max_steps - 1:
            result_type = "timeout"

    result["rollout"] = {
        "max_steps": max_steps,
        "final_step": step + 1 if 'step' in dir() else max_steps,
        "best_progress": best_progress,
        "final_progress": current_progress if 'current_progress' in dir() else 0.0,
    }

    if effective_success_mode == "state_match":
        done, rt = evaluator.check_primitive_success(
            primitive=segment,
            current_step=step + 1 if 'step' in dir() else max_steps,
            terminated=terminated if 'terminated' in dir() else False,
            timeout_steps=max_steps,
        )
        success = str(rt).startswith("success")
        result_type = str(rt)

    result["success"] = success
    result["result_type"] = result_type

    return result


def _reconfigure_for_segment(
    evaluator: SubTaskEvaluator,
    sample: Dict[str, Any],
) -> Dict[str, Any]:
    """Mutate ``evaluator.cfg`` and per-segment caches for the given sample.

    The evaluator is task-bound at construction time, so this helper refuses to
    switch tasks. It updates per-segment cfg fields, releases stale rawdata
    caches, ensures output directories exist, reloads the task instance, and
    (re)opens the video writer if requested. Returns a context dict with the
    resolved per-segment paths used by :func:`run_segment_on_env`.
    """
    cfg = evaluator.cfg
    task_name = str(sample["task_name"])
    if task_name != str(cfg.task.name):
        raise RuntimeError(
            f"_reconfigure_for_segment cannot switch tasks "
            f"({cfg.task.name} -> {task_name}); rebuild the evaluator for a different task."
        )

    demo_id = _as_demo_id(sample["demo_id"])
    segment_level = str(sample.get("segment_level", cfg.get("segment_level", "skill")))
    if "segment_idx" in sample:
        segment_idx = int(sample["segment_idx"])
    elif "skill_idx" in sample:
        segment_idx = int(sample["skill_idx"])
    else:
        raise KeyError("sample must contain segment_idx or skill_idx")
    segment_max_steps = sample.get("segment_max_steps")
    if segment_max_steps is None:
        segment_max_steps = sample.get("dynamic_max_steps")
    if segment_max_steps is not None:
        segment_max_steps = int(segment_max_steps)
    success_mode = str(sample.get("success_mode", cfg.get("success_mode", "segment_predicates")))
    dry_run = bool(sample.get("dry_run", cfg.get("dry_run", False)))
    write_video = bool(sample.get("write_video", cfg.get("write_video", False)))
    test_hidden = bool(sample.get("test_hidden", cfg.get("test_hidden", False)))
    expected_skill = sample.get("expected_skill", sample.get("skill"))
    log_path = Path(str(sample["log_path"])).expanduser()

    # Mutate cfg in place so existing helpers (e.g. run_single_segment) see the new values.
    cfg.demo_id = demo_id
    cfg.segment_level = segment_level
    cfg.segment_idx = segment_idx
    if segment_max_steps is not None:
        cfg.segment_max_steps = segment_max_steps
    cfg.success_mode = success_mode
    cfg.dry_run = dry_run
    cfg.write_video = write_video
    cfg.test_hidden = test_hidden
    cfg.log_path = str(log_path)
    if expected_skill is not None:
        cfg.expected_skill = str(expected_skill)

    # Release any rawdata caches lingering from a previous segment to avoid
    # reusing a stale HDF5 handle for a different demo_id.
    raw = getattr(evaluator, "current_rawdata_hdf5", None)
    if raw is not None:
        try:
            raw.close()
        except Exception:
            pass
        evaluator.current_rawdata_hdf5 = None
    evaluator.current_primitive_state_cache = None
    evaluator.current_demo_id = None
    evaluator.current_demo_data = None

    # Ensure per-segment output directories exist.
    metrics_path = log_path / "metrics"
    metrics_path.mkdir(parents=True, exist_ok=True)
    review_dir = log_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    video_dir: Optional[Path] = None
    if write_video:
        video_dir = log_path / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)

    instance_id = _get_instance_id_from_demo_id(demo_id)

    # Mirror the original main() flow: reset → load_task_instance → reset.
    evaluator.reset()
    evaluator.load_task_instance(instance_id, test_hidden=test_hidden)

    video_name: Optional[str] = None
    if write_video and video_dir is not None:
        video_name = str(
            video_dir / f"{task_name}_demo{demo_id}_{segment_level}{segment_idx:03d}.mp4"
        )
        evaluator.video_writer = create_video_writer(
            fpath=video_name,
            resolution=(448, 672),
        )
    else:
        # Drop any leftover writer from a previous segment.
        evaluator.video_writer = None

    evaluator.reset()

    return {
        "task_name": task_name,
        "demo_id": demo_id,
        "instance_id": instance_id,
        "segment_level": segment_level,
        "segment_idx": segment_idx,
        "segment_max_steps": segment_max_steps,
        "success_mode": success_mode,
        "dry_run": dry_run,
        "write_video": write_video,
        "log_path": log_path,
        "metrics_path": metrics_path,
        "review_dir": review_dir,
        "video_name": video_name,
    }


def run_segment_on_env(
    evaluator: SubTaskEvaluator,
    sample: Dict[str, Any],
    *,
    write_metrics: bool = True,
) -> Dict[str, Any]:
    """Run one segment on a long-lived evaluator and return the metrics dict.

    Mirrors the per-segment block of :func:`__main__`: reconfigure the
    evaluator, run the rollout, pad the review-video tail, optionally write
    the metrics JSON, flush the video writer, and release rawdata caches.
    Always cleans up cached HDF5 handles, even if the rollout raised.
    """
    ctx = _reconfigure_for_segment(evaluator, sample)
    cfg = evaluator.cfg
    try:
        result = run_single_segment(
            evaluator=evaluator,
            demo_id=ctx["demo_id"],
            segment_level=ctx["segment_level"],
            segment_idx=ctx["segment_idx"],
            success_mode=ctx["success_mode"],
            dry_run=ctx["dry_run"],
            segment_max_steps=ctx["segment_max_steps"],
            review_dir=ctx["review_dir"],
        )

        result["task_name"] = ctx["task_name"]
        result["instance_id"] = ctx["instance_id"]

        video_name = ctx["video_name"]
        if ctx["write_video"] and video_name is not None:
            review_artifacts = result.setdefault("review_artifacts", {})
            rollout = result.get("rollout") or {}
            rollout_frame_count_estimate = int(rollout.get("final_step") or 0)
            video_rate = 30
            min_meaningful_frames = int(cfg.get("review_video_min_meaningful_frames", 30))
            tail_frames_appended = _pad_review_video_tail(
                evaluator,
                current_frames=rollout_frame_count_estimate,
                min_frames=min_meaningful_frames,
            )
            estimated_frames = rollout_frame_count_estimate + tail_frames_appended
            review_artifacts.update(
                {
                    "video_path": video_name,
                    "video_rollout_frame_count_estimate": rollout_frame_count_estimate,
                    "video_tail_frames_appended": tail_frames_appended,
                    "video_frame_count_estimate": estimated_frames,
                    "video_duration_s_estimate": estimated_frames / float(video_rate) if estimated_frames else 0.0,
                    "video_min_meaningful_frames": min_meaningful_frames,
                    "video_too_short_for_review": estimated_frames < min_meaningful_frames,
                    "video_early_stop_reason": result.get("result_type") if rollout_frame_count_estimate < min_meaningful_frames else None,
                }
            )

        if write_metrics:
            out_path = (
                ctx["metrics_path"]
                / f"segment_eval_{ctx['task_name']}_{ctx['demo_id']}_{ctx['segment_level']}{ctx['segment_idx']:03d}.json"
            )
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            result["_metrics_path"] = str(out_path)
            logger.info(f"Saved metrics to {out_path}")

        if ctx["write_video"] and video_name is not None:
            # Flush + close the writer so the mp4 is finalized before we move on.
            evaluator.video_writer = None
            logger.info(f"Saved video to {video_name}")

        return result
    finally:
        raw = getattr(evaluator, "current_rawdata_hdf5", None)
        if raw is not None:
            try:
                raw.close()
            except Exception:
                pass
            evaluator.current_rawdata_hdf5 = None
        evaluator.current_primitive_state_cache = None


def _build_sample_from_cli_config(config: DictConfig) -> Dict[str, Any]:
    """Build a sample dict from the Hydra CLI config so the CLI entrypoint
    can route through ``run_segment_on_env``.
    """
    segment_level = config.get("segment_level", "primitive")
    segment_idx = config.get("segment_idx", None)
    if segment_idx is None:
        raise RuntimeError("Missing required config: segment_idx")
    segment_idx = int(segment_idx)

    demo_id_cfg = config.get("demo_id", None)
    if demo_id_cfg is not None:
        demo_id = _as_demo_id(demo_id_cfg)
    else:
        demo_data_path = config.get("demo_data_path", None)
        if demo_data_path is None:
            raise RuntimeError("demo_data_path must be specified (points to 2025-challenge-demos)")
        demo_ids = get_demo_ids_for_task(
            demo_data_path=str(demo_data_path),
            task_name=config.task.name,
            limit=1,
        )
        if not demo_ids:
            raise RuntimeError(f"No demos found for task {config.task.name}")
        demo_id = demo_ids[0]

    return {
        "task_name": str(config.task.name),
        "demo_id": demo_id,
        "segment_level": segment_level,
        "segment_idx": segment_idx,
        "segment_max_steps": config.get("segment_max_steps", None),
        "success_mode": config.get("success_mode", "predicate_subgoal"),
        "dry_run": bool(config.get("dry_run", False)),
        "write_video": bool(config.get("write_video", False)),
        "test_hidden": bool(config.get("test_hidden", False)),
        "log_path": str(Path(str(config.log_path)).expanduser()),
        "expected_skill": config.get("expected_skill", None),
    }


if __name__ == "__main__":
    register_omegaconf_resolvers()

    src = getsourcefile(lambda: 0) or __file__
    with hydra.initialize_config_dir(
        f"{Path(src).parents[0]}/configs",
        version_base="1.1",
    ):
        config = hydra.compose("eval_segment_config.yaml", overrides=sys.argv[1:])

    OmegaConf.resolve(config)
    OmegaConf.set_struct(config, False)

    if config.get("demo_data_path", None) is None:
        logger.error("demo_data_path must be specified (points to 2025-challenge-demos)")
        sys.exit(1)

    with gm.unlocked():
        gm.HEADLESS = config.headless

    try:
        sample = _build_sample_from_cli_config(config)
    except RuntimeError as exc:
        logger.error(str(exc))
        sys.exit(1)

    with SubTaskEvaluator(config) as evaluator:
        run_segment_on_env(evaluator, sample, write_metrics=True)
