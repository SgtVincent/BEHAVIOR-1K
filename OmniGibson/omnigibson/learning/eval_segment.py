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
from inspect import getsourcefile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
from omegaconf import DictConfig, OmegaConf

from omnigibson.learning.eval_subtask_reset import SubTaskEvaluator, get_demo_ids_for_task
from omnigibson.learning.utils.config_utils import register_omegaconf_resolvers
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
    build_auto_mined_predicates,
    build_template_predicates,
    eval_segment_predicates,
    predicate_window_satisfied,
)
from omnigibson.macros import gm


logger = logging.getLogger("segment_evaluator")
logger.setLevel(logging.INFO)


def _as_demo_id(x: Any) -> str:
    s = str(x)
    if s.isdigit() and len(s) < 8:
        return s.zfill(8)
    return s


def _get_instance_id_from_demo_id(demo_id: str) -> int:
    return int(demo_id) // 10 % 1000


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
        evaluator.current_demo_id = demo_id
        evaluator.current_rawdata_hdf5 = evaluator.load_rawdata_hdf5(demo_id)
        if evaluator.current_rawdata_hdf5 is None:
            evaluator.current_primitive_state_cache = evaluator.load_primitive_state_cache(demo_id)

        restored_start, method_start, _ = restore_and_eval_predicates(evaluator, start_frame)
        restored_end, method_end, _ = restore_and_eval_predicates(evaluator, end_frame)
        if not restored_start or not restored_end:
            return {
                "demo_id": demo_id,
                "segment_level": segment_level,
                "segment_idx": segment_idx,
                "segment_desc": segment_desc,
                "frame_duration": [int(start_frame), int(end_frame)],
                "restore": {
                    "start": {"restored": restored_start, "method": method_start},
                    "end": {"restored": restored_end, "method": method_end},
                },
                "success_mode": str(success_mode),
                "success": False,
                "result_type": "restore_failed",
            }

        template_specs, metric_debug = build_template_predicates(segment_level, segment, evaluator.env)
        restored_start_for_compare, _, _ = restore_and_eval_predicates(evaluator, start_frame)
        if not restored_start_for_compare:
            return {
                "demo_id": demo_id,
                "segment_level": segment_level,
                "segment_idx": segment_idx,
                "segment_desc": segment_desc,
                "frame_duration": [int(start_frame), int(end_frame)],
                "restore": {
                    "start": {"restored": False, "method": method_start},
                    "end": {"restored": True, "method": method_end},
                },
                "success_mode": str(success_mode),
                "success": False,
                "result_type": "restore_failed",
            }
        start_truth_map, start_trace = eval_segment_predicates(evaluator.env, template_specs)
        restored_end, method_end, _ = restore_and_eval_predicates(evaluator, end_frame)
        end_truth_map, end_trace = eval_segment_predicates(evaluator.env, template_specs)
        auto_specs = build_auto_mined_predicates(
            segment_level,
            segment,
            evaluator.env,
            start_truth=start_truth_map,
            end_truth=end_truth_map,
        )
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
                    "start": {"restored": True, "method": method_start},
                    "end": {"restored": True, "method": method_end},
                },
                "success_mode": str(success_mode),
                "effective_success_mode": "segment_predicates",
                "success": False,
                "result_type": "no_predicates_generated",
            }

        # Segment rollout must start from the segment start frame, not the end frame used for target metric capture.
        restored_rollout_start, _, _ = restore_and_eval_predicates(evaluator, start_frame)
        if not restored_rollout_start:
            return {
                "demo_id": demo_id,
                "segment_level": segment_level,
                "segment_idx": segment_idx,
                "segment_desc": segment_desc,
                "frame_duration": [int(start_frame), int(end_frame)],
                "restore": {
                    "start": {"restored": False, "method": method_start},
                    "end": {"restored": True, "method": method_end},
                },
                "success_mode": str(success_mode),
                "effective_success_mode": "segment_predicates",
                "success": False,
                "result_type": "restore_failed_before_rollout",
            }

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
        success = False
        result_type = "timeout"

        window_mode = str(evaluator.cfg.get("segment_predicate_window_mode", "anytime"))
        last_k = int(evaluator.cfg.get("segment_predicate_last_k", 20))
        min_consecutive = int(evaluator.cfg.get("segment_predicate_min_consecutive", 1))
        combine_mode = str(metric_debug.get("combine_mode", "all_of"))
        start_all_satisfied = (
            any(item.get("satisfied", False) for item in start_trace)
            if combine_mode == "any_of"
            else all(item.get("satisfied", False) for item in start_trace)
        ) if len(start_trace) > 0 else False
        require_unsatisfied_at_start = bool(metric_debug.get("require_unsatisfied_at_start", True))
        activation_armed = not (require_unsatisfied_at_start and start_all_satisfied)
        result["predicate_debug"]["start_all_satisfied"] = start_all_satisfied
        result["predicate_debug"]["require_unsatisfied_at_start"] = require_unsatisfied_at_start

        for step in range(max_steps):
            evaluator.obs["_meta"] = meta
            terminated, truncated = evaluator.step()
            _, step_trace = eval_segment_predicates(evaluator.env, predicate_specs)
            trace_history.append(step_trace)
            if not activation_armed:
                step_satisfied_now = (
                    any(item.get("satisfied", False) for item in step_trace)
                    if combine_mode == "any_of"
                    else all(item.get("satisfied", False) for item in step_trace)
                ) if len(step_trace) > 0 else False
                if not step_satisfied_now:
                    activation_armed = True

            evaluator._video_primitive_progress = (step + 1) / float(max_steps)
            if evaluator.cfg.write_video:
                evaluator._write_video()

            if activation_armed and predicate_window_satisfied(
                trace_history,
                mode=window_mode,
                last_k=last_k,
                min_consecutive=min_consecutive,
                combine_mode=combine_mode,
            ):
                success = True
                result_type = "predicate_satisfied"
                break

            if truncated:
                success = False
                result_type = "truncated"
                break

            if terminated:
                success = True
                result_type = "env_terminated"
                break

        result["rollout"] = {
            "max_steps": max_steps,
            "final_step": step + 1 if "step" in dir() else max_steps,
            "predicate_window_mode": window_mode,
            "combine_mode": combine_mode,
        }
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

    segment_level = config.get("segment_level", "primitive")
    segment_idx = config.get("segment_idx", None)
    if segment_idx is None:
        logger.error("Missing required config: segment_idx")
        sys.exit(1)
    segment_idx = int(segment_idx)

    demo_id_cfg = config.get("demo_id", None)
    segment_max_steps = config.get("segment_max_steps", None)
    if segment_max_steps is not None:
        segment_max_steps = int(segment_max_steps)

    success_mode = config.get("success_mode", "predicate_subgoal")
    dry_run = config.get("dry_run", False)

    gm.HEADLESS = config.headless

    log_root = Path(str(config.log_path)).expanduser()
    metrics_path = log_root / "metrics"
    metrics_path.mkdir(parents=True, exist_ok=True)

    video_dir = None
    if config.write_video:
        video_dir = log_root / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)

    demo_data_path = config.get("demo_data_path", None)
    if demo_data_path is None:
        logger.error("demo_data_path must be specified (points to 2025-challenge-demos)")
        sys.exit(1)

    if demo_id_cfg is not None:
        demo_id = _as_demo_id(demo_id_cfg)
    else:
        demo_ids = get_demo_ids_for_task(
            demo_data_path=str(demo_data_path),
            task_name=config.task.name,
            limit=1,
        )
        if not demo_ids:
            logger.error(f"No demos found for task {config.task.name}")
            sys.exit(1)
        demo_id = demo_ids[0]

    instance_id = _get_instance_id_from_demo_id(demo_id)

    with SubTaskEvaluator(config) as evaluator:
        evaluator.reset()
        evaluator.load_task_instance(instance_id, test_hidden=config.test_hidden)

        video_name = None
        if config.write_video and video_dir is not None:
            video_name = str(
                video_dir / f"{config.task.name}_demo{demo_id}_{segment_level}{segment_idx:03d}.mp4"
            )
            evaluator.video_writer = create_video_writer(
                fpath=video_name,
                resolution=(448, 672),
            )

        evaluator.reset()

        result = run_single_segment(
            evaluator=evaluator,
            demo_id=demo_id,
            segment_level=segment_level,
            segment_idx=segment_idx,
            success_mode=success_mode,
            dry_run=dry_run,
            segment_max_steps=segment_max_steps,
        )

        result["task_name"] = config.task.name
        result["instance_id"] = instance_id

        out_path = metrics_path / f"segment_eval_{config.task.name}_{demo_id}_{segment_level}{segment_idx:03d}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        logger.info(f"Saved metrics to {out_path}")
        if config.write_video and video_name is not None:
            evaluator.video_writer = None
            logger.info(f"Saved video to {video_name}")

        raw = getattr(evaluator, "current_rawdata_hdf5", None)
        if raw is not None:
            try:
                raw.close()
            except Exception:
                pass
            evaluator.current_rawdata_hdf5 = None

        evaluator.current_primitive_state_cache = None
