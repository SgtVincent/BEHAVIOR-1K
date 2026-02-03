"""Single-primitive evaluator for BEHAVIOR-1K demos.

This script is meant for debugging / unit-style testing of one primitive.
It restores the world + robot state at the selected primitive boundary
(prefer rawdata full state, else primitive-state cache, else robot-only),
then runs the policy until termination or timeout.

Restore sources (in order):
  1) rawdata_path (full world state from HDF5)
  2) primitive_state_cache_dir (full world state from .npz cache)
  3) demo parquet (robot-only proprioception)

Typical usage:
  python OmniGibson/omnigibson/learning/eval_primitive.py \
    policy=websocket task.name=turning_on_radio \
    env_wrapper._target_=omnigibson.learning.wrappers.wbvima_wrapper.WBVIMAWrapper \
    demo_data_path=/path/to/2025-challenge-demos \
    rawdata_path=/path/to/2025-challenge-rawdata \
    demo_id=00000010 primitive_idx=0 \
    log_path=./eval_logs/primitive_eval

Notes:
- `primitive_idx` refers to the primitive index *after sorting by start frame*.
- If you want to rely only on cached world states, omit `rawdata_path` and make sure
  caches exist at: <demo_data_path>/meta/primitive_states/task-XXXX/episode_YYYYYYYY.npz
  (see OmniGibson/scripts/learning/extract_primitive_world_states.py)

"""

from __future__ import annotations

import hydra
import json
import logging
import os
import sys
from inspect import getsourcefile
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf

from omnigibson.learning.eval_subtask_reset import SubTaskEvaluator, get_demo_ids_for_task
from omnigibson.learning.utils.config_utils import register_omegaconf_resolvers
from omnigibson.learning.utils.obs_utils import create_video_writer
from omnigibson.macros import gm


logger = logging.getLogger("primitive_evaluator")
logger.setLevel(20)


def _as_demo_id(x: Any) -> str:
    s = str(x)
    # Accept "10" or 10 etc.
    if s.isdigit() and len(s) < 8:
        return s.zfill(8)
    return s


def _get_instance_id_from_demo_id(demo_id: str) -> int:
    # demo_id encodes the instance (demo_id // 10 % 1000), same logic as subtask_eval.py
    return int(demo_id) // 10 % 1000


def run_single_primitive(
    evaluator: SubTaskEvaluator,
    demo_id: str,
    primitive_idx: int,
    primitive_max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    # Load annotations + demo data (for primitive definitions and fallback robot restoration)
    annotations = evaluator.load_demo_annotations(demo_id)
    if annotations is None:
        return {"error": "no_annotations"}

    evaluator.current_demo_data = evaluator.load_demo_lowdim_data(demo_id)
    if evaluator.current_demo_data is None:
        return {"error": "no_demo_data"}

    evaluator.current_demo_id = demo_id

    # Load restoration sources
    evaluator.current_rawdata_hdf5 = evaluator.load_rawdata_hdf5(demo_id)
    if evaluator.current_rawdata_hdf5 is None:
        evaluator.current_primitive_state_cache = evaluator.load_primitive_state_cache(demo_id)

    primitives = annotations.get("primitive_annotation", [])
    if not primitives:
        return {"error": "no_primitives"}

    primitives = sorted(primitives, key=lambda x: x["frame_duration"][0])

    if primitive_idx < 0 or primitive_idx >= len(primitives):
        return {
            "error": "primitive_idx_out_of_range",
            "primitive_idx": primitive_idx,
            "n_primitives": len(primitives),
        }

    primitive = primitives[primitive_idx]
    start_frame, end_frame = primitive["frame_duration"]
    primitive_desc = primitive.get("primitive_description", ["unknown"])[0]

    # Optional policy guidance.
    # These are forwarded as obs['_meta'] each step.
    # Keep this evaluator policy-agnostic: only include generic primitive metadata.
    # `expected_skill` is an optional *hint* some policies may choose to use.
    expected_skill = evaluator.cfg.get("expected_skill", None)
    meta = {
        "primitive_idx": int(primitive_idx),
        "primitive_desc": primitive_desc,
    }
    if expected_skill is not None:
        meta["expected_skill"] = str(expected_skill)

    # Video overlay info
    evaluator._video_n_primitives = len(primitives)
    evaluator._video_primitive_idx = primitive_idx + 1
    evaluator._video_primitive_desc = primitive_desc
    evaluator._video_primitive_progress = 0.0

    logger.info("\n" + "=" * 60)
    logger.info(f"Evaluating SINGLE primitive idx={primitive_idx} / {len(primitives)-1}")
    logger.info(f"Demo: {demo_id}")
    logger.info(f"Primitive: {primitive_desc}")
    logger.info(f"Frames: {start_frame} - {end_frame}")
    logger.info("=" * 60)

    # Always restore at the chosen primitive start
    logger.info(f"Restoring state to primitive start frame {start_frame}")
    restored, method = evaluator._try_restore_to_frame(int(start_frame))
    logger.info(f"State restore result: restored={restored} method={method}")

    if not restored:
        result = {
            "demo_id": demo_id,
            "primitive_idx": primitive_idx,
            "primitive_desc": primitive_desc,
            "frame_duration": [int(start_frame), int(end_frame)],
            "restore": {"restored": False, "method": method},
            "success": False,
            "result_type": "restore_failed",
        }
        return result

    # Reset policy for a clean primitive attempt
    evaluator.policy.reset()

    # Refresh obs after restoration (important for wrappers injecting task obs)
    evaluator.obs = evaluator._preprocess_obs(evaluator._get_obs_for_policy())

    # Execute primitive (inject meta every step so it survives env.step() updates)
    if primitive_max_steps is None:
        primitive_max_steps = evaluator.get_primitive_timeout(primitive)
    max_steps = int(primitive_max_steps)

    success = False
    result_type = "timeout"
    best_state_errors: Optional[Dict[str, float]] = None
    final_state_errors: Optional[Dict[str, float]] = None
    for step in range(max_steps):
        evaluator.obs["_meta"] = meta
        terminated, truncated = evaluator.step()

        # Compute state-match errors for debugging and (optional) primitive-level success.
        # This is stored on the evaluator by `check_primitive_success` as well, but we keep
        # explicit tracking here so the single-primitive JSON is self-contained.
        done, rt = evaluator.check_primitive_success(
            primitive=primitive,
            current_step=step + 1,
            terminated=terminated,
            timeout_steps=max_steps,
        )

        state_errors = getattr(evaluator, "_last_primitive_state_errors", None)
        if isinstance(state_errors, dict):
            final_state_errors = state_errors
            if best_state_errors is None:
                best_state_errors = dict(state_errors)
            else:
                # Keep the best (minimum) error seen so far per metric
                for k, v in state_errors.items():
                    try:
                        best_state_errors[k] = float(min(best_state_errors.get(k, float("inf")), float(v)))
                    except Exception:
                        pass

        evaluator._video_primitive_progress = (step + 1) / float(max_steps)
        if evaluator.cfg.write_video:
            evaluator._write_video()

        if truncated:
            success = False
            result_type = "truncated"
            break

        if done:
            success = str(rt).startswith("success")
            result_type = str(rt)
            break

    result = {
        "demo_id": demo_id,
        "primitive_idx": primitive_idx,
        "primitive_desc": primitive_desc,
        "frame_duration": [int(start_frame), int(end_frame)],
        "restore": {"restored": True, "method": method},
        "success": bool(success),
        "result_type": str(result_type),
    }

    # Add primitive-level success debugging info (if available)
    thresholds = None
    try:
        thresholds = evaluator.get_primitive_success_thresholds()
    except Exception:
        thresholds = None
    if thresholds is not None or best_state_errors is not None or final_state_errors is not None:
        result["primitive_success_debug"] = {
            "criterion": "state_match" if evaluator.cfg.get("primitive_success_use_state_match", True) else "env_terminated",
            "thresholds": thresholds,
            "best_state_errors": best_state_errors,
            "final_state_errors": final_state_errors,
            "success_reason": getattr(evaluator, "_last_primitive_success_reason", None),
        }

    return result


if __name__ == "__main__":
    register_omegaconf_resolvers()

    # Load config
    src = getsourcefile(lambda: 0) or __file__
    with hydra.initialize_config_dir(
        f"{Path(src).parents[0]}/configs",
        version_base="1.1",
    ):
        config = hydra.compose("eval_primitive_config.yaml", overrides=sys.argv[1:])

    OmegaConf.resolve(config)
    OmegaConf.set_struct(config, False)

    # Required args
    primitive_idx = config.get("primitive_idx", None)
    if primitive_idx is None:
        logger.error("Missing required config: primitive_idx (e.g., primitive_idx=0)")
        sys.exit(1)
    primitive_idx = int(primitive_idx)

    # Optional args
    demo_id_cfg = config.get("demo_id", None)
    primitive_max_steps = config.get("primitive_max_steps", None)
    if primitive_max_steps is not None:
        primitive_max_steps = int(primitive_max_steps)

    # Set headless mode
    gm.HEADLESS = config.headless

    # Create output folders
    # `log_path` is required by config, but keep typing / runtime robust.
    log_root = Path(str(config.log_path)).expanduser()
    metrics_path = log_root / "metrics"
    metrics_path.mkdir(parents=True, exist_ok=True)

    video_dir = None
    if config.write_video:
        video_dir = log_root / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)

    # Determine demo_id
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

    # Run
    with SubTaskEvaluator(config) as evaluator:
        evaluator.reset()
        evaluator.load_task_instance(instance_id, test_hidden=config.test_hidden)

        # Setup video writer
        video_name = None
        if config.write_video and video_dir is not None:
            video_name = str(video_dir / f"{config.task.name}_demo{demo_id}_prim{primitive_idx:03d}.mp4")
            evaluator.video_writer = create_video_writer(
                fpath=video_name,
                resolution=(448, 672),
            )

        evaluator.reset()

        result = run_single_primitive(
            evaluator=evaluator,
            demo_id=demo_id,
            primitive_idx=primitive_idx,
            primitive_max_steps=primitive_max_steps,
        )
        result["task_name"] = config.task.name
        result["instance_id"] = instance_id

        out_path = metrics_path / f"primitive_eval_{config.task.name}_{demo_id}_prim{primitive_idx:03d}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        logger.info(f"Saved metrics to {out_path}")
        if config.write_video and video_name is not None:
            evaluator.video_writer = None  # type: ignore[assignment]  # flush
            logger.info(f"Saved video to {video_name}")

        # Cleanup raw data handle if opened
        raw = getattr(evaluator, "current_rawdata_hdf5", None)
        if raw is not None:
            try:
                raw.close()
            except Exception:
                pass
            evaluator.current_rawdata_hdf5 = None

        evaluator.current_primitive_state_cache = None