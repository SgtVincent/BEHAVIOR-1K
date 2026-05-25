"""Batch primitive evaluator for BEHAVIOR-1K demos.

This script is a thin wrapper around `eval_primitive.py`'s single-primitive runner.
It selects matching primitives by `primitive_desc`, runs a set of (demo_id, primitive_idx)
trials, writes per-trial JSON metrics files, and emits a `summary.json`.

Typical usage (client side; policy served separately):

  python OmniGibson/omnigibson/learning/eval_primitive_batch.py \
    policy=websocket task.name=make_pizza \
    env_wrapper._target_=omnigibson.learning.wrappers.wbvima_wrapper.WBVIMAWrapper \
    demo_data_path=/mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-demos \
    rawdata_path=/mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-rawdata \
    primitive_desc="pick up from" \
    primitive_max_steps=400 \
    log_path=./eval_logs/batch_act_rawdata_T400_d3_p2_stateMatch

"""

from __future__ import annotations

import hydra
import json
import logging
import sys
import time
from collections import Counter
from inspect import getsourcefile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import OmegaConf

from omnigibson.learning.eval_primitive import _as_demo_id, _get_instance_id_from_demo_id, run_single_primitive
from omnigibson.learning.eval_subtask_reset import SubTaskEvaluator, get_demo_ids_for_task
from omnigibson.learning.utils.config_utils import register_omegaconf_resolvers
from omnigibson.learning.utils.obs_utils import create_video_writer
from omnigibson.macros import gm


logger = logging.getLogger("primitive_batch_evaluator")
logger.setLevel(20)


def _norm_desc(s: Any) -> str:
    s = str(s).lower().strip()
    s = s.replace("_", " ").replace("-", " ")
    s = " ".join(s.split())
    return s


def _primitive_matches(desc: str, query: str) -> bool:
    # substring match after normalization (e.g., "pick_up_from" matches "pick up from")
    return _norm_desc(query) in _norm_desc(desc)


def _get_frame_start(prim: Dict[str, Any]) -> int:
    """Extract start frame from primitive, handling malformed frame_duration."""
    fd = prim.get("frame_duration", [0, 0])
    if not fd:
        return 0
    first = fd[0]
    # Handle nested list case like [[8664, 9403], 12222]
    if isinstance(first, list):
        return first[0] if first else 0
    return first


def build_plan(
    evaluator: SubTaskEvaluator,
    demo_ids: List[str],
    primitive_desc_query: str,
    max_primitives_per_demo: int,
) -> List[Dict[str, Any]]:
    plan: List[Dict[str, Any]] = []
    for demo_id in demo_ids:
        annotations = evaluator.load_demo_annotations(demo_id)
        if annotations is None:
            logger.warning(f"Skipping demo {demo_id}: no annotations")
            continue
        primitives = annotations.get("primitive_annotation", [])
        if not primitives:
            logger.warning(f"Skipping demo {demo_id}: no primitives")
            continue
        primitives = sorted(primitives, key=_get_frame_start)

        matched: List[int] = []
        for idx, prim in enumerate(primitives):
            desc = prim.get("primitive_description", ["unknown"])[0]
            if _primitive_matches(desc, primitive_desc_query):
                matched.append(int(idx))
                if len(matched) >= int(max_primitives_per_demo):
                    break

        if not matched:
            logger.warning(f"No primitives matched '{primitive_desc_query}' in demo {demo_id}")
            continue

        for primitive_idx in matched:
            plan.append({"demo_id": demo_id, "primitive_idx": int(primitive_idx)})

    return plan


if __name__ == "__main__":
    register_omegaconf_resolvers()

    src = getsourcefile(lambda: 0) or __file__
    with hydra.initialize_config_dir(
        f"{Path(src).parents[0]}/configs",
        version_base="1.1",
    ):
        config = hydra.compose("eval_primitive_batch_config.yaml", overrides=sys.argv[1:])

    OmegaConf.resolve(config)
    OmegaConf.set_struct(config, False)

    # Required args
    if config.get("demo_data_path", None) is None:
        logger.error("demo_data_path must be specified (points to 2025-challenge-demos)")
        sys.exit(1)
    if config.get("primitive_desc", None) is None:
        logger.error("primitive_desc must be specified (e.g., primitive_desc=\"pick up from\")")
        sys.exit(1)
    if config.get("log_path", None) is None:
        logger.error("log_path must be specified")
        sys.exit(1)

    primitive_desc_query = str(config.primitive_desc)
    max_primitives_per_demo = int(config.get("max_primitives_per_demo", 1))

    # Optional
    primitive_max_steps = config.get("primitive_max_steps", None)
    if primitive_max_steps is not None:
        primitive_max_steps = int(primitive_max_steps)

    # Set headless mode
    gm.HEADLESS = bool(config.headless)

    # Output folders
    log_root = Path(str(config.log_path)).expanduser()
    metrics_path = log_root / "metrics"
    metrics_path.mkdir(parents=True, exist_ok=True)
    video_dir = None
    if bool(config.write_video):
        video_dir = log_root / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)

    # Select demo ids
    demo_ids_cfg = config.get("demo_ids", None)
    if demo_ids_cfg is not None:
        demo_ids = [_as_demo_id(x) for x in list(demo_ids_cfg)]
    else:
        demo_ids = get_demo_ids_for_task(
            demo_data_path=str(config.demo_data_path),
            task_name=config.task.name,
            limit=config.get("num_demos", None),
        )
    if not demo_ids:
        logger.error(f"No demos found for task {config.task.name}")
        sys.exit(1)

    # Run
    t0 = time.time()
    results: List[Dict[str, Any]] = []

    with SubTaskEvaluator(config) as evaluator:
        evaluator.reset()

        plan = build_plan(
            evaluator=evaluator,
            demo_ids=demo_ids,
            primitive_desc_query=primitive_desc_query,
            max_primitives_per_demo=max_primitives_per_demo,
        )
        if not plan:
            logger.error("Empty plan: no (demo_id, primitive_idx) pairs to run")
            sys.exit(1)

        logger.info(f"Batch plan has {len(plan)} trials")

        for item in plan:
            demo_id = str(item["demo_id"])
            primitive_idx = int(item["primitive_idx"])
            instance_id = _get_instance_id_from_demo_id(demo_id)

            evaluator.reset()
            evaluator.load_task_instance(instance_id, test_hidden=bool(config.test_hidden))

            # Setup video writer per trial
            video_name = None
            if bool(config.write_video) and video_dir is not None:
                video_name = str(video_dir / f"{config.task.name}_demo{demo_id}_prim{primitive_idx:03d}.mp4")
                evaluator.video_writer = create_video_writer(
                    fpath=video_name,
                    resolution=(448, 672),
                )

            evaluator.reset()

            trial = run_single_primitive(
                evaluator=evaluator,
                demo_id=demo_id,
                primitive_idx=primitive_idx,
                primitive_max_steps=primitive_max_steps,
            )
            trial["task_name"] = config.task.name
            trial["instance_id"] = int(instance_id)

            out_path = metrics_path / f"primitive_eval_{config.task.name}_{demo_id}_prim{primitive_idx:03d}.json"
            with open(out_path, "w") as f:
                json.dump(trial, f, indent=2)

            # Flush video
            if bool(config.write_video) and video_name is not None:
                evaluator.video_writer = None  # type: ignore[assignment]

            # Close raw handle if opened (run_single_primitive opens per demo)
            raw = getattr(evaluator, "current_rawdata_hdf5", None)
            if raw is not None:
                try:
                    raw.close()
                except Exception:
                    pass
                evaluator.current_rawdata_hdf5 = None
            evaluator.current_primitive_state_cache = None

            results.append(
                {
                    "demo_id": demo_id,
                    "primitive_idx": primitive_idx,
                    "success": bool(trial.get("success", False)),
                    "result_type": str(trial.get("result_type", "unknown")),
                    "restore_method": str(trial.get("restore", {}).get("method", "unknown")),
                    "metrics_path": str(out_path),
                }
            )

            logger.info(
                f"Trial demo={demo_id} prim={primitive_idx:03d}: "
                f"success={results[-1]['success']} type={results[-1]['result_type']} restore={results[-1]['restore_method']}"
            )

        # IMPORTANT: write summary BEFORE evaluator context exits.
        # `Evaluator.__exit__` calls `og.shutdown()`, which may terminate the process.
        elapsed_s = time.time() - t0
        by_result_type = Counter([r["result_type"] for r in results])
        by_restore_method = Counter([r["restore_method"] for r in results])
        n_success = sum(1 for r in results if r["success"])

        summary = {
            "task": str(config.task.name),
            "primitive_desc": str(primitive_desc_query),
            "demo_ids": list(demo_ids),
            "plan": plan,
            "primitive_max_steps": primitive_max_steps,
            "rawdata_path": config.get("rawdata_path", None),
            "n_trials": len(results),
            "n_success": int(n_success),
            "success_rate": float(n_success / max(1, len(results))),
            "by_result_type": dict(by_result_type),
            "by_restore_method": dict(by_restore_method),
            "elapsed_s": float(elapsed_s),
            "results": results,
        }

        with open(log_root / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved summary to {log_root / 'summary.json'}")