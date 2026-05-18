"""Batch golden-rule evaluator for BEHAVIOR-1K demos.

This script runs full-episode golden-rule evaluation across multiple demos,
following the GT skill plan for each demo.  It produces per-demo JSON metrics
and a summary.json with aggregate statistics.

Typical usage (client side; policy served separately via serve_golden_rule.py):

  python OmniGibson/omnigibson/learning/eval_golden_rule_batch.py \
    policy=websocket task.name=turning_on_radio \
    demo_data_path=/mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-demos \
    num_demos=10 \
    log_path=./eval_logs/golden_rule_batch

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
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf

from omnigibson.learning.eval_golden_rule import GoldenRuleEvaluator, get_demo_ids_for_task
from omnigibson.learning.utils.config_utils import register_omegaconf_resolvers
from omnigibson.learning.utils.obs_utils import create_video_writer
from omnigibson.macros import gm


logger = logging.getLogger("golden_rule_batch_evaluator")
logger.setLevel(logging.INFO)


def _as_demo_id(x: Any) -> str:
    """Normalize a demo identifier to a string."""
    s = str(x).strip()
    # If it looks like an integer, zero-pad to 8 digits
    try:
        return f"{int(s):08d}"
    except ValueError:
        return s


def _get_instance_id_from_demo_id(demo_id: str) -> int:
    """Extract instance_id from demo_id (same heuristic as eval_primitive.py)."""
    try:
        return int(demo_id) // 10 % 1000
    except ValueError:
        return 0


if __name__ == "__main__":
    register_omegaconf_resolvers()

    src = getsourcefile(lambda: 0) or __file__
    with hydra.initialize_config_dir(
        f"{Path(src).parents[0]}/configs",
        version_base="1.1",
    ):
        config = hydra.compose("eval_golden_rule_batch_config.yaml", overrides=sys.argv[1:])

    OmegaConf.resolve(config)
    OmegaConf.set_struct(config, False)

    # Required args
    if config.get("demo_data_path", None) is None:
        logger.error("demo_data_path must be specified (points to 2025-challenge-demos)")
        sys.exit(1)
    if config.get("log_path", None) is None:
        logger.error("log_path must be specified")
        sys.exit(1)

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

    logger.info(f"Evaluating {len(demo_ids)} demo(s) for task {config.task.name}")

    # Run
    t0 = time.time()
    all_results: List[Dict[str, Any]] = []

    with GoldenRuleEvaluator(config) as evaluator:
        for demo_idx, demo_id in enumerate(demo_ids):
            logger.info("")
            logger.info("=" * 60)
            logger.info(f"Evaluating demo {demo_idx + 1}/{len(demo_ids)}: {demo_id}")
            logger.info("=" * 60)

            evaluator.reset()
            instance_id = _get_instance_id_from_demo_id(demo_id)
            evaluator.load_task_instance(instance_id, test_hidden=bool(config.test_hidden))

            # Setup video writer per demo
            video_name = None
            if bool(config.write_video) and video_dir is not None:
                video_name = str(video_dir / f"{config.task.name}_golden_rule_demo{demo_id}.mp4")
                evaluator.video_writer = create_video_writer(
                    fpath=video_name,
                    resolution=(448, 672),
                )

            evaluator.reset()
            results = evaluator.run_episode(demo_id=demo_id)
            results["instance_id"] = instance_id
            results["task_name"] = config.task.name

            # Write per-demo metrics
            out_path = metrics_path / f"golden_rule_{config.task.name}_{demo_id}.json"
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

            # Flush video
            if bool(config.write_video) and video_name is not None:
                evaluator.video_writer = None  # type: ignore[assignment]
                logger.info(f"Saved video to {video_name}")

            # Close rawdata handle if opened
            raw = getattr(evaluator, "current_rawdata_hdf5", None)
            if raw is not None:
                try:
                    raw.close()
                except Exception:
                    pass
                evaluator.current_rawdata_hdf5 = None
            evaluator.current_primitive_state_cache = None

            # Record summary row
            skill_results = results.get("skill_results", [])
            n_skills = results.get("n_skills", 0)
            n_success = results.get("n_skill_successes", 0)
            endtoend = results.get("endtoend_success", False)

            all_results.append(
                {
                    "demo_id": demo_id,
                    "instance_id": instance_id,
                    "n_skills": n_skills,
                    "n_skill_successes": n_success,
                    "skill_success_rate": float(n_success / max(1, n_skills)),
                    "endtoend_success": endtoend,
                    "total_steps": results.get("total_steps", 0),
                    "terminated": results.get("terminated", False),
                    "truncated": results.get("truncated", False),
                    "skill_results": [
                        {
                            "description": desc,
                            "success": success,
                            "result": result_type,
                        }
                        for desc, success, result_type in skill_results
                    ],
                    "metrics_path": str(out_path),
                }
            )

            logger.info(
                f"Demo {demo_id}: {n_success}/{n_skills} skills succeeded, "
                f"end-to-end={endtoend}, steps={results.get('total_steps', 0)}"
            )

        # Write aggregate summary BEFORE context exit (og.shutdown may terminate)
        elapsed_s = time.time() - t0

        total_skills = sum(r["n_skills"] for r in all_results)
        total_skill_successes = sum(r["n_skill_successes"] for r in all_results)
        total_endtoend_successes = sum(1 for r in all_results if r["endtoend_success"])

        # Per-skill-type breakdown
        skill_type_counter: Counter = Counter()
        skill_success_counter: Counter = Counter()
        for r in all_results:
            for sr in r.get("skill_results", []):
                desc = sr["description"]
                skill_type_counter[desc] += 1
                if sr["success"]:
                    skill_success_counter[desc] += 1

        skill_type_breakdown = {
            desc: {
                "attempts": skill_type_counter[desc],
                "successes": skill_success_counter[desc],
                "success_rate": float(skill_success_counter[desc] / skill_type_counter[desc]),
            }
            for desc in sorted(skill_type_counter.keys())
        }

        summary = {
            "task_name": str(config.task.name),
            "n_demos": len(demo_ids),
            "demo_ids": demo_ids,
            "total_skills": total_skills,
            "total_skill_successes": total_skill_successes,
            "skill_success_rate": float(total_skill_successes / max(1, total_skills)),
            "total_episodes": len(all_results),
            "successful_episodes": total_endtoend_successes,
            "endtoend_success_rate": float(total_endtoend_successes / max(1, len(all_results))),
            "elapsed_s": float(elapsed_s),
            "skill_type_breakdown": skill_type_breakdown,
            "per_demo_results": all_results,
        }

        with open(log_root / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("")
        logger.info("=" * 60)
        logger.info("BATCH GOLDEN RULE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Demos: {len(all_results)}")
        logger.info(f"Skills: {total_skill_successes}/{total_skills} ({summary['skill_success_rate']:.1%})")
        logger.info(f"End-to-end: {total_endtoend_successes}/{len(all_results)} ({summary['endtoend_success_rate']:.1%})")
        logger.info(f"Elapsed: {elapsed_s:.1f}s")
        logger.info(f"Summary saved to {log_root / 'summary.json'}")
        logger.info("=" * 60)
