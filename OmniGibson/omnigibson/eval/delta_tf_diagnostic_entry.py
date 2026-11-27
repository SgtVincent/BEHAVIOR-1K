"""Minimal official-entry adapter for the live-target first-plan diagnostic.

This peer module does NOT implement a parallel evaluation loop, model-serving
stack, replay loader or source/frame0 restore. It mirrors
``omnigibson.eval.eval``'s official cfg shape and delegates physics stepping,
metrics and video to the upstream ``BatchedEvaluator`` engine via
:class:`PrimitiveEvaluator`.

Scope and claims (fail-closed labels):
- The policy builds one synthetic row from live link poses: small nonzero
  left/right/chest local translations, zero base displacement, grip held at
  the live finger qpos. It exercises the official CuRobo MotionGen plan ->
  Controller execution path on a real live robot.
- This is a first-plan smoke diagnostic only. It is NOT a converted
  demonstration replay, NOT a source/frame0 identity check, NOT a scientific
  task pass, and NOT a time-faithful tracking certificate.
- No original episode data, manifest, npy, checkpoint, stats or preprocessing
  is read here. Moving base and changing grip remain upstream-boundary
  gates; this diagnostic keeps base fixed and grip unchanged.
"""
import argparse
import json
import os
from pathlib import Path

from omegaconf import OmegaConf

from omnigibson.eval.delta_tf_policy import PrimitiveEvaluator, persist_run_failure
from omnigibson.eval.delta_tf_replay_entry import _assert_baseline_worktree
from omnigibson.eval.evaluator import resolve_instance_ids
from omnigibson.eval.utils.eval_utils import DEFAULT_EVAL_SEED, seed_everything
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import create_module_logger

logger = create_module_logger(module_name=__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", required=True, help="BEHAVIOR task name.")
    parser.add_argument(
        "--robot-config",
        required=True,
        type=str,
        help="Peer joint-profile YAML; the q_to_action path requires an absolute-position "
        "JointController / HolonomicBaseJointController profile.",
    )
    parser.add_argument(
        "--tool-links",
        required=True,
        type=str,
        help='JSON mapping, e.g. \'{"left": "link_l", "right": "link_r", "chest": "link_c"}\'.',
    )
    parser.add_argument("--instance-indices", type=int, nargs="+", default=[0])
    parser.add_argument("--mode", choices=("train", "public_test", "hidden_test"), default="public_test")
    parser.add_argument("--num-rollouts", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", default="/tmp/b1k_delta_tf_diagnostic")
    parser.add_argument(
        "--write-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Diagnostic run keeps a decodable MP4 per rollout.",
    )
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="This slice requires one logical environment.",
    )
    parser.add_argument(
        "--offsets-json",
        default=None,
        help='Optional JSON override, e.g. \'{"left":[0.02,0,0],"right":[0,0.02,0],'
        '"chest":[0,0,0.01]}\'. Metres, local link frame; rotations stay zero.',
    )
    return parser.parse_args()


def main() -> None:
    _assert_baseline_worktree()
    args = parse_args()
    if args.num_envs != 1:
        raise ValueError("The live-target diagnostic slice only supports --num-envs 1.")

    gm.HEADLESS = args.headless
    if args.headless and os.getenv("OMNIGIBSON_KEEP_VIEWER_CAMERA", "0") != "1":
        gm.RENDER_VIEWER_CAMERA = False

    seed = seed_everything(DEFAULT_EVAL_SEED)
    instance_ids = resolve_instance_ids(args.task_name, args.instance_indices, mode=args.mode)
    if len(instance_ids) != args.num_envs:
        raise ValueError(
            f"Evaluation requires exactly one instance per logical environment: "
            f"got {len(instance_ids)} instance indices and --num-envs={args.num_envs}."
        )

    tool_links = json.loads(args.tool_links)
    if set(tool_links) != {"left", "right", "chest"}:
        raise ValueError("tool_links must map exactly left, right and chest.")

    offsets = None
    if args.offsets_json:
        offsets = json.loads(args.offsets_json)

    robot_config = OmegaConf.load(str(Path(args.robot_config).expanduser()))

    model_cfg = {
        "_target_": "omnigibson.eval.delta_tf_policy.LiveTargetDiagnosticPolicy",
        "tool_links": tool_links,
        "representation": "pairwise_delta",
    }
    if offsets:
        model_cfg["offsets"] = offsets
    cfg = OmegaConf.create(
        {
            "env_wrapper": {"_target_": "omnigibson.eval.wrappers.DefaultWrapper"},
            "policy_name": "live_target_diagnostic",
            "model": model_cfg,
            "headless": args.headless,
            "partial_scene_load": True,
            "max_steps": args.max_steps,
            "write_video": args.write_video,
            "mode": args.mode,
            "seed": seed,
            "num_envs": args.num_envs,
            "task": {"name": args.task_name},
            "robot": robot_config,
        }
    )

    json_dir = os.path.join(os.path.expanduser(args.output_dir), "json")
    video_dir = os.path.join(os.path.expanduser(args.output_dir), "videos")
    os.makedirs(json_dir, exist_ok=True)
    if args.write_video:
        os.makedirs(video_dir, exist_ok=True)

    results = []
    with PrimitiveEvaluator(cfg) as evaluator:
        for rollout_id in range(args.num_rollouts):
            try:
                rollout_results = evaluator.run(
                    [int(i) for i in instance_ids],
                    write_video=args.write_video,
                    video_path=video_dir,
                    metrics_dir=json_dir,
                    rollout_id=rollout_id,
                    video_fps=args.video_fps,
                )
            except Exception as exc:
                # Persist receipt + failure JSON and flush any open video writer,
                # then re-raise: the error is not swallowed and no action/step
                # is synthesized.
                persist_run_failure(evaluator, json_dir, args.task_name, rollout_id, exc)
                raise
            results.extend(rollout_results.values())
            if hasattr(evaluator.policy, "bound_receipt"):
                receipt = evaluator.policy.bound_receipt()
                receipt["exhaustion_notice"] = getattr(evaluator, "_exhaustion_notice", None)
                receipt_path = os.path.join(json_dir, f"{args.task_name}_receipt_{rollout_id}.json")
                with open(receipt_path, "w", encoding="utf-8") as f:
                    json.dump(receipt, f, indent=2, default=float)

    n = len(results)
    n_success = sum(r["success"] for r in results)
    mean_q = sum(r.get("q_score", {}).get("final", 0.0) for r in results) / n if n else 0.0
    logger.info(
        f"Diagnostic summary: {n_success}/{n} success | mean q_score={mean_q:.3f} "
        f"| task={args.task_name} | this is a live-target smoke run, not a replay/science pass."
    )


if __name__ == "__main__":
    main()
