"""Minimal official-entry adapter for the original frozen-model websocket slice.

This peer module does NOT rebuild an evaluation loop, model server, checkpoint
loader, stats reader or preprocessing stack. It mirrors
``omnigibson.eval.eval``'s official cfg shape and delegates physics stepping,
metrics and video to the upstream :class:`BatchedEvaluator` engine.

The frozen model server is launched separately (unchanged original serving
directory, original checkpoint + stats + preprocessing); this entry only opens
the client-side websocket contract through the peer :class:`CheckpointWsPolicy`.

Boundaries (segment 2 WIP):
- Single logical environment only.
- The server row is a physical23 / IK-target23 plan; without a bound
  native-joint decoder seam the policy holds the row and takes zero steps.
- No source-joint fallback, padding, or fabricated planning certificate.
- Source/frame0 identity, moving-base / gripper phases, held-object planning
  and action-clock preservation remain later gates; this entry is not a
  challenge-observation-compliant model benchmark by itself.
"""
import argparse
import json
import os
from pathlib import Path

from omegaconf import OmegaConf

from omnigibson.eval.evaluator import BatchedEvaluator, resolve_instance_ids
from omnigibson.eval.utils.eval_utils import DEFAULT_EVAL_SEED, seed_everything
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import create_module_logger

logger = create_module_logger(module_name=__name__)

_REPO_EVAL_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", required=True, help="BEHAVIOR task name.")
    parser.add_argument(
        "--robot-config",
        type=str,
        default=str(_REPO_EVAL_DIR / "r1pro.yaml"),
        help="Eval robot config YAML. Defaults to the shipped upstream r1pro profile.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Frozen model server host.")
    parser.add_argument("--port", type=int, default=8000, help="Frozen model server port.")
    parser.add_argument(
        "--action-chunk-size",
        type=int,
        default=0,
        help="Client-side open-loop chunk replay length; <= 1 keeps one request per step.",
    )
    parser.add_argument("--instance-indices", type=int, nargs="+", default=[0])
    parser.add_argument("--mode", choices=("train", "public_test", "hidden_test"), default="public_test")
    parser.add_argument("--num-rollouts", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", default="/tmp/b1k_brvla_ws_eval")
    parser.add_argument(
        "--write-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Debug run keeps a decodable MP4 per rollout.",
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
        "--env-wrapper",
        default="omnigibson.eval.wrappers.DefaultWrapper",
        help="Target path of the EnvironmentWrapper to apply.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_envs != 1:
        raise ValueError("The frozen-model websocket slice only supports --num-envs 1.")

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

    robot_config = OmegaConf.load(str(Path(args.robot_config).expanduser()))

    model_cfg = {
        "_target_": "omnigibson.eval.brvla_ws_policy.CheckpointWsPolicy",
        "host": args.host,
        "port": args.port,
        "action_chunk_size": args.action_chunk_size,
    }
    cfg = OmegaConf.create(
        {
            "env_wrapper": {"_target_": args.env_wrapper},
            "policy_name": "brvla_ws",
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
    with BatchedEvaluator(cfg) as evaluator:
        for rollout_id in range(args.num_rollouts):
            rollout_results = evaluator.run(
                [int(i) for i in instance_ids],
                write_video=args.write_video,
                video_path=video_dir,
                metrics_dir=json_dir,
                rollout_id=rollout_id,
                video_fps=args.video_fps,
            )
            results.extend(rollout_results.values())

    n = len(results)
    n_success = sum(r["success"] for r in results)
    mean_q = sum(r.get("q_score", {}).get("final", 0.0) for r in results) / n if n else 0.0
    logger.info(f"Model WS summary: {n_success}/{n} success | mean q_score={mean_q:.3f} | task={args.task_name}")


if __name__ == "__main__":
    main()
