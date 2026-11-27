"""Minimal official-entry adapter for the fixed-base delta-TF replay slice.

This peer module does NOT implement a parallel evaluation loop, websocket server
or model-serving stack. It mirrors ``omnigibson.eval.eval``'s official cfg shape
and delegates physics stepping, metrics and video to the upstream
``BatchedEvaluator`` engine via :class:`PrimitiveEvaluator`.

WIP boundaries (same as ``delta_tf_policy.py``):
- Single logical environment only.
- Converted execution23 rows are read from the original ``manifest.json`` +
  ``actions.npy`` directory (``brvla_saved_execution23_v1``, pairwise_delta),
  or an optional compatibility npz shim; replayed through upstream cuRobo
  MotionGen planning and feedback execution. This is a tracking diagnostic, not
  a time-faithful replay certificate.
- Moving base / changing gripper width are rejected before planning.
- Finite input exhaustion is a diagnostic exit, never task success.

No checkpoint, stats, preprocessing or chunk/history contract is wired here
yet. Original-model physical eval and the three-can closed-loop task remain
separate later gates.
"""
import argparse
import json
import os
from pathlib import Path

from omegaconf import OmegaConf

import omnigibson
from omnigibson.eval.delta_tf_policy import PrimitiveEvaluator, persist_run_failure
from omnigibson.eval.evaluator import resolve_instance_ids
from omnigibson.eval.utils.eval_utils import DEFAULT_EVAL_SEED, seed_everything
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import create_module_logger

logger = create_module_logger(module_name=__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _assert_baseline_worktree() -> None:
    """Reject a shared editable install resolving outside this baseline worktree.

    The runner shell prepends the worktree ``OmniGibson`` directory to
    ``PYTHONPATH``; this assertion fails closed if the imported package still
    points elsewhere. No shared installation is modified.
    """
    expected_pkg = _REPO_ROOT / "OmniGibson" / "omnigibson"
    loaded = Path(omnigibson.__file__).resolve()
    if not str(loaded).startswith(str(expected_pkg) + os.sep):
        raise SystemExit(
            f"omnigibson resolved outside baseline worktree: {loaded} "
            f"(expected under {expected_pkg})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", required=True, help="BEHAVIOR task name.")
    parser.add_argument(
        "--robot-config",
        required=True,
        type=str,
        help=(
            "Path to the eval robot config YAML. This slice's q_to_action path "
            "requires a JointController / HolonomicBaseJointController absolute-position "
            "profile; the MultiFinger gripper profile is not compatible."
        ),
    )
    parser.add_argument("--converted-dir", default=None, help="Original converted episode directory holding manifest.json + actions.npy.")
    parser.add_argument("--manifest-sha256", default=None, help="Expected manifest.json sha256 (optional pin).")
    parser.add_argument("--actions-sha256", default=None, help="Expected actions.npy sha256 (optional pin).")
    parser.add_argument("--npz-path", default=None, help="Compatibility replay npz (alternative to --converted-dir).")
    parser.add_argument("--array-key", default=None, help="Array key inside the compatibility npz.")
    parser.add_argument(
        "--tool-links",
        required=True,
        type=str,
        help='JSON mapping, e.g. \'{"left": "link_l", "right": "link_r", "chest": "link_c"}\'.',
    )
    parser.add_argument(
        "--representation",
        choices=("pairwise_delta", "chunk_cumulative"),
        default="pairwise_delta",
    )
    parser.add_argument("--instance-indices", type=int, nargs="+", default=[0])
    parser.add_argument("--mode", choices=("train", "public_test", "hidden_test"), default="public_test")
    parser.add_argument("--num-rollouts", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", default="/tmp/b1k_delta_tf_replay")
    parser.add_argument(
        "--write-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Debug replay keeps a decodable MP4 per rollout.",
    )
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="This slice requires one logical environment.",
    )
    return parser.parse_args()


def main() -> None:
    _assert_baseline_worktree()
    args = parse_args()
    if args.num_envs != 1:
        raise ValueError("The delta-TF replay slice only supports --num-envs 1.")

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

    if not args.converted_dir and not args.npz_path:
        raise ValueError("Provide either --converted-dir or --npz-path/--array-key.")
    if args.npz_path and not args.array_key:
        raise ValueError("--array-key is required with --npz-path.")

    robot_config = OmegaConf.load(str(Path(args.robot_config).expanduser()))

    model_cfg = {
        "_target_": "omnigibson.eval.delta_tf_policy.DeltaTFReplayPolicy",
        "tool_links": tool_links,
        "representation": args.representation,
    }
    if args.converted_dir:
        model_cfg.update({
            "converted_dir": args.converted_dir,
            "manifest_sha256": args.manifest_sha256,
            "actions_sha256": args.actions_sha256,
        })
    else:
        model_cfg.update({
            "npz_path": args.npz_path,
            "array_key": args.array_key,
        })
    cfg = OmegaConf.create(
        {
            "env_wrapper": {"_target_": "omnigibson.eval.wrappers.DefaultWrapper"},
            "policy_name": "delta_tf_replay",
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
    logger.info(f"Replay summary: {n_success}/{n} success | mean q_score={mean_q:.3f} | task={args.task_name}")


if __name__ == "__main__":
    main()
