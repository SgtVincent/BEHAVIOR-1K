"""Serve an il_lib policy checkpoint over websocket with configurable port.

Why this exists:
- `baselines/il_lib/serve.py` hardcodes `port=8000`, which prevents running multiple
  policy servers in parallel for multi-GPU evaluation.

Usage (run from `baselines/il_lib` so Hydra search path plugin is found):

  /home/ubuntu/miniconda3/envs/behavior/bin/python \
    /home/ubuntu/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/serve_il_lib_policy.py \
    --host 0.0.0.0 --port 8001 \
    robot=r1pro task=behavior task.name=make_pizza arch=dp3 \
    ckpt_path=/path/to/ckpt.pth

"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
import shutil
import tempfile

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from omnigibson.learning.utils.network_utils import WebsocketPolicyServer


logger = logging.getLogger("il_lib_policy_server")
logger.setLevel(20)


def main(argv: list[str]) -> int:
    # fix pytorch 2.6 weights_only = True by default issue
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    os.environ.setdefault("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "0")

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args, overrides = ap.parse_known_args(argv)

    # Import il_lib lazily so this file can live in BEHAVIOR-1K without requiring
    # il_lib to be installed in editor / lint contexts.
    import il_lib

    import importlib

    cfg_utils = importlib.import_module("il_lib.utils.config_utils")
    training_utils = importlib.import_module("il_lib.utils.training_utils")

    register_omegaconf_resolvers = getattr(cfg_utils, "register_omegaconf_resolvers")
    load_state_dict = getattr(training_utils, "load_state_dict")
    load_torch = getattr(training_utils, "load_torch")

    register_omegaconf_resolvers()

    # Locate il_lib config dir robustly (works for editable install)
    il_lib_file = getattr(il_lib, "__file__", None)
    if il_lib_file:
        cfg_dir = Path(il_lib_file).resolve().parents[0] / "configs"
    else:
        # Namespace package fallback
        cfg_dir = Path(il_lib.__path__[0]).resolve() / "configs"
    if not cfg_dir.exists():
        raise FileNotFoundError(f"Could not find il_lib configs dir at: {cfg_dir}")

    # Ensure the config search path includes OmniGibson learning configs (robot/task groups).
    # We do this without relying on Hydra plugins by building a temporary config root
    # containing symlinks to both il_lib configs and OmniGibson configs.
    import omnigibson as og

    og_cfg_dir = Path(og.__path__[0]).resolve() / "learning" / "configs"
    if not og_cfg_dir.exists():
        raise FileNotFoundError(f"Could not find OmniGibson learning configs dir at: {og_cfg_dir}")

    tmp_root = Path(tempfile.mkdtemp(prefix="il_lib_hydra_cfg_"))
    try:
        # il_lib config files / groups
        os.symlink(str(cfg_dir / "base_config.yaml"), str(tmp_root / "base_config.yaml"))
        os.symlink(str(cfg_dir / "arch"), str(tmp_root / "arch"))
        os.symlink(str(cfg_dir / "eval"), str(tmp_root / "eval"))

        # OmniGibson config groups referenced by il_lib defaults
        os.symlink(str(og_cfg_dir / "robot"), str(tmp_root / "robot"))
        os.symlink(str(og_cfg_dir / "task"), str(tmp_root / "task"))

        with hydra.initialize_config_dir(str(tmp_root), version_base="1.1"):
            cfg = hydra.compose("base_config.yaml", overrides=list(overrides))
    finally:
        # Best-effort cleanup
        try:
            shutil.rmtree(tmp_root)
        except Exception:
            pass

    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    policy = instantiate(cfg.module, _recursive_=False)
    if cfg.get("ckpt_path", None):
        ckpt = load_torch(
            cfg.ckpt_path,
            map_location="cpu",
        )
        load_state_dict(
            policy,
            ckpt["state_dict"],
            strict=True,
        )
    else:
        logger.info("No ckpt_path provided; serving randomly initialized policy.")

    policy = policy.to("cuda")
    policy.eval()

    policy_wrapper = instantiate(cfg.policy_wrapper)
    policy_wrapper.policy = policy

    logger.info(f"Starting websocket server on {args.host}:{args.port}")
    server = WebsocketPolicyServer(
        policy=policy_wrapper,
        host=str(args.host),
        port=int(args.port),
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))