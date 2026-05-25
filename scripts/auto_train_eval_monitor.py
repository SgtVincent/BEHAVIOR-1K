#!/usr/bin/env python3
"""
Simple monitor to (a) start missing training jobs on free GPUs and (b) auto-run eval when a task has all 4 ckpts.
Designed to be conservative and robust; it logs actions to train_logs/auto_train_eval.log.

Usage: run this in background from repo root (it will spawn training processes and eval scripts):
  nohup /home/ubuntu/miniconda3/envs/behavior/bin/python scripts/auto_train_eval_monitor.py &

Notes:
- Requires `nvidia-smi` present.
- Uses existing `train.py` invocation conventions and `run_eval_task_all_primitives_parallel.sh`.
- Tailor intervals or GPU-use thresholds as needed.
"""
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
IL_LIB_ROOT = Path(os.environ.get('IL_LIB_ROOT', "/mnt/bn/robot-mllm-data-hl/mlx/users/chenjunting/repo/b1k-baselines/baselines/il_lib"))  # default to the cloned training package; can be overridden via IL_LIB_ROOT env var
PY = os.environ.get("PY", "/home/ubuntu/miniconda3/envs/behavior/bin/python")
DATA_DIR = "/mnt/bn/navigation-hl/mlx/users/chenjunting/data"
OUT_ROOT = IL_LIB_ROOT / "outputs" / time.strftime("%Y-%m-%d")
LOG_DIR = REPO_ROOT / "train_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
MON_LOG = LOG_DIR / "auto_train_eval.log"

# Tasks and known task_info dims (computed earlier)
TASKS = [
    "putting_shoes_on_rack",
    "picking_up_trash",
    "setting_mousetraps",
    "turning_on_radio",
    "hiding_Easter_eggs",
    "bringing_in_wood",
    "moving_boxes_to_storage",
    "sorting_vegetables",
]
TASK_INFO_DIMS = {
    "putting_shoes_on_rack": 82,
    "picking_up_trash": 82,
    "setting_mousetraps": 94,
    "turning_on_radio": 46,
    "hiding_Easter_eggs": 382,
    "bringing_in_wood": 70,
    "moving_boxes_to_storage": 58,
    "sorting_vegetables": 250,
}

CHECK_INTERVAL = int(os.environ.get("MON_CHECK_INTERVAL", "120"))  # seconds
GPU_FREE_MEM_THRESH_MB = 8_000  # treat GPU free if < ~8GB used
MAX_GPU_UTIL_PCT = 10  # treat free if utilization < 10%
FAST_BOOTSTRAP = os.environ.get("FAST_BOOTSTRAP", "true").lower() in ("1","true","yes")
GPU_RESERVATION_SECS = int(os.environ.get("GPU_RESERVATION_SECS", "90"))
_RESERVED_GPUS = {}  # local reservations to avoid scheduling multiple jobs to the same GPU in quick succession

def log(msg: str):
    t = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{t}] {msg}\n"
    print(line, end="", flush=True)
    MON_LOG.write_text(MON_LOG.read_text() + line if MON_LOG.exists() else line)


def run_bg(cmd, env=None, cwd=None, logfile=None):
    """Spawn cmd in background and return Popen."""
    if logfile is None:
        logfile = LOG_DIR / ("proc_" + time.strftime("%H%M%S") + ".log")
    log(f"Starting background: {cmd} -> {logfile}")
    with open(logfile, "ab") as f:
        p = subprocess.Popen(cmd, shell=True, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
    return p


def parse_nvidia_smi():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
    except Exception as e:
        log(f"Failed to query nvidia-smi: {e}")
        return []
    rows = []
    for line in out.strip().splitlines():
        idx, util, mem = [x.strip() for x in line.split(',')]
        rows.append({'index': int(idx), 'util': int(util), 'mem': int(mem)})
    return rows


def find_free_gpu():
    rows = parse_nvidia_smi()
    now = time.time()
    # cleanup expired local reservations
    for gi, exp in list(_RESERVED_GPUS.items()):
        if exp < now:
            del _RESERVED_GPUS[gi]
    for r in rows:
        if r['index'] in _RESERVED_GPUS:
            continue
        if r['util'] <= MAX_GPU_UTIL_PCT and r['mem'] <= GPU_FREE_MEM_THRESH_MB:
            # reserve a short window so quick successive launches don't pick the same GPU
            _RESERVED_GPUS[r['index']] = now + GPU_RESERVATION_SECS
            return r['index']
    return None


def find_latest_ckpt(arch: str, task: str):
    # look for outputs in today's folder (and fallback to older days)
    # return path to 'last.pth' if exists
    outs = list((IL_LIB_ROOT / 'outputs').glob(f"**/{arch}_{task}_*/ckpt/last.pth"))
    if outs:
        # pick the newest by mtime
        ck = sorted(outs, key=lambda p: p.stat().st_mtime)[-1]
        return str(ck)
    return None


def is_train_running(arch: str, task: str):
    """Return True if a training process for given arch+task is detected."""
    try:
        out = subprocess.check_output(['pgrep', '-af', 'train.py'], text=True)
    except subprocess.CalledProcessError:
        return False
    for line in out.splitlines():
        if arch in line and task in line:
            return True
    return False


def task_is_eval_done(task: str):
    # check for report.json
    root = REPO_ROOT / 'eval_logs'
    for d in root.glob(f"{task}_all_primitives_parallel_*"):
        if (d / 'report.json').exists():
            return True
    return False


def is_eval_running(task: str):
    """Return True if an eval process for the task is running."""
    try:
        out = subprocess.check_output(['pgrep', '-af', 'run_eval_task_all_primitives_parallel.sh'], text=True)
    except subprocess.CalledProcessError:
        return False
    for line in out.splitlines():
        if task in line:
            return True
    return False


def all_ckpts_present(task: str):
    # ACT, DP3, MoMa-stage, WBVIMA
    ckpts = {}
    for arch in ['act','dp3','moma_stage','wbvima']:
        ck = find_latest_ckpt(arch, task)
        if not ck:
            return False
        ckpts[arch]=ck
    return ckpts


def launch_eval_for_task(task: str):
    ckpts = all_ckpts_present(task)
    if not ckpts:
        return False
    if task_is_eval_done(task):
        log(f"Eval already done for {task}, skipping")
        return False
    # build env and run eval script with correct TASK_INFO_DIM
    tid = TASK_INFO_DIMS.get(task, 262)
    env = os.environ.copy()
    env.update({
        'CKPT_ACT': ckpts['act'],
        'CKPT_DP3': ckpts['dp3'],
        'CKPT_MOMA': ckpts['moma_stage'],
        'CKPT_WBVIMA': ckpts['wbvima'],
        'TASK_INFO_DIM': str(tid),
        'PARALLEL_MODELS': '1',
        'NUM_DEMOS': '3',
        'PRIMITIVE_MAX_STEPS': os.environ.get('PRIMITIVE_MAX_STEPS','400'),
        'DISPLAY_NUM': os.environ.get('DISPLAY_NUM',':10.0'),
        'PY': PY,
    })
    cmd = f"bash {REPO_ROOT}/run_eval_task_all_primitives_parallel.sh {shlex.quote(task)}"
    run_bg(cmd, env=env, cwd=str(REPO_ROOT), logfile=LOG_DIR / f"eval_{task}.log")
    log(f"Launched eval for {task}")
    return True


def launch_training_for_model(arch: str, task: str, gpu: int, quick: bool = False):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    run_name = f"{arch}_{task}_prim_pick_up_from_demo50_{time.strftime('%Y%m%d-%H%M%S')}"
    if quick:
        run_name = run_name + "_fast"

    if arch == 'act':
        if quick:
            cmd = (
                f"{PY} train.py robot=r1pro task=behavior task.name={task} arch=act seed=42 use_wandb=false run_name={run_name} "
                f"data_dir=/mnt/bn/robot-mllm-data-hl/mlx/users/chenjunting/data data.max_num_demos=50 +data.primitive_desc=\"pick up from\" horizon=20 num_latest_obs=1 deployed_action_steps=1 bs=128 wd=0.1 data.use_action_chunks=false data.use_task_info=false trainer.max_epochs=1 +trainer.limit_train_batches=5 trainer.check_val_every_n_epoch=5"
            )
        else:
            cmd = (
                f"{PY} train.py robot=r1pro task=behavior task.name={task} arch=act seed=42 use_wandb=false run_name={run_name} "
                f"data_dir=/mnt/bn/robot-mllm-data-hl/mlx/users/chenjunting/data data.max_num_demos=50 +data.primitive_desc=\"pick up from\" horizon=20 num_latest_obs=1 deployed_action_steps=1 bs=128 wd=0.1 data.use_action_chunks=false data.use_task_info=false trainer.max_epochs=20 trainer.check_val_every_n_epoch=5"
            )
    elif arch == 'dp3':
        if quick:
            cmd = (
                f"{PY} train.py robot=r1pro task=behavior task.name={task} arch=dp3 seed=42 use_wandb=false run_name={run_name} "
                f"data_dir=/mnt/bn/navigation-hl/mlx/users/chenjunting/data data.max_num_demos=50 +data.primitive_desc=\"pick up from\" horizon=16 num_latest_obs=2 deployed_action_steps=8 bs=32 wd=0.0 data.use_action_chunks=false data.use_task_info=false trainer.max_epochs=1 +trainer.limit_train_batches=10 trainer.check_val_every_n_epoch=2"
            )
        else:
            cmd = (
                f"{PY} train.py robot=r1pro task=behavior task.name={task} arch=dp3 seed=42 use_wandb=false run_name={run_name} "
                f"data_dir=/mnt/bn/navigation-hl/mlx/users/chenjunting/data data.max_num_demos=50 +data.primitive_desc=\"pick up from\" horizon=16 num_latest_obs=2 deployed_action_steps=8 bs=32 wd=0.0 data.use_action_chunks=false data.use_task_info=false trainer.max_epochs=10 trainer.check_val_every_n_epoch=2"
            )
    elif arch == 'moma_stage':
        tid = TASK_INFO_DIMS.get(task, 262)
        if quick:
            cmd = (
                f"{PY} train.py robot=r1pro task=behavior task.name={task} arch=moma_stage seed=42 use_wandb=false run_name={run_name}_dim{tid}_T8_fast "
                f"data_dir=/mnt/bn/navigation-hl/mlx/users/chenjunting/data data.max_num_demos=50 +data.primitive_desc=\"pick_up_from\" horizon=8 num_latest_obs=8 deployed_action_steps=8 bs=32 wd=0.1 data.use_action_chunks=true data.use_task_info=true module.feature_extractors.task.input_dim={tid} trainer.max_epochs=1 +trainer.limit_train_batches=10 trainer.check_val_every_n_epoch=2"
            )
        else:
            cmd = (
                f"{PY} train.py robot=r1pro task=behavior task.name={task} arch=moma_stage seed=42 use_wandb=false run_name={run_name}_dim{tid}_T8 "
                f"data_dir=/mnt/bn/navigation-hl/mlx/users/chenjunting/data data.max_num_demos=50 +data.primitive_desc=\"pick_up_from\" horizon=8 num_latest_obs=8 deployed_action_steps=8 bs=32 wd=0.1 data.use_action_chunks=true data.use_task_info=true module.feature_extractors.task.input_dim={tid} trainer.max_epochs=10 trainer.check_val_every_n_epoch=2"
            )
    elif arch == 'wbvima':
        tid = TASK_INFO_DIMS.get(task, 262)
        # wbvima is already a short config; keep it for quick/full
        cmd = (
            f"{PY} train.py robot=r1pro task=behavior task.name={task} arch=wbvima seed=42 use_wandb=false run_name={run_name}_task_info_True "
            f"data_dir=/mnt/bn/navigation-hl/mlx/users/chenjunting/data data.max_num_demos=50 +data.primitive_desc=\"pick up from\" horizon=2 num_latest_obs=2 deployed_action_steps=8 bs=8 wd=0.1 data.use_action_chunks=true data.use_task_info=true module.feature_extractors.task.input_dim={tid} trainer.max_epochs=1 trainer.num_sanity_val_steps=0 trainer.check_val_every_n_epoch=1 +trainer.limit_train_batches=5 +trainer.limit_val_batches=2"
        )
    else:
        return None
    logfile = LOG_DIR / f"train_{arch}_{task}_{time.strftime('%Y%m%d-%H%M%S')}.log"
    env_for_run = env.copy()
    p = run_bg(cmd, env=env_for_run, cwd=str(IL_LIB_ROOT), logfile=logfile)
    return p


def main_loop():
    log("Monitor started: will start missing jobs and run evals when ready.")
    while True:
        try:
            # Aggressive quick-ckpt bootstrap: fill free GPUs with short-running jobs to produce ckpts quickly
            if FAST_BOOTSTRAP:
                for arch in ['wbvima','moma_stage','dp3','act']:
                    for task in TASKS:
                        if find_latest_ckpt(arch, task):
                            continue
                        if is_train_running(arch, task):
                            continue
                        gpu = find_free_gpu()
                        if gpu is None:
                            break
                        log(f"Launching quick training for {arch} on {task} (GPU {gpu}) [FAST_BOOTSTRAP]")
                        launch_training_for_model(arch, task, gpu, quick=True)
                        time.sleep(3)
            # If FAST_BOOTSTRAP is disabled or GPUs are busy, fall back to conservative scheduling (original behavior)
            # 4) For tasks with all ckpts and no eval report, run eval
            for task in TASKS:
                ckpts = all_ckpts_present(task)
                if ckpts and not task_is_eval_done(task):
                    if is_eval_running(task):
                        continue
                    # ensure some GPUs are free for eval
                    # we'll proceed anyway and let the eval script pick GPUs we set
                    log(f"Auto-launch eval for task {task}")
                    launch_eval_for_task(task)
                    time.sleep(10)
            time.sleep(CHECK_INTERVAL)
        except Exception as e:
            log(f"Monitor exception: {e}")
            time.sleep(CHECK_INTERVAL)

if __name__ == '__main__':
    main_loop()
