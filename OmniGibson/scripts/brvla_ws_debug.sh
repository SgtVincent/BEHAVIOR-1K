#!/usr/bin/env bash
# Versioned runner for the original frozen-model websocket debug slice.
#
# Scope: single-env official evaluator against an unchanged original model
# server over the frozen msgpack websocket contract. The original checkpoint,
# stats, preprocessing and chunk/history clock live in the server process;
# this client slice loads no weights or stats itself.
#
# The server must already be running (original serving directory, original
# pinned commit). This script only launches the official evaluator client.
# Without a bound native-joint decoder seam the policy holds each physical23
# row and takes zero simulator steps; that boundary is checked in the policy,
# not bypassed here.
#
# Environment: activate the per-run isolated overlay BEFORE calling this
# script. Do not modify a shared conda. The overlay must carry the pinned
# upstream extras (see OmniGibson/setup.py "primitives" extra):
#   ninja~=1.13.0
#   warp-lang==1.12.0
#   nvidia-curobo @ git+https://github.com/StanfordVL/curobo@78612f45cef52c3fa0298de243a54cd7ca614414
# cuRobo v2 / MotionPlanner is deliberately NOT in this path.
#
# Required inputs (no private paths are hardcoded here):
#   TASK_NAME   BEHAVIOR task name, e.g. turning_on_radio
#   HOST        Frozen model server host (default 127.0.0.1)
#   PORT        Frozen model server port (default 8000)
#
# Optional:
#   ROBOT_CONFIG       Eval robot config yaml (default: eval/r1pro.yaml)
#   ACTION_CHUNK_SIZE  Client chunk replay length, <=1 disables (default 0)
#   OUTPUT_DIR         (default: /tmp/b1k_brvla_ws_eval)
#   INSTANCE_INDICES   (default: 0)
#   MAX_STEPS          (default: unset = official timeout)
#   NUM_ROLLOUTS       (default: 1)
#   VIDEO_FPS          (default: 30)
#   ENV_WRAPPER        (default: omnigibson.eval.wrappers.DefaultWrapper)
#   OMNI_PYTHON        (default: python)
#   BASELINE_COMMIT    Sha gate (default pinned to this slice's anchor commit)
set -euo pipefail

# The anchor commit that must be an ancestor of the checked-out HEAD. Follow-up
# baseline commits on top of it are allowed; an unrelated/missing slice aborts.
SLICE_COMMIT="${BASELINE_COMMIT:-d7363c7d5b36ad661c43c3283243fea48ed15df5}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

: "${TASK_NAME:?TASK_NAME is required}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
ROBOT_CONFIG="${ROBOT_CONFIG:-$REPO_ROOT/OmniGibson/omnigibson/eval/r1pro.yaml}"
ACTION_CHUNK_SIZE="${ACTION_CHUNK_SIZE:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/b1k_brvla_ws_eval}"
INSTANCE_INDICES="${INSTANCE_INDICES:-0}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-1}"
VIDEO_FPS="${VIDEO_FPS:-30}"
ENV_WRAPPER="${ENV_WRAPPER:-omnigibson.eval.wrappers.DefaultWrapper}"
OMNI_PYTHON="${OMNI_PYTHON:-python}"

# Sha pre-flight gate: refuse to run unless the pinned anchor commit is an
# ancestor of the checked-out HEAD.
HEAD_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
if ! git -C "$REPO_ROOT" merge-base --is-ancestor "$SLICE_COMMIT" "$HEAD_COMMIT"; then
  echo "BASELINE SHA MISMATCH: head=$HEAD_COMMIT missing ancestor=$SLICE_COMMIT" >&2
  exit 2
fi

# Prefer this worktree's OmniGibson package over any shared editable install;
# the entry asserts the resolved package file lives under this worktree.
# No shared conda / site-packages is modified.
export PYTHONPATH="$REPO_ROOT/OmniGibson${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUTPUT_DIR"

CMD=(
  "$OMNI_PYTHON" -m omnigibson.eval.brvla_ws_entry
  --task-name "$TASK_NAME"
  --robot-config "$ROBOT_CONFIG"
  --host "$HOST"
  --port "$PORT"
  --action-chunk-size "$ACTION_CHUNK_SIZE"
  --env-wrapper "$ENV_WRAPPER"
  --instance-indices $INSTANCE_INDICES
  --num-rollouts "$NUM_ROLLOUTS"
  --output-dir "$OUTPUT_DIR"
  --video-fps "$VIDEO_FPS"
  --write-video
)
if [[ -n "${MAX_STEPS:-}" ]]; then
  CMD+=(--max-steps "$MAX_STEPS")
fi

echo "Running frozen-model websocket debug (commit $HEAD_COMMIT):"
printf ' %q' "${CMD[@]}"
echo
exec "${CMD[@]}"
