#!/usr/bin/env bash
# Versioned runner for the fixed-base delta-TF replay debug slice.
#
# Scope: single-env converted execution23 replay through the official
# CuRoboMotionGenerator / controller / evaluator stack. This is a tracking
# diagnostic, not a time-faithful replay certificate, not original-model
# physical eval, not the three-can closed-loop task.
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
#   TASK_NAME        BEHAVIOR task name, e.g. collecting_childrens_toys
#   CONVERTED_DIR    Original converted episode dir: manifest.json + actions.npy
#                    (schema brvla_saved_execution23_v1, pairwise_delta)
#   TOOL_LINKS_JSON  JSON: {"left":"<link>","right":"<link>","chest":"<link>"}
#                    Link names must match the live robot and the cuRobo profile.
# Optional compatibility:
#   NPZ_PATH / ARRAY_KEY  Wrapped npz shim preserving the original npy sha;
#                         used only when CONVERTED_DIR is unset.
#
# Optional:
#   REPRESENTATION   pairwise_delta | chunk_cumulative (default: pairwise_delta)
#   MANIFEST_SHA256  Expected manifest.json sha256 pin (converted-dir lane)
#   ACTIONS_SHA256  Expected actions.npy sha256 pin (converted-dir lane)
#   ROBOT_CONFIG     Peer joint-profile yaml (default: eval/r1pro_delta_tf_debug.yaml)
#   OUTPUT_DIR       (default: /tmp/b1k_delta_tf_replay)
#   INSTANCE_INDICES (default: 0)
#   MAX_STEPS        (default: unset = official timeout)
#   NUM_ROLLOUTS     (default: 1)
#   VIDEO_FPS        (default: 30)
#   OMNI_PYTHON      (default: python)
#   BASELINE_COMMIT  Sha gate (default pinned to this slice's commit)
set -euo pipefail

# The slice commit that must be an ancestor of the checked-out HEAD. Follow-up
# baseline commits on top of it are allowed; an unrelated/missing slice aborts.
SLICE_COMMIT="${BASELINE_COMMIT:-7688958b8ce4079c193d6f73e2c8517a147c1a6a}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

: "${TASK_NAME:?TASK_NAME is required}"
: "${TOOL_LINKS_JSON:?TOOL_LINKS_JSON is required}"
if [[ -z "${CONVERTED_DIR:-}" && -z "${NPZ_PATH:-}" ]]; then
  echo "Either CONVERTED_DIR or NPZ_PATH/ARRAY_KEY is required" >&2
  exit 2
fi
if [[ -n "${NPZ_PATH:-}" && -z "${ARRAY_KEY:-}" ]]; then
  echo "ARRAY_KEY is required with NPZ_PATH" >&2
  exit 2
fi

REPRESENTATION="${REPRESENTATION:-pairwise_delta}"
ROBOT_CONFIG="${ROBOT_CONFIG:-$REPO_ROOT/OmniGibson/omnigibson/eval/r1pro_delta_tf_debug.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/b1k_delta_tf_replay}"
INSTANCE_INDICES="${INSTANCE_INDICES:-0}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-1}"
VIDEO_FPS="${VIDEO_FPS:-30}"
OMNI_PYTHON="${OMNI_PYTHON:-python}"

# Sha pre-flight gate: refuse to run unless the pinned slice commit is an
# ancestor of the checked-out HEAD.
HEAD_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
if ! git -C "$REPO_ROOT" merge-base --is-ancestor "$SLICE_COMMIT" "$HEAD_COMMIT"; then
  echo "BASELINE SHA MISMATCH: head=$HEAD_COMMIT missing ancestor=$SLICE_COMMIT" >&2
  exit 2
fi

# Managed scratch: /dev/shm lane when available, otherwise the run output
# dir's tmp subtree. Peer cuRobo profile and the warp shim live here only;
# nothing is written to a shared site-customize or the NAS.
RUN_TMP="${B1K_RUN_TMPDIR:-}"
if [[ -z "$RUN_TMP" ]]; then
  if [[ -d /dev/shm && -w /dev/shm ]]; then
    RUN_TMP="/dev/shm/b1k_replay_${USER:-user}_$$"
  else
    RUN_TMP="$OUTPUT_DIR/tmp"
  fi
fi
mkdir -p "$RUN_TMP/peer_profile" "$RUN_TMP/warp_shim" "$RUN_TMP/tmp"
export TMPDIR="$RUN_TMP/tmp"
export B1K_PEER_PROFILE_DIR="$RUN_TMP/peer_profile"

# Per-run warp 1.12 preload shim (retained 303 content, sha256
# b516931a65a594d1ea0af3dbe79c6e6cb10d80f5a419b1f84d55493bce427054);
# scratch-only, never written to shared site-packages.
if [[ "${WARP_SHIM:-1}" != "0" ]]; then
  cat > "$RUN_TMP/warp_shim/sitecustomize.py" <<'SHIM'
import sys
import torch as _torch  # noqa: F401
for _m in list(sys.modules):
    if _m == "warp" or _m.startswith("warp."):
        del sys.modules[_m]
import warp as _wp
import warp.torch  # noqa: F401
_wp.init()
SHIM
  SHIM_SHA="$(sha256sum "$RUN_TMP/warp_shim/sitecustomize.py" | cut -d' ' -f1)"
  echo "Per-run warp shim sha256: $SHIM_SHA ($RUN_TMP/warp_shim/sitecustomize.py)"
  export PYTHONPATH="$RUN_TMP/warp_shim${PYTHONPATH:+:$PYTHONPATH}"
fi

# Prefer this worktree's OmniGibson package over any shared editable install;
# the entry asserts the resolved package file lives under this worktree.
# No shared conda / site-packages is modified.
export PYTHONPATH="$REPO_ROOT/OmniGibson${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUTPUT_DIR"

CMD=(
  "$OMNI_PYTHON" -m omnigibson.eval.delta_tf_replay_entry
  --task-name "$TASK_NAME"
  --robot-config "$ROBOT_CONFIG"
  --tool-links "$TOOL_LINKS_JSON"
  --representation "$REPRESENTATION"
  --instance-indices $INSTANCE_INDICES
  --num-rollouts "$NUM_ROLLOUTS"
  --output-dir "$OUTPUT_DIR"
  --video-fps "$VIDEO_FPS"
  --write-video
)
if [[ -n "${CONVERTED_DIR:-}" ]]; then
  CMD+=(--converted-dir "$CONVERTED_DIR")
  [[ -n "${MANIFEST_SHA256:-}" ]] && CMD+=(--manifest-sha256 "$MANIFEST_SHA256")
  [[ -n "${ACTIONS_SHA256:-}" ]] && CMD+=(--actions-sha256 "$ACTIONS_SHA256")
else
  CMD+=(--npz-path "$NPZ_PATH" --array-key "$ARRAY_KEY")
fi
if [[ -n "${MAX_STEPS:-}" ]]; then
  CMD+=(--max-steps "$MAX_STEPS")
fi

echo "Running delta-TF replay debug (commit $HEAD_COMMIT):"
printf ' %q' "${CMD[@]}"
echo
exec "${CMD[@]}"
