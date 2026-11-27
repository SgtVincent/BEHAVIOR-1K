#!/usr/bin/env bash
# Versioned runner for the live-target first-plan diagnostic slice.
#
# Scope: single-env synthetic live small-offset target through the official
# CuRoboMotionGenerator / Controller / evaluator stack. This is a first-plan
# smoke diagnostic ONLY: it is not a converted demonstration replay, not a
# source/frame0 identity check, not a scientific task pass, and not a
# time-faithful tracking certificate. No original episode data, manifest,
# npy, checkpoint, stats or preprocessing is read.
#
# Environment: activate the per-run isolated overlay BEFORE calling this
# script. Do not modify a shared conda / site-packages. The overlay must carry
# the pinned upstream extras (see OmniGibson/setup.py "primitives" extra):
#   ninja~=1.13.0
#   warp-lang==1.12.0
#   nvidia-curobo @ git+https://github.com/StanfordVL/curobo@78612f45cef52c3fa0298de243a54cd7ca614414
# cuRobo v2 / MotionPlanner is deliberately NOT in this path.
#
# Required inputs (no private paths are hardcoded here):
#   TASK_NAME        BEHAVIOR task name, e.g. collecting_childrens_toys
#   TOOL_LINKS_JSON  JSON: {"left":"<link>","right":"<link>","chest":"<link>"}
#                    Link names must match the live robot and the cuRobo profile.
#
# Optional:
#   ROBOT_CONFIG       Peer joint-profile yaml (default: eval/r1pro_delta_tf_debug.yaml)
#   OFFSETS_JSON       {"left":[x,y,z],"right":[...],"chest":[...]} metres, local frame
#   OUTPUT_DIR         Persistent video/json dir (default: /tmp/b1k_delta_tf_diagnostic)
#   B1K_RUN_TMPDIR     Managed scratch dir; defaults to a /dev/shm lane when
#                      writable, else <OUTPUT_DIR>/tmp. Run-time assets never go
#                      to NAS or an unrelated shared /tmp location.
#   WARP_SHIM          1 (default) writes the per-run warp 1.12 preload
#                      sitecustomize under the scratch dir; 0 disables it.
#   INSTANCE_INDICES   (default: 0)
#   MAX_STEPS          (default: unset = official timeout)
#   NUM_ROLLOUTS       (default: 1)
#   VIDEO_FPS          (default: 30)
#   OMNI_PYTHON        (default: python)
#   BASELINE_COMMIT    Sha gate (default pinned to this slice's ancestor commit)
set -euo pipefail

# The slice commit that must be an ancestor of the checked-out HEAD. Follow-up
# baseline commits on top of it are allowed; an unrelated/missing slice aborts.
SLICE_COMMIT="${BASELINE_COMMIT:-5232673fce5a11494a7c976bc0f4641b3c7c5d58}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

: "${TASK_NAME:?TASK_NAME is required}"
: "${TOOL_LINKS_JSON:?TOOL_LINKS_JSON is required}"

ROBOT_CONFIG="${ROBOT_CONFIG:-$REPO_ROOT/OmniGibson/omnigibson/eval/r1pro_delta_tf_debug.yaml}"
OFFSETS_JSON="${OFFSETS_JSON:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/b1k_delta_tf_diagnostic}"
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
    RUN_TMP="/dev/shm/b1k_diag_${USER:-user}_$$"
  else
    RUN_TMP="$OUTPUT_DIR/tmp"
  fi
fi
mkdir -p "$RUN_TMP/peer_profile" "$RUN_TMP/warp_shim" "$RUN_TMP/tmp" "$OUTPUT_DIR/json" "$OUTPUT_DIR/videos"
export TMPDIR="$RUN_TMP/tmp"
export B1K_PEER_PROFILE_DIR="$RUN_TMP/peer_profile"

# Per-run warp 1.12 preload shim. Content is the retained 303 field shim
# (sha256 b516931a65a594d1ea0af3dbe79c6e6cb10d80f5a419b1f84d55493bce427054);
# it lives only in this run's scratch dir and is never written to a shared
# site-packages sitecustomize.
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
export PYTHONPATH="$REPO_ROOT/OmniGibson${PYTHONPATH:+:$PYTHONPATH}"

CMD=(
  "$OMNI_PYTHON" -m omnigibson.eval.delta_tf_diagnostic_entry
  --task-name "$TASK_NAME"
  --robot-config "$ROBOT_CONFIG"
  --tool-links "$TOOL_LINKS_JSON"
  --instance-indices $INSTANCE_INDICES
  --num-rollouts "$NUM_ROLLOUTS"
  --output-dir "$OUTPUT_DIR"
  --video-fps "$VIDEO_FPS"
  --write-video
)
[[ -n "$OFFSETS_JSON" ]] && CMD+=(--offsets-json "$OFFSETS_JSON")
if [[ -n "${MAX_STEPS:-}" ]]; then
  CMD+=(--max-steps "$MAX_STEPS")
fi

echo "Running live-target diagnostic (commit $HEAD_COMMIT) — smoke only, not a replay/science pass:"
printf ' %q' "${CMD[@]}"
echo
exec "${CMD[@]}"
