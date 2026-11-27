# Official BEHAVIOR-1K 2026 baseline — initial implementation slice

Status: **WIP; not a release and not business acceptance.**

Base: `v3.9.3-post1`, `bd049de3119acdcdf2334fe9e1ebe060fa20c108`.
Branch: `baseline/b1k2026-official`.

## Design

Keep upstream core files unchanged. Add peer modules in the existing package:

- `omnigibson/action_primitives/delta_tf_action_primitives.py`
  - `DeltaTFDecoder` uses upstream `transform_utils` for Euler/quaternion/SE3.
  - `DeltaTFActionPrimitives` derives from `StarterSemanticActionPrimitives` /
    `BaseActionPrimitiveSet` and calls upstream `CuRoboMotionGenerator` plus
    upstream feedback execution. No custom IK/trajectory optimizer/exporter.
- `omnigibson/eval/delta_tf_policy.py`
  - `DeltaTFReplayPolicy` derives from `LocalPolicy`.
  - `PrimitiveEvaluator` derives from `BatchedEvaluator`, changing only its
    policy factory to supply the already created environment and robot.
  - Upstream evaluator remains owner of physics stepping, metrics and video.

Official primitives dependency remains the unmodified `setup.py` extra:
`StanfordVL/curobo@78612f45cef52c3fa0298de243a54cd7ca614414` (MotionGen API).
cuRobo v2 is deliberately NOT part of this branch's execution path.

Sources:
- https://behavior.stanford.edu/reference/action_primitives/curobo.html
- https://behavior.stanford.edu/reference/controllers/ik_controller.html#controllers.ik_controller.InverseKinematicsController.add_member
- https://behavior.stanford.edu/omnigibson/controllers.html

## Input and authority

The decoder input is execution23: left pose6, right pose6, two gripper widths in
metres, base dx/dy/dyaw, chest pose6. This is NOT the native23 joint command.
Use explicit `pairwise_delta` for saved conversion rows or `chunk_cumulative`
for model chunk targets. Anchor rebinding is explicit at each run/chunk, never
silently refreshed per control step.

No source joint-angle fallback, state teleport, disabled collision checks or
fabricated planner certification is allowed. `add_member` registers an IK
controller member; it is not the solving operation. Original checkpoint, stats,
preprocessing and chunk/history clock contracts must be retained for model eval.

## Current limitations (intentional fail-closed boundaries)

This first code slice only supports fixed-base, unchanged-gripper pose targets
with all requested left/right/chest tool links present in the upstream profile.
Changing the base or gripper currently raises `NotImplementedError` before plan
execution. The planner result must also preserve those fixed joints.

Consequently, this slice **cannot yet replay an entire converted demonstration,
complete original-model physical eval or reproduce picking_up_trash**. No model
server/Agent policy is connected here yet. Source initial-state migration,
source/frame0 identity, moving-base/gripper phases, held-object planning,
model action-clock preservation and three-can termination remain open.

Waypoint feedback execution is not equivalent to preserving each source row's
fixed simulation duration. A tracking-only diagnostic cannot grant time-faithful
replay acceptance. StopIteration on finite exhaustion is a diagnostic exit,
not task success. Scene-geometry-based engineering tests are not automatically
challenge-observation-compliant model benchmarks.

## Validation performed

- Both new Python modules compile syntactically.
- Existing remote Python, CPU only: AST-isolated decoder and exact upstream
  transformation function bodies; not a full installed-module import.
- Pairwise target accumulation; cumulative targets not accumulated twice;
  malformed-width and nonfinite inputs reject; upstream Euler rotation checked
  independently against SciPy. Max rotation-entry difference
  `5.960464477539063e-08`.
- **No new GPU run, no physical step, no new video, no business PASS.**
- No package was installed locally; no shared remote environment was changed.

## Next acceptance

1. Confirm old owners quiesced and preserve WIP / untracked / runtime references.
2. Bind a single official source/dependency/config installation; verify real
   imports and actual loaded consumer paths in the authorized remote isolation.
3. Exercise one real delta-TF target with collision-aware planning, physical
   feedback and decodable video; then add only the proven missing base/gripper
   bindings using upstream modules.
4. Connect original BRVLA input/model contracts and original Agent + MoMaGen
   decision sources without replacing them with recorded source joints.
5. Separately validate converted replay, original-model physical execution and
   original three-can closed-loop task. Only then call a commit a validated
   baseline and archive 2025 chapters.

The current branch preserves code for continuation, not a claim of successful
migration. Old branches, worktrees, raw failures and existing resource processes
were not deleted or stopped.
