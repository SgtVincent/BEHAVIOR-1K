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
- `omnigibson/eval/delta_tf_replay_entry.py`
  - Peer runner that mirrors the official `omnigibson.eval.eval` cfg shape and
    delegates to `PrimitiveEvaluator`. It does not rebuild an eval loop,
    websocket server or model-serving stack. Upstream `eval.py` is unchanged.
- `omnigibson/eval/brvla_ws_policy.py`
  - `CheckpointWsPolicy` derives from upstream `WebsocketPolicy`; transport
    reuses the upstream `WebsocketClientPolicy` (healthz, handshake, msgpack
    envelope, optional open-loop chunk replay).
  - Reset-time metadata gate (layout / horizon / chunk / history) fails closed.
    Official 61-d proprio maps to declared legacy 256 wire slices; RGB3 frames
    forward; no padding, no source-joint fallback.
  - The physical23 / IK-target23 row is held at the controller boundary until a
    bound native-joint decoder seam exists; zero env steps otherwise.
  - Checkpoint, stats and preprocessing stay in the unchanged frozen server;
    this client loads no weights or stats.
- `omnigibson/eval/brvla_ws_entry.py`
  - Peer runner mirroring the official eval cfg shape; upstream
    `BatchedEvaluator` owns stepping, metrics and video.
- `scripts/brvla_ws_debug.sh`
  - Sha-gated versioned runner for the frozen-model websocket debug slice.

Official primitives dependency remains the unmodified `setup.py` extra:
`StanfordVL/curobo@78612f45cef52c3fa0298de243a54cd7ca614414` (MotionGen API).
cuRobo v2 is deliberately NOT part of this branch's execution path.

Sources:
- https://behavior.stanford.edu/reference/action_primitives/curobo.html
- https://behavior.stanford.edu/reference/controllers/ik_controller.html#controllers.ik_controller.InverseKinematicsController.add_member
- https://behavior.stanford.edu/omnigibson/controllers.html

## Input and authority

The decoder input is execution23: left pose6, right pose6, two gripper widths in
metres (frozen range 0..0.05 m; width is never finger qpos), base dx/dy/dyaw,
chest pose6. This is NOT the native23 joint command.
Use explicit `pairwise_delta` for saved conversion rows or `chunk_cumulative`
for model chunk targets. Anchor rebinding is explicit at each run/chunk, never
silently refreshed per control step.

Saved converted episode contract (`brvla_saved_execution23_v1`,
`pairwise_delta`, `<f4`, Nx23, 30 Hz): `manifest.json` + `actions.npy`, columns
left 0:6 / right 6:12 / continuous_gripper 12:14 / move 14:17 / chest 17:23;
fixed run-local start anchors live in `chunks[0].anchors`; chest link is
`torso_link4` (manifest `provenance.chest_link`). The replay lane reads the
original npy with schema/representation/shape/digest gates; the model lane's
`physical23_same_anchor_cumulative` rows are never mixed into this loader.

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

- Both original new Python modules compile syntactically.
- Existing remote Python, CPU only: AST-isolated decoder and exact upstream
  transformation function bodies; not a full installed-module import.
- Pairwise target accumulation; cumulative targets not accumulated twice;
  malformed-width and nonfinite inputs reject; upstream Euler rotation checked
  independently against SciPy. Max rotation-entry difference
  `5.960464477539063e-08`.
- Round 2026-11-27 (local Mac, CPU only, no torch/omnigibson installed):
  - `py_compile` passes for all three peer modules.
  - AST-isolated import resolution: every `omnigibson.*` import in the peer
    modules resolves to an existing upstream module and defined/exported name;
    third-party imports (torch, numpy, gymnasium, hydra, omegaconf) are not
    resolved locally.
  - Upstream base classes, evaluator factory, controller view, robot attributes
    and entry helpers were read against the `bd049de` tree and matched.
  - Diff contains only public behavior.stanford.edu documentation URLs; no NAS,
    home/user path, checkpoint, weight, token or credential is committed.
- **No new GPU run, no physical step, no new video, no business PASS.**
- No package was installed locally; no shared remote environment was changed.
- Round 2026-11-27 (frozen-model WS slice, local Mac, CPU only):
  - `py_compile` passes for `brvla_ws_policy.py` and `brvla_ws_entry.py`.
  - AST-isolated import resolution: every `omnigibson.*` import resolves to an
    existing upstream module and defined name (`WebsocketPolicy`,
    `BatchedEvaluator`, `resolve_instance_ids`, `DEFAULT_EVAL_SEED`,
    `seed_everything`, `gm`, `create_module_logger`).
  - Proprio slices were re-verified against upstream
    `eval_utils.PROPRIOCEPTION_INDICES["R1Pro"]`; the legacy 256 slices and the
    frozen metadata constants are read from the retained original bridge
    sources, not re-invented.
  - New files carry no NAS/home path, checkpoint, weight, token or credential.
- **Round 2 also has no GPU run, no physical step, no model forward and no
  business PASS.** The decoder seam is absent, so every server row is held and
  zero simulator steps are taken by design.
- Round 2026-11-27 (round 3, local Mac, CPU only):
  - Replay loader reads the original `manifest.json` + `actions.npy` directory
    and gates schema (`brvla_saved_execution23_v1`), representation
    (`pairwise_delta`), shape, float32 dtype and manifest/action digest; an
    optional compatibility npz shim remains as a fallback. Manifest field
    names, slices, anchors, dt and `chest_link=torso_link4` were checked against
    the retained original converted-manifest sample.
  - Gripper widths are gated to the frozen 0..0.05 m range; widths are metres,
    not finger joint qpos.
  - WS policy parity with the retained bridge contract: `set_action_dim` refuses
    any width other than 23; image lineage records `channels`; every bound
    proprio slice records a 64-hex `source_sha256`; the boundary guard rejects
    a decoder output that is not tagged `joint-native23`.
  - `py_compile`, `bash -n` and AST-isolated import resolution pass. No GPU run,
    no physical step, no live-server handshake, no business PASS.
- Round 2026-11-27 (round 4, field trace 302 + live readback 303, local Mac CPU):
  - Field first GPU trace reached policy construction and failed gate (a):
    official DEFAULT cuRobo kinematics `link_names` advertised only
    `left_eef_link` (ee) + `right_eef_link`; the requested chest
    `torso_link4` was missing. No constraint was silently dropped.
  - Live readback 303 (sha256 `6aeff519…48fbfa1`) fixes the robot truth:
    robot_r1 / r1pro, action_dim 25, 28 qpos; controller order
    base, trunk, arm_left, gripper_left, arm_right, gripper_right; base
    HolonomicBaseJointController velocity dof [0,1,5]; trunk/arms
    JointController position; two fingers per hand dof [24,25] / [26,27];
    no delta, no impedance. `torso_link4`, left/right EEF and base_link
    exist among 44 links. Joint position/velocity limits are null and must
    be read live from the official controller; no defaults are filled.
  - Gate (a) minimal fix: `_build_peer_curobo_profile` loads the official
    DEFAULT yaml (PyYAML `safe_load`), appends only `torso_link4` to
    `kinematics.link_names` (de-duplicated), and dumps a per-run temp peer
    yaml; the official `CuRoboMotionGenerator` receives it through the
    official `robot_cfg_path` override with `use_default_embodiment_only`.
    Upstream core/yaml are untouched; the shared install is not written to;
    no 700-line static copy is maintained. Receipt records source yaml path,
    source sha256, added link, resulting link_names and peer yaml path.
  - Editable-binding patch: both runner shells prepend the worktree
    `OmniGibson` dir to `PYTHONPATH`; both entries fail closed if the loaded
    `omnigibson.__file__` resolves outside this worktree. No shared conda /
    site-customize is modified.
  - Gate (b) moving-base and gate (c) changing gripper width remain
    `NotImplementedError` before planning; neither channel is deleted or
    skipped. Row0 yaw 4.563e-6 and 7918/13333 moving rows, 5651/13333
    grip-changing rows from 303 are real unsupported motion, not removable
    channels.
  - `py_compile`, `bash -n` and AST-isolated resolution pass; a local
    yaml load/merge/dump round-trip mirror test passes. **No new physical
    step, no new plan, no video, no business PASS.** The warp 1.12 per-run
    preload shim stays field-side versioned and must later be bound into
    the baseline shell; it is never written to the shared site-customize.

## Open items (not yet verified)

- Full installed-module import and real consumer-path loading in an authorized
  remote isolation; local Mac has no torch/omnigibson and must not install deps.
- Robot/controller profile binding: the `q_to_action` path requires an
  absolute-position JointController / HolonomicBaseJointController profile. The
  shipped eval `r1pro.yaml` uses MultiFinger grippers and is not compatible;
  a joint-profile robot config must be bound for this slice.
- Embodiment: DEFAULT is assumed; the available cuRobo embodiments depend on
  the external robot curobo asset. A missing DEFAULT now raises an explicit
  error listing available embodiments instead of an implicit KeyError.
- Gripper qpos vs metre-width unit contract is unverified against a real
  controller; the unchanged-gripper guard may need a controller-command-level
  comparison.
- Input exhaustion / planner failure currently propagates as an uncaught
  exception through the official loop; no diagnostic result JSON is recorded
  yet. This must not be reported as task success.
- Original checkpoint model inference, source/frame0 identity, moving-base /
  gripper phases, held-object planning, action-clock preservation and the
  three-can termination remain open.
- Converted-dir lane: the manifest fixed-start anchors are recorded in the
  receipt but their agreement with the live frame0 pose is not yet checked
  against a real reset; the 303 readback has `anchor_residual_base_relative=null`
  because the probe v2 did not compute it. Real anchor binding is still done
  in the primitive `bind_anchors` step via the official inverse transform;
  this null is not a frame0 pass. Tool link names (chest = `torso_link4`) and
  the six-group controller layout were confirmed by live readback 303; joint
  position/velocity limits read null there and must be read live from the
  official controller — no default is filled. Finger qpos ↔ metre-width real
  range is still unverified.
- Beyond the fixed-base / unchanged-gripper first slice, moving-base and
  grasp-phase width changes must be bound through upstream Controller/MotionGen;
  metres width must be normalised into the controller range, never sent as
  finger qpos or used to skip a grasp phase.
- Frozen-model WS slice (round 2), not yet verified on a live server:
  - The native-joint decoder seam is absent; `CheckpointActionBlocked` holds
    every row and takes zero steps until a verified IK-target23 → joint-native23
    binding exists along the official Controller path.
  - Server metadata handshake (`action_layout`, `horizon=32`,
    `chunk_generated=50`, `history_every_control_step`) has only been checked
    against retained receipts; a live metadata frame must confirm it.
  - Camera role → sensor prim names default to the r1pro profile and must be
    re-bound against the live robot config; the legacy 256 slice table and the
    profile-zero EE/base slots need a live proprio canary.
  - Client-side action-chunk replay (`action_chunk_size > 1`) is unused by
    default and unverified against the frozen server's chunk envelope.
  - The original server must be launched from its unchanged serving directory
    with the original checkpoint, stats and preprocessing; this client slice
    does not start or reload that server.

## Minimal real-GPU entry plan (not executed locally)

1. Remote, authorized isolation only: install the pinned upstream extras
   (`StanfordVL/curobo@78612f45...`, MotionGen API; cuRobo v2 stays out of the
   execution path). Do not touch local Mac conda or shared remote envs.
2. Bind one joint-profile R1Pro robot config (absolute-position JointController
   arms/trunk/grippers, HolonomicBaseJointController base); confirm the cuRobo
   profile advertises the requested left/right/chest links and the available
   embodiment.
3. Launch one single-env debug rollout through the versioned peer runner:
   `bash OmniGibson/scripts/delta_tf_replay_debug.sh` with `TASK_NAME`,
   `CONVERTED_DIR` (manifest.json + actions.npy) and `TOOL_LINKS_JSON` exported
   (see script header; optional `MANIFEST_SHA256` / `ACTIONS_SHA256` pins, or the
   npz compatibility fallback). It sha-gates to this slice's commit, binds the
   peer joint-profile `eval/r1pro_delta_tf_debug.yaml`, and calls
   `python -m omnigibson.eval.delta_tf_replay_entry` with `--write-video`.
   Keep `write_video=True` and retain a decodable MP4 plus the result JSON.
4. First gate: one real delta-TF target reaches collision-aware planning,
   physical feedback execution and a decodable video; fixed base/gripper joints
   are preserved. Only then add proven missing base/gripper bindings via
   upstream modules.
5. Later gates, in order: original BRVLA input/model contracts and source/frame0
   lineage; original Agent + MoMaGen decision sources (MoMaGen pinned commit,
   native module direct call; no numpy port, no source-joint fallback); converted
   replay, original-model physical execution, and the original three-can
   closed-loop task. Only after these pass is a commit called a validated
   baseline; 2025 chapters stay archived-frozen until then.

The current branch preserves code for continuation, not a claim of successful
migration. Old branches, worktrees, raw failures and existing resource processes
were not deleted or stopped.
