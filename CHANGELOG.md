# BRVLA Conversion Pipeline — Changelog

> Auto-generated from `git log origin/main..HEAD`; 67 commits covering the B1K→BRVLA data conversion pipeline, Stage-C physical replay verification, and related tooling.

---

## FK Conversion Pipeline (Core Adapter + Batch Infrastructure)

- **Minimal FK-only adapter** — mature restore to 3-row FK conversion, restored frame validation, clock delta zero assertion, IK-neutralized vendored IK (`3ff6921`)
- **Whole-demo continuous FK** — frame0 single restore with `[0,T)` continuous FK chain, segment post-hoc slicing, demo-level data bind for unlabeled prefix (`0bcc7cc3c`)
- **Multi-unit orchestration** — single `-m multi` entrypoint for ordered segment lists, mixed segment+demo units in one evaluator, CSV-driven first-batch pre-selection (`e27795d`, `3f229ef`, `e31f05f`, `80894d9`)
- **Hydra entrypoint wiring** — scope=segment|demo|multi via composed config; explicit segment-level binding via mature loaders (`74bf684`, `5a84e1c4`, `8cf86fb0c`)
- **Per-unit log_path + sample-key contract gate** — AST-scanned mature sample key contract, per-unit output isolation, multi/demo/mixed unit coverage (`7da821a3d`)
- **Conversion gate hardening** — single conversion gate shared by production FK reader and B' reader; cross-source agreement; production reader red/green mutation tests (`c8d021a6f`)

## IK Conversion Pipeline (Differential IK → Iterative IK)

- **Window-2 differential IK adapter** — BRVLA 3-row → native23 via DLS single-step IK, solve-calls tracking, fallback detection, gripper ±1 exact restoration (`98822db`)
- **End-effector pose error measurement** — per-limb quantile summary, fixed vs chained seeding, first-class wall-clock for FK and IK windows (`c69b29b3d`, `c5dea5d9`, `aa0bdc3ab`, `96b6b509d`)
- **First-target iterative IK** — explicit N-iteration first-target solve with world restore + joint perturbation retry, ~49-row convergence for distant targets (`11f7e414`, `4f2f17bf0`)
- **Iterative IK evidence hardening** — live canary failure evidence preservation, production canary failure seam closure, retry-perturb exception handling (`9b030b3ec`, `4abb7ba4f`, `61204730a`, `9caecea56`, `1c410901`, `4d7b04271`)
- **CachedFunctions compatibility fix** — do not membership-test EE control dict; use direct subscript with KeyError catch (`e1bd7d52c`)
- **IK receipt double-index fix** — remove redundant `[i]` on already-indexed SavedTargets pose matrix, caught at n≥5 rows (`9471731d6`)

## Stage-C Physical Replay Verification

- **Stage-C A/B replay gate** — physical A (original actions) and B (converted actions, live IK) replay with per-arm video separation, A/B exact-set policy/arm whitelist, rollout budget capping to converted length (`7d4de762b`, `1e20aadef`, `eb0ecf945`)
- **Conversion fidelity vs physical tracking split** — separate EE position fidelity (pre-Kit kinematic roundtrip, p95≤2cm/max≤10cm) from physical tracking (post env.step with controller+contact, un-gated first-run measurement only) (`29aaa24a5`)
- **B-phase enter before first live IK solve** — isolate physical B phase entry without full dump/load snapshot, verifying solve reads only `phase`/`fresh_control_dict()`/`ik` (`c8980ac92`)
- **Rollout metrics from real fixture** — derive rehearsal metrics from production fixture files, subset guard for rehearsal-vs-production key parity, rollout.final_step as authoritative step count (`73e7b28b1`)
- **Target poses by limb key** — reuse production SavedTargets for rehearsal, map limb→task keys via EE_TASK_KEYS, eliminate hand-written staging payloads (`1fbf5e96d`)
- **Stage-C partial failure evidence persistence** — complete result JSON written before any gate raise, no evidence gap between failure detection and evidence capture (`521dc107f`)

## Pre-Kit Preflight Band

- **Pre-Kit preflight band** — hoist all non-sim checks (run_root existence, CSV/annotation readability, key contract, disk space) before Kit cold start; zero-GPU path to Kit-construct-line (`17cdade72`)
- **Real launch validates preflight marker** — preflight marker non-idempotent overwrite fix; two-step runner usage covered by test; marker existence+depth check after preflight (`136b0e7ea`, `9b70fa5bb`)
- **Production pre-Kit annotation loader guard** — guard against stub leak from rehearsal injection; production asserts real loader in all branches (`cf0c80438`)
- **Annotation reconcile + digest binding + disk gate** — annotation-vs-manifest reconcile, plan_digest isolation, disk space gate as fail-closed pre-Kit check (`ae1fcd7b7`, `a94befc4a`, `acc0cb623`)
- **Restored objects initialization fix** — zero-duration pending-object init before Toggle state load in cross-demo scene.restore (`2a3dc2571`)

## GT-Plan Text Injection

- **Minimal GT-plan P/O text injection** — segment success eval with plan/outcome text injected into observation, config-driven gated behind `gt_plan_experiment_enabled` (`4962c0e`, `21397e95`, `a6c39bf1`)
- **GT plan scalar type preservation** — fix websocket int→float type widening across policy request/response conversion (`627c7526`)

## Executed-Action Trace (Per-Step)

- **Default-off per-step executed-action trace** — record policy_returned_action vs executed_action per environment step, renaming raw_policy_action for clarity (`1dc37257`, `f42857e7`)
- **Live evaluator runtime config capture** — bind resolved Hydra config into trace metrics for provenance (`d14d3e03`)

## Memory Telemetry & Serving

- **Memory telemetry in serving response** — return KV cache and session memory metrics with each policy response, tear down only sessions spawned by current request (`bb38c30d`, `67ed410c`)
- **Absolute environment step with each policy request** — send env_step counter for server-side alignment and diagnostics (`ddbece20b`)

## Paired Smoke Driver

- **Minimal B1K→BRVLA paired smoke driver** — mature A replay → converted [N,23] → B file replay, exhaustion net with B budget following A rows, wall-clock instrumentation, optional smoke_steps (`dcef7730`, `55566927`, `a03c4496`)
- **Segment idx / restore frame from composed config** — config-driven runtime parameters, eliminating hard-coded debug singletons (`d13e5998`)

## Evidence, Gates & Durable Finalization

- **Durable FAIL finalization** — shutdown-safe summary, staged clocks, production-style raising gates that persist evidence before exit (`85cef3f4f`, `63b56130a`)
- **Scientific rows_receipt validation** — forbid NOT_AVAILABLE leaves; require parseable structured evidence for every row (`2693efb1f`)
- **Write-video fail-closed** — video output hardened to fail-closed; IK neutralization guard for FK-only runs (`e7ae01bac`)
- **Addict MacroDict-safe renderer probe** — early serialization gate before Kit shutdown, preventing probe crash from MacroDict auto-key behavior (`f17ef1e2c`)
- **Real construction side-effect evidence** — override gate with solved-override counting; production vs rehearsal identity check (`5e8f0138a`)

---

## Commit Listing (chronological)

| Commit | Date | Description |
|--------|------|-------------|
| `d14d3e031` | 2026-10-08 | feat(trace): bind live evaluator.cfg resolved runtime config into metric |
| `dcef7730c` | 2026-10-09 | feat(eval): minimal B1K→BRVLA paired smoke driver |
| `555669272` | 2026-10-09 | fix(brvla-smoke): correct import-order check + S1/S2 checkout realpath gate |
| `a03c44963` | 2026-10-09 | feat(brvla-smoke): optional smoke_steps, B budget follows A rows, exhaustion net, wall-clock |
| `d13e59981` | 2026-10-09 | fix(brvla-smoke): read segment idx/restore frame from composed config |
| `3ff692189` | 2026-10-10 | feat(brvla-fk): minimal FK-only adapter (mature restore→3-row FK) |
| `e7ae01bac` | 2026-10-10 | fix(brvla-fk): write_video fail-closed + neutralize vendored IK |
| `5e8f0138a` | 2026-10-10 | fix(brvla-fk): real construction side-effect evidence + solve override gate |
| `9fe1268e2` | 2026-10-10 | fix(brvla-fk): bind segment/demo/rawdata via mature loaders before restore |
| `3f4d8582a` | 2026-10-10 | refactor(brvla-fk): mature _as_demo_id + require explicit segment_level |
| `98822db93` | 2026-10-10 | feat(brvla-fk): window-2 differential IK adapter (BRVLA→native23) |
| `f17ef1e2c` | 2026-10-10 | fix(brvla-fk): addict MacroDict-safe renderer probe + early serialization gate |
| `2693efb1f` | 2026-10-10 | test(brvla-fk): forbid NOT_AVAILABLE leaves in scientific rows_receipt |
| `63b56130a` | 2026-10-10 | fix(brvla-fk): shutdown-safe durable summary/finalization, staged clocks |
| `85cef3f4f` | 2026-10-10 | fix(brvla-fk): durable FAIL finalization on production-style raising gates |
| `f30f7c7a8` | 2026-10-10 | feat(brvla-fk): window-2 IK hydra entrypoint + read-only window1 gates |
| `96b6b509d` | 2026-10-12 | feat(brvla-ik): unpin whole-segment rows, add fixed+chained IK seeding and first-class wall clock |
| `aa0bdc3ab` | 2026-10-12 | feat(brvla-fk): unpin FK whole-segment rows + first-class FK wall clock; share window/timing with IK |
| `9471731d6` | 2026-10-12 | fix(brvla-ik): do not double-index SavedTargets pose matrix in IK receipt |
| `c69b29b3d` | 2026-10-12 | feat(brvla-ik): record end-effector pose error (the real conversion fidelity metric) |
| `c5dea5d95` | 2026-10-12 | refactor(brvla-ik): per-limb independent EE pose-error quantile summary |
| `e1bd7d52c` | 2026-10-12 | fix(brvla-ik): do not membership-test the CachedFunctions EE control dict |
| `80894d9a9` | 2026-10-12 | feat(brvla-fk): whole-demo continuous FK + same-process multi-unit + first-batch CSV/stable-id |
| `e31f05f41` | 2026-10-12 | feat(brvla-fk): wire scope=segment|demo|multi into the Hydra -m entrypoint |
| `3f229ef0c` | 2026-10-12 | feat(brvla-fk): -m multi unit_scope=segment via explicit ordered segment list |
| `e27795d06` | 2026-10-12 | feat(brvla-fk): one -m multi mixed spec (segment units + one demo, one evaluator) |
| `74bf684ca` | 2026-10-12 | fix(brvla-fk): read annotations[f"{level}_annotation"] (mature key), not "skill" |
| `5a84e1c4` | 2026-10-12 | fix(brvla-fk): create multi-unit top run root + no-Kit rehearsal gate |
| `8cf86fb0c` | 2026-10-13 | test(brvla-fk): top-gate-specific missing-parent assertion + argv single source |
| `7d4de762b` | 2026-10-13 | feat(brvla): add Stage-C physical A/B replay gate (CPU/no-Kit rehearsal only) |
| `29aaa24a5` | 2026-10-13 | fix(brvla): make Stage-C EE numeric gate real, split fidelity vs tracking, wire production video gate, reachable failure branches, corpus aggregation |
| `c8d021a6f` | 2026-10-13 | fix(brvla): single conversion gate shared by both readers; cross-source agreement; production B' reader red/green tests |
| `7da821a3d` | 2026-10-13 | fix(brvla): per-unit log_path + mature sample-key contract gate for multi/demo/mixed units |
| `194ec9bd6` | 2026-10-14 | feat(brvla): stage2 runtime distinct-segment gate (offline two-unit verifier) |
| `1e20aadef` | 2026-10-14 | fix(brvla): stage-C per-arm video separation + exact-set A/B whitelist |
| `73e7b28b1` | 2026-10-14 | fix(brvla): stage-C reads rollout.final_step; rehearsal derives metrics from real fixture + subset guard |
| `c8980ac92` | 2026-10-14 | fix(brvla): stage-C arm-B enters physical B phase before first live IK solve |
| `eb0ecf945` | 2026-10-14 | fix(brvla): stage-C caps B rollout to converted length; exhausted timeout is a scientific negative |
| `1fbf5e96d` | 2026-10-14 | fix(brvla): stage-C reads target poses by limb key, rehearsal derives from real SavedTargets |
| `17cdade72` | 2026-10-15 | feat(brvla): pre-Kit preflight band for FK batch; hoist all non-sim checks before Kit |
| `cf0c80438` | 2026-10-15 | fix(brvla): guard production pre-Kit annotation loader against stub leak |
| `136b0e7ea` | 2026-10-15 | fix(brvla): real launch validates preflight marker, never overwrites it |
| `9b70fa5bb` | 2026-10-15 | test(brvla): cover runner preflight→real-launch two-step sequence |
| `ae1fcd7b7` | 2026-10-15 | feat(brvla): final pre-475 batch gates — annotation reconcile, digest binding, disk gate |
| `a94befc4a` | 2026-10-15 | fix(brvla): fail-closed batch gates per review (grouped reconcile, digest isolation, disk) |
| `acc0cb623` | 2026-10-15 | fix(brvla): review2 batch gate isolation (prod helper, unoverrideable sha) |
| `0bcc7cc3c` | 2026-10-15 | fix(brvla): whole-demo FK uses demo-level data bind; allow unlabeled prefix |
| `2a3dc2571` | 2026-10-15 | fix(sim): initialize restored objects before state load |
| `521dc107f` | 2026-10-15 | fix(brvla): persist Stage-C partial failure evidence |
| `11f7e4141` | 2026-10-15 | feat(brvla): add explicit first-target iterative IK |
| `4f5c7907e` | 2026-10-15 | fix(brvla): harden iterative IK evidence contracts |
| `9b030b3ec` | 2026-10-15 | fix(brvla): close iterative canary evidence gaps |
| `4abb7ba4f` | 2026-10-15 | fix(brvla): preserve live canary failure evidence |
| `61204730a` | 2026-10-15 | fix(brvla): test production canary failure seam |
| `9caecea56` | 2026-10-15 | fix(brvla): persist every canary cleanup failure |
| `4f2f17bf0` | 2026-10-15 | feat(canary): retry iterative IK with world restore and joint perturbation |
| `1c4109010` | 2026-10-17 | fix(canary): set work_error on restore/perturb exceptions, reset restore_receipt per retry cycle |
| `4d7b04271` | 2026-10-17 | fix(canary): wrap joint positions in torch.tensor for set_joint_positions |

### GT-Plan & Trace commits (interleaved in mainline)

| Commit | Description |
|--------|-------------|
| `67ed410ce` | test(serving): assert the MoMA teardown hook, and probe the real handler |
| `bb38c30df` | feat(serving): return memory telemetry, and tear down only sessions we spawned |
| `627c75266` | fix(eval): preserve GT plan scalar types across websocket conversion |
| `ddbece20b` | feat(eval): send the absolute environment step with each policy request |
| `4962c0e86` | feat(eval_segment): minimal GT-plan P/O text injection for segment success eval |
| `a6c39bf12` | fix(eval_segment): define/import the GT-plan install helper wired in 4962c0e |
| `21397e953` | fix(eval_segment): gate pre-rollout component gate on gt_plan_experiment_enabled |
| `1dc37257e` | feat(eval): default-off per-step executed-action trace (Path A) |
| `f42857e7d` | refactor(trace): rename raw_policy_action → policy_returned_action |

---

*Generated from `git log origin/main..HEAD` at {{generation_timestamp}}*