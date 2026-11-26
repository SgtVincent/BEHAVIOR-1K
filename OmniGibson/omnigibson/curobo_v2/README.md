# curobo_v2 — cuRobo v2 executor lane (2026 R1Pro)

Integrator-owned binding behind the existing `moma.mp_plugin` manifest/loader.
This package defines **no** second planner RPC, no empty ABC replacement, and no
silent cuRobo v1 fallback. The BR replay side consumes the same sealed
`native23_joint` slices; the `execution23_se3_delta → native23` conversion lives
in the `brvla_bridge` owner lane.

## Layout

| File | Responsibility |
|------|----------------|
| `contracts.py` | Frozen native23 slices, R1Pro joint/frame inventory, target normalization. Stdlib only. |
| `errors.py` | Classified fail-closed exceptions. |
| `backend.py` | Lazy v2 backend: torch/curobo imports land only in `_bind` on the GPU-bound stage. |
| `executor.py` | `.plan(target) -> {"action23": TaggedAction, "certificate": {...}}`. |
| `factory.py` | `make_curobo_v2_executor(**kwargs)`, the manifest factory. |
| `tests/` | CPU contract, fail-closed, factory wiring and import-guard tests; `run_tests.py` stdlib runner. |

## Manifest wiring (private, outside git)

```json
{
  "moma": {
    "executor": {
      "module": "omnigibson.curobo_v2.factory",
      "factory": "make_curobo_v2_executor",
      "version": "0.1.0-pre",
      "required": true,
      "kwargs": {
        "robot_profile": "r1pro",
        "device_uuid": "GPU-...",
        "robot_config_yaml": "/private/.../r1pro.curobo.yaml",
        "canary_evidence": {"canary_pass": false, "receipt_sha256": null}
      }
    }
  }
}
```

The runner calls `factory(**kwargs, cfg=..., robot=...)`. Private URDF / mesh /
config paths, device UUIDs and receipt digests live in the external runtime
manifest; never hard-code NAS model paths or internal domains in public source.

## Boundary rules

- Top-level modules must stay importable without torch / curobo / omni / warp.
  The MoMa doctor classifies the plugin statically; heavy imports are deferred
  to backend bind.
- An unbound or uncertified backend raises a classified blocker. The executor
  never returns a zero-action placeholder, an implicit `ready=false` success,
  or a GT-challenge answer.
- Certificate `status == "optimal"` is emitted only on a finite, solver-succeeded
  plan; the object provider strictly rejects anything else.
- native23 order: `base3 | trunk4 | armL7 | gripL1 | armR7 | gripR1`. Planner
  owns `trunk4 + armL7 + armR7` (18 active joints); holonomic base velocity and
  continuous gripper width arrive from the navigation/grasp seams.
- The franka example gate (GPU0, 2026-11-26) validates the v2 cuda_core backend
  shape only. It is **not** R1Pro evidence for frames, collision spheres or tool
  contact. R1Pro frames/spheres/tool-contact must be verified on the R1Pro URDF
  in the GPU-bound canary.

## Test how-to (remote 2026 env)

```bash
python3 omnigibson/curobo_v2/tests/run_tests.py
```

Status: CPU contract/fail-closed/guard/provenance tests pass. R1Pro config, FK/IK,
standalone MP and one left-arm adapter plan were proven on the prior GPU
allocation (scope: left arm, self-collision only; no live scene, no SDK step).
A post-receipt certificate provenance patch awaits re-verification on the next
allocation; right-arm, dual-arm and live-scene plans remain open.
