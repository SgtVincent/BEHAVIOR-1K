# brvla_bridge (CPU contract layer)

Port of the frozen BRVLA serving contract onto the 2026 BEHAVIOR-1K codebase
(post1 `bd049de3`). This revision is **CPU-only**: it defines contracts,
schemas, mapping provider seam, transport, and the websocket protocol boundary.

## Boundaries

- No model loading/forward, simulator, `env.step`, Kit/Isaac import, GPU job,
  or real controller installation in this layer.
- Checkpoint/stats/data paths and hashes live in a private runtime manifest
  (`BRVLA_MANIFEST_PATH`), never hardcoded in public source.
- The 61 -> legacy 45 mapping is fail-closed for unverified slots. The base-move
  frame was confirmed by the GPU1 live proprio sample (robot-local `[vx, vy, wz]`);
  the torso SE3 chain binds at the verified CPU FK layer once the manifest URDF
  identity and four-joint adapter chain are injected. The live CPU-FK vs
  simulator-trunk relative-pose canary and the physical tracking gate are
  separate later layers. No slot is zero-filled or bridged from the old
  256-wide vector.
- Gate A (physical replay) must run through the official post1 cuRobo seam.
  Legacy DLS/stock IK implementations cannot substitute for it; their receipt
  status is recorded in the backend registry.

## Layout

- `contracts.py` frozen widths, layouts, gates, 2026 evaluator entry, wire keys
- `schema.py` canary observation bundle validation
- `state_mapping.py` mapping provider interface (fail-closed)
- `torso_fk.py` verified four-joint trunk CPU FK chain
- `model_session.py` manifest identity gate + provider assembly factory
- `transport.py` execution23 anchor queue, SE3 helpers, base velocity
- `ws_protocol.py` post1 request/response, chunk cache, metadata gate, envelope
- `backend.py` controller backend receipts and replay gate
- `manifest.py` private frozen-asset manifest loader
- `tests/` stdlib runner, positive and negative cases

## Tests

Run on the remote 2026 environment only:

```
python -B brvla_bridge/tests/run_tests.py
```

The runner has no pytest dependency. Do not claim a green result before this
executes remotely.
