# integrations.b1k2026 — sim-free public contract layer

This directory holds the public, dependency-free contract code shared by the
BRVLA bridge, MoMa planning, and motion-executor seams for the 2026
BEHAVIOR-1K evaluation line.

## Status

- Contract layer: CPU unittest covered (`tests/integrations_b1k2026`).
- Concrete adapters (policy model, motion planner, simulator backend) are
  **UNVERIFIED** until a bound resource passes the environment canary:
  build → reset → first frame → `env.step`, with debug video retained.
- This package imports no third-party dependency. It must not import Isaac Sim,
  torch, cuRobo, or model weights at module import time.

## Boundaries

- Three 23-d action schemas are independent tags; a bare vector without its
  schema tag is rejected.
- Chunked actions are opt-in: when `chunk_size > 0`, sequence length `L <= K`;
  `chunk_size == 0` means one action per request.
- Observation resolution is fixed at environment creation; no inference-time
  resizing in the adapter. Tiled observations are reserved for performance
  research and are disabled in formal eval.
- cuRobo v1 (Stanford fork snapshot, NVIDIA License) and cuRoboV2 (Apache-2.0,
  rewritten public API) are separate backends behind the executor seam; one
  must not be substituted for the other silently.

## License

MIT, matching the upstream BEHAVIOR-1K license. Third-party restricted
manifests remain private and are not vendored here.
