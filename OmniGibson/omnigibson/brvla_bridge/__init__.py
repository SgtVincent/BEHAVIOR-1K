"""BRVLA 2026 candidate portable package.

Scope (this revision, CPU-only contract layer):
    - frozen contract constants and read-only identity references
    - private frozen-asset manifest loader (paths/hashes stay outside public source)
    - canary observation schema validation
    - 2026 -> legacy state45 mapping provider interface (fail-closed)
    - execution23 transport and anchor queue (stdlib port)
    - WS protocol boundary: post1 request/response contract, chunk cache,
      metadata gate, msgpack ndarray envelope
    - controller backend receipt registry; replay gate requires the official
      cuRobo primitive receipt

Out of scope until the integration lane provides the frozen base/feature
worktree and a bound GPU lane:
    - model loading/forward, simulator/env.step, websocket client implementation,
      real controller installation, stats/weights handling.

No torch / numpy / omnigibson / isaac / kit / msgpack import at package import.
Tests are executed on the remote 2026 environment, never installed or run on
this Mac.
"""

__version__ = "0.1.0-candidate"

from . import (  # noqa: F401
    backend,
    caller,
    contracts,
    delta_tf,
    errors,
    lineage,
    managed_policy,
    manifest,
    model_session,
    native_action,
    plan_wrapper,
    planner_grid,
    replay_policy,
    scope,
    schema,
    state_mapping,
    torso_fk,
    transport,
    ws_protocol,
)
