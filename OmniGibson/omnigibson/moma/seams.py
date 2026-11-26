"""MoMa four-layer seams against the shared integration contract.

Layers (INTERFACE_SPEC.md §5):

- **provider**: task / scene / target supply; the MoMa line uses the
  object-centric MoMaGen-style provider.
- **executor**: turn a planned target into a 2026 controller command tagged
  ``native23_joint``. The official post1 ``CuRoboMotionGenerator`` is the
  intended MP binding; the baseline QP / MoMaGen provider is a second
  implementation under the same contract and must not silently substitute
  for cuRobo (gap G1, integration-owned license/build).
- **observer**: read observations through the wrapper contract; privileged
  oracle fields stay in the isolated RSI profile (gap G9).
- **sim-adapter**: isolate every simulator coupling point. Sim-free leaf
  modules are unit-testable headless; controller/frames/DOF/Jacobian/dt/
  grasp/reset re-binding happens in the 2026 stack canary (gaps G2/G3/G8).

This module only declares the seam surface and its boundary rules. It does
not import omnigibson, torch, numba or cvxpy.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from .schema import TaggedAction


class MoMaObjectProvider(ABC):
    """Provider layer: object-centric target retargeting."""

    @abstractmethod
    def target(self, index: int, live_object_world: Any) -> Dict[str, Any]:
        """Return the live-aligned target pose(s), grippers and provider digest."""

    @abstractmethod
    def action(self, index: int, live_object_world: Any, executor: "MoMaExecutor") -> Dict[str, Any]:
        """Plan through the executor and return the tagged action plus certificate."""


class MoMaExecutor(ABC):
    """Executor layer: planned target -> bounded controller command.

    Implementations must return a :class:`TaggedAction` with the schema label
    fixed to ``native23_joint`` for the MoMa line. A solver failure must be
    surfaced as a rejected/unknown outcome, never as a zero-action success.
    """

    @abstractmethod
    def plan(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """Return ``{"action23": TaggedAction, "certificate": {...}}``."""


class MoMaObserver(ABC):
    """Observer layer: read-only live observations for predicates and S2."""

    @abstractmethod
    def observe(self, robot: Any, cans: Dict[str, Any], bucket: Any,
                pose_fn: Any, inside_type: Any, contact_type: Any) -> Dict[str, Any]:
        """Return the sanitized observation dict; unknown reads become UNKNOWN entries."""

    @abstractmethod
    def transitions(self, previous: Dict[str, Any], current: Dict[str, Any],
                    can_names: list[str]) -> list[Dict[str, Any]]:
        """Derive inside/grasp release events between two consecutive observations."""


class SimAdapter(ABC):
    """Sim-adapter layer: the single boundary touching the OmniGibson runtime.

    All methods below are re-bound against post1 ``bd049de3`` in the GPU-bound
    canary. Until then the sim-free layers must not call through this seam.
    """

    @abstractmethod
    def create_env(self, config: Dict[str, Any]) -> Any:
        """Create the 2026 environment (G3 re-bind)."""

    @abstractmethod
    def restore_initial_state(self, env: Any, source_cursor: int, init_metadata: Any) -> None:
        """Restore the same initial state as the a28 baseline (G8 identity check)."""

    @abstractmethod
    def step(self, env: Any, action: TaggedAction) -> Dict[str, Any]:
        """Apply the tagged command through the real 2026 controller pipeline."""

    @abstractmethod
    def read_live_spec(self, env: Any, step_index: int) -> Dict[str, Any]:
        """Read robot/object/TCP live spec from the runtime; raises if the live reader is unbound."""
