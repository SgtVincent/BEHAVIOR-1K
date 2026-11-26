"""Environment adapter bound to the live 2026 OmniGibson environment.

Verified post1 surface (probe evidence ``43_moma_caller_detail`` /
``46_moma_registry``):

- ``env.step(action, n_render_iterations=1)`` -> ``(obs, reward, terminated,
  truncated, info)``; the wrapper forwards transparently.
- ``env.reset(**kwargs)`` -> ``(obs, ...)``.
- ``env.task.success[env_idx]`` is the BDDL success boolean; a BDDL dict is
  never coerced to bool by this adapter.
- ``env.task.object_scopes[env_idx]`` maps BDDL instance name to the live object.
- ``env.scenes[env_idx].robots[0]`` is the single evaluation robot.

The adapter does not import the simulator; the live environment is
duck-typed. Scene/object predicate resolution (``Inside`` / ``ContactBodies``)
is injected by the observer rather than hardcoded here.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


class EnvAdapterUnboundError(RuntimeError):
    """Raised before a live environment is bound."""


class SdkEnvironmentAdapter:
    """Single-environment adapter over the shared OmniGibson wrapper."""

    def __init__(self, env: Any = None, env_idx: int = 0) -> None:
        self._env: Any = env
        self._env_idx = int(env_idx)

    def bind(self, env: Any, env_idx: int = 0) -> None:
        self._env = env
        self._env_idx = int(env_idx)

    @property
    def bound(self) -> bool:
        return self._env is not None

    @property
    def env(self) -> Any:
        if self._env is None:
            raise EnvAdapterUnboundError("environment adapter not bound")
        return self._env

    @property
    def env_idx(self) -> int:
        return self._env_idx

    # -- lifecycle --------------------------------------------------------

    def reset(self, **kwargs: Any) -> Any:
        return self.env.reset(**kwargs)

    def step(self, action: Any, n_render_iterations: int = 1) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        return self.env.step(action, n_render_iterations=n_render_iterations)

    # -- task / scene ------------------------------------------------------

    def task_success(self) -> bool:
        return bool(self.env.task.success[self._env_idx])

    def object_scope(self) -> Dict[str, Any]:
        return dict(self.env.task.object_scopes[self._env_idx] or {})

    def scene(self) -> Any:
        return self.env.scenes[self._env_idx]

    def robot(self) -> Any:
        robots = self.scene().robots
        if len(robots) != 1:
            raise ValueError(f"evaluation assumes one robot, got {len(robots)}")
        return robots[0]

    def task(self) -> Any:
        return self.env.task
