# Copyright (c) 2026 B1K2026 integration line
# SPDX-License-Identifier: MIT
"""Structural seams between the eval harness and concrete backends.

These protocols are sim-free. Concrete backends (BRVLA bridge, MoMa planning,
cuRobo v1/v2 executor, Isaac Sim adapter) implement the same shape but are
certified separately. A backend that has not passed the environment canary
must be reported ``UNVERIFIED``; calling it raises ``UnverifiedAdapter``.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol, runtime_checkable

from .errors import UnverifiedAdapter
from .schemas import ActionCommand
from .ws_contract import WsPolicyRequest, WsPolicyResponse

__all__ = [
    "AdapterStatus",
    "PolicyProvider",
    "MotionExecutor",
    "SimulatorAdapter",
]


class AdapterStatus(Enum):
    UNVERIFIED = "unverified"
    BUILD_PENDING = "build_pending"
    CANARY_PASS = "canary_pass"


@runtime_checkable
class PolicyProvider(Protocol):
    """Maps a policy request to a contract response (model side)."""

    status: AdapterStatus

    def compute(self, request: WsPolicyRequest) -> WsPolicyResponse: ...


@runtime_checkable
class MotionExecutor(Protocol):
    """Executes one action command; returns an execution receipt token."""

    status: AdapterStatus

    def execute(self, command: ActionCommand) -> str: ...


@runtime_checkable
class SimulatorAdapter(Protocol):
    """Thin facade over the simulation backend.

    Resolution and observation wrapper are fixed at environment creation time;
    the adapter must not resize observations at inference time. Model-side
    compliant 224-resize is a separate responsibility of the provider.
    """

    status: AdapterStatus

    def reset(self, task_name: str, instance_index: int) -> dict[str, Any]: ...

    def step(self) -> dict[str, Any]: ...

    def first_frame(self) -> dict[str, Any]: ...


def require_canary(adapter: Any) -> None:
    """Gate an adapter invocation on its canary status.

    CPU tests, sealed manifests, or a configured-but-unbuilt backend do not
    satisfy the gate. Raises ``UnverifiedAdapter`` unless status is
    ``CANARY_PASS``.
    """

    status = getattr(adapter, "status", AdapterStatus.UNVERIFIED)
    if status is not AdapterStatus.CANARY_PASS:
        raise UnverifiedAdapter(
            f"{type(adapter).__name__} status={status.value}; "
            "canary (build + reset + first frame + step) not passed"
        )
