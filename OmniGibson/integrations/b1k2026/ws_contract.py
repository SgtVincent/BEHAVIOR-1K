# Copyright (c) 2026 B1K2026 integration line
# SPDX-License-Identifier: MIT
"""Websocket policy request/response contract mirrors.

Message shapes follow the official post1 eval websocket: msgpack ``action``
``[action_dim]`` plus optional ``action_chunk`` ``[L, action_dim]``. The
chunk-size flag is opt-in: ``chunk_size == 0`` means one action per request;
when ``chunk_size > 0`` the chunk sequence length ``L`` must satisfy ``L <= K``.
This contract does not describe an "unlimited chunk" and does not re-introduce
the tiled-observation path in formal eval.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .errors import ChunkLengthExceeded, ContractViolation
from .schemas import ActionCommand

__all__ = ["EvalProfile", "WsPolicyRequest", "WsPolicyResponse"]


class EvalProfile(Enum):
    CHALLENGE = "challenge"
    RSI_RESEARCH = "rsi_research"


@dataclass(frozen=True)
class WsPolicyRequest:
    task_name: str
    instance_index: int
    request_id: str
    profile: EvalProfile
    privileged_obs: bool = False

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ContractViolation("request_id is required for correlation")
        if self.instance_index < 0:
            raise ContractViolation("instance_index must be >= 0")
        if self.profile is EvalProfile.CHALLENGE and self.privileged_obs:
            raise ContractViolation(
                "challenge profile must not carry privileged observations"
            )


@dataclass(frozen=True)
class WsPolicyResponse:
    request_id: str
    action: ActionCommand | None = None
    action_chunk: tuple[ActionCommand, ...] = ()
    chunk_size: int = 0

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ContractViolation("response must echo the request_id")
        if self.chunk_size < 0:
            raise ContractViolation("chunk_size must be >= 0")
        if self.chunk_size == 0 and self.action_chunk:
            raise ContractViolation(
                "chunk_size == 0 means per-step; action_chunk must be empty"
            )
        if self.action is None and not self.action_chunk:
            raise ContractViolation("response needs action or action_chunk")
        if self.chunk_size > 0:
            if len(self.action_chunk) > self.chunk_size:
                raise ChunkLengthExceeded(
                    f"chunk length L={len(self.action_chunk)} exceeds K="
                    f"{self.chunk_size}"
                )
        for cmd in self.action_chunk:
            if cmd.request_id is not None and cmd.request_id != self.request_id:
                raise ContractViolation("chunk command request_id mismatch")
