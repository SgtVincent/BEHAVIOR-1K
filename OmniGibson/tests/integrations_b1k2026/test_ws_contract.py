# Copyright (c) 2026 B1K2026 integration line
# SPDX-License-Identifier: MIT
"""CPU contract tests for websocket request/response and chunk rules."""

import unittest

from integrations.b1k2026.errors import (
    ChunkLengthExceeded,
    ContractViolation,
)
from integrations.b1k2026.schemas import ACTION_DIM, ActionCommand, ActionSchema
from integrations.b1k2026.ws_contract import (
    EvalProfile,
    WsPolicyRequest,
    WsPolicyResponse,
)


def _cmd(request_id: str) -> ActionCommand:
    return ActionCommand(
        schema=ActionSchema.NATIVE23_JOINT,
        values=[0.0] * ACTION_DIM,
        request_id=request_id,
    )


class TestWsContract(unittest.TestCase):
    def test_per_step_response(self) -> None:
        resp = WsPolicyResponse(
            request_id="r1", action=_cmd("r1"), chunk_size=0
        )
        self.assertEqual(resp.action_chunk, ())
        self.assertEqual(resp.chunk_size, 0)

    def test_chunk_within_k_ok(self) -> None:
        resp = WsPolicyResponse(
            request_id="r2",
            action=None,
            action_chunk=(_cmd("r2"), _cmd("r2")),
            chunk_size=3,
        )
        self.assertEqual(len(resp.action_chunk), 2)

    def test_chunk_exceeding_k_rejected(self) -> None:
        with self.assertRaises(ChunkLengthExceeded):
            WsPolicyResponse(
                request_id="r3",
                action=None,
                action_chunk=tuple(_cmd("r3") for _ in range(4)),
                chunk_size=3,
            )

    def test_chunk_size_zero_forbids_chunk(self) -> None:
        with self.assertRaises(ContractViolation):
            WsPolicyResponse(
                request_id="r4",
                action=None,
                action_chunk=(_cmd("r4"),),
                chunk_size=0,
            )

    def test_response_without_action_or_chunk_rejected(self) -> None:
        with self.assertRaises(ContractViolation):
            WsPolicyResponse(request_id="r5")

    def test_challenge_rejects_privileged_obs(self) -> None:
        with self.assertRaises(ContractViolation):
            WsPolicyRequest(
                task_name="hello",
                instance_index=0,
                request_id="r6",
                profile=EvalProfile.CHALLENGE,
                privileged_obs=True,
            )

    def test_rsi_profile_allows_opt_in_privileged_flag(self) -> None:
        req = WsPolicyRequest(
            task_name="hello",
            instance_index=1,
            request_id="r7",
            profile=EvalProfile.RSI_RESEARCH,
            privileged_obs=True,
        )
        self.assertTrue(req.privileged_obs)


if __name__ == "__main__":
    unittest.main()
