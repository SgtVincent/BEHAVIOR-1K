# Copyright (c) 2026 B1K2026 integration line
# SPDX-License-Identifier: MIT
"""CPU contract tests for seam conformance and canary gating."""

import unittest

from integrations.b1k2026.errors import UnverifiedAdapter
from integrations.b1k2026.seams import (
    AdapterStatus,
    MotionExecutor,
    PolicyProvider,
    SimulatorAdapter,
    require_canary,
)


class _DummyProvider:
    status = AdapterStatus.CANARY_PASS

    def compute(self, request):  # noqa: D401 - structural dummy
        return None


class _UnverifiedExecutor:
    status = AdapterStatus.UNVERIFIED

    def execute(self, command):
        return "receipt"


class TestSeams(unittest.TestCase):
    def test_provider_conforms_to_protocol(self) -> None:
        self.assertIsInstance(_DummyProvider(), PolicyProvider)

    def test_executor_conforms_to_protocol(self) -> None:
        self.assertIsInstance(_UnverifiedExecutor(), MotionExecutor)

    def test_simulator_adapter_is_structural(self) -> None:
        class _Adapter:
            status = AdapterStatus.UNVERIFIED

            def reset(self, task_name, instance_index):
                return {}

            def step(self):
                return {}

            def first_frame(self):
                return {}

        self.assertIsInstance(_Adapter(), SimulatorAdapter)

    def test_canary_gate_blocks_unverified(self) -> None:
        with self.assertRaises(UnverifiedAdapter):
            require_canary(_UnverifiedExecutor())

    def test_canary_gate_allows_passed(self) -> None:
        require_canary(_DummyProvider())  # no exception


if __name__ == "__main__":
    unittest.main()
