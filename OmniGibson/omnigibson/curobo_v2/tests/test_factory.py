# Copyright (c) 2026 B1K2026 integration line
# SPDX-License-Identifier: Apache-2.0
"""Factory / manifest wiring and heavy-import isolation tests."""
from __future__ import annotations

import pathlib
import sys
import unittest

_HERE = pathlib.Path(__file__).resolve().parent
_OMNIGIBSON_DIR = _HERE.parents[2]
sys.path.insert(0, str(_OMNIGIBSON_DIR))

from curobo_v2.executor import CuroboV2Executor  # noqa: E402
from curobo_v2.factory import (  # noqa: E402
    PLUGIN_VERSION,
    make_curobo_v2_executor,
)

FORBIDDEN_MODULES = ("torch", "cuda", "curobo", "warp", "cvxpy", "pytorch3d", "omni")


class FactoryWiringTest(unittest.TestCase):
    def test_factory_builds_executor(self) -> None:
        executor = make_curobo_v2_executor(
            robot_profile="r1pro",
            device_uuid="GPU-test",
            cfg={"run_id": "probe"},
            robot=None,
        )
        self.assertIsInstance(executor, CuroboV2Executor)
        self.assertTrue(callable(executor.plan))
        profile = executor.profile()
        self.assertEqual(profile["context_keys"], ["cfg", "device_uuid", "robot", "robot_profile"])
        self.assertEqual(profile["backend"]["device_uuid"], "GPU-test")

    def test_factory_accepts_empty_context(self) -> None:
        executor = make_curobo_v2_executor()
        self.assertIsInstance(executor, CuroboV2Executor)

    def test_plugin_version_string(self) -> None:
        self.assertIsInstance(PLUGIN_VERSION, str)
        self.assertTrue(PLUGIN_VERSION)

    def test_no_heavy_modules_imported(self) -> None:
        for name in FORBIDDEN_MODULES:
            self.assertNotIn(
                name,
                sys.modules,
                f"top-level import pulled heavy dependency: {name}",
            )


if __name__ == "__main__":
    unittest.main()
