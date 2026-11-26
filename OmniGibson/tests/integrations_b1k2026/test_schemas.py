# Copyright (c) 2026 B1K2026 integration line
# SPDX-License-Identifier: MIT
"""CPU contract tests for action schema tags and width validation."""

import unittest

from integrations.b1k2026.errors import SchemaMismatch
from integrations.b1k2026.schemas import (
    ACTION_DIM,
    ActionCommand,
    ActionSchema,
)


def _vec(value: float = 0.0) -> list[float]:
    return [value] * ACTION_DIM


class TestActionSchema(unittest.TestCase):
    def test_three_schemas_are_distinct(self) -> None:
        tags = {s.value for s in ActionSchema}
        self.assertEqual(
            tags,
            {
                "execution23_se3_delta",
                "native23_joint",
                "bridge_decoded23_ik_pose",
            },
        )
        self.assertEqual(len(set(ActionSchema)), 3)

    def test_valid_command_constructs(self) -> None:
        for schema in ActionSchema:
            cmd = ActionCommand(schema=schema, values=_vec(0.1), request_id="r1")
            self.assertEqual(len(cmd.values), ACTION_DIM)
            self.assertEqual(cmd.schema, schema)

    def test_wrong_width_rejected(self) -> None:
        with self.assertRaises(SchemaMismatch):
            ActionCommand(
                schema=ActionSchema.NATIVE23_JOINT, values=[0.0] * 22
            )

    def test_non_finite_rejected(self) -> None:
        with self.assertRaises(SchemaMismatch):
            ActionCommand(
                schema=ActionSchema.EXECUTION23_SE3_DELTA,
                values=[float("nan")] * ACTION_DIM,
            )

    def test_missing_schema_tag_rejected(self) -> None:
        with self.assertRaises(SchemaMismatch):
            ActionCommand(schema=None, values=_vec())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
