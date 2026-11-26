# Copyright (c) 2026 B1K2026 integration line
# SPDX-License-Identifier: MIT
"""Frozen 23-d action schema contracts and 2026 state slice offsets.

Three independent 23-d action schemas are preserved as distinct enum tags.
A bare 23-vector without its schema tag is rejected at the contract boundary;
the schemas must not be collapsed into a generic "IK target / joint" pair.

Widths other than 23 (proprio256, joint22, legacy 35, state45) are state or
legacy-side only and must not be substituted for an action schema here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from .errors import SchemaMismatch

__all__ = ["ACTION_DIM", "ActionSchema", "ActionCommand", "StateSlices"]

ACTION_DIM = 23


class ActionSchema(Enum):
    """Independent 23-d action representations (no aliasing between them)."""

    #: L pose6 [0:6] / R pose6 [6:12] / gripper2 [12:14] / move3 [14:17] /
    #: chest pose6 [17:23]. Physical base link SE3 delta (pos3 + euler3) plus
    #: body-frame dx/dy/dyaw; pairwise deltas, fixed anchors; no joint targets
    #: or progress terms.
    EXECUTION23_SE3_DELTA = "execution23_se3_delta"

    #: base3 / trunk4 / armL7 / gripL1 / armR7 / gripR1. Base in body-frame
    #: velocity; trunk/arm absolute joint qpos; gripper continuous position.
    NATIVE23_JOINT = "native23_joint"

    #: base3 / trunk6 pose / armL6 pose / gripL1 / armR6 pose / gripR1.
    #: IK absolute_pose goals in the physical base frame; slices generated
    #: dynamically from the post-installation command_dim.
    BRIDGE_DECODED23_IK_POSE = "bridge_decoded23_ik_pose"


def _coerce_finite(values: Sequence[float]) -> tuple[float, ...]:
    try:
        coerced = tuple(float(v) for v in values)
    except (TypeError, ValueError) as exc:
        raise SchemaMismatch("action values must be numeric") from exc
    if any(not math.isfinite(v) for v in coerced):
        raise SchemaMismatch("action values must be finite (no NaN/inf)")
    return coerced


@dataclass(frozen=True)
class ActionCommand:
    """A 23-d action tagged with its owning schema.

    Attributes:
        schema: The independent schema tag. Required; default ``None`` raises.
        values: Exactly 23 finite floats.
        request_id: Correlation token echoed from the websocket request.
    """

    schema: ActionSchema
    values: tuple[float, ...]
    request_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.schema, ActionSchema):
            raise SchemaMismatch("schema must be an ActionSchema tag")
        if len(self.values) != ACTION_DIM:
            raise SchemaMismatch(
                f"{self.schema.value}: expected {ACTION_DIM} dims, "
                f"got {len(self.values)}"
            )
        object.__setattr__(self, "values", _coerce_finite(self.values))


class StateSlices:
    """2026 observation velocity slice offsets (v3.9.2 onward).

    These offsets are carried for the adapter that consumes state; they are
    pending re-verification against a running environment before any state-side
    consumer relies on them.
    """

    BASE_VEL = slice(0, 3)
    ARM_LEFT_VEL = slice(10, 17)
    GRIPPER_LEFT_VEL = slice(26, 28)
    ARM_RIGHT_VEL = slice(35, 42)
    GRIPPER_RIGHT_VEL = slice(51, 53)
    TRUNK_VEL = slice(57, 61)
