"""Action schema labels and tagged-action validation for the MoMa planning line.

Contract reference: ``contracts/INTERFACE_SPEC.md`` §4. Three 23-dimensional
vectors share a width but **not** a schema; transactions, manifests and logs
must carry the schema label. Padding one schema onto another is forbidden.

Baseline MoMa actions use ``joint-native23`` (stock controller command). The
BRVLA line uses ``IK-target23``; they must not be pooled or bridged by padding.

NOTE (gap G2): the baseline DOF order below is recorded from the a28 frozen
checkpoint. The 2026 R1Pro controller definition must be re-verified before
any slice is actually consumed by an executor; do not treat the listed order
as verified against post1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

JOINT_NATIVE23 = "joint-native23"
IK_TARGET23 = "IK-target23"
EXECUTION23_SE3 = "execution23_se3"

ALLOWED_SCHEMAS = frozenset({JOINT_NATIVE23, IK_TARGET23, EXECUTION23_SE3})

ACTION_DIM = 23

# Baseline a28 native23 composition (pending G2 recheck against 2026 R1Pro):
#   base 3 velocity; trunk 4 position; armL 7 position; gripL 1 continuous;
#   armR 7 position; gripR 1 continuous.
NATIVE23_GROUPS = (
    ("base", 3),
    ("trunk", 4),
    ("arm_left", 7),
    ("gripper_left", 1),
    ("arm_right", 7),
    ("gripper_right", 1),
)
NATIVE23_PENDING_G2_RECHECK = True


def _as_finite_vector(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] != ACTION_DIM:
        raise ValueError(f"action23_invalid_shape: expected ({ACTION_DIM},), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("action23_nonfinite")
    return arr


@dataclass(frozen=True)
class TaggedAction:
    """A 23-d command bound to its schema label.

    ``meta`` is free-form accounting (step, primitive, provider digest, ...).
    No schema label is attached by implicit conversion; callers must construct
    the tagged action explicitly.
    """

    label: str
    vector: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.label not in ALLOWED_SCHEMAS:
            raise ValueError(f"unknown_action_schema: {self.label!r}")
        object.__setattr__(self, "vector", _as_finite_vector(self.vector))

    def assert_compatible(self, other_label: str) -> None:
        """Reject pooling/padding between different schema types."""
        if other_label not in ALLOWED_SCHEMAS:
            raise ValueError(f"unknown_action_schema: {other_label!r}")
        if other_label != self.label:
            raise ValueError(
                f"schema_mismatch: {self.label} cannot be bridged to {other_label}; "
                "same width does not imply same schema"
            )

    def to_list(self) -> list[float]:
        return [float(x) for x in self.vector]


def native23_action(vector: Any, meta: Optional[Dict[str, Any]] = None) -> TaggedAction:
    """Construct a MoMa ``joint-native23`` tagged action."""
    return TaggedAction(JOINT_NATIVE23, vector, meta or {})
