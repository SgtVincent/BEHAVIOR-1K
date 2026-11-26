"""Action schema labels and tagged-action validation for the MoMa planning line.

Contract reference: ``contracts/INTERFACE_SPEC.md`` §4. Three 23-dimensional
vectors share a width but **not** a schema; transactions, manifests and logs
must carry the schema label. Padding one schema onto another is forbidden.

Canonical labels (review-aligned):

- ``native23_joint`` — stock controller command (MoMa line).
- ``bridge_decoded23_ik_pose`` — IK-bridge decoded absolute SE3 targets.
- ``execution23_se3_delta`` — execution-row physical-base SE3 delta command.

Legacy spellings recorded from the 2025 review snapshot resolve through
``LEGACY_LABEL_ALIASES`` to the canonical label; they are **not** a fourth
schema. Unknown labels always reject.

NOTE (gap G2): the baseline DOF order below is recorded from the a28 frozen
checkpoint. The 2026 R1Pro controller definition must be re-verified before
any slice is actually consumed by an executor; do not treat the listed order
as verified against post1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

NATIVE23_JOINT = "native23_joint"
BRIDGE_DECODED23_IK_POSE = "bridge_decoded23_ik_pose"
EXECUTION23_SE3_DELTA = "execution23_se3_delta"

ALLOWED_SCHEMAS = frozenset({NATIVE23_JOINT, BRIDGE_DECODED23_IK_POSE, EXECUTION23_SE3_DELTA})

# Frozen legacy aliases: 2025 review snapshot spellings. They normalize to the
# canonical label on construction; they never open a cross-schema bridge.
LEGACY_LABEL_ALIASES: Dict[str, str] = {
    "joint-native23": NATIVE23_JOINT,
    "IK-target23": BRIDGE_DECODED23_IK_POSE,
    "execution23_se3": EXECUTION23_SE3_DELTA,
}

# Deprecated exported names, kept only for source compatibility. New code
# must use the canonical constants above.
JOINT_NATIVE23 = NATIVE23_JOINT  # deprecated alias
IK_TARGET23 = BRIDGE_DECODED23_IK_POSE  # deprecated alias
EXECUTION23_SE3 = EXECUTION23_SE3_DELTA  # deprecated alias

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


def normalize_label(label: str) -> str:
    """Resolve a canonical or legacy-alias schema label; reject anything else."""
    if label in ALLOWED_SCHEMAS:
        return label
    if label in LEGACY_LABEL_ALIASES:
        return LEGACY_LABEL_ALIASES[label]
    raise ValueError(f"unknown_action_schema: {label!r}")


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
    the tagged action explicitly. The stored label is always canonical.
    """

    label: str
    vector: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", normalize_label(self.label))
        object.__setattr__(self, "vector", _as_finite_vector(self.vector))

    def assert_compatible(self, other_label: str) -> None:
        """Reject pooling/padding between different schema types."""
        other = normalize_label(other_label)
        if other != self.label:
            raise ValueError(
                f"schema_mismatch: {self.label} cannot be bridged to {other}; "
                "same width does not imply same schema"
            )

    def to_list(self) -> list[float]:
        return [float(x) for x in self.vector]


def native23_action(vector: Any, meta: Optional[Dict[str, Any]] = None) -> TaggedAction:
    """Construct a MoMa ``native23_joint`` tagged action."""
    return TaggedAction(NATIVE23_JOINT, vector, meta or {})
