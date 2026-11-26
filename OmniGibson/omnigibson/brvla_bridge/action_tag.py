"""Tagged-action boundary between model output and the native controller.

The frozen server emits a 23-d row carrying the physical layout
``physical23_same_anchor_cumulative``. That row is an IK-target plan
(:data:`contracts.IK_TARGET23`), **not** a joint command for the stock R1Pro
controller. The two share the 23 width but not the semantics; handing the
physical row to the native joint controller is a wrong-semantics step and is
refused here.

Boundary contract:

* :func:`tag_model_output` wraps a raw server row as an :class:`TaggedAction`
  labeled :data:`contracts.IK_TARGET23`, recording the physical layout.
* A decoder seam converts that tagged target into a verified
  :data:`contracts.JOINT_NATIVE23` vector through the fixed-anchor delta-TF,
  the certified planner, and the time grid. Until such a decoder is bound,
  the consumer guard holds the action and takes zero simulator steps.
* :func:`assert_native23_joint` is the last check before the official
  ``env.step``: anything that is not a verified joint-native23 tagged action
  raises :class:`~brvla_bridge.errors.ControlledActionBlocked`.

Stdlib-only at import. No torch, numpy, omnigibson, or simulator import.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from . import contracts
from ._arr import is_finite, to_plain
from .errors import ControlledActionBlocked


@dataclass(frozen=True)
class TaggedAction:
    """A 23-d command together with its schema label and provenance.

    Attributes:
        label: one of :data:`contracts.SCHEMA_LABELS`.
        vector: the 23-d plain float command.
        layout: the pinned physical layout recorded on the server row.
        provenance: decoder / planner / current-state binding evidence
            (group sources, source sha256, planner segment, target row).
    """

    label: str
    vector: List[float]
    layout: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)

    def is_native_joint(self) -> bool:
        return self.label == contracts.JOINT_NATIVE23


def _plain_row(row: Sequence[Any]) -> List[float]:
    plain = to_plain(row)
    if isinstance(plain, list) and len(plain) == 1 and isinstance(plain[0], list):
        plain = plain[0]
    if not isinstance(plain, list):
        raise ControlledActionBlocked("action row is not a flat 23-d sequence")
    if len(plain) != contracts.ACTION23:
        raise ControlledActionBlocked(
            f"action width {len(plain)} != {contracts.ACTION23}; refuse to reshape")
    if not is_finite(plain):
        raise ControlledActionBlocked("action row contains non-finite values")
    return [float(v) for v in plain]


def tag_model_output(
    row: Sequence[Any],
    *,
    layout: str = contracts.SERVER_METADATA_LAYOUT,
    provenance: Optional[Dict[str, Any]] = None,
) -> TaggedAction:
    """Wrap a frozen-server response row as an IK-target23 tagged action.

    The row is validated for width and finiteness only; it is never relabeled
    as a joint-native23 command here.
    """
    vector = _plain_row(row)
    if layout != contracts.SERVER_METADATA_LAYOUT:
        raise ControlledActionBlocked(
            f"unexpected physical layout {layout!r}; pinned "
            f"{contracts.SERVER_METADATA_LAYOUT!r}")
    return TaggedAction(
        label=contracts.IK_TARGET23,
        vector=vector,
        layout=layout,
        provenance=dict(provenance or {}),
    )


def tag_native23(
    vector: Sequence[Any],
    *,
    provenance: Optional[Dict[str, Any]] = None,
    layout: Optional[str] = None,
) -> TaggedAction:
    """Wrap a verified planner/grid output as a joint-native23 tagged action."""
    row = _plain_row(vector)
    return TaggedAction(
        label=contracts.JOINT_NATIVE23,
        vector=row,
        layout=layout,
        provenance=dict(provenance or {}),
    )


def assert_native23_joint(tagged: Any) -> List[float]:
    """Consumer guard: refuse to step anything that is not joint-native23.

    Runs immediately before the official ``env.step``. An IK-target23 or
    physical23 row of the same width raises
    :class:`~brvla_bridge.errors.ControlledActionBlocked`; the caller must
    hold the action and take zero simulator steps.
    """
    if not isinstance(tagged, TaggedAction):
        raise ControlledActionBlocked(
            "consumer guard received an untagged action; no decoder provenance")
    if tagged.label != contracts.JOINT_NATIVE23:
        raise ControlledActionBlocked(
            f"refuse to step {tagged.label!r} as joint-native23; "
            "decode through the planner seam before env.step")
    row = _plain_row(tagged.vector)
    return row
