"""native23 action assembly with explicit per-group source bindings.

The verified R1Pro stock controller order is
``base3 | trunk4 | arm_left7 | gripper_left1 | arm_right7 | gripper_right1``
(see :data:`brvla_bridge.contracts.NATIVE23_GROUPS`). Each group records where
its values came from: a planned group, a replay row, or an unproven seam. An
unproven group fails closed; planner zero values are never substituted for a
gripper, base, or arm joint that has no verified source.

Stdlib-only at import.
"""
import math
from typing import Any, Dict, List, Optional, Sequence

from . import contracts
from ._arr import is_finite, to_plain
from .errors import ContractViolation

GROUP_WIDTHS: Dict[str, int] = {
    "base": 3,
    "trunk": 4,
    "arm_left": 7,
    "gripper_left": 1,
    "arm_right": 7,
    "gripper_right": 1,
}

SOURCE_PLANNER = "planner"
SOURCE_REPLAY = "replay"
SOURCE_CURRENT_Q = "current_q"
SOURCE_MISSING = "missing"

# Sources a native23 group may carry after fail-closed validation.
KNOWN_SOURCES = (SOURCE_PLANNER, SOURCE_REPLAY, SOURCE_CURRENT_Q, SOURCE_MISSING)


def group_slices() -> Dict[str, slice]:
    """Offset slices for the native23 controller order."""
    slices: Dict[str, slice] = {}
    offset = 0
    for group in contracts.NATIVE23_GROUPS:
        width = GROUP_WIDTHS[group]
        slices[group] = slice(offset, offset + width)
        offset += width
    if offset != contracts.ACTION23:
        raise ContractViolation(f"native23 group widths sum {offset} != {contracts.ACTION23}")
    return slices


def _check_group(group: str, values: Sequence[Any], source: str) -> List[float]:
    width = GROUP_WIDTHS[group]
    plain = to_plain(values)
    if source not in KNOWN_SOURCES:
        raise ContractViolation(
            f"native23 group '{group}' has unknown source {source!r}")
    if source == SOURCE_MISSING:
        raise ContractViolation(
            f"native23 group '{group}' has no verified source; do not zero-fill "
            "from a planner default. Resolve its seam before replay.")
    if len(plain) != width:
        raise ContractViolation(
            f"native23 group '{group}' width {len(plain)} != {width}")
    if not is_finite(plain):
        raise ContractViolation(f"native23 group '{group}' contains non-finite values")
    return [float(v) for v in plain]


def bind_current_q(
    arm: str,
    qpos: Sequence[Any],
    *,
    source_sha256: str,
    source_kind: str = "live_sdk_qpos",
) -> Dict[str, Any]:
    """Bind an inactive arm's current joint q from the live SDK state.

    The held arm keeps its current configuration under collision authority;
    its group is never defaulted to zero. The live qpos vector must carry a
    source sha256 so the binding is auditable. A missing or non-finite qpos
    fails closed instead of substituting a planner default.
    """
    if arm not in ("left", "right"):
        raise ContractViolation(f"unknown arm {arm!r}")
    group = f"arm_{arm}"
    plain = to_plain(qpos)
    width = GROUP_WIDTHS[group]
    if len(plain) != width or not is_finite(plain):
        raise ContractViolation(
            f"current-q binding for {group} width/finite invalid")
    if not source_sha256 or not isinstance(source_sha256, str):
        raise ContractViolation(
            f"current-q binding for {group} requires a source sha256")
    return {
        "group": group,
        "values": [float(v) for v in plain],
        "source": SOURCE_CURRENT_Q,
        "source_kind": source_kind,
        "source_sha256": source_sha256,
    }


def assemble_native23(
    *,
    base_vel: Optional[Sequence[Any]] = None,
    trunk: Optional[Sequence[Any]] = None,
    arm_left: Optional[Sequence[Any]] = None,
    gripper_left: Optional[Sequence[Any]] = None,
    arm_right: Optional[Sequence[Any]] = None,
    gripper_right: Optional[Sequence[Any]] = None,
    sources: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Assemble the 23-d controller vector with recorded group provenance.

    ``sources`` maps each group to ``planner`` or ``replay``. A group whose
    source is missing/None raises :class:`ContractViolation`; no placeholder,
    zero, or planner-default fallback is applied.
    """
    sources = dict(sources or {})
    groups = {
        "base": base_vel,
        "trunk": trunk,
        "arm_left": arm_left,
        "gripper_left": gripper_left,
        "arm_right": arm_right,
        "gripper_right": gripper_right,
    }
    checked: Dict[str, List[float]] = {}
    for group in contracts.NATIVE23_GROUPS:
        source = sources.get(group)
        if source is None:
            source = SOURCE_PLANNER if groups[group] is not None else SOURCE_MISSING
        checked[group] = _check_group(group, groups[group], source)
        sources[group] = source

    vector: List[float] = []
    for group in contracts.NATIVE23_GROUPS:
        vector.extend(checked[group])
    if len(vector) != contracts.ACTION23 or not is_finite(vector):
        raise ContractViolation("assembled native23 vector is invalid")
    return {
        "vector": vector,
        "sources": dict(sources),
        "group_widths": dict(GROUP_WIDTHS),
    }


def extract_group(vector: Sequence[Any], group: str) -> List[float]:
    """Pull one group out of an assembled native23 vector."""
    plain = to_plain(vector)
    if len(plain) != contracts.ACTION23 or not is_finite(plain):
        raise ContractViolation("finite 23-d native23 vector required")
    sl = group_slices()[group]
    return [float(v) for v in plain[sl]]


def gripper_width_to_finger(value: Any) -> float:
    """Normalize a replay continuous gripper width into the stock range.

    The replay row carries a width in meters over the frozen 0.05 m range; the
    stock single-finger controller expects the same normalized command that the
    verified adapter packs. Range clipping is a physical limit, not a fallback.
    """
    v = float(value)
    if not math.isfinite(v):
        raise ContractViolation("gripper value non-finite")
    return min(1.0, max(0.0, v / contracts.GRIPPER_RANGE_M))
