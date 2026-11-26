"""CPU replica of the frozen adapter torso forward-kinematics chain.

Verbatim port of ``br_state45_adapter.py`` (sha `8f7bb7b9…30f78f`) helpers
``trunk_chain`` / ``trunk_fk`` / ``rot_axis`` / ``R_euler``. stdlib only:
``xml.etree.ElementTree`` replaces lxml for chain parsing, and all trig is
plain ``math`` so the contract layer has no numpy/torch install requirement.

The URDF path and its sha256 are private manifest material; this module never
hardcodes an asset path. The chain is loaded by the integration wrapper from
the manifest-injected URDF, then handed to :class:`TorsoFk`. If the chain or
URDF identity is not verified, the torso SE3 slots stay pending fail-closed
(see :mod:`state_mapping`); they are never zero-filled.
"""
import math
import xml.etree.ElementTree as ET
from typing import List, Sequence, Tuple

from .errors import ContractViolation

TRUNK_JOINTS = ("torso_joint1", "torso_joint2", "torso_joint3", "torso_joint4")

Mat4 = List[List[float]]
Chain = List[Tuple[str, List[float], List[float]]]


def _rot_axis(axis: Sequence[float], a: float) -> List[List[float]]:
    ax = [float(v) for v in axis]
    norm = math.sqrt(sum(v * v for v in ax))
    if norm <= 1e-12:
        raise ContractViolation("rotation axis is zero")
    ax = [v / norm for v in ax]
    x, y, z = ax
    c, s = math.cos(a), math.sin(a)
    C = 1.0 - c
    return [
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ]


def load_trunk_chain(urdf_path: str) -> Chain:
    """Parse the four torso joints from a URDF; order 1..4 by trailing digit."""
    try:
        tree = ET.parse(urdf_path)
    except (ET.ParseError, OSError) as exc:
        raise ContractViolation(f"cannot parse URDF: {exc}") from exc
    out: Chain = []
    for joint in tree.getroot().iter("joint"):
        name = joint.get("name")
        if name not in TRUNK_JOINTS:
            continue
        origin = joint.find("origin")
        xyz = [0.0, 0.0, 0.0]
        if origin is not None and origin.get("xyz"):
            xyz = [float(v) for v in origin.get("xyz").split()]
        axis = joint.find("axis")
        ax = [0.0, 0.0, 1.0]
        if axis is not None and axis.get("xyz"):
            ax = [float(v) for v in axis.get("xyz").split()]
        out.append((name, xyz, ax))
    out.sort(key=lambda item: int(item[0][-1]))
    if len(out) != len(TRUNK_JOINTS):
        raise ContractViolation(
            f"trunk chain: expected {len(TRUNK_JOINTS)} joints, found {len(out)}"
        )
    return out


def trunk_fk(chain: Chain, q: Sequence[float]) -> Mat4:
    """Forward kinematics for one trunk configuration."""
    if len(q) != len(chain):
        raise ContractViolation(f"trunk q length {len(q)} != chain {len(chain)}")
    T: Mat4 = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    for (_, xyz, axis), v in zip(chain, q):
        M: Mat4 = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
        R = _rot_axis(axis, float(v))
        for i in range(3):
            M[i][:3] = R[i]
        M[0][3], M[1][3], M[2][3] = xyz
        T = _mat_mul(T, M)
    return T


def _mat_mul(a: Mat4, b: Mat4) -> Mat4:
    out: Mat4 = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for k in range(4):
            if a[i][k] == 0.0:
                continue
            for j in range(4):
                out[i][j] += a[i][k] * b[k][j]
    return out


def _inv_se3(T: Mat4) -> Mat4:
    R = [row[:3] for row in T[:3]]
    p = [T[i][3] for i in range(3)]
    Rt = [[R[j][i] for j in range(3)] for i in range(3)]
    out: Mat4 = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    for i in range(3):
        for j in range(3):
            out[i][j] = Rt[i][j]
        out[i][3] = -sum(Rt[i][k] * p[k] for k in range(3))
    return out


def R_euler(R: Sequence[Sequence[float]]) -> List[float]:
    """Euler angles (adapter convention) from a 3x3 rotation matrix."""
    sy = math.sqrt(R[0][0] ** 2 + R[1][0] ** 2)
    if sy >= 1e-6:
        return [
            math.atan2(R[2][1], R[2][2]),
            math.atan2(-R[2][0], sy),
            math.atan2(R[1][0], R[0][0]),
        ]
    return [
        math.atan2(-R[1][2], R[1][1]),
        math.atan2(-R[2][0], sy),
        0.0,
    ]


class TorsoFk:
    """Frame-0-relative torso SE3 provider, pinned to a loaded chain.

    The first observed trunk configuration anchors frame 0, matching the
    adapter's ``base = trunk_fk(chain, q[0]); inv = inv(base); rel = inv @ T``.
    """

    def __init__(self, chain: Chain):
        if len(chain) != len(TRUNK_JOINTS):
            raise ContractViolation("torso chain incomplete")
        self._chain = chain
        self._base: Mat4 | None = None

    def reset(self) -> None:
        self._base = None

    def relative_se3(self, trunk_qpos: Sequence[float]) -> List[float]:
        T = trunk_fk(self._chain, trunk_qpos)
        if self._base is None:
            self._base = T
        rel = _mat_mul(_inv_se3(self._base), T)
        pos = [rel[i][3] for i in range(3)]
        euler = R_euler([row[:3] for row in rel[:3]])
        return pos + euler
