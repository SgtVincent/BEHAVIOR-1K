"""Fixed-anchor SE3 target reconstruction for the execution23 replay seam.

This module is the CPU side of the delta-TF bridge between pinned saved
execution23 rows and the native R1Pro controller / cuRobo v2 planner. The
semantics are ported interface-by-interface from the frozen S2 adapter and
converted-action producer; no step loop, simulator, IK solve, or model forward
is implemented here.

Saved-row contract (``brvla_saved_execution23_v1``):

- Columns: ``left_pose`` 0:6, ``right_pose`` 6:12, ``continuous_gripper``
  12:14, ``move`` 14:17, ``chest_pose`` 17:23.
- Representation ``pairwise_delta``: each limb row is a right-multiply SE3
  delta against the previous command target; the very first delta references
  the fixed run-local start anchor.
- Pose frame: physical base link; units metres / radians; Euler
  ``rx_ry_rz_matrix_Rz_Ry_Rx``.
- The anchor is rebuilt once at frame 0 and never refreshed from a live pose.
- Continuous gripper width is carried through without accumulation; the move
  channels are body-frame ``dx, dy, dyaw`` per step.

Stdlib only at import: no torch, numpy, omnigibson, isaac, or curobo. Inputs
are plain sequences; non-finite values, wrong widths, or unrecognized
representations fail closed.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import contracts
from ._arr import is_finite, to_plain
from .errors import ContractViolation, Port2026Error

LIMBS = ("left", "right", "chest")

POSE_SLICES: Dict[str, slice] = {
    "left": slice(0, 6),
    "right": slice(6, 12),
    "chest": slice(17, 23),
}
GRIPPER_SLICE = slice(12, 14)
MOVE_SLICE = slice(14, 17)

REPRESENTATION_PAIRWISE = "pairwise_delta"
REPRESENTATION_CUMULATIVE = "chunk_cumulative"
REPRESENTATIONS = (REPRESENTATION_PAIRWISE, REPRESENTATION_CUMULATIVE)

#: Pinned producer source hashes recorded by the converted manifest. These
#: are provenance fingerprints, not asset paths.
PRODUCER_SOURCE_SHA256: Dict[str, str] = {
    "br1_vla_transforms.py": "67aeebb3ce659d7326436353a162321e1fd85996a00d767375111a13259ab045",
    "rotation_util.py": "2f61fb23611d85d6a9164fd2aedcc887110f2d07aa997f714f22b266da743ec1",
}


class TargetExhausted(Port2026Error):
    """The saved command sequence is exhausted; no last-row repeat."""


class DeltaTFViolation(ContractViolation):
    """A reconstruction input violates the frozen anchor / SE3 contract."""


# ---------------------------------------------------------------------------
# SE3 geometry helpers (plain 4x4 nested lists; RzRyRx Euler, producer util)
# ---------------------------------------------------------------------------


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def euler_to_rot3(euler: Sequence[float]) -> List[List[float]]:
    """Rotation matrix for ``(rx, ry, rz)`` with ``R = Rz @ Ry @ Rx``."""
    if len(euler) != 3:
        raise DeltaTFViolation(f"euler width {len(euler)} != 3")
    rx, ry, rz = (float(v) for v in euler)
    if not all(math.isfinite(v) for v in (rx, ry, rz)):
        raise DeltaTFViolation("euler angles non-finite")
    ca, sa = math.cos(rx), math.sin(rx)
    cb, sb = math.cos(ry), math.sin(ry)
    cc, sc = math.cos(rz), math.sin(rz)
    return [
        [cb * cc, -sc * ca + cc * sb * sa, sc * sa + cc * sb * ca],
        [sc * cb, cc * ca + sc * sb * sa, -cc * sa + sc * sb * ca],
        [-sb, cb * sa, cb * ca],
    ]


def pose6_to_matrix(pose6: Sequence[float]) -> List[List[float]]:
    """Build a 4x4 homogeneous pose from ``(x, y, z, rx, ry, rz)``."""
    if len(pose6) != 6:
        raise DeltaTFViolation(f"pose width {len(pose6)} != 6")
    values = [float(v) for v in pose6]
    if not all(math.isfinite(v) for v in values):
        raise DeltaTFViolation("pose non-finite")
    rot = euler_to_rot3(values[3:6])
    mat: List[List[float]] = [row[:] + [0.0] for row in rot]
    mat.append([0.0, 0.0, 0.0, 1.0])
    for i in range(3):
        mat[i][3] = values[i]
    return mat


def rot3_to_euler(rot: Sequence[Sequence[float]]) -> List[float]:
    """Euler ``(rx, ry, rz)`` back from a 3x3 rotation (producer convention)."""
    sy = math.sqrt(rot[0][0] ** 2 + rot[1][0] ** 2)
    if sy >= 1e-6:
        rx = math.atan2(rot[2][1], rot[2][2])
        ry = math.atan2(-rot[2][0], sy)
        rz = math.atan2(rot[1][0], rot[0][0])
    else:
        rx = math.atan2(-rot[1][2], rot[1][1])
        ry = math.atan2(-rot[2][0], sy)
        rz = 0.0
    return [_wrap_angle(rx), _wrap_angle(ry), _wrap_angle(rz)]


def matrix_to_pose6(mat: Sequence[Sequence[float]]) -> List[float]:
    """Decompose a 4x4 homogeneous pose back to ``(pos3, euler3)``."""
    if len(mat) != 4 or any(len(row) != 4 for row in mat):
        raise DeltaTFViolation("matrix must be 4x4")
    rot = [list(row[:3]) for row in mat[:3]]
    pos = [mat[i][3] for i in range(3)]
    if not all(math.isfinite(v) for row in rot for v in row) or not all(math.isfinite(v) for v in pos):
        raise DeltaTFViolation("matrix non-finite")
    return pos + rot3_to_euler(rot)


def mat_mul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> List[List[float]]:
    out: List[List[float]] = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for k in range(4):
            if a[i][k] == 0.0:
                continue
            for j in range(4):
                out[i][j] += float(a[i][k]) * float(b[k][j])
    return out


def inv_se3(mat: Sequence[Sequence[float]]) -> List[List[float]]:
    rot = [list(row[:3]) for row in mat[:3]]
    pos = [float(mat[i][3]) for i in range(3)]
    rot_t = [[rot[j][i] for j in range(3)] for i in range(3)]
    out: List[List[float]] = [[0.0] * 4 for _ in range(4)]
    for i in range(3):
        for j in range(3):
            out[i][j] = rot_t[i][j]
        out[i][3] = -sum(rot_t[i][k] * pos[k] for k in range(3))
    out[3][3] = 1.0
    return out


# ---------------------------------------------------------------------------
# Pairwise-delta accumulation (pinned producer accumulator, numpy-free port)
# ---------------------------------------------------------------------------


def accumulate_pose_series(deltas: Sequence[Sequence[float]]) -> List[List[float]]:
    """Right-multiply SE3 accumulation over one limb's pairwise deltas.

    ``cum[i] = cum[i-1] @ delta[i]``; the first cumulative pose is the first
    delta composed with identity. Returns cumulative pose6 rows in physical
    base frame.
    """
    if len(deltas) == 0:
        raise DeltaTFViolation("empty pose delta series")
    cumulative: List[List[List[float]]] = []
    for i, delta in enumerate(deltas):
        d_mat = pose6_to_matrix(delta)
        if i == 0:
            cumulative.append(d_mat)
        else:
            cumulative.append(mat_mul(cumulative[-1], d_mat))
    return [matrix_to_pose6(m) for m in cumulative]


def accumulate_move_series(deltas: Sequence[Sequence[float]]) -> List[List[float]]:
    """Body-frame per-step ``dx, dy, dyaw`` -> cumulative global ``x, y, yaw``.

    The lateral displacement is rotated into the world frame using the previous
    accumulated yaw (pinned producer ``accumulate_move_delta_pose``).
    """
    if len(deltas) == 0:
        raise DeltaTFViolation("empty move delta series")
    x = y = 0.0
    prev_yaw = 0.0
    out: List[List[float]] = []
    for step in deltas:
        if len(step) != 3:
            raise DeltaTFViolation(f"move width {len(step)} != 3")
        dx, dy, dyaw = (float(v) for v in step)
        if not all(math.isfinite(v) for v in (dx, dy, dyaw)):
            raise DeltaTFViolation("move delta non-finite")
        c, s = math.cos(prev_yaw), math.sin(prev_yaw)
        x += dx * c - dy * s
        y += dx * s + dy * c
        yaw = prev_yaw + dyaw
        out.append([x, y, yaw])
        prev_yaw = yaw
    return out


# ---------------------------------------------------------------------------
# Fixed-anchor target set
# ---------------------------------------------------------------------------


class FixedAnchorTargets:
    """Rebuild absolute command targets from pinned execution23 rows.

    Args:
        rows: saved execution23 matrix (``N x 23``), plain nested lists.
        anchors: ``left`` / ``right`` / ``chest`` fixed run-local start pose6.
        dt_seconds: saved control step dt (1/30 for the sealed replay episode).
        representation: ``pairwise_delta`` (accumulate against the anchor) or
            ``chunk_cumulative`` (rows already cumulative; accumulator skipped).
    """

    def __init__(
        self,
        rows: Sequence[Sequence[float]],
        anchors: Dict[str, Sequence[float]],
        dt_seconds: float,
        representation: str = REPRESENTATION_PAIRWISE,
    ) -> None:
        plain_rows = [list(to_plain(row)) for row in rows]
        if not plain_rows:
            raise DeltaTFViolation("saved rows empty")
        for row in plain_rows:
            if len(row) != contracts.ACTION23:
                raise DeltaTFViolation(f"row width {len(row)} != {contracts.ACTION23}")
            if not is_finite(row):
                raise DeltaTFViolation("saved row non-finite")
        self._rows: List[List[float]] = [[float(v) for v in row] for row in plain_rows]

        if representation not in REPRESENTATIONS:
            raise DeltaTFViolation(f"unknown representation: {representation!r}")
        self._representation = representation

        dt = float(dt_seconds)
        if not math.isfinite(dt) or dt <= 0.0:
            raise DeltaTFViolation("dt must be finite and positive")
        self._dt = dt

        self._anchors: Dict[str, List[float]] = {}
        for limb in LIMBS:
            if limb not in anchors:
                raise DeltaTFViolation(f"anchor missing limb: {limb}")
            anchor = list(to_plain(anchors[limb]))
            if len(anchor) != 6 or not is_finite(anchor):
                raise DeltaTFViolation(f"anchor {limb} must be finite pose6")
            self._anchors[limb] = [float(v) for v in anchor]

        self._target_matrices: Dict[str, List[List[List[float]]]] = {}
        self._body_velocity: List[List[float]] = []
        self._rebuild()

    # -- reconstruction -----------------------------------------------------

    def _rebuild(self) -> None:
        if self._representation == REPRESENTATION_PAIRWISE:
            for limb in LIMBS:
                sl = POSE_SLICES[limb]
                cumulative6 = accumulate_pose_series(
                    [row[sl] for row in self._rows]
                )
                anchor_mat = pose6_to_matrix(self._anchors[limb])
                self._target_matrices[limb] = [
                    mat_mul(anchor_mat, pose6_to_matrix(cum)) for cum in cumulative6
                ]
            cumulative_move = accumulate_move_series(
                [row[MOVE_SLICE] for row in self._rows]
            )
        else:
            for limb in LIMBS:
                sl = POSE_SLICES[limb]
                anchor_mat = pose6_to_matrix(self._anchors[limb])
                self._target_matrices[limb] = [
                    mat_mul(anchor_mat, pose6_to_matrix(row[sl]))
                    for row in self._rows
                ]
            cumulative_move = [list(row[MOVE_SLICE]) for row in self._rows]

        # Recover body-frame per-step displacement from the cumulative SE(2)
        # track; divide by dt to obtain body velocity (pinned producer).
        previous: List[List[float]] = [[0.0, 0.0, 0.0]]
        previous += cumulative_move[:-1]
        for i, current in enumerate(cumulative_move):
            px, py, pyaw = previous[i]
            diff = [current[0] - px, current[1] - py, current[2] - pyaw]
            c, s = math.cos(pyaw), math.sin(pyaw)
            body = [
                c * diff[0] + s * diff[1],
                -s * diff[0] + c * diff[1],
                diff[2],
            ]
            self._body_velocity.append([v / self._dt for v in body])

    # -- public seam --------------------------------------------------------

    @property
    def length(self) -> int:
        return len(self._rows)

    @property
    def dt_seconds(self) -> float:
        return self._dt

    @property
    def representation(self) -> str:
        return self._representation

    def anchors(self) -> Dict[str, List[float]]:
        return {limb: list(values) for limb, values in self._anchors.items()}

    def target(self, index: int) -> Dict[str, Any]:
        """Return the decoded command target for one saved row.

        ``pose_matrices`` holds the fixed-anchor absolute 4x4 SE3 target for
        each limb; ``base_velocity_body`` is the body-frame velocity; gripper
        widths pass through without accumulation. Missing or out-of-range
        indices raise :class:`TargetExhausted`; no row is repeated on exhaustion.
        """
        if type(index) is not int or index < 0 or index >= len(self._rows):
            raise TargetExhausted(
                f"converted target exhausted at index {index}; no last-row repeat"
            )
        row = self._rows[index]
        return {
            "source_row": index,
            "pose_matrices": {
                limb: [list(r) for r in self._target_matrices[limb][index]]
                for limb in LIMBS
            },
            "continuous_gripper": [float(v) for v in row[GRIPPER_SLICE]],
            "base_velocity_body": list(self._body_velocity[index]),
            "dt_seconds": self._dt,
            "representation": self._representation,
        }

    def receipt(self) -> Dict[str, Any]:
        return {
            "rows": len(self._rows),
            "dt_seconds": self._dt,
            "representation": self._representation,
            "anchors": self.anchors(),
            "producer_source_sha256": dict(PRODUCER_SOURCE_SHA256),
        }
