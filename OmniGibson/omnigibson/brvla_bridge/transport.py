"""Stdlib port of the frozen BR action transport (execution23 -> fixed-anchor goals).

Faithful to ``br_action_transport_v1.py``: cumulative SE(3) rows are converted to
absolute pose goals against previous commanded anchors. Fresh first plan uses
current physical-base poses; later plans reuse the last executed command goals.
No IK, simulator, learned policy, or clipping lives here.
"""
import math
from collections import deque

from . import contracts
from ._arr import is_finite, shape_of, to_plain
from .errors import ContractViolation

LAYOUT = contracts.LAYOUT_EXECUTION23


def _mat_mul(a, b):
    out = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for k in range(4):
            if a[i][k] == 0.0:
                continue
            for j in range(4):
                out[i][j] += a[i][k] * b[k][j]
    return out


def pose_matrix(p):
    p = to_plain(p)
    if len(p) != 6 or not all(math.isfinite(float(x)) for x in p):
        raise ContractViolation("finite pose6 required")
    x, y, z, rx, ry, rz = (float(v) for v in p)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    R = [
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ]
    T = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    T[0][:3], T[1][:3], T[2][:3] = R[0], R[1], R[2]
    T[0][3], T[1][3], T[2][3] = x, y, z
    return T


def check_transform(T):
    T = to_plain(T)
    if shape_of(T) != (4, 4) or not is_finite(T):
        raise ContractViolation("4x4 finite SE3 required")
    if list(T[3]) != [0.0, 0.0, 0.0, 1.0]:
        raise ContractViolation("invalid SE3 last row")
    R = [row[:3] for row in T[:3]]
    tol = 2e-5
    for i in range(3):
        for j in range(3):
            dot = sum(R[i][k] * R[j][k] for k in range(3))
            if not math.isclose(dot, 1.0 if i == j else 0.0, abs_tol=tol):
                raise ContractViolation("rotation not orthonormal")
    det = (R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1])
           - R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0])
           + R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]))
    if not math.isclose(det, 1.0, abs_tol=tol):
        raise ContractViolation("rotation determinant != 1")
    return [row[:] for row in T]


def inverse_cumulative_move(move, dt):
    move = to_plain(move)
    if shape_of(move) != (len(move), 3) or not is_finite(move):
        raise ContractViolation("finite [N,3] base move required")
    if not math.isfinite(float(dt)) or float(dt) <= 0:
        raise ContractViolation("invalid move/dt")
    prev = [[0.0, 0.0, 0.0]] + [list(r) for r in move[:-1]]
    vel = []
    for cur, pr in zip(move, prev):
        d = [cur[0] - pr[0], cur[1] - pr[1], cur[2] - pr[2]]
        d[2] = math.atan2(math.sin(d[2]), math.cos(d[2]))
        c, s = math.cos(pr[2]), math.sin(pr[2])
        vel.append([(c * d[0] + s * d[1]) / dt, (-s * d[0] + c * d[1]) / dt, d[2] / dt])
    return vel


def decompose_execution23(row):
    plain = to_plain(row)
    if len(plain) != contracts.ACTION23 or not is_finite(plain):
        raise ContractViolation("finite 23-d execution23 row required")
    return {name: list(plain[sl]) for name, sl in contracts.EXECUTION23_SLICES.items()}


def step_base_velocity(move, prev, dt=contracts.CONTROL_DT):
    move, prev = to_plain(move), to_plain(prev)
    if shape_of(move) != (3,) or shape_of(prev) != (3,):
        raise ContractViolation("base move/prev must be 3-vectors")
    if not is_finite(move) or not is_finite(prev):
        raise ContractViolation("non-finite base move")
    if not math.isfinite(float(dt)) or float(dt) <= 0:
        raise ContractViolation("invalid dt")
    dx = move[0] - prev[0]
    dy = move[1] - prev[1]
    dth = math.atan2(math.sin(move[2] - prev[2]), math.cos(move[2] - prev[2]))
    yaw = prev[2]
    c, s = math.cos(yaw), math.sin(yaw)
    return [(c * dx + s * dy) / dt, (-s * dx + c * dy) / dt, dth / dt]


class ChunkGoals:
    """Anchor state for a receding-horizon plan; anchors are commanded goals."""

    def __init__(self, anchors, horizon=contracts.HORIZON, dt=contracts.CONTROL_DT):
        if not isinstance(horizon, int) or not 1 <= horizon <= contracts.CHUNK_SIZE:
            raise ContractViolation("invalid horizon")
        if set(anchors) != {"left", "right", "chest"}:
            raise ContractViolation("three limb anchors required")
        self.anchors = {k: check_transform(v) for k, v in anchors.items()}
        self.horizon = horizon
        self.dt = dt
        self.queue = deque()
        self.step_index = 0
        self.plan_steps = []
        self.last_goal = None

    def needs_plan(self):
        return not self.queue

    def accept(self, chunk, layout):
        if layout != LAYOUT:
            raise ContractViolation("action layout/representation unverified")
        if self.queue:
            raise ContractViolation("refusing to overwrite unexecuted actions")
        a = to_plain(chunk)
        if shape_of(a) != (contracts.CHUNK_SIZE, contracts.ACTION23) or not is_finite(a):
            raise ContractViolation("expected finite 50x23 chunk")
        base_vel = inverse_cumulative_move([r[14:17] for r in a], self.dt)
        snapshot = {k: [row[:] for row in v] for k, v in self.anchors.items()}
        for i in range(self.horizon):
            poses = {
                key: _mat_mul(snapshot[key], pose_matrix(a[i][start:start + 6]))
                for key, start in [("left", 0), ("right", 6), ("chest", 17)]
            }
            self.queue.append({
                "pose_matrices": poses,
                "continuous_gripper": list(a[i][12:14]),
                "base_velocity_body": base_vel[i],
                "plan_step": self.step_index,
                "chunk_row": i,
                "source_row": self.step_index + i,
                "dt_seconds": self.dt,
            })
        self.plan_steps.append(self.step_index)

    def peek(self):
        if not self.queue:
            raise ContractViolation("new plan required")
        return self.queue[0]

    def commit_step(self):
        """Commit only after exactly one successful env.step, not after decode."""
        if not self.queue:
            raise ContractViolation("cannot commit from empty queue")
        goal = self.queue.popleft()
        self.anchors = {k: [row[:] for row in v] for k, v in goal["pose_matrices"].items()}
        self.last_goal = goal
        self.step_index += 1
        return goal


class ProprioHistory:
    """Last-4 contiguous proprio rows; early window pads by repeating row 0."""

    def __init__(self, width=contracts.PROPRIO256):
        self.rows = deque(maxlen=contracts.HISTORY_LEN)
        self.width = width
        self.last_step = None

    def append(self, step, row):
        if not isinstance(step, int) or (self.last_step is not None and step != self.last_step + 1):
            raise ContractViolation("non-contiguous proprio history")
        plain = to_plain(row)
        if len(plain) != self.width or not is_finite(plain):
            raise ContractViolation("invalid proprio row")
        self.rows.append(plain)
        self.last_step = step

    def window(self):
        if not self.rows:
            raise ContractViolation("empty history")
        pad = [list(self.rows[0])] * (contracts.HISTORY_LEN - len(self.rows))
        return pad + [list(r) for r in self.rows]
