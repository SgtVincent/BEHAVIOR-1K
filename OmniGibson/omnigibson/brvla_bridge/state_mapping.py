"""State mapping provider interface (2026 state -> legacy state45).

The frozen pipeline consumes a 45-d state history window. 2026 data ships two
different source widths:

    demos (LeRobot v3 parquet): observation.state width 61
    rawdata (HDF5):            state width 623

Verified slices come from post1 ``eval/utils/eval_utils.py``
``PROPRIOCEPTION_INDICES["R1Pro"]``; the 45-slot assembly mirrors the frozen
``br_state45_adapter.py`` (sha `8f7bb7b9…30f78f`). The base-move frame was
confirmed by the GPU1 live proprio sample (v3 receipt): ``base_qvel`` 0:3 is
already robot-local ``[vx, vy, wz]``. The torso SE3 provider binds at the CPU
FK layer once the manifest URDF identity and frozen-adapter chain cross-check
pass; a live CPU-FK vs simulator-trunk relative-pose canary and the tracking
gate are separate layers and do not hold the CPU binding pending. Unverified
slots stay fail-closed: they are never zero-filled, padded, or bridged from the
old 256-wide vector.
"""
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from . import contracts
from ._arr import is_finite, to_plain
from .errors import ContractViolation, MappingNotDeclaredError, Port2026Error

# post1 R1Pro 61-d proprio slices (eval_utils PROPRIOCEPTION_INDICES["R1Pro"]).
POST1_R1PRO_SLICES = {
    "base_qvel": slice(0, 3),
    "arm_left_qpos": slice(3, 10),
    "arm_left_qvel": slice(10, 17),
    "eef_left_pos": slice(17, 20),
    "eef_left_quat": slice(20, 24),
    "gripper_left_qpos": slice(24, 26),
    "gripper_left_qvel": slice(26, 28),
    "arm_right_qpos": slice(28, 35),
    "arm_right_qvel": slice(35, 42),
    "eef_right_pos": slice(42, 45),
    "eef_right_quat": slice(45, 49),
    "gripper_right_qpos": slice(49, 51),
    "gripper_right_qvel": slice(51, 53),
    "trunk_qpos": slice(53, 57),
    "trunk_qvel": slice(57, 61),
}

# Frozen adapter state45 slot roles (adapter L15-25).
SLOT_ROLES = (
    ["left_ee_se3"] * 6
    + ["right_ee_se3"] * 6
    + ["gripper_width_left", "gripper_width_right"]
    + ["arm_left_qpos"] * 7
    + ["arm_right_qpos"] * 7
    + ["trunk_qpos"] * 4
    + ["finger_left_qpos"] * 2
    + ["finger_right_qpos"] * 2
    + ["base_move"] * 3
    + ["torso_se3"] * 6
)

GRIPPER_RANGE_M = contracts.GRIPPER_RANGE_M


@dataclass(frozen=True)
class FieldMapping:
    source_index: Optional[int]
    role: str
    status: str  # "bound" | "unknown_pending_remote"
    note: str = ""


@dataclass(frozen=True)
class SchemaDescriptor:
    source_name: str
    source_width: int
    target_width: int
    fields: List[FieldMapping]

    def unknown_fields(self) -> List[FieldMapping]:
        return [f for f in self.fields if f.status == "unknown_pending"]


class StateMappingProvider:
    """Abstract mapping implementation registered by the integration lane."""

    def descriptor(self) -> SchemaDescriptor:
        raise NotImplementedError

    def map_state(self, row) -> list:
        """Map one source-state row to the frozen 45-d state."""
        raise NotImplementedError


class UnmappedProvider(StateMappingProvider):
    """Fail-closed placeholder used until the remote field mapping is reviewed."""

    def __init__(self, source_name: str, source_width: int):
        self._descriptor = SchemaDescriptor(
            source_name=source_name,
            source_width=source_width,
            target_width=contracts.STATE45,
            fields=[
                FieldMapping(None, f"legacy_state45_slot_{i}", "unknown_pending",
                             "mapping to legacy 45 not verified against remote 2026 data")
                for i in range(contracts.STATE45)
            ],
        )

    def descriptor(self) -> SchemaDescriptor:
        return self._descriptor

    def map_state(self, row) -> list:
        raise MappingNotDeclaredError(
            f"{self._descriptor.source_name}: {self._descriptor.source_width}->45 mapping "
            "is not declared; do not zero-fill or reuse the 256 slice"
        )


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _mean(values: Sequence[float]) -> float:
    return sum(float(v) for v in values) / len(values)


class OnlineState45Provider(StateMappingProvider):
    """post1 61-d proprio -> frozen 45-d state, verified slots only.

    Args:
        source_width: live proprio width (61 for the post1 R1Pro vector).
        velocity_provider: optional callable returning robot-local
            ``[vx, vy, wz]``. The frame was confirmed on the GPU1 v3 proprio
            sample: the 61-d ``base_qvel`` slice 0:3 is already robot-local, so
            the wrapper composes the provider from that slice; the provider
            never reaches into privileged objects. When ``None`` slots 36:39
            fail closed.
        torso_fk: optional :class:`~brvla_bridge.torso_fk.TorsoFk` loaded from
            the manifest-injected, sha-verified URDF. Until the chain is bound,
            slots 39:45 fail closed.
        dt: control step (frozen 1/30).
        remove_eef: online topology ``remove_eef_state=true``; EE slots 0:12
            are forced zero by profile, not by an unverified FK.
    """

    def __init__(
        self,
        velocity_provider: Optional[Callable[[], List[float]]] = None,
        torso_fk=None,
        dt: float = contracts.CONTROL_DT,
        source_width: int = contracts.STATE61,
        remove_eef: bool = True,
    ):
        if not remove_eef:
            raise ContractViolation(
                "remove_eef=False is unsupported: the post1 61-d proprio carries no "
                "EE pose source, so slots 0:12 must not be faked as bound. The frozen "
                "online topology is remove_eef_state=true (EE slots forced zero by profile)."
            )
        self._source_width = source_width
        self._velocity_provider = velocity_provider
        self._torso_fk = torso_fk
        self._dt = float(dt)
        self._remove_eef = remove_eef
        self._rows: List[List[float]] = []

    # -- descriptor ---------------------------------------------------------

    def _status(self, role: str) -> str:
        if role.endswith("ee_se3"):
            return "ee_zero_profile"
        if role == "base_move":
            return "bound" if self._velocity_provider is not None else "unknown_pending"
        if role == "torso_se3":
            return "bound_cpu_fk" if self._torso_fk is not None else "unknown_pending"
        return "bound"

    def descriptor(self) -> SchemaDescriptor:
        fields = [
            FieldMapping(None, role, self._status(role),
                          "post1 61-d verified slice" if self._status(role) == "bound" else
                          ("EE forced zero under remove_eef_state topology" if self._status(role) == "ee_zero_profile"
                           else ("manifest URDF identity + CPU FK chain cross-checked; live sim-trunk "
                                 "relative-pose canary and tracking gate remain separate"
                                 if self._status(role) == "bound_cpu_fk"
                                 else "source context not verified; fail-closed")))
            for role in SLOT_ROLES
        ]
        return SchemaDescriptor(
            source_name="post1_r1pro_proprio61",
            source_width=self._source_width,
            target_width=contracts.STATE45,
            fields=fields,
        )

    def pending_slots(self) -> List[str]:
        return [role for role, f in zip(SLOT_ROLES, self.descriptor().fields)
                if f.status == "unknown_pending"]

    # -- mapping ------------------------------------------------------------

    def _validate_row(self, row) -> List[float]:
        plain = to_plain(row)
        if len(plain) != self._source_width:
            raise MappingNotDeclaredError(
                f"post1 proprio width {len(plain)} != {self._source_width}")
        if not is_finite(plain):
            raise MappingNotDeclaredError("post1 proprio contains non-finite values")
        return [float(v) for v in plain]

    def _fill(self, row: List[float]) -> List[float]:
        out = [0.0] * contracts.STATE45
        sl = POST1_R1PRO_SLICES
        grip_l = row[sl["gripper_left_qpos"]]
        grip_r = row[sl["gripper_right_qpos"]]

        # 12:14 normalized gripper width (two-finger mean / 0.05, clip [0,1]).
        out[12] = _clip01(_mean(grip_l) / GRIPPER_RANGE_M)
        out[13] = _clip01(_mean(grip_r) / GRIPPER_RANGE_M)

        # 14:36 j22: armL7, armR7, trunk4, fingerL2, fingerR2.
        out[14:21] = row[sl["arm_left_qpos"]]
        out[21:28] = row[sl["arm_right_qpos"]]
        out[28:32] = row[sl["trunk_qpos"]]
        out[32:34] = list(grip_l)
        out[34:36] = list(grip_r)

        pending = self.pending_slots()
        if self._velocity_provider is not None:
            vel = [float(v) for v in self._velocity_provider()]
            if len(vel) != 3 or not all(math.isfinite(v) for v in vel):
                raise MappingNotDeclaredError("velocity provider must return finite 3-vector")
            out[36:39] = [vel[0] * self._dt, vel[1] * self._dt, vel[2] * self._dt]
        if self._torso_fk is not None:
            out[39:45] = list(self._torso_fk.relative_se3(row[sl["trunk_qpos"]]))

        if pending:
            raise MappingNotDeclaredError(
                "state45 slots still pending fail-closed: " + ", ".join(pending))
        return out

    def map_state(self, row) -> list:
        return self._fill(self._validate_row(row))

    def append(self, row) -> List[float]:
        """Append a live proprio row and return the current 45-d state."""
        validated = self._validate_row(row)
        state = self._fill(validated)
        self._rows.append(state)
        return state

    def history(self) -> List[List[float]]:
        """Oldest-first last-4 window; early steps repeat row 0 (adapter pack_history)."""
        if not self._rows:
            raise MappingNotDeclaredError("no proprio history recorded")
        window = list(self._rows[-contracts.HISTORY_LEN:])
        pad = [list(self._rows[0])] * (contracts.HISTORY_LEN - len(window))
        return pad + window

    def reset(self) -> None:
        self._rows.clear()
        if self._torso_fk is not None:
            self._torso_fk.reset()


_REGISTRY = {}


def register_provider(name: str, provider: StateMappingProvider) -> None:
    if not isinstance(provider, StateMappingProvider):
        raise Port2026Error("provider must implement StateMappingProvider")
    if name in _REGISTRY:
        raise Port2026Error(f"provider already registered: {name}")
    _REGISTRY[name] = provider


def get_provider(name: str) -> StateMappingProvider:
    if name not in _REGISTRY:
        raise Port2026Error(f"unknown state mapping provider: {name}")
    return _REGISTRY[name]


def clear_registry() -> None:
    """Test-only helper; never called from production wiring."""
    _REGISTRY.clear()


def validate_mapped_state(row) -> list:
    plain = to_plain(row)
    if len(plain) != contracts.STATE45:
        raise MappingNotDeclaredError(f"mapped state width must be {contracts.STATE45}, got {len(plain)}")
    if not is_finite(plain):
        raise MappingNotDeclaredError("mapped state contains non-finite values")
    return plain
