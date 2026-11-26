"""Controller backend interface and receipt registry.

Gate A (converted-data physical replay) must run through the official
post1 :class:`~omnigibson.action_primitives.curobo.CuRoboMotionGenerator` seam.
The legacy stock IK bridge and the DLS active-set v2 solver are historical
clues only: they may be registered for comparison, but a replay run is refused
unless an integrator-issued receipt for the official cuRobo backend is present.
No backend may silently replace another.

License/version verification belongs to the git integrator. The cuRobo fork is
NVIDIA-license proprietary and has no static release version; it may live only
in a private dependency manifest (``private_manifest``), never published.
``unknown_pending`` blocks any use. This package must not label the pinned fork
as "cuRobo v2 release".
"""
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from .errors import BackendNotDeclaredError, Port2026Error

# Official post1 pinned source (post1 extras_require["primitives"]).
CUROBO_BACKEND = "curobo_motion_gen"
CUROBO_PIN_REF = "StanfordVL/curobo@78612f45cef52c3fa0298de243a54cd7ca614414"
STOCK_IK = "stock_ik"
DLS_ACTIVE_SET_V2 = "dls_active_set_v2"

# Gate A (converted-data physical replay) backend requirement.
REPLAY_GATE_BACKEND = CUROBO_BACKEND

LICENSE_STATUSES = {"public_verified", "private_manifest", "unknown_pending"}


@dataclass(frozen=True)
class BackendReceipt:
    kind: str
    version: Optional[str]
    license_status: str
    source_sha256: Optional[str]
    declared_by: str
    notes: str = ""

    def __post_init__(self):
        if self.license_status not in LICENSE_STATUSES:
            raise Port2026Error(f"invalid license_status: {self.license_status}")
        if self.kind == CUROBO_BACKEND and self.license_status == "unknown_pending":
            raise BackendNotDeclaredError(
                "official cuRobo receipt: license/version must be resolved before use"
            )


class ControllerBackend:
    """Abstract backend installed against one robot; concrete impl lives remote."""

    def anchors(self) -> Dict[str, list]:
        raise NotImplementedError

    def decode(self, goal: Dict[str, Any]) -> list:
        raise NotImplementedError

    def profile(self) -> Dict[str, Any]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


_REGISTRY: Dict[str, tuple[BackendReceipt, Callable[..., ControllerBackend]]] = {}


def register_backend(receipt: BackendReceipt, factory: Callable[..., ControllerBackend]) -> None:
    if not isinstance(receipt, BackendReceipt):
        raise Port2026Error("receipt required")
    if receipt.kind in _REGISTRY:
        raise Port2026Error(f"backend already registered: {receipt.kind}")
    _REGISTRY[receipt.kind] = (receipt, factory)


def get_receipt(kind: str) -> BackendReceipt:
    if kind not in _REGISTRY:
        raise BackendNotDeclaredError(f"no receipt/backend registered: {kind}")
    return _REGISTRY[kind][0]


def build_backend(kind: str, *args, **kwargs) -> ControllerBackend:
    if kind not in _REGISTRY:
        raise BackendNotDeclaredError(f"no receipt/backend registered: {kind}")
    return _REGISTRY[kind][1](*args, **kwargs)


def require_replay_backend() -> BackendReceipt:
    """Gate A closure: official cuRobo receipt must be registered and resolved."""
    receipt = get_receipt(REPLAY_GATE_BACKEND)
    if receipt.license_status == "unknown_pending":
        raise BackendNotDeclaredError("official cuRobo license/version unresolved")
    if receipt.kind == CUROBO_BACKEND and receipt.version is None:
        raise BackendNotDeclaredError("cuRobo receipt must record the resolved build version")
    return receipt


def clear_registry() -> None:
    """Test-only helper; never called from production wiring."""
    _REGISTRY.clear()
