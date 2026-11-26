"""MoMa Agent+MP planning adapter for the B1K 2026 challenge line.

Seam layers: provider / executor / observer / sim-adapter (INTERFACE_SPEC §5).
Sim-free leaves are importable without omnigibson/torch; simulator coupling
stays behind the :class:`~omnigibson.moma.seams.SimAdapter` boundary.
"""
from .schema import (  # noqa: F401
    NATIVE23_JOINT,
    BRIDGE_DECODED23_IK_POSE,
    EXECUTION23_SE3_DELTA,
    LEGACY_LABEL_ALIASES,
    normalize_label,
    TaggedAction,
    native23_action,
)
from .seams import (  # noqa: F401
    MoMaObjectProvider,
    MoMaExecutor,
    MoMaObserver,
    SimAdapter,
)

__all__ = [
    "NATIVE23_JOINT",
    "BRIDGE_DECODED23_IK_POSE",
    "EXECUTION23_SE3_DELTA",
    "LEGACY_LABEL_ALIASES",
    "normalize_label",
    "TaggedAction",
    "native23_action",
    "MoMaObjectProvider",
    "MoMaExecutor",
    "MoMaObserver",
    "SimAdapter",
]
