"""SDK boundary adapters: live robot and environment read/write surface.

Every adapter is duck-typed and sim-free; the live OmniGibson object is bound
at the GPU canary. Until bound, methods raise ``AccessorUnboundError`` /
``EnvAdapterUnboundError`` instead of returning zero-action placeholders.
"""
from .robot_accessor import (  # noqa: F401
    SdkRobotAccessor,
    GraspReadout,
    AccessorUnboundError,
)
from .env_adapter import SdkEnvironmentAdapter, EnvAdapterUnboundError  # noqa: F401

__all__ = [
    "SdkRobotAccessor",
    "GraspReadout",
    "AccessorUnboundError",
    "SdkEnvironmentAdapter",
    "EnvAdapterUnboundError",
]
