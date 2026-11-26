# Copyright (c) 2026 B1K2026 integration line
# SPDX-License-Identifier: Apache-2.0
"""cuRobo v2 executor lane: MoMa plugin seam + BR replay native23 boundary.

This package is the integrator-owned 2026 binding behind the existing
``moma.mp_plugin`` manifest/loader. It deliberately defines no second planner
RPC, no empty ABC replacement, and no v1 fallback.

Boundary rules (see ``contracts.py``):

- top-level modules must stay importable without torch / curobo / omni / warp;
- heavy planner imports are deferred to :class:`CuroboV2Backend._bind`;
- an unbound or uncertified backend fails closed; no zero-action placeholder;
- native23 slice order and R1Pro joint/frame inventory are sealed here.
"""
from __future__ import annotations

from .backend import CuroboV2Backend
from .contracts import (
    ACTION_DIM,
    BRIDGE_DECODED23_IK_POSE,
    EXECUTION23_SE3_DELTA,
    NATIVE23_GROUPS,
    NATIVE23_JOINT,
    NATIVE23_SLICES,
    PlannerGoal,
    TaggedAction,
    assemble_native23,
    native23_action,
    parse_target,
)
from .errors import (
    AdapterNotCertified,
    BackendImportFailed,
    BackendUnavailable,
    CuroboV2Error,
    PlanRejected,
    RobotProfileUnbound,
)
from .executor import CuroboV2Executor
from .factory import PLUGIN_VERSION, make_curobo_v2_executor

__all__ = [
    "ACTION_DIM",
    "BRIDGE_DECODED23_IK_POSE",
    "EXECUTION23_SE3_DELTA",
    "NATIVE23_GROUPS",
    "NATIVE23_JOINT",
    "NATIVE23_SLICES",
    "PLUGIN_VERSION",
    "PlannerGoal",
    "TaggedAction",
    "CuroboV2Backend",
    "CuroboV2Executor",
    "CuroboV2Error",
    "AdapterNotCertified",
    "BackendImportFailed",
    "BackendUnavailable",
    "PlanRejected",
    "RobotProfileUnbound",
    "assemble_native23",
    "make_curobo_v2_executor",
    "native23_action",
    "parse_target",
]

__version__ = PLUGIN_VERSION
