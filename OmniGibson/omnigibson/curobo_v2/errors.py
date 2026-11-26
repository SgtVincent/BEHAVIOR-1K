# Copyright (c) 2026 B1K2026 integration line
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed exception types for the cuRobo v2 executor lane.

A solver/backend gap must surface as a classified blocker. The executor never
returns a zero-action placeholder or an implicit ``ready=false`` success when
the backend is unbound or the plan is rejected.
"""
from __future__ import annotations


class CuroboV2Error(RuntimeError):
    """Base class for this lane's classified failures."""

    reason: str = "curobo_v2_error"

    def __init__(self, detail: str = "") -> None:
        super().__init__(f"{self.reason}: {detail}" if detail else self.reason)
        self.detail = detail


class BackendUnavailable(CuroboV2Error):
    """The planner backend is not bound to a verified robot profile."""

    reason = "backend_unavailable"


class BackendImportFailed(CuroboV2Error):
    """Lazy cuRobo v2 / torch import failed in the GPU-bound stage."""

    reason = "backend_import_failed"


class RobotProfileUnbound(CuroboV2Error):
    """The R1Pro URDF / sphere-map / tool-frame binding has not been verified."""

    reason = "robot_profile_unbound"


class PlanRejected(CuroboV2Error):
    """The planner returned no finite collision-free trajectory for the goal."""

    reason = "plan_rejected"


class AdapterNotCertified(CuroboV2Error):
    """The adapter has no canary (build + reset + first frame + step) evidence."""

    reason = "adapter_not_certified"
