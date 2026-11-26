# Copyright (c) 2026 B1K2026 integration line
# SPDX-License-Identifier: Apache-2.0
"""MoMa-compatible executor wrapping the lazy cuRobo v2 backend.

Contract honored by the MoMa plugin loader / object provider:

- executor exposes :meth:`plan(target)`;
- a successful plan returns
  ``{"action23": TaggedAction, "certificate": {"status": "optimal", ...}}``;
- provider strictly requires ``certificate.status == "optimal"``;
- backend unbound, adapter uncertified, unparseable target or rejected plan
  all raise a classified blocker. The executor never returns a zero-action
  placeholder, an implicit ``ready=false`` success, or a GT-challenge answer.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .backend import CuroboV2Backend
from .contracts import ContractViolation, PlannerGoal, TaggedAction, native23_action, parse_target
from .errors import (
    AdapterNotCertified,
    BackendUnavailable,
    CuroboV2Error,
    PlanRejected,
)


class CuroboV2Executor:
    """Plan-side executor for the R1Pro dual-arm lane.

    ``context`` is what the MoMa runner injects (``cfg``, ``robot``) plus the
    private manifest kwargs. ``cfg``/``robot`` are retained for seam hooks but
    no simulator object is touched at construction.
    """

    status: str = "unverified"

    def __init__(
        self,
        backend: Optional[CuroboV2Backend] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._context = dict(context or {})
        self._backend = backend or CuroboV2Backend(**self._backend_kwargs(self._context))
        self.status = "unverified"

    @staticmethod
    def _backend_kwargs(context: Dict[str, Any]) -> Dict[str, Any]:
        keys = {
            "robot_profile",
            "arm_profile",
            "device_uuid",
            "robot_config_yaml",
            "urdf_path",
            "asset_path",
            "cache_dir",
            "canary_evidence",
            "private_manifest",
        }
        return {k: context[k] for k in keys if k in context}

    # -- certification seam ------------------------------------------------

    def certify(self, evidence: Dict[str, Any]) -> None:
        """Attach a verified canary receipt; the backend may then serve plans."""
        evidence = dict(evidence or {})
        if not evidence.get("canary_pass") or not evidence.get("receipt_sha256"):
            raise AdapterNotCertified("canary receipt incomplete")
        self._backend._canary_evidence.update(evidence)  # noqa: SLF001
        self.status = "canary_pass"

    # -- MoMa executor contract ---------------------------------------------

    def plan(self, target: Dict[str, Any]) -> Dict[str, Any]:
        if self.status != "canary_pass":
            raise AdapterNotCertified("executor has no canary evidence; refusing to plan")
        if not isinstance(target, dict):
            raise ContractViolation("target_must_be_dict")

        try:
            goal: PlannerGoal = parse_target(target)
            result = self._backend.plan_goal(goal)
        except ContractViolation:
            raise
        except CuroboV2Error:
            raise
        except Exception as exc:  # wrap unexpected planner failure as classified blocker
            raise BackendUnavailable(f"unexpected backend failure: {exc}") from exc

        certificate = result.get("certificate") or {}
        if not result.get("success") or certificate.get("status") != "optimal":
            raise PlanRejected(
                certificate.get("reason") or result.get("error") or "planner rejected goal"
            )

        vector = result.get("native23")
        if vector is None:
            raise PlanRejected("plan returned no native23 command")

        action23: TaggedAction = native23_action(vector, meta=result.get("trajectory") or {})
        return {
            "action23": action23,
            "certificate": certificate,
        }

    # -- introspection / teardown -------------------------------------------

    def profile(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "backend": self._backend.profile(),
            "context_keys": sorted(self._context.keys()),
        }

    def close(self) -> None:
        self._backend.close()
        self.status = "unverified"
