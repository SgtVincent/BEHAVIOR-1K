"""Receipt certificate and scope verification for the planner seam.

A `canary_pass` / `gate_pass` boolean alone is not acceptance evidence. This
module re-derives the gate from the structured receipt: stage statuses, optimal
certificate, finite native23 vector, receipt/config sha back-links, device
identity source, and the certified arm / scene scope. A consumer declares
which arms and whether a live scene its replay needs; every missing capability
is listed explicitly and the report fails closed.

Stdlib-only at import. No planner, simulator, torch, numpy, or omnigibson
import.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .errors import ContractViolation

ARM_LEFT = "left"
ARM_RIGHT = "right"
ARM_BOTH = "both"
ALLOWED_ARMS = (ARM_LEFT, ARM_RIGHT, ARM_BOTH)


@dataclass(frozen=True)
class ScopeReport:
    pass_: bool
    missing: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)

    def require(self) -> None:
        if not self.pass_:
            raise ContractViolation(
                "replay scope not certified: " + "; ".join(self.missing))


def _stage(receipt: Dict[str, Any], name: str) -> Dict[str, Any]:
    stage = receipt.get("stages", {}).get(name)
    return stage if isinstance(stage, dict) else {}


def _outputs(receipt: Dict[str, Any], stage: str) -> Dict[str, Any]:
    out = _stage(receipt, stage).get("outputs")
    return out if isinstance(out, dict) else {}


def _certificate(receipt: Dict[str, Any]) -> Dict[str, Any]:
    for stage in ("executor_plan", "backend_plan"):
        cert = _outputs(receipt, stage).get("certificate")
        if isinstance(cert, dict) and cert:
            return cert
    return {}


def _trajectory_meta(receipt: Dict[str, Any]) -> Dict[str, Any]:
    meta = _outputs(receipt, "executor_plan").get("meta")
    if isinstance(meta, dict) and meta:
        return meta
    traj = _outputs(receipt, "backend_plan").get("trajectory")
    return traj if isinstance(traj, dict) else {}


def verify_receipt(
    receipt: Any,
    *,
    required_arm: str = ARM_LEFT,
    require_live_scene: bool = False,
    expected_receipt_sha: Optional[str] = None,
    expected_config_sha: Optional[str] = None,
    expected_device_uuid: Optional[str] = None,
) -> ScopeReport:
    """Re-derive acceptance gates from a planner canary receipt.

    Args:
        receipt: parsed receipt dict.
        required_arm: arm(s) this replay consumer needs: ``left``, ``right``,
            or ``both``. A left-only certificate only passes the ``left``
            requirement.
        require_live_scene: True when the replay needs live-scene collision
            planning; a ``scene_model_null`` certificate fails closed.
        expected_receipt_sha: sha256 the certificate's receipt back-link must
            equal, when pinned by the private manifest.
        expected_config_sha: sha256 the certificate's config yml back-link must
            equal.
        expected_device_uuid: device UUID the certificate must record.
    """
    missing: List[str] = []
    evidence: Dict[str, Any] = {}

    if not isinstance(receipt, dict):
        return ScopeReport(False, ["receipt_not_object"], {"type": type(receipt).__name__})

    if required_arm not in ALLOWED_ARMS:
        raise ContractViolation(f"unknown required_arm: {required_arm}")

    # -- stage statuses -----------------------------------------------------
    statuses = {}
    for stage in ("runtime", "bind", "backend_plan", "executor_plan"):
        statuses[stage] = _stage(receipt, stage).get("status")
    evidence["stage_statuses"] = statuses
    for stage, status in statuses.items():
        if status != "ok":
            missing.append(f"stage_{stage}_not_ok:{status}")

    if receipt.get("gate_pass") is not True:
        missing.append("gate_pass_not_true")

    outputs = _outputs(receipt, "executor_plan")
    backend = _outputs(receipt, "backend_plan")
    cert = _certificate(receipt)
    traj = _trajectory_meta(receipt)

    # -- plan result / certificate ------------------------------------------
    if backend.get("success") is not True:
        missing.append("backend_plan_success_false")
    if outputs.get("vector_finite") is not True:
        missing.append("native23_vector_not_finite")
    if outputs.get("vector_width") not in (None, 23):
        missing.append(f"native23_vector_width_{outputs.get('vector_width')}")
    if cert.get("status") != "optimal":
        missing.append(f"certificate_status_{cert.get('status')}")

    evidence["certificate"] = {
        "status": cert.get("status"),
        "bound_arm": cert.get("bound_arm"),
        "primary_tool": cert.get("primary_tool"),
        "certified_scope": cert.get("certified_scope"),
        "right_arm_planned": cert.get("right_arm_planned"),
        "device_uuid": cert.get("device_uuid"),
        "device_uuid_source": cert.get("device_uuid_source"),
        "receipt": cert.get("receipt"),
        "config_yml_sha256": cert.get("config_yml_sha256"),
        "position_error_mm": cert.get("position_error_mm"),
    }

    # -- back-link sha gates -------------------------------------------------
    if expected_receipt_sha is not None:
        if cert.get("receipt") != expected_receipt_sha:
            missing.append("receipt_sha_mismatch")
    if expected_config_sha is not None:
        if cert.get("config_yml_sha256") != expected_config_sha:
            missing.append("config_sha_mismatch")
    if expected_device_uuid is not None:
        if cert.get("device_uuid") != expected_device_uuid:
            missing.append("device_uuid_mismatch")

    # -- arm scope ------------------------------------------------------------
    bound_arm = cert.get("bound_arm")
    right_planned = cert.get("right_arm_planned")
    evidence["bound_arm"] = bound_arm
    evidence["right_arm_planned"] = right_planned
    if required_arm in (ARM_LEFT, ARM_BOTH) and bound_arm not in (ARM_LEFT, ARM_BOTH):
        missing.append("left_arm_unproven")
    if required_arm in (ARM_RIGHT, ARM_BOTH) and right_planned is not True:
        missing.append("right_arm_unproven")

    # -- live scene scope -----------------------------------------------------
    scope_text = str(cert.get("certified_scope") or "")
    evidence["certified_scope"] = scope_text
    if require_live_scene and "scene_model_null" in scope_text:
        missing.append("live_scene_unproven")

    # -- trajectory shape ----------------------------------------------------
    evidence["trajectory"] = {
        "waypoints": traj.get("waypoints"),
        "batch": traj.get("batch"),
        "horizon": traj.get("horizon"),
        "dt": traj.get("dt"),
    }
    if traj.get("batch") not in (None, 1):
        missing.append(f"trajectory_batch_{traj.get('batch')}")
    if traj.get("horizon") not in (None, 1):
        missing.append(f"trajectory_horizon_{traj.get('horizon')}")
    if traj.get("waypoints") in (None, 0):
        missing.append("trajectory_no_waypoints")

    return ScopeReport(not missing, sorted(missing), evidence)


def required_capabilities(required_arm: str, live_scene: bool) -> List[str]:
    """Human-readable capability list for receipts and manifest pins."""
    caps = ["left"] if required_arm in (ARM_LEFT, ARM_BOTH) else []
    if required_arm in (ARM_RIGHT, ARM_BOTH):
        caps.append("right")
    if live_scene:
        caps.append("live_scene")
    return caps
