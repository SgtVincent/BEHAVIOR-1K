"""Stdlib tests for receipt certificate and scope verification.

The gate re-derives acceptance from structured receipt fields, not the
``canary_pass``/``gate_pass`` bool: stage statuses, optimal certificate, finite
23-d vector, receipt/config sha back-links, device identity, arm and live-scene
scope. No planner, simulator, torch, or numpy is imported.
"""
from brvla_bridge import scope
from brvla_bridge.errors import ContractViolation


def _receipt(**cert_overrides):
    certificate = {
        "status": "optimal",
        "bound_arm": "left",
        "primary_tool": "left_gripper_link",
        "certified_scope": "left_arm_only; self_collision; scene_model_null",
        "right_arm_planned": False,
        "device_uuid": "GPU-abc",
        "device_uuid_source": "manifest_argument_not_live_probe",
        "receipt": "receipt-sha",
        "config_yml_sha256": "config-sha",
        "position_error_mm": 0.00043,
    }
    certificate.update(cert_overrides)
    return {
        "gate_pass": True,
        "stages": {
            "runtime": {"status": "ok"},
            "bind": {"status": "ok"},
            "backend_plan": {"status": "ok", "outputs": {
                "success": True,
                "trajectory": {"waypoints": 21, "batch": 1, "horizon": 1, "dt": 0.025},
                "certificate": certificate,
            }},
            "executor_plan": {"status": "ok", "outputs": {
                "vector_width": 23,
                "vector_finite": True,
                "certificate": certificate,
                "meta": {"waypoints": 21, "batch": 1, "horizon": 1, "dt": 0.025},
            }},
        },
    }


def test_left_only_receipt_passes_left_consumer():
    report = scope.verify_receipt(_receipt(), required_arm="left")
    assert report.pass_ is True
    assert report.missing == []
    assert report.evidence["bound_arm"] == "left"


def test_right_requirement_fails_when_right_unplanned():
    report = scope.verify_receipt(_receipt(), required_arm="right")
    assert report.pass_ is False
    assert "right_arm_unproven" in report.missing


def test_dual_requirement_fails_on_left_only_cert():
    report = scope.verify_receipt(_receipt(), required_arm="both")
    assert report.pass_ is False
    assert "right_arm_unproven" in report.missing


def test_live_scene_requirement_fails_on_null_scene():
    report = scope.verify_receipt(_receipt(), require_live_scene=True)
    assert report.pass_ is False
    assert "live_scene_unproven" in report.missing


def test_gate_pass_bool_is_not_enough():
    receipt = _receipt()
    receipt["gate_pass"] = False
    report = scope.verify_receipt(receipt, required_arm="left")
    assert "gate_pass_not_true" in report.missing


def test_non_optimal_certificate_fails():
    report = scope.verify_receipt(_receipt(status="rejected"), required_arm="left")
    assert "certificate_status_rejected" in report.missing


def test_receipt_sha_backlink_is_checked():
    report = scope.verify_receipt(_receipt(), required_arm="left",
                                  expected_receipt_sha="other-sha")
    assert "receipt_sha_mismatch" in report.missing


def test_config_sha_backlink_is_checked():
    report = scope.verify_receipt(_receipt(), required_arm="left",
                                  expected_config_sha="other-sha")
    assert "config_sha_mismatch" in report.missing


def test_device_uuid_is_checked():
    report = scope.verify_receipt(_receipt(), required_arm="left",
                                  expected_device_uuid="GPU-other")
    assert "device_uuid_mismatch" in report.missing


def test_right_certified_receipt_serves_right_consumer():
    receipt = _receipt(bound_arm="both", right_arm_planned=True,
                       certified_scope="dual_arm; live_scene_known")
    report = scope.verify_receipt(receipt, required_arm="both",
                                  require_live_scene=True)
    assert report.pass_ is True, report.missing


def test_non_object_receipt_fails_closed():
    report = scope.verify_receipt("not-a-dict")
    assert report.pass_ is False
    assert "receipt_not_object" in report.missing


def test_scope_require_raises():
    report = scope.verify_receipt(_receipt(), required_arm="right")
    try:
        report.require()
    except ContractViolation:
        return
    raise AssertionError("require() must raise on failed scope")
