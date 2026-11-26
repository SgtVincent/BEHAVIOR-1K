"""Stdlib tests for the ManagedReplayPolicy evaluator seam.

Covers pin/anchor/scope construction, plain native23 composition through the
wrapper, left-only unproven-right fail-closed, and the empty-receipt block.
Forward tensor construction is lazy (torch); these tests exercise the plain
composition path only. No planner, simulator, torch, numpy, or omnigibson.
"""
from brvla_bridge import contracts, managed_policy
from brvla_bridge.errors import ContractViolation

DT = 1.0 / 30.0


def _row() -> list:
    row = [0.0] * contracts.ACTION23
    row[12:14] = [0.4, 0.6]
    return row


def _anchors() -> dict:
    return {
        "left": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
        "right": [-0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
        "chest": [0.0, 0.0, 0.9, 0.0, 0.0, 0.0],
    }


def _receipt(**overrides):
    certificate = {
        "status": "optimal", "bound_arm": "left", "primary_tool": "left_gripper_link",
        "certified_scope": "left_arm_only; self_collision; scene_model_null",
        "right_arm_planned": False, "device_uuid": "GPU-abc",
        "device_uuid_source": "manifest_argument_not_live_probe",
        "receipt": "r", "config_yml_sha256": "c",
        "scene_identity": None, "source_epoch": None,
    }
    certificate.update(overrides)
    return {
        "gate_pass": True,
        "stages": {
            "runtime": {"status": "ok"}, "bind": {"status": "ok"},
            "backend_plan": {"status": "ok", "outputs": {"success": True, "certificate": certificate}},
            "executor_plan": {"status": "ok", "outputs": {
                "vector_width": 23, "vector_finite": True, "certificate": certificate,
                "meta": {"waypoints": 3, "batch": 1, "horizon": 1, "dt": 0.025},
            }},
        },
    }


class FakeExecutor:
    def __init__(self):
        self.calls = []

    def plan(self, goal):
        self.calls.append(goal)
        return {"certificate": {}, "trajectory": {
            "native23_waypoints": [[float(k + 1)] * contracts.ACTION23 for k in range(3)],
            "dt": 0.025, "batch": 1, "horizon": 1, "waypoints": 3,
        }}


def _policy(receipt, **kwargs):
    kwargs.setdefault("rows", [_row(), _row()])
    kwargs.setdefault("anchors", _anchors())
    kwargs.setdefault("receipt", receipt)
    kwargs.setdefault("executor", FakeExecutor())
    kwargs.setdefault("dt_seconds", DT)
    return managed_policy.ManagedReplayPolicy(**kwargs)


def test_construction_records_scope_and_episode():
    policy = _policy(_receipt())
    receipt = policy.bound_receipt()
    assert receipt["policy"] == "ManagedReplayPolicy"
    assert receipt["episode_length"] == 2
    assert receipt["scope"]["pass"] is True
    assert receipt["inner"]["schema"] == "brvla_saved_execution23_v1"


def test_left_policy_blocks_unproven_right_group():
    policy = _policy(_receipt())
    try:
        policy._next_native23()
    except ContractViolation as exc:
        assert "arm_right" in str(exc)
    else:
        raise AssertionError("left scope must fail closed on unproven right group")


def test_dual_certified_policy_composes_native23():
    receipt = _receipt(bound_arm="both", right_arm_planned=True,
                       certified_scope="dual_arm; live_scene_known",
                       scene_identity="scene_ep0475_001", source_epoch="epoch-1")
    policy = _policy(receipt, required_arm="both", require_live_scene=True)
    out = policy._next_native23()
    assert len(out["vector"]) == contracts.ACTION23
    assert out["sources"]["base"] == "replay"
    assert out["sources"]["arm_right"] == "planner"
    assert out["vector"][14] == 0.4
    assert out["vector"][22] == 0.6


def test_empty_receipt_blocks_construction():
    try:
        _policy({})
    except ContractViolation:
        return
    raise AssertionError("empty receipt must fail closed")


def test_reset_clears_plan_cache():
    receipt = _receipt(bound_arm="both", right_arm_planned=True,
                       certified_scope="dual_arm; live_scene_known",
                       scene_identity="scene_ep0475_001", source_epoch="epoch-1")
    policy = _policy(receipt, required_arm="both", require_live_scene=True)
    policy._next_native23()
    policy.reset()
    assert policy._plan is None
    assert policy._chunk_index == 0
