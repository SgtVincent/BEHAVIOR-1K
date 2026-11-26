"""Stdlib tests for the managed replay composition wrapper.

Covers goal construction (left-only certificate sends only the left row),
plan -> grid -> native23 assembly, replay-sourced base/gripper groups, the
right-arm unproven fail-closed gap, missing-executor and missing-waypoint
gates. No planner, simulator, torch, numpy, or omnigibson is imported.
"""
from brvla_bridge import contracts, delta_tf, native_action, plan_wrapper, scope
from brvla_bridge.errors import BackendNotDeclaredError, ContractViolation

DT = 1.0 / 30.0


def _row(gripper=(0.4, 0.6), move=(0.0, 0.0, 0.0)) -> list:
    row = [0.0] * contracts.ACTION23
    row[12:14] = list(gripper)
    row[14:17] = list(move)
    return row


def _anchors() -> dict:
    return {
        "left": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
        "right": [-0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
        "chest": [0.0, 0.0, 0.9, 0.0, 0.0, 0.0],
    }


def _targets() -> delta_tf.FixedAnchorTargets:
    return delta_tf.FixedAnchorTargets([_row(), _row()], _anchors(), DT)


def _receipt(**overrides):
    certificate = {
        "status": "optimal", "bound_arm": "left", "primary_tool": "left_gripper_link",
        "certified_scope": "left_arm_only; self_collision; scene_model_null",
        "right_arm_planned": False, "device_uuid": "GPU-abc",
        "device_uuid_source": "manifest_argument_not_live_probe",
        "receipt": "r", "config_yml_sha256": "c", "position_error_mm": 0.00043,
        "scene_identity": "scene_ep0475_001", "source_epoch": "epoch-1",
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
    def __init__(self, waypoints=None, with_trajectory=True):
        self.calls = []
        self.waypoints = waypoints or [[float(k + 1)] * contracts.ACTION23 for k in range(3)]
        self.with_trajectory = with_trajectory

    def plan(self, goal):
        self.calls.append(goal)
        if not self.with_trajectory:
            return {"action23": None, "certificate": {}}
        return {"action23": None, "certificate": {}, "trajectory": {
            "native23_waypoints": self.waypoints, "dt": 0.025,
            "batch": 1, "horizon": 1, "waypoints": len(self.waypoints),
        }}


def test_left_only_goal_sends_single_row():
    report = scope.verify_receipt(_receipt(), required_arm="left")
    wrapper = plan_wrapper.ManagedReplayWrapper(
        targets=_targets(), scope_report=report, executor=FakeExecutor())
    goal = wrapper.planner_goal(wrapper.targets_at(0))
    assert len(goal["target_pose8x4_world"]) == 1
    assert len(goal["grippers"]) == 2
    assert len(goal["base_vel"]) == 3


def test_left_scope_fails_closed_on_unproven_right_group():
    report = scope.verify_receipt(_receipt(), required_arm="left")
    wrapper = plan_wrapper.ManagedReplayWrapper(
        targets=_targets(), scope_report=report, executor=FakeExecutor())
    plan = wrapper.plan_row(0)
    try:
        wrapper.assemble_step(plan, 0)
    except ContractViolation as exc:
        assert "arm_right" in str(exc)
    else:
        raise AssertionError("unproven right group must fail closed")


def test_inactive_right_arm_holds_live_current_q():
    report = scope.verify_receipt(_receipt(), required_arm="left")
    wrapper = plan_wrapper.ManagedReplayWrapper(
        targets=_targets(), scope_report=report, executor=FakeExecutor())
    held = [0.5] * 7
    wrapper.bind_current_q("right", held, source_sha256="sha-current-q")
    plan = wrapper.plan_row(0)
    out = wrapper.assemble_step(plan, 0)
    assert out["sources"]["arm_right"] == native_action.SOURCE_CURRENT_Q
    assert out["vector"][15:22] == held
    assert out["held_current_q"]["arm_right"]["source_sha256"] == "sha-current-q"


def test_right_certified_plan_assembles_full_native23():
    receipt = _receipt(bound_arm="both", right_arm_planned=True,
                       certified_scope="dual_arm; live_scene_known")
    report = scope.verify_receipt(receipt, required_arm="both", require_live_scene=True)
    executor = FakeExecutor()
    wrapper = plan_wrapper.ManagedReplayWrapper(
        targets=_targets(), scope_report=report, executor=executor)
    goal = wrapper.planner_goal(wrapper.targets_at(0))
    assert len(goal["target_pose8x4_world"]) == 2
    plan = wrapper.plan_row(0)
    out = wrapper.assemble_step(plan, 0)
    assert len(out["vector"]) == contracts.ACTION23
    assert out["sources"]["base"] == "replay"
    assert out["sources"]["gripper_left"] == "replay"
    assert out["sources"]["trunk"] == "planner"
    assert out["sources"]["arm_right"] == "planner"
    # Replay gripper width overwrites the planner row grip slot.
    assert out["vector"][14] == 0.4
    assert out["vector"][22] == 0.6


def test_missing_executor_fails_closed():
    report = scope.verify_receipt(_receipt(), required_arm="left")
    wrapper = plan_wrapper.ManagedReplayWrapper(
        targets=_targets(), scope_report=report, executor=None)
    try:
        wrapper.plan_row(0)
    except BackendNotDeclaredError:
        return
    raise AssertionError("missing executor must raise BackendNotDeclaredError")


def test_plan_without_waypoints_fails_closed():
    report = scope.verify_receipt(_receipt(), required_arm="left")
    wrapper = plan_wrapper.ManagedReplayWrapper(
        targets=_targets(), scope_report=report,
        executor=FakeExecutor(with_trajectory=False))
    try:
        wrapper.plan_row(0)
    except ContractViolation as exc:
        assert "trajectory" in str(exc)
    else:
        raise AssertionError("plan without trajectory must fail closed")


def test_meta_without_waypoints_fails_closed():
    class MetaOnly(FakeExecutor):
        def plan(self, goal):
            self.calls.append(goal)
            return {"certificate": {}, "meta": {"dt": 0.025}}

    report = scope.verify_receipt(_receipt(), required_arm="left")
    wrapper = plan_wrapper.ManagedReplayWrapper(
        targets=_targets(), scope_report=report, executor=MetaOnly())
    try:
        wrapper.plan_row(0)
    except ContractViolation as exc:
        assert "waypoints" in str(exc)
    else:
        raise AssertionError("plan without waypoints must fail closed")


def test_failed_scope_blocks_wrapper_construction():
    report = scope.verify_receipt(_receipt(), required_arm="right")
    try:
        plan_wrapper.ManagedReplayWrapper(
            targets=_targets(), scope_report=report, executor=FakeExecutor())
    except ContractViolation:
        return
    raise AssertionError("failed scope must block wrapper construction")
