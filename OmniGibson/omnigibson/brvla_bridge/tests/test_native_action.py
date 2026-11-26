"""Stdlib tests for native23 group assembly with explicit source provenance.

Covers controller-order layout, per-group source recording, missing-group
fail-closed, non-finite and width gates, and group extraction. No planner,
simulator, torch, or numpy is imported.
"""
from brvla_bridge import contracts, native_action
from brvla_bridge.errors import ContractViolation
from brvla_bridge.tests._assert import raises


def _full_groups():
    return {
        "base_vel": [1.0, 2.0, 3.0],
        "trunk": [4.0, 5.0, 6.0, 7.0],
        "arm_left": [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
        "gripper_left": [0.2],
        "arm_right": [15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0],
        "gripper_right": [0.3],
    }


def test_assemble_follows_controller_order():
    out = native_action.assemble_native23(**_full_groups())
    assert len(out["vector"]) == contracts.ACTION23
    assert out["vector"][:3] == [1.0, 2.0, 3.0]
    assert out["vector"][3:7] == [4.0, 5.0, 6.0, 7.0]
    assert out["vector"][7:14] == [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
    assert out["vector"][14] == 0.2
    assert out["vector"][15:22] == [15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0]
    assert out["vector"][22] == 0.3


def test_sources_are_recorded_per_group():
    out = native_action.assemble_native23(
        **_full_groups(),
        sources={"base": "replay", "trunk": "planner", "arm_left": "planner",
                 "gripper_left": "replay", "arm_right": "planner",
                 "gripper_right": "replay"},
    )
    assert out["sources"]["base"] == "replay"
    assert out["sources"]["gripper_left"] == "replay"
    assert out["sources"]["arm_right"] == "planner"


def test_missing_group_fails_closed_without_zero_fill():
    groups = _full_groups()
    groups["arm_right"] = None
    sources = {g: "missing" if g == "arm_right" else "planner"
               for g in native_action.GROUP_WIDTHS}
    try:
        native_action.assemble_native23(**groups, sources=sources)
    except ContractViolation as exc:
        assert "arm_right" in str(exc)
    else:
        raise AssertionError("missing arm_right group must fail closed")


def test_non_finite_group_rejected():
    groups = _full_groups()
    groups["trunk"] = [4.0, float("nan"), 6.0, 7.0]
    with raises(ContractViolation):
        native_action.assemble_native23(**groups)


def test_width_mismatch_rejected():
    groups = _full_groups()
    groups["arm_left"] = [8.0, 9.0]
    with raises(ContractViolation):
        native_action.assemble_native23(**groups)


def test_extract_group_round_trip():
    out = native_action.assemble_native23(**_full_groups())
    assert native_action.extract_group(out["vector"], "trunk") == [4.0, 5.0, 6.0, 7.0]
    assert native_action.extract_group(out["vector"], "base") == [1.0, 2.0, 3.0]


def test_bind_current_q_records_live_qpos_provenance():
    entry = native_action.bind_current_q("right", [0.1] * 7, source_sha256="abc123")
    assert entry["group"] == "arm_right"
    assert entry["source"] == native_action.SOURCE_CURRENT_Q
    assert entry["source_sha256"] == "abc123"
    assert len(entry["values"]) == 7


def test_bind_current_q_rejects_invalid_qpos():
    with raises(ContractViolation):
        native_action.bind_current_q("left", [0.1] * 3, source_sha256="abc")
    with raises(ContractViolation):
        native_action.bind_current_q("left", [float("nan")] * 7, source_sha256="abc")
    with raises(ContractViolation):
        native_action.bind_current_q("left", [0.1] * 7, source_sha256="")
    with raises(ContractViolation):
        native_action.bind_current_q("torso", [0.1] * 4, source_sha256="abc")


def test_inactive_arm_holds_current_q_without_zero_fill():
    groups = _full_groups()
    held = [0.5] * 7
    groups["arm_right"] = held
    sources = {g: "planner" for g in native_action.GROUP_WIDTHS}
    sources["arm_right"] = native_action.SOURCE_CURRENT_Q
    out = native_action.assemble_native23(**groups, sources=sources)
    assert out["sources"]["arm_right"] == native_action.SOURCE_CURRENT_Q
    assert out["vector"][15:22] == held


def test_unknown_group_source_rejected():
    groups = _full_groups()
    sources = {g: "planner" for g in native_action.GROUP_WIDTHS}
    sources["trunk"] = "telepathy"
    with raises(ContractViolation):
        native_action.assemble_native23(**groups, sources=sources)
