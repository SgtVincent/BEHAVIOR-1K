"""Stdlib tests for the managed DemoReplayPolicy seam.

Torch/numpy wrapping is exercised on the remote 2026 environment; these cases
cover pin gates, row order, reset, exhaustion, bound identity, and the audit
receipt without importing torch or numpy.
"""
import json
import os
import tempfile

from brvla_bridge import replay_policy
from brvla_bridge.errors import ContractViolation
from brvla_bridge.tests._assert import raises


def _row(value: float) -> list:
    return [value] * 23


def _manifest() -> dict:
    return {
        "schema": replay_policy.SAVED_SCHEMA,
        "representation": replay_policy.SAVED_REPRESENTATION,
        "shape": [3, 23],
        "dt_seconds": 1.0 / 30,
        "columns": {
            "left_pose": [0, 6],
            "right_pose": [6, 12],
            "continuous_gripper": [12, 14],
            "move": [14, 17],
            "chest_pose": [17, 23],
        },
        "convention": {"chunk_anchor": "fixed_run_local_start_pose_not_updated_each_step"},
        "chunks": [{"anchors": {"left": [0.1] * 6, "right": [0.2] * 6, "chest": [0.3] * 6}}],
    }


def _policy(**kwargs) -> replay_policy.DemoReplayPolicy:
    rows = kwargs.pop("rows", [_row(1.0), _row(2.0), _row(3.0)])
    manifest = kwargs.pop("manifest", _manifest())
    return replay_policy.DemoReplayPolicy(rows=rows, manifest=manifest, **kwargs)


def test_row_order_reset_and_alias():
    policy = _policy()
    assert policy._next_row() == _row(1.0)
    assert policy._next_row() == _row(2.0)
    policy.reset()
    assert policy._next_row() == _row(1.0)
    assert policy._index == 1


def test_exhausted_segment_never_repeats():
    policy = _policy(end_frame=2)
    policy._next_row()
    policy._next_row()
    with raises(replay_policy.ReplayExhausted):
        policy._next_row()


def test_start_frame_offset():
    policy = _policy(start_frame=1)
    assert policy._next_row() == _row(2.0)


def test_wide_rejected():
    with raises(replay_policy.ReplayPinMismatch):
        replay_policy.DemoReplayPolicy(rows=[[0.0] * 22], manifest=_manifest())


def test_non_finite_rejected():
    bad = _row(1.0)
    bad[0] = float("nan")
    with raises(replay_policy.ReplayPinMismatch):
        replay_policy.DemoReplayPolicy(rows=[bad], manifest=_manifest())


def test_load_demo_bound_identity():
    policy = _policy(demo_id="00212380", task_name="turning_on_radio", instance_id=301)
    policy.load_demo("00212380")
    assert policy._index == 0
    with raises(replay_policy.ReplayPinMismatch):
        policy.load_demo("00219999")


def test_missing_converted_dir_rejected():
    with raises(ContractViolation):
        replay_policy.DemoReplayPolicy()


def test_receipt_records_anchor_and_dim():
    policy = _policy(demo_id="00212380", task_name="turning_on_radio", instance_id=301)
    policy.set_action_dim(23)
    receipt = policy.bound_receipt()
    assert receipt["schema"] == replay_policy.SAVED_SCHEMA
    assert receipt["representation"] == replay_policy.SAVED_REPRESENTATION
    assert receipt["shape"] == [3, 23]
    assert receipt["anchors"]["left"] == [0.1] * 6
    assert receipt["bound"] == {
        "task_name": "turning_on_radio", "demo_id": "00212380", "instance_id": 301}
    assert receipt["evaluator_action_dim"] == 23
    assert receipt["not_model_eval"] is True


def test_trace_writes_bound_and_row_events():
    fd, trace = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    try:
        policy = _policy(trace_path=trace)
        policy._next_row()
        events = [json.loads(line) for line in open(trace, encoding="utf-8")]
        assert events[0]["event"] == "saved_policy_bound"
        assert events[1]["event"] == "saved_row_selected"
        assert events[1]["source_row"] == 0
    finally:
        os.remove(trace)
