"""CPU tests for the S2 sanitizer, terminal predicate, validation and
request-idempotency state machine.

Ported scope: CPU-2 (test_a21_external_s2_policy_cpu). Stdlib/numpy only; no
simulator or network import.
"""
import json

import pytest

from omnigibson.moma.s2.contract import (
    ARGS,
    FORBIDDEN,
    RequestTracker,
    capped_observation,
    full_task_terminal,
    sanitized,
    strict_json,
    validate,
)

CANS = ["can_of_soda_114", "can_of_soda_115", "can_of_soda_113"]
BUCKET = "trash_can_116"

CFG = {
    "can_names": CANS,
    "bucket_name": BUCKET,
    "allowed_primitives": [
        "continue_nominal_tracking", "stabilize_and_brake", "open_right_gripper",
        "retract_to_pregrasp", "reanchor_to_live_object", "approach_object",
        "close_and_acquire_grasp",
    ],
    "primitive_caps": {
        "continue_nominal_tracking": 3000,
        "stabilize_and_brake": 1800,
        "open_right_gripper": 10,
        "retract_to_pregrasp": 50,
        "reanchor_to_live_object": 200,
        "approach_object": 300,
        "close_and_acquire_grasp": 100,
    },
}


def _schema():
    return {
        "approach_object": {
            "max_steps": 300,
            "min_steps": 1,
            "allowed_objects": CANS,
            "allowed_stop_conditions": None,
        },
        "stabilize_and_brake": {
            "max_steps": 1800,
            "min_steps": 1,
            "allowed_objects": CANS + [BUCKET],
            "allowed_stop_conditions": None,
        },
    }


def _observation(**overrides):
    obs = {
        "runner_state": "decision_required",
        "decision_required": True,
        "active_request": None,
        "robot_state": {
            "arms": {
                "left": {"object_in_hand": None, "constraint_present": False,
                         "release_counter": 0, "valid_sdk_hold": False},
                "right": {"object_in_hand": None, "constraint_present": False,
                          "release_counter": 0, "valid_sdk_hold": False},
            }
        },
        "allowed_primitives_schema": _schema(),
        "task_progress": {},
        "objects": {},
    }
    obs.update(overrides)
    return obs


def test_strict_json_rejects_duplicate_keys():
    with pytest.raises(ValueError, match="duplicate_json_key"):
        strict_json('{"a": 1, "a": 2}')


def test_strict_json_rejects_nonfinite_constant():
    with pytest.raises(ValueError, match="nonfinite_json"):
        strict_json('{"a": NaN}')


def test_sanitized_strips_forbidden_keys_recursively():
    obs = {
        "source_cursor": 818,
        "next_primitive": "approach_object",
        "nested": {"intervention_mode": "none", "ok": 1},
        "list": [{"failure_index": 3, "keep": 2}],
    }
    clean = sanitized(obs)
    assert "source_cursor" not in clean
    assert "next_primitive" not in clean
    assert clean["nested"] == {"ok": 1}
    assert clean["list"] == [{"keep": 2}]
    assert FORBIDDEN  # sanity: non-empty registry


def test_validate_accepts_valid_primitive():
    obs = _observation()
    args = {"primitive": "approach_object", "target_object": CANS[0], "max_steps": 120}
    validate(args, obs, CFG)  # does not raise


def test_validate_rejects_unknown_argument():
    obs = _observation()
    args = {"primitive": "approach_object", "target_object": CANS[0], "max_steps": 120,
            "shell": "rm -rf"}
    with pytest.raises(ValueError, match="unknown_tool_arguments"):
        validate(args, obs, CFG)


def test_validate_rejects_disallowed_primitive():
    obs = _observation()
    args = {"primitive": "teleport_object", "target_object": CANS[0], "max_steps": 10}
    with pytest.raises(ValueError, match="primitive_not_allowed"):
        validate(args, obs, CFG)


def test_validate_rejects_non_idle_boundary():
    obs = _observation(active_request={"request_id": "x"})
    args = {"primitive": "approach_object", "target_object": CANS[0], "max_steps": 120}
    with pytest.raises(ValueError, match="not_an_idle_decision_boundary"):
        validate(args, obs, CFG)


def _default_objects():
    return {
        **{c: {"inside_bucket": True, "contact_bodies": []} for c in CANS},
        BUCKET: {"on_ground": True, "contact_bodies": ["/floors_0/groundPlane"]},
    }


def _progress(progress_overrides):
    progress = {
        "bddl_done": {"success": True},
        "bucket_ever_held": True,
        "bucket_release_observed": True,
        "can_ever_held": {c: True for c in CANS},
        "can_release_observed": {c: True for c in CANS},
        "all_cans_inside": {c: True for c in CANS},
    }
    progress.update(progress_overrides)
    return progress


def _terminal_obs(task_progress=None, objects=None, **progress_overrides):
    progress = task_progress if task_progress is not None else _progress(progress_overrides)
    obs = _observation(
        task_progress=progress,
        objects=objects or _default_objects(),
    )
    # Empty-hand terminal contract: release_counter is None on an empty hand.
    for arm in ("left", "right"):
        obs["robot_state"]["arms"][arm]["release_counter"] = None
    return obs


def test_full_task_terminal_requires_all_conditions():
    assert full_task_terminal(_terminal_obs(), CFG) is True


def test_full_task_terminal_rejects_truthy_but_not_success_bddl_dict():
    obs = _terminal_obs(task_progress={"bddl_done": {"success": False}})
    assert full_task_terminal(obs, CFG) is False


def test_full_task_terminal_rejects_missing_can_inside():
    obs = _terminal_obs(objects={
        **{c: {"inside_bucket": True} for c in CANS},
        **{CANS[1]: {"inside_bucket": False}},
        BUCKET: {"on_ground": True, "contact_bodies": ["/floors_0/groundPlane"]},
    })
    assert full_task_terminal(obs, CFG) is False


def test_full_task_terminal_rejects_non_empty_hands():
    obs = _terminal_obs()
    obs["robot_state"]["arms"]["left"]["object_in_hand"] = CANS[2]
    assert full_task_terminal(obs, CFG) is False


def test_request_tracker_idempotent_replay_blocks_same_pending_id():
    tracker = RequestTracker("run-1")
    req = {"request_id": "s2api_aaa", "run_id": "run-1", "primitive": "approach_object"}
    assert tracker.begin(req) == "accepted_pending"
    with pytest.raises(ValueError, match="unresolved_same_id"):
        tracker.begin(req)
    receipt = {"request_id": "s2api_aaa", "run_id": "run-1", "status": "completed"}
    tracker.receipt(receipt)
    assert tracker.begin(req) == "replayed_completed"


def test_request_tracker_rejects_identity_mismatch():
    tracker = RequestTracker("run-1")
    tracker.begin({"request_id": "s2api_bb", "run_id": "run-1"})
    with pytest.raises(ValueError, match="receipt_identity_mismatch"):
        tracker.receipt({"request_id": "s2api_bb", "run_id": "other-run"})


def test_request_tracker_blocks_new_id_while_active():
    tracker = RequestTracker("run-1")
    tracker.begin({"request_id": "s2api_1", "run_id": "run-1"})
    with pytest.raises(ValueError, match="active_request_blocks_new_id"):
        tracker.begin({"request_id": "s2api_2", "run_id": "run-1"})
