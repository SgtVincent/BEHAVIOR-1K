"""CPU/loopback tests for the frozen a28 rule S2 service.

Real HTTP over 127.0.0.1 only: the ChatCompletions client posts a frozen-schema
request and the rule service answers one schema-valid tool call. No simulator,
no MP executor, no env.step; a missing executor stays an explicit blocker.
"""
import json
import urllib.error

import pytest

from omnigibson.moma.config import RunnerConfig
from omnigibson.moma.runner import MomaRunner
from omnigibson.moma.sdk import SdkEnvironmentAdapter, SdkRobotAccessor
from omnigibson.moma.s2.client import ChatCompletionsClient
from omnigibson.moma.s2.contract import validate
from omnigibson.moma.s2.rule_service import RuleSelectionError, RuleS2Service, select_rule

CANS = ["can_of_soda_114", "can_of_soda_115", "can_of_soda_113"]
BUCKET = "trash_can_116"
MODEL = "fake-embodied-s2-native-v2"

S2_CFG = {
    "can_names": CANS,
    "bucket_name": BUCKET,
    "allowed_primitives": [
        "continue_nominal_tracking", "stabilize_and_brake", "open_right_gripper",
        "retract_to_pregrasp", "reanchor_to_live_object", "approach_object",
        "close_and_acquire_grasp",
    ],
    "primitive_caps": {
        "continue_nominal_tracking": 12000,
        "stabilize_and_brake": 120,
        "open_right_gripper": 120,
        "retract_to_pregrasp": 240,
        "reanchor_to_live_object": 240,
        "approach_object": 800,
        "close_and_acquire_grasp": 60,
    },
}


def _arm(object_in_hand, constraint_present, valid_sdk_hold):
    return {
        "object_in_hand": object_in_hand,
        "constraint_present": constraint_present,
        "release_counter": None,
        "valid_sdk_hold": valid_sdk_hold,
    }


def _obs(handed: str = "none"):
    if handed == "left_can":
        left, right = _arm(CANS[0], True, True), _arm(None, False, False)
    elif handed == "right_can":
        left, right = _arm(None, False, False), _arm(CANS[0], True, True)
    else:
        left, right = _arm(None, False, False), _arm(None, False, False)
    return {
        "runner_state": "decision_required",
        "decision_required": True,
        "active_request": None,
        "robot_state": {"arms": {"left": left, "right": right}},
        "allowed_primitives_schema": {
            "continue_nominal_tracking": {
                "allowed_objects": [None, *CANS, BUCKET],
                "min_steps": 1, "max_steps": 15000, "default_max_steps": 3000,
                "allowed_stop_conditions": ["next_grasp_check_or_episode_end", "episode_end"],
            },
            "approach_object": {
                "allowed_objects": CANS, "min_steps": 10, "max_steps": 800,
                "default_max_steps": 600, "allowed_stop_conditions": None,
            },
        },
        "objects": {c: {"inside_bucket": False, "contact_bodies": []} for c in CANS},
    }


def test_select_rule_tracks_held_object_nominally():
    rule_id, args = select_rule(_obs("left_can"), S2_CFG)
    assert rule_id.startswith("held_nominal_tracking")
    assert args["primitive"] == "continue_nominal_tracking"
    assert args["target_object"] == CANS[0]
    assert "stop_condition" in args
    validate(args, _obs("left_can"), S2_CFG)


def test_select_rule_falls_back_to_first_schema_valid_primitive():
    rule_id, args = select_rule(_obs("none"), S2_CFG)
    assert rule_id.startswith("rule_continue_nominal_tracking")
    assert args["target_object"] == CANS[0]
    assert args["stop_condition"] in ("next_grasp_check_or_episode_end", "episode_end")
    validate(args, _obs("none"), S2_CFG)


def test_select_rule_rejects_schema_without_valid_target():
    obs = _obs("none")
    obs["allowed_primitives_schema"] = {
        "approach_object": {"allowed_objects": [None], "min_steps": 10, "max_steps": 800},
    }
    with pytest.raises(RuleSelectionError):
        select_rule(obs, S2_CFG)


def test_real_loopback_http_call_completes_decision():
    service = RuleS2Service(model_name=MODEL, cfg=S2_CFG)
    host, port = service.start()
    try:
        client = ChatCompletionsClient(
            base_url=service.base_url, model=MODEL, prompt_text="test",
            cfg=S2_CFG, history_turns=8, timeout_s=5.0,
        )
        obs = _obs("none")
        decision = client.decide(obs, [])
        assert decision["args"]["primitive"] == "continue_nominal_tracking"
        assert decision["args"]["target_object"] == CANS[0]
        validate(decision["args"], obs, S2_CFG)
        assert decision["call_id"].startswith("call_")
        assert decision["request_sha256"]
        assert len(service.requests) == 1
        assert service.requests[0]["model"] == MODEL
    finally:
        service.stop()


def test_loopback_service_healthz_identity():
    import urllib.request

    service = RuleS2Service(model_name=MODEL, cfg=S2_CFG)
    _, _ = service.start()
    try:
        url = service.base_url.removesuffix("/v1") + "/healthz"
        with urllib.request.urlopen(url, timeout=5) as response:
            health = json.loads(response.read())
        assert health == {"ok": True, "model": MODEL}
    finally:
        service.stop()


def test_rejected_observation_surfaces_http_error_not_fake_action():
    service = RuleS2Service(model_name=MODEL, cfg=S2_CFG)
    service.start()
    try:
        client = ChatCompletionsClient(
            base_url=service.base_url, model=MODEL, prompt_text="test", cfg=S2_CFG,
        )
        bad_obs = _obs("none")
        bad_obs["allowed_primitives_schema"] = {
            "approach_object": {"allowed_objects": [None], "min_steps": 10, "max_steps": 800},
        }
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            client.decide(bad_obs, [])
        assert exc_info.value.code == 400
    finally:
        service.stop()


def test_runner_without_executor_reports_blocker_and_steps_nothing(tmp_path):
    cfg = RunnerConfig(run_id="run-rule-boundary", output_dir=str(tmp_path), max_steps=2)
    runner = MomaRunner(
        cfg,
        env_adapter=SdkEnvironmentAdapter(),
        robot_accessor=SdkRobotAccessor(),
        observer_fn=lambda *a, **k: _obs("none"),
        transition_fn=lambda prev, current, names: [],
        executor=None,
        plugin_manifest=None,
        live_reads={"s2_cfg": S2_CFG},
    )
    report = runner.prepare()
    assert report.ready is False
    assert any(b["kind"] == "executor" for b in report.blockers)
    assert runner.operator is None
