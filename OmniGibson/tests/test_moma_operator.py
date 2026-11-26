"""CPU tests for the MoMa episode operator: lifecycle guards, real consumer
grasp records, boundary decision and step recording.

Sim-free: env/robot/observer/executor/S2 are duck-typed fakes.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from omnigibson.moma.operator import (
    EpisodeConfig,
    MomaEpisodeOperator,
    EpisodeImmutableAfterStart,
    ExecutorAdapterMissing,
    ExecutorPlanRejected,
)
from omnigibson.moma.sdk import SdkEnvironmentAdapter, SdkRobotAccessor
from omnigibson.moma.schema import NATIVE23_JOINT, native23_action
from omnigibson.moma.s2.client import ChatCompletionsClient
from omnigibson.moma.s2.rule_service import RuleS2Service

CANS = ["can_of_soda_114", "can_of_soda_115", "can_of_soda_113"]
BUCKET = "trash_can_116"

CFG = {
    "can_names": CANS,
    "bucket_name": BUCKET,
    "allowed_primitives": ["approach_object", "stabilize_and_brake"],
    "primitive_caps": {"approach_object": 300, "stabilize_and_brake": 1800},
}

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


def _loopback_obs():
    return {
        "runner_state": "decision_required",
        "decision_required": True,
        "active_request": None,
        "robot_state": {"arms": {
            "left": {"object_in_hand": None, "constraint_present": False,
                     "release_counter": None, "valid_sdk_hold": False},
            "right": {"object_in_hand": None, "constraint_present": False,
                      "release_counter": None, "valid_sdk_hold": False},
        }},
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


class _GraspState:
    def __init__(self, name: str, value: int) -> None:
        self.name = name
        self.value = value

    def __int__(self) -> int:
        return self.value


class FakeRobot:
    arm_names = ["left", "right"]

    def __init__(self) -> None:
        self._ag_obj_in_hand = {"left": None, "right": None}
        self._ag_release_counter = {"left": None, "right": None}
        self._ag_grasp_counter = {"left": 0, "right": 0}
        self._ag_obj_constraints = {"left": None, "right": None}
        self._ag_obj_constraint_params = {"left": None, "right": None}
        self._ag_freeze_gripper = {"left": False, "right": False}

    def is_grasping(self, arm="default", candidate_obj=None):
        return _GraspState("FALSE", -1)


class FakeTask:
    def __init__(self, scope: dict) -> None:
        self.success = [False]
        self.object_scopes = [scope]


class FakeEnv:
    def __init__(self, scope: dict) -> None:
        self.task = FakeTask(scope)
        self.scenes = [object()]
        self.steps: list[list[float]] = []
        self.reset_calls = 0

    def reset(self, **kwargs):
        self.reset_calls += 1
        return {}

    def step(self, action, n_render_iterations=1):
        self.steps.append(list(action))
        return {}, 0.0, False, False, {}


class FakeExecutor:
    def __init__(self) -> None:
        self.plans: list[dict] = []

    def plan(self, args):
        self.plans.append(args)
        command = np.full(23, 0.05, dtype=np.float64)
        return {"action23": native23_action(command), "certificate": {"primitive": args["primitive"]}}


class FakeS2:
    def __init__(self) -> None:
        self.calls = 0

    def decide(self, obs, recent):
        self.calls += 1
        return {
            "args": {"primitive": "approach_object", "target_object": CANS[0], "max_steps": 120},
            "call_id": "call-1",
            "latency_s": 0.01,
            "request_sha256": "ab" * 32,
        }


def _scope():
    return {**{c: SimpleNamespace(name=c) for c in CANS}, BUCKET: SimpleNamespace(name=BUCKET)}


def _terminal_obs():
    return {
        "runner_state": "decision_required",
        "decision_required": True,
        "active_request": None,
        "robot_state": {"arms": {
            "left": {"object_in_hand": None, "constraint_present": False, "release_counter": None, "valid_sdk_hold": False},
            "right": {"object_in_hand": None, "constraint_present": False, "release_counter": None, "valid_sdk_hold": False},
        }},
        "allowed_primitives_schema": {
            "approach_object": {"max_steps": 300, "min_steps": 1, "allowed_objects": CANS, "allowed_stop_conditions": None},
        },
        "task_progress": {
            "bddl_done": {"success": True},
            "bucket_ever_held": True,
            "bucket_release_observed": True,
            "can_ever_held": {c: True for c in CANS},
            "can_release_observed": {c: True for c in CANS},
            "all_cans_inside": {c: True for c in CANS},
        },
        "objects": {
            **{c: {"inside_bucket": True, "contact_bodies": []} for c in CANS},
            BUCKET: {"on_ground": True, "contact_bodies": ["/floors_0/groundPlane"]},
        },
    }


_UNSET = object()


def _make_operator(obs, executor=_UNSET, s2=_UNSET):
    scope = _scope()
    env = FakeEnv(scope)
    adapter = SdkEnvironmentAdapter(env, 0)
    robot = SdkRobotAccessor(FakeRobot())
    if executor is _UNSET:
        executor = FakeExecutor()
    if s2 is _UNSET:
        s2 = FakeS2()

    def observe_fn(robot_, cans, bucket, pose_fn=None, inside_type=None, contact_type=None):
        return obs

    def transition_fn(prev, current, can_names):
        return []

    cfg = EpisodeConfig(run_id="run-test", can_names=CANS, bucket_name=BUCKET, max_steps=10)
    op = MomaEpisodeOperator(
        cfg,
        adapter,
        robot,
        observe_fn=observe_fn,
        transition_fn=transition_fn,
        executor=executor,
        s2_client=s2,
        live_reads={"s2_cfg": CFG},
    )
    return op, env, executor, s2


def test_reset_clears_episode_bookkeeping():
    op, env, _, _ = _make_operator({})
    op.reset_episode()
    assert env.reset_calls == 1
    assert op.step_count == 0
    assert op.records == []


def test_one_time_initial_load_then_post_start_reset_blocked():
    loaded = []
    op, _, _, _ = _make_operator({})
    op.reset_episode(perform_initial_load=True, load_state_fn=lambda: loaded.append(1))
    assert loaded == [1]

    with pytest.raises(EpisodeImmutableAfterStart, match="only once"):
        op.restore_initial_state(lambda: loaded.append(2))

    op.step()
    with pytest.raises(EpisodeImmutableAfterStart, match="already started"):
        op.reset_episode()
    with pytest.raises(EpisodeImmutableAfterStart, match="forbidden"):
        op.restore_initial_state(lambda: loaded.append(3))


def test_step_records_real_consumer_grasp_and_tagged_action():
    obs = _terminal_obs()
    obs["task_progress"]["bddl_done"]["success"] = False  # not terminal: exercise the step path
    op, env, executor, s2 = _make_operator(obs)
    op.reset_episode()
    record = op.step()

    assert record.action_label == NATIVE23_JOINT
    assert record.action_dim == 23
    assert len(env.steps) == 1 and len(env.steps[0]) == 23
    assert record.s2["call_id"] == "call-1"
    assert record.arms["left"]["object_in_hand"] is None
    assert executor.plans[0]["primitive"] == "approach_object"
    assert s2.calls == 1


def test_terminal_observation_short_circuits_executor():
    op, env, executor, s2 = _make_operator(_terminal_obs())
    op.reset_episode()
    record = op.step()
    assert record.terminal is True
    assert env.steps == []
    assert executor.plans == []
    assert s2.calls == 0


def test_observe_missing_scope_objects_return_blocked_not_keyerror():
    scope = {BUCKET: SimpleNamespace(name=BUCKET)}  # all cans absent from scope
    env = FakeEnv(scope)
    adapter = SdkEnvironmentAdapter(env, 0)
    robot = SdkRobotAccessor(FakeRobot())
    op = MomaEpisodeOperator(
        EpisodeConfig(run_id="run-blocked", can_names=CANS, bucket_name=BUCKET, max_steps=10),
        adapter,
        robot,
        observe_fn=lambda *a, **k: {},
        transition_fn=lambda *a: [],
        executor=FakeExecutor(),
        s2_client=FakeS2(),
        live_reads={"s2_cfg": CFG},
    )
    obs = op.observe()
    assert obs["runner_state"] == "blocked"
    assert obs["decision_required"] is False
    for can in CANS:
        assert obs["objects"][can]["status"] == "BLOCKED"
        assert obs["objects"][can]["error"] == "object_not_in_scope"
    assert set(obs["robot_state"]["arms"]) == {"left", "right"}


def test_blocked_observation_short_circuits_before_s2_and_step():
    scope = {BUCKET: SimpleNamespace(name=BUCKET)}  # cans absent: blocked scope
    env = FakeEnv(scope)
    adapter = SdkEnvironmentAdapter(env, 0)
    robot = SdkRobotAccessor(FakeRobot())
    s2 = FakeS2()
    op = MomaEpisodeOperator(
        EpisodeConfig(run_id="run-blocked-step", can_names=CANS, bucket_name=BUCKET, max_steps=10),
        adapter,
        robot,
        observe_fn=lambda *a, **k: {},
        transition_fn=lambda *a: [],
        executor=FakeExecutor(),
        s2_client=s2,
        live_reads={"s2_cfg": CFG},
    )
    op.reset_episode()
    record = op.step()

    assert record.terminal is False
    assert record.action_label is None
    assert env.steps == []
    assert s2.calls == 0
    assert any(e["type"] == "boundary_blocked" for e in record.events)


def test_closed_loop_requires_bddl_success_and_terminal_gate():
    op, _, _, _ = _make_operator(_terminal_obs())
    op.reset_episode()
    # FakeEnv task.success stays False: BDDL alone is not enough, and a False
    # BDDL dict must never be truthy-coerced.
    assert op.closed_loop_success() is False


def test_loopback_http_decision_reaches_operator_consumer():
    """Real local HTTP: rule service -> S2 client -> operator consumer -> env.step."""
    service = RuleS2Service(model_name=MODEL, cfg=S2_CFG)
    service.start()
    try:
        client = ChatCompletionsClient(
            base_url=service.base_url, model=MODEL, prompt_text="test",
            cfg=S2_CFG, history_turns=8, timeout_s=5.0,
        )
        op, env, executor, _ = _make_operator(_loopback_obs(), s2=client)
        op.reset_episode()
        record = op.step()

        assert record.action_label == NATIVE23_JOINT
        assert record.action_dim == 23
        assert len(env.steps) == 1 and len(env.steps[0]) == 23
        assert executor.plans[0]["primitive"] == "continue_nominal_tracking"
        assert executor.plans[0]["target_object"] == CANS[0]
        assert record.s2["call_id"].startswith("call_")
        assert record.s2["request_sha256"]
        assert len(service.requests) == 1
    finally:
        service.stop()


def test_missing_executor_blocks_before_decision_and_step():
    """No MP adapter: explicit blocker, no S2 round-trip, no env.step."""
    service = RuleS2Service(model_name=MODEL, cfg=S2_CFG)
    service.start()
    try:
        client = ChatCompletionsClient(
            base_url=service.base_url, model=MODEL, prompt_text="test",
            cfg=S2_CFG, history_turns=8, timeout_s=5.0,
        )
        op, env, _, _ = _make_operator(_loopback_obs(), executor=None, s2=client)
        op.reset_episode()
        with pytest.raises(ExecutorAdapterMissing):
            op.step()
        assert env.steps == []
        assert service.requests == []
    finally:
        service.stop()


class _PlanReturningExecutor:
    def __init__(self, plan):
        self.plan_value = plan
        self.calls = 0

    def plan(self, args):
        self.calls += 1
        return self.plan_value


class _RaisingExecutor:
    def plan(self, args):
        raise RuntimeError("solver unreachable target")


def _plan_test_operator(executor):
    op, env, _, _ = _make_operator(_loopback_obs(), executor=executor)
    op.reset_episode()
    return op, env


def test_plan_missing_action23_rejected_without_step():
    op, env = _plan_test_operator(_PlanReturningExecutor({"certificate": {"ok": True}}))
    with pytest.raises(ExecutorPlanRejected, match="action23_missing"):
        op.step()
    assert env.steps == []


def test_plan_zero_vector_rejected_as_placeholder():
    zero = native23_action(np.zeros(23))
    op, env = _plan_test_operator(
        _PlanReturningExecutor({"action23": zero, "certificate": {"ok": True}})
    )
    with pytest.raises(ExecutorPlanRejected, match="zero_action_placeholder"):
        op.step()
    assert env.steps == []


def test_plan_missing_certificate_rejected():
    command = np.full(23, 0.05)
    op, env = _plan_test_operator(
        _PlanReturningExecutor({"action23": native23_action(command)})
    )
    with pytest.raises(ExecutorPlanRejected, match="certificate_missing"):
        op.step()
    assert env.steps == []


def test_executor_exception_reclassified_without_step():
    op, env = _plan_test_operator(_RaisingExecutor())
    with pytest.raises(ExecutorPlanRejected, match="executor_raised"):
        op.step()
    assert env.steps == []
