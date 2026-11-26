"""CPU tests for the concrete MoMa runner: prepare wiring, episode loop,
finalize report and blocker handling. Sim-free duck-typed fakes only.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from omnigibson.moma.config import RunnerConfig
from omnigibson.moma.episode_writer import EpisodeWriter
from omnigibson.moma.runner import MomaRunner
from omnigibson.moma.sdk import SdkEnvironmentAdapter, SdkRobotAccessor
from omnigibson.moma.schema import native23_action

CANS = ["can_of_soda_114", "can_of_soda_115", "can_of_soda_113"]
BUCKET = "trash_can_116"


class _GraspState:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __int__(self):
        return self.value


class FakeRobot:
    arm_names = ["left", "right"]

    def __init__(self):
        self._ag_obj_in_hand = {"left": None, "right": None}
        self._ag_release_counter = {"left": None, "right": None}
        self._ag_grasp_counter = {"left": 0, "right": 0}
        self._ag_obj_constraints = {"left": None, "right": None}
        self._ag_obj_constraint_params = {"left": None, "right": None}
        self._ag_freeze_gripper = {"left": False, "right": False}

    def is_grasping(self, arm="default", candidate_obj=None):
        return _GraspState("FALSE", -1)


class FakeTask:
    def __init__(self, scope):
        self.success = [False]
        self.object_scopes = [scope]


class FakeEnv:
    def __init__(self, scope):
        self.task = FakeTask(scope)
        self.scenes = [object()]
        self.steps = []
        self.reset_calls = 0

    def reset(self, **kwargs):
        self.reset_calls += 1
        return {}

    def step(self, action, n_render_iterations=1):
        self.steps.append(list(action))
        return {}, 0.0, False, False, {}


class FakeExecutor:
    def __init__(self):
        self.plans = []

    def plan(self, args):
        self.plans.append(args)
        return {"action23": native23_action(np.zeros(23)), "certificate": {"primitive": args["primitive"]}}


class FakeS2:
    def __init__(self):
        self.calls = 0

    def decide(self, obs, recent):
        self.calls += 1
        return {"args": {"primitive": "approach_object", "target_object": CANS[0], "max_steps": 120},
                "call_id": "call-1", "latency_s": 0.01, "request_sha256": "ab" * 32}


def _scope():
    return {**{c: SimpleNamespace(name=c) for c in CANS}, BUCKET: SimpleNamespace(name=BUCKET)}


def _obs(terminal: bool):
    obs = {
        "runner_state": "decision_required",
        "decision_required": True,
        "active_request": None,
        "allowed_primitives_schema": {
            "approach_object": {"max_steps": 300, "min_steps": 1, "allowed_objects": CANS,
                                "allowed_stop_conditions": None},
        },
        "task_progress": {
            "bddl_done": {"success": terminal},
            "bucket_ever_held": True,
            "bucket_release_observed": True,
            "can_ever_held": {c: True for c in CANS},
            "can_release_observed": {c: True for c in CANS},
            "all_cans_inside": {c: True for c in CANS},
        },
        "objects": {
            **{c: {"inside_bucket": terminal, "contact_bodies": []} for c in CANS},
            BUCKET: {"on_ground": True, "contact_bodies": ["/floors_0/groundPlane"]},
        },
    }
    return obs


def _make_runner(tmp_path, obs, *, executor=None, s2=None, manifest=None, max_steps=2):
    scope = _scope()
    env = FakeEnv(scope)
    adapter = SdkEnvironmentAdapter(env, 0)
    robot = SdkRobotAccessor(FakeRobot())
    cfg = RunnerConfig(run_id="run-test", output_dir=str(tmp_path), max_steps=max_steps)

    def observe_fn(robot_, cans, bucket, **kw):
        return obs

    def transition_fn(prev, current, can_names):
        return []

    return MomaRunner(
        cfg,
        env_adapter=adapter,
        robot_accessor=robot,
        observer_fn=observe_fn,
        transition_fn=transition_fn,
        executor=executor,
        s2_client=s2,
        plugin_manifest=manifest,
        live_reads={"s2_cfg": {"can_names": CANS, "bucket_name": BUCKET,
                               "allowed_primitives": ["approach_object"],
                               "primitive_caps": {"approach_object": 300}}},
    ), env


def test_prepare_wires_injected_parts_and_builds_operator(tmp_path):
    runner, _ = _make_runner(tmp_path, _obs(False), executor=FakeExecutor(), s2=FakeS2())
    report = runner.prepare()
    assert report.ready is True
    assert report.blockers == []
    assert runner.operator is not None
    assert "executor_injected" in report.bound
    assert report.config_digest


def test_prepare_missing_executor_without_manifest_is_blocked(tmp_path):
    runner, _ = _make_runner(tmp_path, _obs(False), executor=None, s2=FakeS2())
    report = runner.prepare()
    assert report.ready is False
    assert any(b["kind"] == "executor" for b in report.blockers)
    assert runner.operator is None


def test_run_episode_runs_to_step_limit_and_writes_result(tmp_path):
    runner, env = _make_runner(tmp_path, _obs(False), executor=FakeExecutor(), s2=FakeS2(), max_steps=2)
    report = runner.prepare()
    assert report.ready
    result = runner.run_episode()
    assert result["issued_actions"] == 2
    assert result["closed_loop_success"] is False
    assert len(env.steps) == 2
    timeline = (Path(tmp_path) / "run-test" / "timeline.jsonl").read_text().strip().splitlines()
    assert len(timeline) == 2
    saved = json.loads((Path(tmp_path) / "run-test" / "result.json").read_text())
    assert saved["run_id"] == "run-test"


def test_terminal_observation_short_circuits_loop(tmp_path):
    runner, env = _make_runner(tmp_path, _obs(True), executor=FakeExecutor(), s2=FakeS2())
    runner.prepare()
    result = runner.run_episode()
    assert result["issued_actions"] == 1
    assert env.steps == []


def test_one_time_initial_state_restore_called_at_reset(tmp_path):
    runner, _ = _make_runner(tmp_path, _obs(False), executor=FakeExecutor(), s2=FakeS2())
    runner.prepare()
    loaded = []
    runner.run_episode(load_state_fn=lambda: loaded.append(1))
    assert loaded == [1]


def test_run_without_prepare_raises(tmp_path):
    runner, _ = _make_runner(tmp_path, _obs(False), executor=FakeExecutor(), s2=FakeS2())
    try:
        runner.run_episode()
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "prepare" in str(exc)
