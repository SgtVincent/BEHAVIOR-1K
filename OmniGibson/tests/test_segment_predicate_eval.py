import omnigibson.learning.utils.segment_predicate_eval as segment_predicate_eval
from omnigibson.learning.utils.segment_predicate_eval import (
    build_template_predicates,
    eval_segment_predicates,
    predicate_window_satisfied,
    trace_has_missing_object,
    trace_missing_objects,
)


class _FakeAttachedState:
    def __init__(self, value: bool) -> None:
        self.value = value

    def get_value(self, other) -> bool:
        return self.value


class _FakeObject:
    def __init__(self, name: str, *, attached: bool = False) -> None:
        self.name = name
        self.states = {segment_predicate_eval.AttachedTo: _FakeAttachedState(attached)}


class _FakeRobot:
    arm_names = ["left", "right"]

    def __init__(self, *, grasped: bool) -> None:
        self.grasped = grasped

    def is_grasping(self, arm: str, candidate_obj) -> bool:
        return self.grasped


class _FakeScene:
    def __init__(self, objects: dict[str, _FakeObject]) -> None:
        self.objects = objects

    def object_registry(self, kind: str, name: str):
        assert kind == "name"
        return self.objects.get(name)

    def get_task_metadata(self, key: str):
        return {}


class _FakeGoalHead:
    def __init__(self, body) -> None:
        self.body = body


class _FakeTask:
    activity_name = "attach_a_camera_to_a_tripod"

    def __init__(self) -> None:
        self.ground_goal_state_options = [[_FakeGoalHead(["attached", "camera", "tripod"])]]


class _FakeEnv:
    def __init__(self, *, attached: bool, grasped: bool) -> None:
        self.scene = _FakeScene(
            {
                "camera": _FakeObject("camera", attached=attached),
                "tripod": _FakeObject("tripod"),
            }
        )
        self.robots = [_FakeRobot(grasped=grasped)]
        self.task = _FakeTask()


def test_trace_missing_objects_extracts_diagnostics() -> None:
    trace = [
        {
            "predicate": "grasped(agent,obj)",
            "metric_type": "predicate",
            "desired": False,
            "value": False,
            "satisfied": True,
            "diagnostics": {"missing_object": "obj"},
        }
    ]

    assert trace_has_missing_object(trace) is True
    assert trace_missing_objects(trace) == [
        {
            "predicate": "grasped(agent,obj)",
            "metric_type": "predicate",
            "desired": False,
            "value": False,
            "satisfied": True,
            "missing_object": "obj",
        }
    ]


def test_predicate_window_satisfied_rejects_missing_object_all_of() -> None:
    history = [
        [
            {
                "predicate": "ontop(missing_obj,surface)",
                "desired": False,
                "value": False,
                "satisfied": True,
                "diagnostics": {"missing_object": "missing_obj"},
            }
        ]
    ]

    assert predicate_window_satisfied(history, combine_mode="all_of") is False


def test_predicate_window_satisfied_rejects_missing_object_any_of() -> None:
    history = [
        [
            {
                "predicate": "touching(tool,target)",
                "desired": True,
                "value": True,
                "satisfied": True,
                "diagnostics": {},
            },
            {
                "predicate": "base_to_object(missing_obj)",
                "desired": True,
                "value": False,
                "satisfied": False,
                "diagnostics": {"missing_object": "missing_obj"},
            },
        ]
    ]

    assert predicate_window_satisfied(history, combine_mode="any_of") is False


def test_predicate_window_satisfied_consecutive_requires_trailing_stability() -> None:
    sat = [{"predicate": "open(drawer)", "desired": True, "value": True, "satisfied": True, "diagnostics": {}}]
    unsat = [{"predicate": "open(drawer)", "desired": True, "value": False, "satisfied": False, "diagnostics": {}}]

    assert predicate_window_satisfied([sat, sat, unsat], mode="consecutive", min_consecutive=2) is False
    assert predicate_window_satisfied([unsat, sat, sat], mode="consecutive", min_consecutive=2) is True


def test_attach_release_composite_rejects_attached_but_still_grasped() -> None:
    env = _FakeEnv(attached=True, grasped=True)
    segment = {"skill_description": "release", "object_id": [["camera"]]}

    specs, debug = build_template_predicates("skill", segment, env)
    _, trace = eval_segment_predicates(env, specs)

    assert debug["task_aware_final_relation_predicates"] == ["attached"]
    assert {item["predicate"] for item in trace} == {"grasped(agent,camera)", "attached(camera,tripod)"}
    assert predicate_window_satisfied([trace], combine_mode=debug["combine_mode"]) is False


def test_attach_release_composite_accepts_attached_and_released() -> None:
    env = _FakeEnv(attached=True, grasped=False)
    segment = {"skill_description": "release", "object_id": [["camera"]]}

    specs, debug = build_template_predicates("skill", segment, env)
    _, trace = eval_segment_predicates(env, specs)

    assert {item["predicate"] for item in trace} == {"grasped(agent,camera)", "attached(camera,tripod)"}
    assert predicate_window_satisfied([trace], combine_mode=debug["combine_mode"]) is True
