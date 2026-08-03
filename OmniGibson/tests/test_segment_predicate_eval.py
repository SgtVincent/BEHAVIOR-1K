import omnigibson.learning.utils.segment_predicate_eval as segment_predicate_eval
from omnigibson.learning.utils.segment_predicate_eval import (
    build_template_predicates,
    eval_segment_predicates,
    predicate_window_satisfied,
    trace_has_missing_object,
    trace_missing_objects,
)


class _FakeBoolState:
    def __init__(self, value: bool) -> None:
        self.value = value

    def get_value(self, *args) -> bool:
        return self.value


class _FakeObject:
    def __init__(
        self,
        name: str,
        *,
        attached: bool = False,
        ontop: bool = False,
        inside: bool = False,
        nextto: bool = False,
        open: bool = False,
        position=(0.0, 0.0, 0.0),
        quat=(0.0, 0.0, 0.0, 1.0),
    ) -> None:
        self.name = name
        self._position = position
        self._quat = quat
        self.states = {
            segment_predicate_eval.AttachedTo: _FakeBoolState(attached),
            segment_predicate_eval.OnTop: _FakeBoolState(ontop),
            segment_predicate_eval.Inside: _FakeBoolState(inside),
            segment_predicate_eval.NextTo: _FakeBoolState(nextto),
            segment_predicate_eval.Open: _FakeBoolState(open),
        }

    def get_position_orientation(self):
        return self._position, self._quat


class _FakeRobot:
    arm_names = ["left", "right"]

    def __init__(
        self,
        *,
        grasped: bool,
        position=(0.0, 0.0, 0.0),
        quat=(0.0, 0.0, 0.0, 1.0),
    ) -> None:
        self.grasped = grasped
        self._position = position
        self._quat = quat

    def is_grasping(self, arm: str, candidate_obj) -> bool:
        return self.grasped

    def get_position_orientation(self):
        return self._position, self._quat


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
    def __init__(
        self,
        *,
        attached: bool = False,
        grasped: bool = False,
        objects: dict[str, _FakeObject] | None = None,
    ) -> None:
        self.scene = _FakeScene(
            objects or {"camera": _FakeObject("camera", attached=attached), "tripod": _FakeObject("tripod")}
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


def test_predicate_window_satisfied_ignores_stale_history_before_min_index_anytime() -> None:
    sat = [{"predicate": "open(drawer)", "desired": True, "value": True, "satisfied": True, "diagnostics": {}}]
    unsat = [{"predicate": "open(drawer)", "desired": True, "value": False, "satisfied": False, "diagnostics": {}}]

    assert predicate_window_satisfied([sat, unsat, unsat], mode="anytime") is True
    assert predicate_window_satisfied([sat, unsat, unsat], mode="anytime", min_history_index=1) is False


def test_predicate_window_satisfied_allows_witness_at_min_index() -> None:
    sat = [{"predicate": "open(drawer)", "desired": True, "value": True, "satisfied": True, "diagnostics": {}}]
    unsat = [{"predicate": "open(drawer)", "desired": True, "value": False, "satisfied": False, "diagnostics": {}}]

    assert predicate_window_satisfied([unsat, sat], mode="anytime", min_history_index=1) is True


def test_generated_product_name_alias_resolves_source_object() -> None:
    env = _FakeEnv(
        objects={
            "zucchini_208": _FakeObject("zucchini_208"),
            "knife_1": _FakeObject("knife_1", nextto=True),
        }
    )

    specs, debug = build_template_predicates(
        "skill",
        {
            "skill_description": "chop",
            "manipulating_object_id": "knife_1",
            "object_id": [["knife_1", "half_zucchini_208_1"]],
        },
        env,
    )
    _, trace = eval_segment_predicates(env, specs)

    assert debug["metric_family"] == "contact_effect_proxy"
    assert len(specs) == 1
    assert trace[0]["predicate"] == "touching(knife_1,half_zucchini_208_1)"
    assert "missing_object" not in trace[0].get("diagnostics", {})


def test_predicate_window_satisfied_consecutive_does_not_cross_min_index() -> None:
    sat = [{"predicate": "open(drawer)", "desired": True, "value": True, "satisfied": True, "diagnostics": {}}]
    unsat = [{"predicate": "open(drawer)", "desired": True, "value": False, "satisfied": False, "diagnostics": {}}]

    assert predicate_window_satisfied(
        [unsat, sat, sat],
        mode="consecutive",
        min_consecutive=2,
        min_history_index=2,
    ) is False
    assert predicate_window_satisfied(
        [unsat, sat, sat, sat],
        mode="consecutive",
        min_consecutive=2,
        min_history_index=2,
    ) is True


def test_predicate_window_satisfied_last_k_ignores_pre_min_index_history() -> None:
    sat = [{"predicate": "open(drawer)", "desired": True, "value": True, "satisfied": True, "diagnostics": {}}]
    unsat = [{"predicate": "open(drawer)", "desired": True, "value": False, "satisfied": False, "diagnostics": {}}]

    assert predicate_window_satisfied([sat, unsat, unsat], mode="last_k", last_k=3) is True
    assert predicate_window_satisfied([sat, unsat, unsat], mode="last_k", last_k=3, min_history_index=1) is False


def test_pour_payload_role_does_not_fallback_to_container() -> None:
    env = _FakeEnv(objects={"pitcher": _FakeObject("pitcher"), "bowl": _FakeObject("bowl")})
    segment = {
        "skill_description": "pour",
        "manipulating_object_id": "pitcher",
        "object_id": [["right", "bowl"]],
    }

    specs, debug = build_template_predicates("skill", segment, env)

    assert specs == []
    assert debug["metric_family"] == "relation_transfer_proxy"
    assert debug["missing_template_roles"] == [
        {"metric_type": "predicate", "metric_name": "ontop", "role": "payload_or_obj"},
        {"metric_type": "predicate", "metric_name": "inside", "role": "payload_or_obj"},
    ]


def test_hand_over_requires_nextto_when_target_is_available() -> None:
    env = _FakeEnv(
        grasped=True,
        objects={
            "meat": _FakeObject("meat", nextto=False, position=(1.0, 2.0, 0.5)),
            "receiver": _FakeObject("receiver"),
        },
    )
    segment = {"skill_description": "hand over", "object_id": [["meat", "receiver"]]}

    specs, debug = build_template_predicates("skill", segment, env)
    _, trace = eval_segment_predicates(env, specs)

    assert debug["metric_family"] == "transfer_pose_proxy"
    assert {item["predicate"] for item in trace} == {
        "object_pose_match(meat)",
        "grasped(agent,meat)",
        "nextto(meat,receiver)",
    }
    nextto_spec = next(spec for spec in specs if spec.name == "nextto")
    assert nextto_spec.params["optional"] is True
    assert predicate_window_satisfied([trace], combine_mode=debug["combine_mode"]) is False


def test_hand_over_skips_optional_nextto_when_target_is_unavailable() -> None:
    env = _FakeEnv(grasped=True, objects={"meat": _FakeObject("meat", position=(1.0, 2.0, 0.5))})
    segment = {"skill_description": "hand over", "object_id": [["meat"]]}

    specs, debug = build_template_predicates("skill", segment, env)

    assert "missing_template_roles" not in debug
    assert [(spec.metric_type, spec.name, spec.args) for spec in specs] == [
        ("object_pose_match", "object_pose_match", ["meat"]),
        ("predicate", "grasped", ["agent", "meat"]),
    ]


def test_articulation_close_registry_requires_five_step_trailing_stability() -> None:
    env = _FakeEnv(objects={"lid": _FakeObject("lid", open=False)})
    segment = {"skill_description": "close lid", "object_id": [["lid"]]}

    specs, debug = build_template_predicates("skill", segment, env)
    _, closed_trace = eval_segment_predicates(env, specs)

    env.scene.objects["lid"] = _FakeObject("lid", open=True)
    _, open_trace = eval_segment_predicates(env, specs)

    assert debug["metric_family"] == "articulation_close"
    assert debug["success_min_consecutive"] == 5
    assert predicate_window_satisfied(
        [closed_trace, closed_trace, closed_trace, closed_trace],
        mode="consecutive",
        min_consecutive=debug["success_min_consecutive"],
        combine_mode=debug["combine_mode"],
    ) is False
    assert predicate_window_satisfied(
        [open_trace, closed_trace, closed_trace, closed_trace, closed_trace, closed_trace],
        mode="consecutive",
        min_consecutive=debug["success_min_consecutive"],
        combine_mode=debug["combine_mode"],
    ) is True


def test_articulation_open_lid_registry_requires_trailing_stability() -> None:
    env = _FakeEnv(objects={"lid": _FakeObject("lid", open=True)})
    segment = {"skill_description": "open lid", "object_id": [["lid"]]}

    specs, debug = build_template_predicates("skill", segment, env)
    _, open_trace = eval_segment_predicates(env, specs)

    env.scene.objects["lid"] = _FakeObject("lid", open=False)
    _, closed_trace = eval_segment_predicates(env, specs)

    assert debug["metric_family"] == "articulation_open"
    assert debug["success_min_consecutive"] == 5
    assert predicate_window_satisfied(
        [open_trace, open_trace, open_trace, open_trace],
        mode="consecutive",
        min_consecutive=debug["success_min_consecutive"],
        combine_mode=debug["combine_mode"],
    ) is False
    assert predicate_window_satisfied(
        [closed_trace, open_trace, open_trace, open_trace, open_trace, open_trace],
        mode="consecutive",
        min_consecutive=debug["success_min_consecutive"],
        combine_mode=debug["combine_mode"],
    ) is True


def test_pull_tray_registry_requires_trailing_stability() -> None:
    env = _FakeEnv(objects={"oven": _FakeObject("oven", open=True)})
    segment = {"skill_description": "pull tray", "object_id": [["oven"]]}

    specs, debug = build_template_predicates("skill", segment, env)
    _, open_trace = eval_segment_predicates(env, specs)

    env.scene.objects["oven"] = _FakeObject("oven", open=False)
    _, closed_trace = eval_segment_predicates(env, specs)

    assert debug["metric_family"] == "articulation_open_proxy"
    assert debug["success_min_consecutive"] == 5
    assert predicate_window_satisfied(
        [open_trace, open_trace, open_trace, open_trace],
        mode="consecutive",
        min_consecutive=debug["success_min_consecutive"],
        combine_mode=debug["combine_mode"],
    ) is False
    assert predicate_window_satisfied(
        [closed_trace, open_trace, open_trace, open_trace, open_trace, open_trace],
        mode="consecutive",
        min_consecutive=debug["success_min_consecutive"],
        combine_mode=debug["combine_mode"],
    ) is True


def test_attach_task_release_accepts_bound_goal_while_still_grasped() -> None:
    env = _FakeEnv(attached=True, grasped=True)
    segment = {"skill_description": "release", "object_id": [["camera"]]}

    specs, debug = build_template_predicates("skill", segment, env)
    _, trace = eval_segment_predicates(env, specs)

    assert debug["task_aware_final_relation_predicates"] == ["attached"]
    assert {item["predicate"] for item in trace} == {"attached(camera,tripod)"}
    assert predicate_window_satisfied([trace], combine_mode=debug["combine_mode"]) is True


def test_attach_task_release_still_requires_bound_attached_goal() -> None:
    env = _FakeEnv(attached=False, grasped=False)
    segment = {"skill_description": "release", "object_id": [["camera"]]}

    specs, debug = build_template_predicates("skill", segment, env)
    _, trace = eval_segment_predicates(env, specs)

    assert {item["predicate"] for item in trace} == {"attached(camera,tripod)"}
    assert predicate_window_satisfied([trace], combine_mode=debug["combine_mode"]) is False
