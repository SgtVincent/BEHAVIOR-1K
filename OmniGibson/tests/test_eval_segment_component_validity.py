from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

import omnigibson.learning.eval_segment as eval_segment
from omnigibson.learning.transition_state_restore import (
    STATE_INVALID_ROLLOUT_COMPONENTS,
    build_restore_component_validity,
)


class _NoRolloutPolicy:
    start_frame = None
    end_frame = None

    def reset(self):
        raise AssertionError("component-invalid segment must return before policy rollout")


class _FakeEvaluator:
    def __init__(self, restore_debug):
        self.policy = _NoRolloutPolicy()
        self.env = SimpleNamespace(task=SimpleNamespace(ground_goal_state_options=[[object()]]))
        self.current_rawdata_hdf5 = None
        self.current_primitive_state_cache = None
        self._last_restore_debug = restore_debug
        # run_single_segment resolves the Gate2 rollout start frame before it can know
        # whether the segment is component-evaluable, so even a segment that returns
        # early must be able to read config. Defaults keep this test on the plain path:
        # no restore override, no perturbation, no object-metric capture, no rollout.
        self.cfg = OmegaConf.create(
            {
                "restore_frame_override": None,
                "perturb_pose": False,
                "compute_object_metrics": False,
                "write_video": False,
                "save_rollout": False,
            }
        )

    def load_demo_lowdim_data(self, demo_id):
        return []

    def load_rawdata_hdf5(self, demo_id):
        return object()

    def load_primitive_state_cache(self, demo_id):
        raise AssertionError("rawdata fixture is available")


@pytest.mark.parametrize(
    ("start_value", "expected_activation_success"),
    [(False, True), (True, False)],
)
def test_run_single_segment_separates_end_state_from_activation_and_rollout(
    monkeypatch, start_value, expected_activation_success
):
    validity = build_restore_component_validity(
        restored=True,
        rigid_state_valid=True,
        assisted_grasp_state_valid=False,
        particle_state_valid=False,
        historical_asset_identity_verified=False,
        source_assisted_grasp_state_present=False,
        source_system_count=1,
    )
    restore_debug = {
        "selected_method": "rawdata",
        "component_validity": validity,
        "rawdata": {"component_validity": validity},
    }
    evaluator = _FakeEvaluator(restore_debug)
    segment = {
        "frame_duration": [10, 20],
        "skill_description": ["open door"],
        "object_id": [["cabinet"]],
    }
    predicate_spec = SimpleNamespace(
        metric_type="predicate",
        name="open",
        args=["cabinet"],
        desired=True,
        source="registry",
        params={"metric_family": "articulation_open"},
    )

    monkeypatch.setattr(
        eval_segment,
        "get_segment",
        lambda evaluator, demo_id, segment_level, segment_idx: (segment, {}),
    )
    monkeypatch.setattr(
        eval_segment,
        "restore_and_eval_predicates",
        lambda evaluator, frame_idx: (True, "rawdata", []),
    )
    monkeypatch.setattr(
        eval_segment,
        "build_template_predicates",
        lambda segment_level, segment, env: (
            [predicate_spec],
            {
                "metric_family": "articulation_open",
                "combine_mode": "all_of",
                "require_unsatisfied_at_start": True,
            },
        ),
    )
    boundary_traces = iter(
        [
            [
                {
                    "predicate": "open(cabinet)",
                    "metric_type": "predicate",
                    "desired": True,
                    "value": start_value,
                    "satisfied": start_value,
                    "source": "registry",
                }
            ],
            [
                {
                    "predicate": "open(cabinet)",
                    "metric_type": "predicate",
                    "desired": True,
                    "value": True,
                    "satisfied": True,
                    "source": "registry",
                }
            ],
        ]
    )

    def fake_eval_segment_predicates(env, specs):
        trace = next(boundary_traces)
        return {"open(cabinet)": bool(trace[0]["value"])}, trace

    monkeypatch.setattr(
        eval_segment,
        "eval_segment_predicates",
        fake_eval_segment_predicates,
    )
    monkeypatch.setattr(
        eval_segment,
        "build_auto_mined_predicates",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(eval_segment, "_capture_review_frame", lambda *args, **kwargs: None)

    result = eval_segment.run_single_segment(
        evaluator=evaluator,
        demo_id="00000010",
        segment_level="skill",
        segment_idx=0,
        success_mode="segment_predicates",
    )

    assert result["result_type"] == STATE_INVALID_ROLLOUT_COMPONENTS
    assert result["success"] is None
    assert result["boundary_evaluation_eligible"] is True
    assert result["boundary_end_state_aggregation_eligible"] is True
    assert result["boundary_evaluation"]["end_state_evaluable"] is True
    assert result["boundary_evaluation"]["end_state_aggregation_eligible"] is True
    assert (
        result["boundary_evaluation"]["end_state_result_type"]
        == "boundary_end_state_success"
    )
    assert result["boundary_evaluation"]["end_state_success"] is True
    assert result["boundary_evaluation"]["require_unsatisfied_at_start"] is True
    assert result["boundary_evaluation"]["activation_evaluable"] is True
    assert result["boundary_evaluation"]["activation_success"] is expected_activation_success
    expected_activation_type = (
        "boundary_activation_success"
        if expected_activation_success
        else "boundary_activation_failure"
    )
    assert result["boundary_evaluation"]["activation_result_type"] == expected_activation_type
    assert result["boundary_evaluation"]["start_satisfied"] is start_value
    assert result["boundary_evaluation"]["end_satisfied"] is True
    assert result["boundary_evaluation"]["start_trace"][0]["value"] is start_value
    assert result["boundary_evaluation"]["end_trace"][0]["value"] is True
    assert result["predicate_debug"]["template_trace_start"][0]["value"] is start_value
    assert result["predicate_debug"]["template_trace_end"][0]["value"] is True
    assert result["rollout_evaluation_eligible"] is False
    assert result["component_evaluability"]["boundary_evaluable"] is True
    assert result["component_evaluability"]["rollout_evaluable"] is False
    assert result["component_evaluability"]["rollout_missing_components"] == [
        "assisted_grasp_state"
    ]
    assert result["rollout"]["rollout_attempted"] is False
    assert result["model_evaluated"] is False
    assert result["model_failure_eligible"] is False
    assert result["aggregation_eligible"] is False
    # Gate2 telemetry must survive on this path too: every result produced after the
    # rollout-start restore records the frame the rollout would have started from.
    assert result["rollout_start_frame"] == 10
    assert result["rollout_start_was_overridden"] is False
    assert result["pose_perturbation"] is None
