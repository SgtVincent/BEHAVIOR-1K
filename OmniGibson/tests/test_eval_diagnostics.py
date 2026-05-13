from omnigibson.learning.utils.eval_diagnostics import ENV_TASK_SUCCESS_BEFORE_SEGMENT_SUCCESS
from omnigibson.learning.utils.eval_diagnostics import RESULT_LIKELY_PROXY_FALSE_POSITIVE
from omnigibson.learning.utils.eval_diagnostics import RESULT_SHORT_PROXY_SUCCESS
from omnigibson.learning.utils.eval_diagnostics import RESULT_SHORT_VIDEO_PROBLEM
from omnigibson.learning.utils.eval_diagnostics import build_pre_satisfied_start_result
from omnigibson.learning.utils.eval_diagnostics import classify_short_proxy_success
from omnigibson.learning.utils.eval_diagnostics import classify_short_video_success
from omnigibson.learning.utils.eval_diagnostics import build_termination_summary
from omnigibson.learning.utils.eval_diagnostics import finalize_segment_predicate_result_type
from omnigibson.learning.utils.eval_diagnostics import update_segment_env_termination_telemetry


def test_build_termination_summary_tracks_prompt_debug_and_stuck_motion() -> None:
    summary = build_termination_summary(
        terminated=False,
        truncated=True,
        env_current_step=6,
        done_info={"success": False, "stuck_motion": True},
        prompt_debug={"selected_skill": None, "fallback_to_task_prompt": True, "fallback_reason": "no_skill_match"},
        generated_subtask="turn to the oven",
    )

    assert summary["termination_reason"] == "stuck_motion"
    assert summary["truncated"] is True
    assert summary["env_current_step"] == 6
    assert summary["prompt_debug"]["fallback_reason"] == "no_skill_match"
    assert summary["generated_subtask"] == "turn to the oven"


def test_build_pre_satisfied_start_result_uses_distinct_result_type() -> None:
    rollout, result_type = build_pre_satisfied_start_result(
        max_steps=25,
        window_mode="anytime",
        combine_mode="all_of",
        start_all_satisfied=True,
        require_unsatisfied_at_start=True,
    )

    assert result_type == "pre_satisfied_start"
    assert rollout["termination_reason"] == "pre_satisfied_start"
    assert rollout["rollout_attempted"] is False
    assert rollout["final_step"] == 0
    assert rollout["start_all_satisfied"] is True
    assert rollout["terminated"] is False
    assert rollout["truncated"] is False
    assert rollout["env_done_success"] is None
    assert rollout["env_terminal_debug"]["termination_reason"] == "pre_satisfied_start"
    assert rollout["env_terminal_debug"]["done_info"] == {}


def test_segment_env_termination_telemetry_records_first_env_success_step() -> None:
    telemetry = {
        "env_terminated_seen": False,
        "env_done_success_seen": False,
        "first_env_terminated_step": None,
        "first_env_done_success_step": None,
        "env_termination_count": 0,
    }

    update_segment_env_termination_telemetry(telemetry, step=1, terminated=True, env_done_success=True)
    update_segment_env_termination_telemetry(telemetry, step=2, terminated=True, env_done_success=True)

    assert telemetry["env_terminated_seen"] is True
    assert telemetry["env_done_success_seen"] is True
    assert telemetry["first_env_terminated_step"] == 1
    assert telemetry["first_env_done_success_step"] == 1
    assert telemetry["env_termination_count"] == 2


def test_segment_predicate_result_type_marks_env_success_before_segment_success() -> None:
    assert (
        finalize_segment_predicate_result_type(
            success=False,
            result_type="timeout",
            env_done_success_seen=True,
        )
        == ENV_TASK_SUCCESS_BEFORE_SEGMENT_SUCCESS
    )
    assert (
        finalize_segment_predicate_result_type(
            success=True,
            result_type="predicate_satisfied",
            env_done_success_seen=True,
        )
        == "predicate_satisfied"
    )


def test_transfer_pose_proxy_short_success_is_likely_false_positive() -> None:
    success, result_type, diagnostics = classify_short_proxy_success(
        success=True,
        result_type="predicate_satisfied",
        final_step=6,
        max_steps=636,
        metric_debug={
            "metric_family": "transfer_pose_proxy",
            "min_rollout_steps_for_success": 30,
            "min_success_step_fraction": 0.10,
            "likely_false_positive_on_short_success": True,
        },
    )

    assert success is False
    assert result_type == RESULT_LIKELY_PROXY_FALSE_POSITIVE
    assert diagnostics["short_proxy_success"] is True
    assert diagnostics["likely_proxy_false_positive"] is True
    assert diagnostics["short_success_required_step"] == 64


def test_geometry_base_facing_short_success_is_caution_not_likely_false_positive() -> None:
    success, result_type, diagnostics = classify_short_proxy_success(
        success=True,
        result_type="predicate_satisfied",
        final_step=6,
        max_steps=636,
        metric_debug={
            "metric_family": "geometry_base_facing",
            "short_success_caution_steps": 30,
            "short_success_caution_step_fraction": 0.10,
        },
    )

    assert success is True
    assert result_type == RESULT_SHORT_PROXY_SUCCESS
    assert diagnostics["short_proxy_success"] is True
    assert diagnostics["likely_proxy_false_positive"] is False
    assert diagnostics["short_success_required_step"] == 64


def test_generic_short_video_success_is_rejected() -> None:
    success, result_type, diagnostics = classify_short_video_success(
        success=True,
        result_type="predicate_satisfied",
        final_step=8,
        max_steps=198,
        min_rollout_steps_for_success=150,
    )

    assert success is False
    assert result_type == RESULT_SHORT_VIDEO_PROBLEM
    assert diagnostics["short_video_problem"] is True
    assert diagnostics["short_video_min_rollout_steps"] == 150


def test_generic_short_video_threshold_is_not_capped_by_max_steps() -> None:
    success, result_type, diagnostics = classify_short_video_success(
        success=True,
        result_type="predicate_satisfied",
        final_step=120,
        max_steps=120,
        min_rollout_steps_for_success=150,
    )

    assert success is False
    assert result_type == RESULT_SHORT_VIDEO_PROBLEM
    assert diagnostics["short_video_problem"] is True
    assert diagnostics["short_video_min_rollout_steps"] == 150
