from __future__ import annotations

from copy import deepcopy
from typing import Any


ENV_TASK_SUCCESS_BEFORE_SEGMENT_SUCCESS = "env_task_success_before_segment_success"
RESULT_SHORT_PROXY_SUCCESS = "short_proxy_success"
RESULT_LIKELY_PROXY_FALSE_POSITIVE = "likely_proxy_false_positive"
RESULT_SHORT_VIDEO_PROBLEM = "short_video_problem"


def build_termination_summary(
    *,
    terminated: bool,
    truncated: bool,
    env_current_step: int,
    done_info: dict[str, Any] | None,
    prompt_debug: dict[str, Any] | None,
    generated_subtask: str | None,
) -> dict[str, Any]:
    done_info = deepcopy(done_info or {})
    prompt_debug = deepcopy(prompt_debug)
    if done_info.get("stuck_motion"):
        termination_reason = "stuck_motion"
    elif truncated:
        termination_reason = "truncated"
    elif terminated:
        termination_reason = "terminated"
    else:
        termination_reason = "running"
    return {
        "termination_reason": termination_reason,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "env_current_step": int(env_current_step),
        "done_info": done_info,
        "prompt_debug": prompt_debug,
        "generated_subtask": generated_subtask,
    }


def build_pre_satisfied_start_result(
    *,
    max_steps: int,
    window_mode: str,
    combine_mode: str,
    start_all_satisfied: bool,
    require_unsatisfied_at_start: bool,
) -> tuple[dict[str, Any], str]:
    env_terminal_debug = {
        "termination_reason": "pre_satisfied_start",
        "terminated": False,
        "truncated": False,
        "env_current_step": 0,
        "done_info": {},
        "prompt_debug": None,
        "generated_subtask": None,
    }
    return (
        {
            "max_steps": int(max_steps),
            "final_step": 0,
            "predicate_window_mode": str(window_mode),
            "combine_mode": str(combine_mode),
            "rollout_attempted": False,
            "termination_reason": "pre_satisfied_start",
            "terminated": False,
            "truncated": False,
            "env_done_success": None,
            "env_terminal_debug": env_terminal_debug,
            "start_all_satisfied": bool(start_all_satisfied),
            "require_unsatisfied_at_start": bool(require_unsatisfied_at_start),
        },
        "pre_satisfied_start",
    )


def update_segment_env_termination_telemetry(
    telemetry: dict[str, Any], *, step: int, terminated: bool, env_done_success: Any
) -> None:
    """Track whole-task env termination/success as segment-level telemetry.

    Segment predicate evaluation has its own success condition. Whole-task success
    can happen earlier than a segment predicate (e.g. an attach relation is already
    satisfied before the final release skill is complete), so callers can use this
    telemetry without necessarily treating env termination as a segment terminal.
    """

    if terminated:
        telemetry["env_terminated_seen"] = True
        telemetry["env_termination_count"] = int(telemetry.get("env_termination_count") or 0) + 1
        if telemetry.get("first_env_terminated_step") is None:
            telemetry["first_env_terminated_step"] = int(step)
    if env_done_success is True:
        telemetry["env_done_success_seen"] = True
        if telemetry.get("first_env_done_success_step") is None:
            telemetry["first_env_done_success_step"] = int(step)


def _success_step_threshold(
    *,
    metric_debug: dict[str, Any] | None,
    max_steps: int,
    min_steps_key: str,
    min_fraction_key: str,
) -> int:
    metric_debug = metric_debug or {}
    threshold = 0
    try:
        threshold = max(threshold, int(metric_debug.get(min_steps_key) or 0))
    except (TypeError, ValueError):
        pass
    try:
        fraction = float(metric_debug.get(min_fraction_key) or 0.0)
    except (TypeError, ValueError):
        fraction = 0.0
    if fraction > 0:
        threshold = max(threshold, int(max_steps * fraction + 0.999999))
    return max(threshold, 0)


def classify_short_proxy_success(
    *,
    success: bool,
    result_type: str,
    final_step: int,
    max_steps: int,
    metric_debug: dict[str, Any] | None,
) -> tuple[bool, str, dict[str, Any]]:
    """Separate very early proxy successes from ordinary predicate_satisfied results.

    Some proxy metrics can become true for a few initial rollout frames even when the
    policy did not complete the semantic skill. Registry entries may set a blocking
    threshold (min_rollout_steps_for_success / min_success_step_fraction) or a
    caution-only threshold (short_success_caution_steps / short_success_caution_step_fraction).
    """

    diagnostics: dict[str, Any] = {
        "short_proxy_success": False,
        "likely_proxy_false_positive": False,
    }
    if not success or result_type != "predicate_satisfied":
        return success, result_type, diagnostics

    metric_debug = metric_debug or {}
    blocking_threshold = _success_step_threshold(
        metric_debug=metric_debug,
        max_steps=max_steps,
        min_steps_key="min_rollout_steps_for_success",
        min_fraction_key="min_success_step_fraction",
    )
    caution_threshold = _success_step_threshold(
        metric_debug=metric_debug,
        max_steps=max_steps,
        min_steps_key="short_success_caution_steps",
        min_fraction_key="short_success_caution_step_fraction",
    )
    threshold = max(blocking_threshold, caution_threshold)
    diagnostics.update(
        {
            "short_success_min_rollout_steps": blocking_threshold or None,
            "short_success_caution_steps": caution_threshold or None,
            "short_success_required_step": threshold or None,
            "short_success_final_step": int(final_step),
            "short_success_max_steps": int(max_steps),
        }
    )
    if threshold <= 0 or int(final_step) >= threshold:
        return success, result_type, diagnostics

    diagnostics["short_proxy_success"] = True
    likely_false_positive = bool(metric_debug.get("likely_false_positive_on_short_success"))
    diagnostics["likely_proxy_false_positive"] = likely_false_positive
    diagnostics["short_success_metric_family"] = metric_debug.get("metric_family")

    if likely_false_positive:
        return False, RESULT_LIKELY_PROXY_FALSE_POSITIVE, diagnostics
    return success, RESULT_SHORT_PROXY_SUCCESS, diagnostics


def classify_short_video_success(
    *,
    success: bool,
    result_type: str,
    final_step: int,
    max_steps: int,
    min_rollout_steps_for_success: int,
) -> tuple[bool, str, dict[str, Any]]:
    """Reject predicate successes that stop before a meaningful rollout video exists.

    This is a generic guard for non-proxy metrics. Proxy-family specific handling should
    run first via :func:`classify_short_proxy_success`; if it already changed the result
    type, this helper leaves that decision intact.
    """

    threshold = max(0, int(min_rollout_steps_for_success or 0))
    diagnostics: dict[str, Any] = {
        "short_video_problem": False,
        "short_video_min_rollout_steps": threshold or None,
        "short_video_final_step": int(final_step),
        "short_video_max_steps": int(max_steps),
    }
    if not success or result_type != "predicate_satisfied" or threshold <= 0 or int(final_step) >= threshold:
        return success, result_type, diagnostics

    diagnostics["short_video_problem"] = True
    return False, RESULT_SHORT_VIDEO_PROBLEM, diagnostics


def finalize_segment_predicate_result_type(
    *, success: bool, result_type: str, env_done_success_seen: bool, classify_env_success_before_segment_success: bool = True
) -> str:
    """Return the final segment result type after env-success telemetry is known."""

    if success or not classify_env_success_before_segment_success or not env_done_success_seen:
        return result_type
    if result_type in {"timeout", "env_terminated"}:
        return ENV_TASK_SUCCESS_BEFORE_SEGMENT_SUCCESS
    return result_type
