from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from omnigibson.learning.utils.eval_diagnostics import (
    ENV_TASK_SUCCESS_BEFORE_SEGMENT_SUCCESS,
    build_termination_summary,
    classify_short_proxy_success,
    classify_short_video_success,
    finalize_segment_predicate_result_type,
    update_segment_env_termination_telemetry,
)
from omnigibson.learning.utils.segment_predicate_eval import (
    MISSING_OBJECT_RESULT_TYPE,
    SegmentPredicate,
    build_template_predicates,
    eval_segment_predicates,
    predicate_trace_satisfied,
    predicate_window_satisfied,
    summarize_predicate_trace,
    trace_missing_objects,
)
from omnigibson.learning.utils.segment_skill_metric_registry import (
    SKILL_METRIC_REGISTRY,
    get_skill_metric_entry,
)

logger = logging.getLogger("skill_completion")
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_skill_completed(
    env,
    skill_desc: str,
    demo_annotations: Optional[Dict] = None,
    object_bindings: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Evaluate whether a skill has been completed in the current environment state.

    This is a lightweight, single-step check (no rollout).  It looks up the skill's
    metric family in SKILL_METRIC_REGISTRY, resolves object bindings, evaluates the
    registered predicates / geometry metrics, and returns a structured result.

    Args:
        env: OmniGibson environment (must expose ``scene``, ``robots``, ``task``, …).
        skill_desc: Natural-language skill description, e.g. ``"pick up from"``.
        demo_annotations: Optional demo annotation dict.  If provided and
            ``object_bindings`` is *not* provided, the annotation is used to
            derive bindings automatically.
        object_bindings: Optional pre-computed mapping from registry ``object_roles``
            to concrete environment object names.

    Returns:
        A dict with keys:
        - ``completed`` (bool): Whether the skill predicates are satisfied *right now*.
        - ``reason`` (str): Human-readable summary.
        - ``metrics`` (Dict): Detailed metric evaluation results.
        - ``result_type`` (str): One of
          ``predicate_satisfied``, ``predicate_unsatisfied``, ``missing_object``,
          ``invalid_object_bindings``, ``unknown_skill``, ``no_predicates``,
          ``env_terminated``.
    """
    if not skill_desc:
        return {
            "completed": False,
            "reason": "Empty skill description",
            "metrics": {},
            "result_type": "unknown_skill",
        }

    entry = get_skill_metric_entry(skill_desc)
    if entry is None:
        return {
            "completed": False,
            "reason": f"Skill '{skill_desc}' not found in SKILL_METRIC_REGISTRY",
            "metrics": {},
            "result_type": "unknown_skill",
        }

    # ------------------------------------------------------------------
    # 1. Build predicate specs
    # ------------------------------------------------------------------
    # Fast path: explicit bindings -> build specs directly from registry entry.
    if object_bindings is not None:
        eval_result = eval_skill_metric(env, entry, object_bindings)
        return {
            "completed": eval_result["success"],
            "reason": (
                f"Skill '{skill_desc}' satisfied"
                if eval_result["success"]
                else f"Skill '{skill_desc}' not satisfied ({eval_result['result_type']})"
            ),
            "metrics": {
                "metric_family": entry.get("metric_family"),
                "success_rule": entry.get("success_rule"),
                "combine_mode": entry.get("combine_mode", "all_of"),
                "trace": eval_result["trace"],
                "truth_map": eval_result["truth_map"],
                "missing_objects": eval_result["missing_objects"],
                "trace_summary": eval_result.get("trace_summary", {}),
                "invalid_role_bindings": eval_result.get("invalid_role_bindings", []),
            },
            "result_type": eval_result["result_type"],
        }

    # Fallback: derive bindings from demo annotations via build_template_predicates.
    segment_level = "skill"
    segment: Dict[str, Any] = {
        f"{segment_level}_description": [skill_desc],
    }
    if demo_annotations is not None:
        segment.update(demo_annotations)

    template_specs, metric_debug = build_template_predicates(segment_level, segment, env)

    if not template_specs:
        return {
            "completed": False,
            "reason": f"No predicates could be built for skill '{skill_desc}'",
            "metrics": {
                "metric_family": entry.get("metric_family"),
                "metric_debug": metric_debug,
            },
            "result_type": "no_predicates",
        }

    # ------------------------------------------------------------------
    # 2. Evaluate predicates / geometry metrics
    # ------------------------------------------------------------------
    truth_map, trace = eval_segment_predicates(env, template_specs)

    # Check for missing objects
    missing = trace_missing_objects(trace)
    if missing:
        missing_names = [m["missing_object"] for m in missing]
        return {
            "completed": False,
            "reason": f"Missing objects: {missing_names}",
            "metrics": {
                "metric_family": entry.get("metric_family"),
                "trace": trace,
                "missing_objects": missing_names,
                "metric_debug": metric_debug,
            },
            "result_type": "missing_object",
        }

    # ------------------------------------------------------------------
    # 3. Determine satisfaction
    # ------------------------------------------------------------------
    combine_mode = str(entry.get("combine_mode", "all_of"))
    trace_summary = summarize_predicate_trace(trace, combine_mode)
    satisfied = predicate_trace_satisfied(trace, combine_mode)
    result_type = "predicate_satisfied" if satisfied else "predicate_unsatisfied"

    # Build a concise reason string from primary BDDL-aligned predicates only.
    if satisfied:
        reason = f"Skill '{skill_desc}' satisfied ({combine_mode})"
    else:
        reason = (
            f"Skill '{skill_desc}' not satisfied; failed primary predicates: "
            f"{trace_summary['failed_primary_predicates']}"
        )

    return {
        "completed": satisfied,
        "reason": reason,
        "metrics": {
            "metric_family": entry.get("metric_family"),
            "success_rule": entry.get("success_rule"),
            "combine_mode": combine_mode,
            "trace": trace,
            "truth_map": truth_map,
            "metric_debug": metric_debug,
            "trace_summary": trace_summary,
        },
        "result_type": result_type,
    }


def get_skill_object_bindings(
    skill_desc: str,
    segment_annotation: Dict,
    env_objects: Dict,
) -> Dict[str, str]:
    """Map registry ``object_roles`` to concrete object names.

    The function extracts object names from *segment_annotation* (using the same
    heuristics as ``extract_segment_args`` in ``segment_predicate_eval``) and
    returns a flat dict ``{role: resolved_name}``.

    Args:
        skill_desc: Skill description string (e.g. ``"place in"``).
        segment_annotation: Annotation dict for the current segment.  Expected keys:
            ``object_id``, ``manipulating_object_id``, ``skill_description`` (or
            ``primitive_description``).
        env_objects: Mapping from object identifiers (as they appear in annotations)
            to actual environment object names.  This is typically derived from the
            scene's object registry or task metadata.

    Returns:
        Dict mapping each ``object_roles`` entry for the skill to a concrete name.
        If a role cannot be resolved, it is omitted from the result.
    """
    entry = get_skill_metric_entry(skill_desc)
    if entry is None:
        return {}

    object_roles = entry.get("object_roles", [])
    if not object_roles:
        return {}

    # Re-use the existing extraction logic by building a minimal segment
    segment_level = "skill"
    segment: Dict[str, Any] = {f"{segment_level}_description": [skill_desc]}
    segment.update(segment_annotation)

    # Import locally to avoid circular issues at module load time
    from omnigibson.learning.utils.segment_predicate_eval import (
        extract_segment_args,
        _resolve_role_name,
    )

    info = extract_segment_args(segment_level, segment)
    bindings: Dict[str, str] = {}
    for role in object_roles:
        resolved = _resolve_role_name(info, role)
        if resolved is None:
            continue
        # Map through env_objects if available
        concrete = env_objects.get(resolved, resolved)
        if concrete:
            bindings[role] = str(concrete)
    return bindings


def eval_skill_metric(
    env,
    metric_entry: Dict[str, Any],
    object_bindings: Dict[str, str],
) -> Dict[str, Any]:
    """Evaluate a single skill metric entry against the current environment state.

    This is a lower-level helper that takes a registry metric entry (or any dict
    with the same shape) and a concrete binding map, builds the corresponding
    ``SegmentPredicate`` list, evaluates them, and returns detailed results.

    Args:
        env: OmniGibson environment.
        metric_entry: A metric entry dict, typically from
            ``SKILL_METRIC_REGISTRY[skill_desc]``.
        object_bindings: Mapping from role tokens to environment object names.

    Returns:
        Dict with keys:
        - ``success`` (bool): Whether the metric entry is satisfied.
        - ``result_type`` (str): ``predicate_satisfied`` or ``predicate_unsatisfied``.
        - ``trace`` (List[Dict]): Primary per-predicate evaluation trace, with
          any non-gating state checks nested under ``auxiliary_diagnostics``.
        - ``truth_map`` (Dict[str, bool]): Predicate-key -> bool mapping.
        - ``missing_objects`` (List[str]): Names of missing primary objects, if any.
        - ``trace_summary`` (Dict): Separate primary failures and auxiliary telemetry.
    """
    metrics = metric_entry.get("metrics", [])
    if not metrics:
        return {
            "success": False,
            "result_type": "no_predicates",
            "trace": [],
            "truth_map": {},
            "missing_objects": [],
            "trace_summary": summarize_predicate_trace([]),
        }

    invalid_role_bindings = []
    for role_group in metric_entry.get("required_distinct_roles", []):
        bindings = {role: object_bindings.get(role) for role in role_group}
        bound_values = [value for value in bindings.values() if value is not None]
        if len(bound_values) != len(role_group):
            invalid_role_bindings.append(
                {
                    "reason": "required_role_missing",
                    "roles": list(role_group),
                    "bindings": bindings,
                }
            )
        elif len(set(bound_values)) != len(bound_values):
            invalid_role_bindings.append(
                {
                    "reason": "required_roles_resolve_to_same_object",
                    "roles": list(role_group),
                    "bindings": bindings,
                }
            )
    if invalid_role_bindings:
        return {
            "success": False,
            "result_type": "invalid_object_bindings",
            "trace": [],
            "truth_map": {},
            "missing_objects": [],
            "invalid_role_bindings": invalid_role_bindings,
            "trace_summary": summarize_predicate_trace([]),
        }

    # Import geometry helpers locally to avoid heavy imports at module load time.
    from omnigibson.learning.utils.segment_predicate_eval import (
        _capture_object_pose,
        _capture_robot_base,
        _object_from_name,
    )
    import numpy as np

    specs: List[SegmentPredicate] = []
    diagnostic_specs: List[SegmentPredicate] = []
    metric_groups = (
        (metrics, specs, True),
        (metric_entry.get("diagnostic_metrics", []), diagnostic_specs, False),
    )
    for metric_group, destination_specs, contributes_to_success in metric_groups:
        for metric in metric_group:
            metric_type = metric.get("type")
            if metric_type == "predicate":
                args = []
                valid = True
                for role in metric.get("args", []):
                    if role == "agent":
                        args.append("agent")
                        continue
                    resolved = object_bindings.get(role)
                    if resolved is None:
                        valid = False
                        break
                    args.append(resolved)
                if not valid:
                    continue
                destination_specs.append(
                    SegmentPredicate(
                        metric_type="predicate",
                        name=metric["name"],
                        args=args,
                        desired=bool(metric["desired"]),
                        source="registry" if contributes_to_success else "registry_diagnostic",
                        params={
                            "metric_family": metric_entry.get("metric_family"),
                            "success_rule": metric_entry.get("success_rule"),
                            "semantic_role": metric.get("semantic_role", metric["name"]),
                            "contributes_to_success": contributes_to_success,
                        },
                    )
                )
            elif contributes_to_success and metric_type in {
                "base_to_object",
                "face_object",
                "object_pose_match",
                "object_orientation_match",
            }:
                resolved = object_bindings.get(metric["role"])
                if resolved is None:
                    continue
                params = dict(metric)
                params["resolved_role_name"] = resolved
                params["semantic_role"] = metric.get("semantic_role", metric_type)
                params["contributes_to_success"] = True
                # Replicate the live-capture logic from build_template_predicates
                # so geometry metrics have correct thresholds / reference poses.
                if metric_type in {"object_pose_match", "object_orientation_match"}:
                    captured = _capture_object_pose(env, resolved)
                    if captured is None:
                        continue
                    params.update(captured)
                elif metric_type == "base_to_object":
                    target_obj = _object_from_name(env, resolved)
                    if target_obj is None:
                        continue
                    robot_base = _capture_robot_base(env)
                    target_pos, _ = target_obj.get_position_orientation()
                    dist = float(
                        np.linalg.norm(np.asarray(robot_base["position"][:2]) - np.asarray(target_pos[:2]))
                    )
                    params["threshold"] = max(
                        float(metric.get("min_threshold", 0.9)),
                        dist + float(metric.get("margin", 0.35)),
                    )
                elif metric_type == "face_object":
                    target_obj = _object_from_name(env, resolved)
                    if target_obj is None:
                        continue
                    robot_base = _capture_robot_base(env)
                    target_pos, _ = target_obj.get_position_orientation()
                    vec = np.asarray(target_pos[:2]) - np.asarray(robot_base["position"][:2])
                    target_yaw = float(np.arctan2(vec[1], vec[0]))
                    end_err = abs(
                        np.arctan2(
                            np.sin(robot_base["yaw"] - target_yaw),
                            np.cos(robot_base["yaw"] - target_yaw),
                        )
                    )
                    params["threshold"] = max(
                        float(metric.get("min_threshold", 0.4)),
                        end_err + float(metric.get("yaw_margin", 0.2)),
                    )
                destination_specs.append(
                    SegmentPredicate(
                        metric_type=metric_type,
                        name=metric_type,
                        args=[resolved],
                        desired=True,
                        source="registry",
                        params=params,
                    )
                )

    if specs and diagnostic_specs:
        specs[0].diagnostic_specs.extend(diagnostic_specs)

    if not specs:
        return {
            "success": False,
            "result_type": "no_predicates",
            "trace": [],
            "truth_map": {},
            "missing_objects": [],
            "trace_summary": summarize_predicate_trace([]),
        }

    truth_map, trace = eval_segment_predicates(env, specs)
    missing = trace_missing_objects(trace)
    missing_names = [m["missing_object"] for m in missing]

    combine_mode = str(metric_entry.get("combine_mode", "all_of"))
    trace_summary = summarize_predicate_trace(trace, combine_mode)
    if missing_names:
        success = False
        result_type = "missing_object"
    else:
        success = predicate_trace_satisfied(trace, combine_mode)
        result_type = "predicate_satisfied" if success else "predicate_unsatisfied"

    return {
        "success": success,
        "result_type": result_type,
        "trace": trace,
        "truth_map": truth_map,
        "missing_objects": missing_names,
        "trace_summary": trace_summary,
    }


# ---------------------------------------------------------------------------
# Rollout-aware evaluation helpers (mirroring eval_segment.py logic)
# ---------------------------------------------------------------------------


def check_skill_completed_rollout(
    env,
    skill_desc: str,
    trace_history: Sequence[List[Dict[str, Any]]],
    *,
    demo_annotations: Optional[Dict] = None,
    object_bindings: Optional[Dict] = None,
    window_mode: str = "anytime",
    last_k: int = 20,
    min_consecutive: int = 1,
    max_steps: int = 500,
    final_step: int = 0,
    env_terminated: bool = False,
    env_truncated: bool = False,
    env_done_success: Optional[bool] = None,
    env_done_success_seen: bool = False,
    min_rollout_steps_for_success: int = 0,
) -> Dict[str, Any]:
    """Check skill completion using a history of per-step predicate traces.

    This is the rollout counterpart of :func:`check_skill_completed`.  It applies
    the same registry lookup and predicate evaluation, but uses
    :func:`predicate_window_satisfied` to decide success over a window of steps.
    It also replicates the short-proxy / short-video / env-termination logic from
    ``eval_segment.py``.

    Args:
        env: OmniGibson environment.
        skill_desc: Skill description.
        trace_history: List of per-step traces (each trace is the output of
            ``eval_segment_predicates`` for that step).
        demo_annotations: Optional segment-level demo annotations.
        object_bindings: Optional pre-computed role -> name bindings.
        window_mode: ``"anytime"``, ``"last_k"``, or ``"consecutive"``.
        last_k: Window size for ``"last_k"`` mode.
        min_consecutive: Required consecutive steps for ``"consecutive"`` mode.
        max_steps: Maximum rollout steps (used for short-success classification).
        final_step: Actual final step reached.
        env_terminated: Whether the environment returned ``terminated``.
        env_truncated: Whether the environment returned ``truncated``.
        env_done_success: Whether the environment done signal indicated success.
        env_done_success_seen: Whether env success was observed at any point.
        min_rollout_steps_for_success: Minimum steps before a success counts.

    Returns:
        Dict with keys ``completed``, ``reason``, ``metrics``, ``result_type``.
    """
    entry = get_skill_metric_entry(skill_desc)
    if entry is None:
        return {
            "completed": False,
            "reason": f"Unknown skill '{skill_desc}'",
            "metrics": {},
            "result_type": "unknown_skill",
        }

    combine_mode = str(entry.get("combine_mode", "all_of"))

    # Window-satisfaction check
    window_satisfied = predicate_window_satisfied(
        trace_history,
        mode=window_mode,
        last_k=last_k,
        min_consecutive=min_consecutive,
        combine_mode=combine_mode,
    )

    # Missing-object guard
    missing_names: List[str] = []
    for step_trace in trace_history:
        for m in trace_missing_objects(step_trace):
            name = m.get("missing_object")
            if name and name not in missing_names:
                missing_names.append(name)

    if missing_names:
        return {
            "completed": False,
            "reason": f"Missing objects: {missing_names}",
            "metrics": {
                "missing_objects": missing_names,
                "trace_history_length": len(trace_history),
            },
            "result_type": "missing_object",
        }

    success = bool(window_satisfied)
    result_type = "predicate_satisfied" if success else "timeout"

    if env_truncated:
        success = False
        result_type = "truncated"
    elif env_terminated:
        success = False
        result_type = "env_terminated"

    # Apply env-success-before-segment-success reclassification
    result_type = finalize_segment_predicate_result_type(
        success=success,
        result_type=result_type,
        env_done_success_seen=env_done_success_seen,
    )

    # Short-proxy / short-video classification
    metric_debug = {
        "metric_family": entry.get("metric_family"),
        "min_rollout_steps_for_success": entry.get("min_rollout_steps_for_success"),
        "min_success_step_fraction": entry.get("min_success_step_fraction"),
        "short_success_caution_steps": entry.get("short_success_caution_steps"),
        "short_success_caution_step_fraction": entry.get("short_success_caution_step_fraction"),
        "likely_false_positive_on_short_success": entry.get("likely_false_positive_on_short_success"),
    }

    success, result_type, proxy_diag = classify_short_proxy_success(
        success=success,
        result_type=result_type,
        final_step=final_step or max_steps,
        max_steps=max_steps,
        metric_debug=metric_debug,
    )
    success, result_type, video_diag = classify_short_video_success(
        success=success,
        result_type=result_type,
        final_step=final_step or max_steps,
        max_steps=max_steps,
        min_rollout_steps_for_success=min_rollout_steps_for_success,
    )

    diagnostics = {**proxy_diag, **video_diag}
    final_trace_summary = summarize_predicate_trace(
        trace_history[-1] if trace_history else [],
        combine_mode,
    )

    reason = (
        f"Skill '{skill_desc}' {result_type}"
        if success
        else f"Skill '{skill_desc}' incomplete ({result_type})"
    )

    return {
        "completed": success,
        "reason": reason,
        "metrics": {
            "metric_family": entry.get("metric_family"),
            "success_rule": entry.get("success_rule"),
            "combine_mode": combine_mode,
            "window_mode": window_mode,
            "trace_history_length": len(trace_history),
            "window_satisfied": window_satisfied,
            "final_step": final_step,
            "max_steps": max_steps,
            "final_trace_summary": final_trace_summary,
            **diagnostics,
        },
        "result_type": result_type,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _apply_object_bindings(
    specs: List[SegmentPredicate],
    bindings: Dict[str, str],
) -> List[SegmentPredicate]:
    """Return a new list of SegmentPredicate with args overridden by *bindings*.

    Only predicate args that correspond to a known role token are replaced;
    ``"agent"`` is left untouched.
    """
    new_specs: List[SegmentPredicate] = []
    for spec in specs:
        new_args = list(spec.args)
        if spec.metric_type == "predicate":
            # The first arg for unary predicates may be a role token;
            # for binary predicates the second arg is typically the object role.
            for i, arg in enumerate(new_args):
                if arg == "agent":
                    continue
                if arg in bindings:
                    new_args[i] = bindings[arg]
        else:
            # Geometry metrics have a single arg that is the resolved role name
            if new_args and new_args[0] in bindings:
                new_args[0] = bindings[new_args[0]]
        new_specs.append(
            SegmentPredicate(
                metric_type=spec.metric_type,
                name=spec.name,
                args=new_args,
                desired=spec.desired,
                source=spec.source,
                params=dict(spec.params),
                diagnostic_specs=_apply_object_bindings(spec.diagnostic_specs, bindings),
            )
        )
    return new_specs
