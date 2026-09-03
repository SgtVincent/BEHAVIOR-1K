"""Pure decision / diagnostic helpers for the golden-rule (GT action replay) arm.

This module deliberately imports **nothing** from ``omnigibson`` (and therefore
nothing from Isaac Sim).  It holds the branch-resolution logic of
``GoldenRuleEvaluator.check_skill_success`` as a side-effect-free function so
that it can be unit-tested directly -- including via
``importlib.util.spec_from_file_location`` -- without starting a simulator.

Why the branch table exists
---------------------------
``check_skill_success`` is a *sequential early-return* criterion:

    env terminated -> std_joint_qpos -> eef+gripper -> joint_qpos_rmse -> timeout

Only the winning branch used to be recorded.  Every branch after the winner was
never executed, which is a different fact from "executed and did not pass":
a branch that never ran carries *zero* information about the criterion, whereas
a branch that ran and rejected the state is evidence about the criterion.  The
GT-action-replay arm exists precisely to measure the second quantity, so the two
must not be merged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

__all__ = [
    "VERDICT_PASS",
    "VERDICT_FAIL",
    "VERDICT_NOT_EVALUATED",
    "VERDICTS",
    "BRANCH_ENV",
    "BRANCH_STD_JOINT_QPOS",
    "BRANCH_EEF_GRIPPER",
    "BRANCH_JOINT_RMSE",
    "BRANCH_TIMEOUT",
    "BRANCH_ORDER",
    "BRANCH_VERDICT_SEMANTICS",
    "SkillDecision",
    "SkillCheckOutcome",
    "evaluate_skill_success_branches",
    "run_skill_success_check",
    "summarize_skill_diagnostics",
]


# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------

VERDICT_PASS = "pass"
VERDICT_FAIL = "fail"
VERDICT_NOT_EVALUATED = "not-evaluated"
VERDICTS: Tuple[str, str, str] = (VERDICT_PASS, VERDICT_FAIL, VERDICT_NOT_EVALUATED)

BRANCH_ENV = "env"
BRANCH_STD_JOINT_QPOS = "std_joint_qpos"
BRANCH_EEF_GRIPPER = "eef_gripper"
BRANCH_JOINT_RMSE = "joint_rmse"
BRANCH_TIMEOUT = "timeout"

#: Evaluation order of the sequential early-return criterion.
BRANCH_ORDER: Tuple[str, ...] = (
    BRANCH_ENV,
    BRANCH_STD_JOINT_QPOS,
    BRANCH_EEF_GRIPPER,
    BRANCH_JOINT_RMSE,
    BRANCH_TIMEOUT,
)

#: Emitted alongside every branch table so a downstream reader never has to
#: guess the polarity of the ``timeout`` row.
BRANCH_VERDICT_SEMANTICS = (
    "Sequential early-return criterion evaluated in BRANCH_ORDER. "
    "'pass' = the branch was reached with usable inputs and its condition fired. "
    "'fail' = the branch was reached with usable inputs and its condition did not fire. "
    "'not-evaluated' = the branch was never reached (an earlier branch returned first), "
    "was disabled by config, or its input metric was absent / non-finite. "
    "'not-evaluated' carries zero information about the criterion and must never be "
    "counted as 'fail'. Polarity note: for the 'timeout' branch 'pass' means the "
    "timeout condition fired, i.e. the skill ended by exhausting its step budget, "
    "which is a skill failure. See 'branch_details' for the per-branch reason."
)

# Detail strings (stable tokens, safe to group on downstream).
_D_ENV_TERMINATED = "env_terminated"
_D_ENV_NOT_TERMINATED = "env_not_terminated"
_D_SHORT_CIRCUIT = "not_reached_short_circuit"
_D_STATE_MATCH_DISABLED = "state_match_disabled"
_D_STATE_ERRORS_UNAVAILABLE = "state_errors_unavailable"
_D_METRIC_UNAVAILABLE = "metric_absent_or_non_finite"
_D_WITHIN_THRESHOLD = "within_threshold"
_D_ABOVE_THRESHOLD = "above_threshold"
_D_BUDGET_EXHAUSTED = "step_budget_exhausted"
_D_BUDGET_REMAINING = "step_budget_remaining"


@dataclass(frozen=True)
class SkillDecision:
    """Outcome of one ``check_skill_success`` evaluation.

    ``is_done`` / ``result`` are the *verdict*: they reproduce the pre-existing
    behaviour byte for byte.  Everything else is diagnostics.
    """

    is_done: bool
    result: str
    success_reason: Optional[str]
    decisive_branch: Optional[str]
    branch_verdicts: Dict[str, str]
    branch_details: Dict[str, str]

    @property
    def success_env(self) -> bool:
        """True iff the environment terminated (task-level success signal)."""
        return self.result == "success_env"

    @property
    def success_state(self) -> bool:
        """True iff a state-match branch accepted the skill."""
        return self.result == "success_state"


def _finite(value: Any) -> Optional[float]:
    """Return ``value`` as a finite float, or ``None`` if absent / non-finite.

    Accepts numpy scalars without importing numpy.
    """
    if value is None:
        return None
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    return as_float if math.isfinite(as_float) else None


def _min_finite(values: Iterable[Any]) -> Optional[float]:
    finite = [f for f in (_finite(v) for v in values) if f is not None]
    return min(finite) if finite else None


def evaluate_skill_success_branches(
    *,
    terminated: bool,
    current_step: int,
    timeout: int,
    state_match_enabled: bool,
    errors: Optional[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
    state_errors_error: Optional[str] = None,
) -> SkillDecision:
    """Resolve the skill-success criterion and record every branch's verdict.

    This mirrors the original control flow of
    ``GoldenRuleEvaluator.check_skill_success`` exactly; the only additions are
    the ``branch_verdicts`` / ``branch_details`` tables and ``decisive_branch``.

    Args:
        terminated: whether the environment terminated this step.
        current_step: steps elapsed inside the current skill.
        timeout: step budget for the current skill.
        state_match_enabled: value of ``primitive_success_use_state_match``.
        errors: output of ``compute_primitive_state_errors``, or ``None``.
            On the ``terminated`` path this is *diagnostics only* and cannot
            influence the returned verdict.
        thresholds: resolved output of ``get_primitive_success_thresholds()``.
        state_errors_error: optional message explaining why ``errors`` is None.

    Returns:
        A :class:`SkillDecision`.
    """
    verdicts: Dict[str, str] = {branch: VERDICT_NOT_EVALUATED for branch in BRANCH_ORDER}
    details: Dict[str, str] = {branch: _D_SHORT_CIRCUIT for branch in BRANCH_ORDER}

    state_branches = (BRANCH_STD_JOINT_QPOS, BRANCH_EEF_GRIPPER, BRANCH_JOINT_RMSE)

    # --- branch 1: environment termination (always evaluated) ---------------
    if terminated:
        verdicts[BRANCH_ENV] = VERDICT_PASS
        details[BRANCH_ENV] = _D_ENV_TERMINATED
        # The verdict is already fixed here.  Any state errors supplied by the
        # caller are diagnostics and must not move the decision.
        return SkillDecision(
            is_done=True,
            result="success_env",
            success_reason="env_terminated",
            decisive_branch=BRANCH_ENV,
            branch_verdicts=verdicts,
            branch_details=details,
        )

    verdicts[BRANCH_ENV] = VERDICT_FAIL
    details[BRANCH_ENV] = _D_ENV_NOT_TERMINATED

    # --- branches 2-4: state match ------------------------------------------
    decided: Optional[SkillDecision] = None

    if not state_match_enabled:
        for branch in state_branches:
            details[branch] = _D_STATE_MATCH_DISABLED
    elif errors is None:
        unavailable = _D_STATE_ERRORS_UNAVAILABLE
        if state_errors_error:
            unavailable = f"{_D_STATE_ERRORS_UNAVAILABLE}:{state_errors_error}"
        for branch in state_branches:
            details[branch] = unavailable
    else:
        # 2. standardised joint qpos RMSE
        std_thr = _finite(thresholds.get("std_joint_qpos_rmse"))
        std_rmse = _finite(errors.get("std_joint_qpos_rmse"))
        if std_rmse is None or std_thr is None:
            details[BRANCH_STD_JOINT_QPOS] = _D_METRIC_UNAVAILABLE
        elif std_rmse <= std_thr:
            verdicts[BRANCH_STD_JOINT_QPOS] = VERDICT_PASS
            details[BRANCH_STD_JOINT_QPOS] = _D_WITHIN_THRESHOLD
            decided = SkillDecision(
                is_done=True,
                result="success_state",
                success_reason="state_match_std_joint_qpos",
                decisive_branch=BRANCH_STD_JOINT_QPOS,
                branch_verdicts=verdicts,
                branch_details=details,
            )
        else:
            verdicts[BRANCH_STD_JOINT_QPOS] = VERDICT_FAIL
            details[BRANCH_STD_JOINT_QPOS] = _D_ABOVE_THRESHOLD

        # 3. eef position + gripper qpos (both must be available)
        if decided is None:
            eef_thr = _finite(thresholds.get("eef_pos"))
            grip_thr = _finite(thresholds.get("gripper_qpos"))
            eef_min = _min_finite(
                (errors.get("eef_left_pos_err"), errors.get("eef_right_pos_err"))
            )
            grip_min = _min_finite(
                (errors.get("gripper_left_qpos_err"), errors.get("gripper_right_qpos_err"))
            )
            if eef_min is None or grip_min is None or eef_thr is None or grip_thr is None:
                details[BRANCH_EEF_GRIPPER] = _D_METRIC_UNAVAILABLE
            elif eef_min <= eef_thr and grip_min <= grip_thr:
                verdicts[BRANCH_EEF_GRIPPER] = VERDICT_PASS
                details[BRANCH_EEF_GRIPPER] = _D_WITHIN_THRESHOLD
                decided = SkillDecision(
                    is_done=True,
                    result="success_state",
                    success_reason="state_match_eef_gripper",
                    decisive_branch=BRANCH_EEF_GRIPPER,
                    branch_verdicts=verdicts,
                    branch_details=details,
                )
            else:
                verdicts[BRANCH_EEF_GRIPPER] = VERDICT_FAIL
                details[BRANCH_EEF_GRIPPER] = _D_ABOVE_THRESHOLD

        # 4. raw joint qpos RMSE
        if decided is None:
            jq_thr = _finite(thresholds.get("joint_qpos_rmse"))
            jq = _finite(errors.get("joint_qpos_rmse"))
            if jq is None or jq_thr is None:
                details[BRANCH_JOINT_RMSE] = _D_METRIC_UNAVAILABLE
            elif jq <= jq_thr:
                verdicts[BRANCH_JOINT_RMSE] = VERDICT_PASS
                details[BRANCH_JOINT_RMSE] = _D_WITHIN_THRESHOLD
                decided = SkillDecision(
                    is_done=True,
                    result="success_state",
                    success_reason="state_match_joint_rmse",
                    decisive_branch=BRANCH_JOINT_RMSE,
                    branch_verdicts=verdicts,
                    branch_details=details,
                )
            else:
                verdicts[BRANCH_JOINT_RMSE] = VERDICT_FAIL
                details[BRANCH_JOINT_RMSE] = _D_ABOVE_THRESHOLD

    if decided is not None:
        return decided

    # --- branch 5: timeout ---------------------------------------------------
    if int(current_step) >= int(timeout):
        verdicts[BRANCH_TIMEOUT] = VERDICT_PASS
        details[BRANCH_TIMEOUT] = _D_BUDGET_EXHAUSTED
        # NOTE: the original code leaves ``success_reason`` unset on timeout.
        # That is preserved; ``decisive_branch`` carries the information instead.
        return SkillDecision(
            is_done=True,
            result="timeout",
            success_reason=None,
            decisive_branch=BRANCH_TIMEOUT,
            branch_verdicts=verdicts,
            branch_details=details,
        )

    verdicts[BRANCH_TIMEOUT] = VERDICT_FAIL
    details[BRANCH_TIMEOUT] = _D_BUDGET_REMAINING
    return SkillDecision(
        is_done=False,
        result="in_progress",
        success_reason=None,
        decisive_branch=None,
        branch_verdicts=verdicts,
        branch_details=details,
    )


@dataclass(frozen=True)
class SkillCheckOutcome:
    """Result of one full skill-success check, verdict plus diagnostics."""

    decision: SkillDecision
    state_errors: Optional[Mapping[str, Any]]
    state_errors_error: Optional[str]
    diagnostics_only: bool


def run_skill_success_check(
    *,
    skill: Any,
    terminated: bool,
    current_step: int,
    timeout: int,
    state_match_enabled: bool,
    thresholds: Mapping[str, Any],
    compute_state_errors: Callable[[Any], Optional[Mapping[str, Any]]],
    diagnose_on_terminated: bool = True,
) -> SkillCheckOutcome:
    """Run the skill-success criterion, gathering diagnostics on every path.

    Spec 2.2.  The original code short-circuited on ``terminated`` *before*
    computing state errors, so ``state_errors`` was ``null`` in the JSON for
    every environment-terminated skill.  Replaying ground-truth actions is
    expected to terminate the environment, so that blanked exactly the rows the
    replay arm exists to inspect.

    The verdict is unchanged.  Two rules keep it that way:

    * ``terminated`` is passed straight through to
      :func:`evaluate_skill_success_branches`, which resolves it first, so any
      state errors computed here are inert.
    * on that path a raising ``compute_state_errors`` is captured into
      ``state_errors_error`` rather than propagating -- a diagnostic must not be
      able to turn a decided skill into a crash.  On the decision path
      exceptions propagate exactly as they did before.

    Args:
        skill: skill annotation, passed opaquely to ``compute_state_errors``.
        terminated: whether the environment terminated this step.
        current_step: steps elapsed inside the current skill.
        timeout: step budget for the current skill.
        state_match_enabled: value of ``primitive_success_use_state_match``.
        thresholds: resolved ``get_primitive_success_thresholds()`` output.
        compute_state_errors: callable returning the state-error mapping.
        diagnose_on_terminated: set False to restore the old, cheaper behaviour
            of skipping the diagnostic computation on the terminated path.

    Returns:
        A :class:`SkillCheckOutcome`.
    """
    diagnostics_only = bool(terminated)
    errors: Optional[Mapping[str, Any]] = None
    errors_error: Optional[str] = None

    # The two paths are gated independently.  On the decision path the gate is
    # `state_match_enabled`, exactly as before.  On the already-decided
    # (`terminated`) path the gate is `diagnose_on_terminated`, so setting that
    # to False genuinely restores the old behaviour of computing nothing --
    # which it would not do if `state_match_enabled` could also open the gate.
    if diagnostics_only:
        if diagnose_on_terminated:
            try:
                errors = compute_state_errors(skill)
            except Exception as exc:  # noqa: BLE001 - must not change the verdict
                errors = None
                errors_error = f"{type(exc).__name__}: {exc}"
    elif state_match_enabled:
        errors = compute_state_errors(skill)

    decision = evaluate_skill_success_branches(
        terminated=terminated,
        current_step=current_step,
        timeout=timeout,
        state_match_enabled=state_match_enabled,
        errors=errors,
        thresholds=thresholds,
        state_errors_error=errors_error,
    )
    return SkillCheckOutcome(
        decision=decision,
        state_errors=errors,
        state_errors_error=errors_error,
        diagnostics_only=diagnostics_only,
    )


def summarize_skill_diagnostics(per_demo_results: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-skill diagnostics across demos, stratified by branch.

    Implements the reporting rule from spec section 4: a ``terminated``-driven
    pass carries no information about the state-match criterion, so pass rates
    must be reported per ``result`` / ``success_reason`` rather than pooled.
    """
    by_result: Dict[str, int] = {}
    by_success_reason: Dict[str, int] = {}
    branch_counts: Dict[str, Dict[str, int]] = {
        branch: {verdict: 0 for verdict in VERDICTS} for branch in BRANCH_ORDER
    }
    n_records = 0
    n_with_branch_table = 0

    for demo in per_demo_results or []:
        if not isinstance(demo, Mapping):
            continue
        for record in demo.get("skill_diagnostics") or []:
            if not isinstance(record, Mapping):
                continue
            n_records += 1
            result = str(record.get("result", ""))
            by_result[result] = by_result.get(result, 0) + 1
            reason = record.get("success_reason")
            reason_key = "null" if reason is None else str(reason)
            by_success_reason[reason_key] = by_success_reason.get(reason_key, 0) + 1

            verdicts = record.get("branch_verdicts")
            if isinstance(verdicts, Mapping):
                n_with_branch_table += 1
                for branch in BRANCH_ORDER:
                    verdict = verdicts.get(branch)
                    if verdict in branch_counts[branch]:
                        branch_counts[branch][verdict] += 1

    return {
        "n_skill_records": n_records,
        "n_records_with_branch_table": n_with_branch_table,
        "by_result": by_result,
        "by_success_reason": by_success_reason,
        "branch_verdict_counts": branch_counts,
        "branch_verdict_semantics": BRANCH_VERDICT_SEMANTICS,
    }
