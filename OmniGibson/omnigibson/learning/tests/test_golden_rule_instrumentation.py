"""Unit tests for the GT-action-replay arm instrumentation.

Covers the four code changes required by
``OVE_L20_repro/gt_plan_audit/GT_ACTION_REPLAY_ARM_SPEC.md`` section 2:

* 2.1 the resolved success thresholds reach the metrics JSON
* 2.2 the ``terminated`` short-circuit no longer blanks ``state_errors``,
      and computing that diagnostic cannot change the verdict
* 2.3 every branch gets an explicit verdict, with ``not-evaluated`` kept
      distinct from ``fail``
* 2.4 switching demos actually switches the replayed parquet

Deliberately simulator-free: nothing here imports ``omnigibson`` or
``isaacsim``.  Modules under test are loaded straight off disk with
``importlib.util.spec_from_file_location``; ``policies.py``'s two omnigibson
imports are satisfied with stubs so the real ``DemoActionReplayPolicy`` runs.

Run with::

    python -m unittest omnigibson.learning.tests.test_golden_rule_instrumentation
    # or, from this directory:
    python -m unittest test_golden_rule_instrumentation -v
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

_LEARNING_DIR = Path(__file__).resolve().parents[1]


def _load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# --- module under test 1: the pure diagnostics helpers ---------------------
grd = _load_module_from_path(
    "_test_golden_rule_diagnostics",
    _LEARNING_DIR / "utils" / "golden_rule_diagnostics.py",
)


# --- module under test 2: the real DemoActionReplayPolicy ------------------
def _install_omnigibson_stubs() -> None:
    """Stub only what ``policies.py`` imports from omnigibson.

    ``policies.py`` needs ``torch_to_numpy`` and ``WebsocketClientPolicy``.
    Neither is exercised by ``DemoActionReplayPolicy``, so trivial stand-ins
    are enough and no simulator is pulled in.
    """
    for mod_name in (
        "omnigibson",
        "omnigibson.learning",
        "omnigibson.learning.utils",
    ):
        if mod_name not in sys.modules:
            mod = types.ModuleType(mod_name)
            mod.__path__ = []  # mark as package
            sys.modules[mod_name] = mod

    if "omnigibson.learning.utils.array_tensor_utils" not in sys.modules:
        m = types.ModuleType("omnigibson.learning.utils.array_tensor_utils")
        m.torch_to_numpy = lambda x: x
        sys.modules["omnigibson.learning.utils.array_tensor_utils"] = m

    if "omnigibson.learning.utils.network_utils" not in sys.modules:
        m = types.ModuleType("omnigibson.learning.utils.network_utils")

        class _StubWebsocketClientPolicy:  # pragma: no cover - never constructed
            def __init__(self, *args, **kwargs):
                raise RuntimeError("stub")

        m.WebsocketClientPolicy = _StubWebsocketClientPolicy
        sys.modules["omnigibson.learning.utils.network_utils"] = m


_install_omnigibson_stubs()
policies = _load_module_from_path("_test_policies", _LEARNING_DIR / "policies.py")


# ---------------------------------------------------------------------------
# Lifting real methods out of un-importable modules
# ---------------------------------------------------------------------------
class _SilentLogger:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass

    def error(self, *a, **k):
        pass

    def debug(self, *a, **k):
        pass


def _extract_methods_from_source(path: Path, class_name: str, names, extra_globals=None):
    """Compile named methods of ``class_name`` straight out of ``path``.

    ``eval_golden_rule.py`` and ``eval_subtask_reset.py`` both import
    omnigibson at module scope, so they cannot be imported here.  The methods
    exercised below are self-contained, so their source is pulled from the AST
    and executed in an isolated namespace.  The point is that the tests then
    run the shipped code: a re-implementation would keep passing after the
    shipped code regressed.
    """
    import ast
    import textwrap

    source = path.read_text()
    tree = ast.parse(source)
    out: Dict[str, Any] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in names:
                    segment = ast.get_source_segment(source, item)
                    assert segment is not None
                    namespace: Dict[str, Any] = {
                        "Optional": Optional,
                        "Any": Any,
                        "Dict": Dict,
                        "logger": _SilentLogger(),
                    }
                    namespace.update(extra_globals or {})
                    exec(textwrap.dedent(segment), namespace)  # noqa: S102
                    out[item.name] = namespace[item.name]
    missing = set(names) - set(out)
    assert not missing, f"methods not found in {path} :: {class_name}: {missing}"
    return out


_EVAL_GOLDEN_RULE_PY = _LEARNING_DIR / "eval_golden_rule.py"
_EVAL_SUBTASK_RESET_PY = _LEARNING_DIR / "eval_subtask_reset.py"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

#: Same keys and defaults as ``get_primitive_success_thresholds()``
#: (eval_subtask_reset.py:489).
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "base_pos": 0.15,
    "yaw": 0.35,
    "eef_pos": 0.12,
    "gripper_qpos": 0.03,
    "std_joint_qpos_rmse": 0.25,
    "joint_qpos_rmse": 0.25,
}

_SKILL = {"description": "press", "frame_duration": [0, 64]}


def _errors(**overrides: float) -> Dict[str, float]:
    """State errors that fail every branch unless overridden."""
    base = {
        "base_pos_err": 9.0,
        "yaw_err": 9.0,
        "eef_left_pos_err": 9.0,
        "eef_right_pos_err": 9.0,
        "gripper_left_qpos_err": 9.0,
        "gripper_right_qpos_err": 9.0,
        "joint_qpos_rmse": 9.0,
        "std_joint_qpos_rmse": 9.0,
    }
    base.update(overrides)
    return base


def _check(
    *,
    terminated: bool = False,
    current_step: int = 0,
    timeout: int = 100,
    state_match_enabled: bool = True,
    errors: Optional[Dict[str, float]] = None,
    thresholds: Optional[Dict[str, float]] = None,
    compute: Any = None,
    diagnose_on_terminated: bool = True,
):
    if compute is None:
        def compute(_skill):  # noqa: ANN001
            return errors
    return grd.run_skill_success_check(
        skill=_SKILL,
        terminated=terminated,
        current_step=current_step,
        timeout=timeout,
        state_match_enabled=state_match_enabled,
        thresholds=thresholds if thresholds is not None else DEFAULT_THRESHOLDS,
        compute_state_errors=compute,
        diagnose_on_terminated=diagnose_on_terminated,
    )


# ===========================================================================
# Spec 2.2 -- terminated path keeps its diagnostics, verdict unchanged
# ===========================================================================
class TestTerminatedDiagnostics(unittest.TestCase):
    def test_state_errors_not_null_on_terminated_path(self):
        """The whole point of 2.2: terminated skills used to record null."""
        errs = _errors()
        outcome = _check(terminated=True, errors=errs)

        self.assertIsNotNone(
            outcome.state_errors,
            "terminated path must still record state errors (spec 2.2)",
        )
        self.assertEqual(outcome.state_errors, errs)
        self.assertTrue(outcome.diagnostics_only)
        self.assertIsNone(outcome.state_errors_error)

    def test_verdict_is_success_env_regardless_of_state_errors(self):
        """Diagnostics must never move the decision (spec 2.2, hard rule)."""
        variants = {
            "errors_fail_everything": _errors(),
            "errors_pass_everything": _errors(
                std_joint_qpos_rmse=0.0,
                eef_left_pos_err=0.0,
                gripper_left_qpos_err=0.0,
                joint_qpos_rmse=0.0,
            ),
            "errors_none": None,
            "errors_empty": {},
        }
        for label, errs in variants.items():
            with self.subTest(errors=label):
                outcome = _check(terminated=True, errors=errs)
                d = outcome.decision
                self.assertTrue(d.is_done)
                self.assertEqual(d.result, "success_env")
                self.assertEqual(d.success_reason, "env_terminated")
                self.assertEqual(d.decisive_branch, grd.BRANCH_ENV)

    def test_verdict_unchanged_even_when_diagnostics_raise(self):
        """A raising diagnostic is captured, not propagated, on this path."""
        def boom(_skill):  # noqa: ANN001
            raise RuntimeError("demo data went away")

        outcome = _check(terminated=True, compute=boom)

        self.assertEqual(outcome.decision.result, "success_env")
        self.assertTrue(outcome.decision.is_done)
        self.assertIsNone(outcome.state_errors)
        self.assertIsNotNone(outcome.state_errors_error)
        self.assertIn("RuntimeError", outcome.state_errors_error)
        # The failure is recorded, and the un-run branches stay 'not-evaluated'
        # rather than being reported as rejections.
        self.assertEqual(
            outcome.decision.branch_verdicts[grd.BRANCH_STD_JOINT_QPOS],
            grd.VERDICT_NOT_EVALUATED,
        )

    def test_exceptions_still_propagate_on_the_decision_path(self):
        """Only the already-decided path swallows; otherwise behave as before."""
        def boom(_skill):  # noqa: ANN001
            raise RuntimeError("demo data went away")

        with self.assertRaises(RuntimeError):
            _check(terminated=False, compute=boom)

    def test_diagnostics_can_be_switched_off(self):
        outcome = _check(terminated=True, errors=_errors(), diagnose_on_terminated=False)
        self.assertIsNone(outcome.state_errors)
        self.assertEqual(outcome.decision.result, "success_env")

    def test_diagnostics_run_on_terminated_even_if_state_match_disabled(self):
        outcome = _check(terminated=True, errors=_errors(), state_match_enabled=False)
        self.assertIsNotNone(outcome.state_errors)
        self.assertEqual(outcome.decision.result, "success_env")


# ===========================================================================
# Spec 2.3 -- explicit per-branch verdicts, not-evaluated != fail
# ===========================================================================
class TestBranchVerdicts(unittest.TestCase):
    def test_all_five_branches_always_present(self):
        for terminated in (True, False):
            with self.subTest(terminated=terminated):
                d = _check(terminated=terminated, errors=_errors()).decision
                self.assertEqual(tuple(d.branch_verdicts), grd.BRANCH_ORDER)
                self.assertEqual(tuple(d.branch_details), grd.BRANCH_ORDER)
                for branch, verdict in d.branch_verdicts.items():
                    self.assertIn(verdict, grd.VERDICTS, f"bad verdict on {branch}")

    def test_terminated_leaves_later_branches_not_evaluated_not_failed(self):
        """The distinction 2.3 is about: never ran != ran and rejected."""
        d = _check(terminated=True, errors=_errors()).decision
        self.assertEqual(d.branch_verdicts[grd.BRANCH_ENV], grd.VERDICT_PASS)
        for branch in (
            grd.BRANCH_STD_JOINT_QPOS,
            grd.BRANCH_EEF_GRIPPER,
            grd.BRANCH_JOINT_RMSE,
            grd.BRANCH_TIMEOUT,
        ):
            self.assertEqual(
                d.branch_verdicts[branch],
                grd.VERDICT_NOT_EVALUATED,
                f"{branch} never ran, so it must not be reported as 'fail'",
            )
            self.assertNotEqual(d.branch_verdicts[branch], grd.VERDICT_FAIL)

    def test_early_state_branch_win_leaves_the_rest_not_evaluated(self):
        d = _check(errors=_errors(std_joint_qpos_rmse=0.01)).decision
        self.assertEqual(d.result, "success_state")
        self.assertEqual(d.success_reason, "state_match_std_joint_qpos")
        self.assertEqual(d.decisive_branch, grd.BRANCH_STD_JOINT_QPOS)
        self.assertEqual(d.branch_verdicts[grd.BRANCH_ENV], grd.VERDICT_FAIL)
        self.assertEqual(d.branch_verdicts[grd.BRANCH_STD_JOINT_QPOS], grd.VERDICT_PASS)
        self.assertEqual(d.branch_verdicts[grd.BRANCH_EEF_GRIPPER], grd.VERDICT_NOT_EVALUATED)
        self.assertEqual(d.branch_verdicts[grd.BRANCH_JOINT_RMSE], grd.VERDICT_NOT_EVALUATED)
        self.assertEqual(d.branch_verdicts[grd.BRANCH_TIMEOUT], grd.VERDICT_NOT_EVALUATED)

    def test_evaluated_and_rejected_is_fail_not_not_evaluated(self):
        """Positive control for the other half of the distinction."""
        d = _check(errors=_errors(), current_step=0, timeout=100).decision
        self.assertEqual(d.result, "in_progress")
        for branch in (
            grd.BRANCH_STD_JOINT_QPOS,
            grd.BRANCH_EEF_GRIPPER,
            grd.BRANCH_JOINT_RMSE,
        ):
            self.assertEqual(
                d.branch_verdicts[branch],
                grd.VERDICT_FAIL,
                f"{branch} ran with usable inputs and rejected -> 'fail'",
            )
            self.assertEqual(d.branch_details[branch], grd._D_ABOVE_THRESHOLD)

    def test_missing_metric_is_not_evaluated_not_fail(self):
        """Absent input is zero information, so it must not count as 'fail'."""
        errs = _errors()
        del errs["std_joint_qpos_rmse"]
        errs["eef_left_pos_err"] = float("inf")
        errs["eef_right_pos_err"] = float("nan")
        d = _check(errors=errs).decision

        self.assertEqual(
            d.branch_verdicts[grd.BRANCH_STD_JOINT_QPOS], grd.VERDICT_NOT_EVALUATED
        )
        self.assertEqual(
            d.branch_details[grd.BRANCH_STD_JOINT_QPOS], grd._D_METRIC_UNAVAILABLE
        )
        self.assertEqual(
            d.branch_verdicts[grd.BRANCH_EEF_GRIPPER], grd.VERDICT_NOT_EVALUATED
        )
        # ...and evaluation still falls through to joint_rmse, as before.
        self.assertEqual(d.branch_verdicts[grd.BRANCH_JOINT_RMSE], grd.VERDICT_FAIL)

    def test_state_match_disabled_is_not_evaluated_with_a_reason(self):
        d = _check(state_match_enabled=False, errors=_errors(), current_step=5, timeout=5).decision
        self.assertEqual(d.result, "timeout")
        for branch in (
            grd.BRANCH_STD_JOINT_QPOS,
            grd.BRANCH_EEF_GRIPPER,
            grd.BRANCH_JOINT_RMSE,
        ):
            self.assertEqual(d.branch_verdicts[branch], grd.VERDICT_NOT_EVALUATED)
            self.assertEqual(d.branch_details[branch], grd._D_STATE_MATCH_DISABLED)

    def test_state_errors_unavailable_is_not_evaluated_with_a_reason(self):
        d = _check(errors=None).decision
        for branch in (
            grd.BRANCH_STD_JOINT_QPOS,
            grd.BRANCH_EEF_GRIPPER,
            grd.BRANCH_JOINT_RMSE,
        ):
            self.assertEqual(d.branch_verdicts[branch], grd.VERDICT_NOT_EVALUATED)
            self.assertTrue(
                d.branch_details[branch].startswith(grd._D_STATE_ERRORS_UNAVAILABLE)
            )

    def test_timeout_branch_polarity_is_documented_and_correct(self):
        fired = _check(errors=_errors(), current_step=100, timeout=100).decision
        self.assertEqual(fired.result, "timeout")
        self.assertEqual(fired.branch_verdicts[grd.BRANCH_TIMEOUT], grd.VERDICT_PASS)
        self.assertEqual(fired.decisive_branch, grd.BRANCH_TIMEOUT)

        not_fired = _check(errors=_errors(), current_step=99, timeout=100).decision
        self.assertEqual(not_fired.result, "in_progress")
        self.assertEqual(not_fired.branch_verdicts[grd.BRANCH_TIMEOUT], grd.VERDICT_FAIL)
        self.assertIsNone(not_fired.decisive_branch)

        self.assertIn("timeout", grd.BRANCH_VERDICT_SEMANTICS)
        self.assertIn("not-evaluated", grd.BRANCH_VERDICT_SEMANTICS)

    def test_success_env_and_success_state_are_separable(self):
        """Spec 2.3: `success` conflates them; result must not."""
        env = _check(terminated=True, errors=_errors()).decision
        state = _check(errors=_errors(std_joint_qpos_rmse=0.01)).decision

        # The pre-existing `success` field is True for both -- that is the bug
        # being worked around, and it is deliberately left intact.
        self.assertTrue(str(env.result).startswith("success"))
        self.assertTrue(str(state.result).startswith("success"))
        # The new booleans separate them.
        self.assertTrue(env.success_env)
        self.assertFalse(env.success_state)
        self.assertFalse(state.success_env)
        self.assertTrue(state.success_state)


# ===========================================================================
# Verdict-preservation: new code must reproduce the original criterion
# ===========================================================================
def _legacy_check(*, terminated, current_step, timeout, state_match_enabled, errors, thr):
    """Verbatim transcription of the pre-change control flow.

    Kept as an oracle so the refactor is checked against behaviour rather than
    against my reading of it.  (Source: eval_golden_rule.py:296-359 at
    d60d650bc, before this branch.)
    """
    if terminated:
        return True, "success_env", "env_terminated"
    if state_match_enabled:
        if errors is not None:
            std_rmse = errors.get("std_joint_qpos_rmse", float("inf"))
            if np.isfinite(std_rmse) and std_rmse <= thr["std_joint_qpos_rmse"]:
                return True, "success_state", "state_match_std_joint_qpos"
            eef_errs = [
                errors.get("eef_left_pos_err", float("inf")),
                errors.get("eef_right_pos_err", float("inf")),
            ]
            grip_errs = [
                errors.get("gripper_left_qpos_err", float("inf")),
                errors.get("gripper_right_qpos_err", float("inf")),
            ]
            has_eef = any(np.isfinite(x) for x in eef_errs)
            has_grip = any(np.isfinite(x) for x in grip_errs)
            if has_eef and has_grip:
                if min(eef_errs) <= thr["eef_pos"] and min(grip_errs) <= thr["gripper_qpos"]:
                    return True, "success_state", "state_match_eef_gripper"
            jq = errors.get("joint_qpos_rmse", float("inf"))
            if np.isfinite(jq) and jq <= thr["joint_qpos_rmse"]:
                return True, "success_state", "state_match_joint_rmse"
    if current_step >= timeout:
        return True, "timeout", None
    return False, "in_progress", None


class TestVerdictPreservation(unittest.TestCase):
    def test_matches_legacy_criterion_across_a_grid(self):
        inf, nan = float("inf"), float("nan")
        error_variants = [
            None,
            {},
            _errors(),
            _errors(std_joint_qpos_rmse=0.25),      # exactly on threshold
            _errors(std_joint_qpos_rmse=0.2501),    # just above
            _errors(std_joint_qpos_rmse=nan),
            _errors(std_joint_qpos_rmse=inf, eef_left_pos_err=0.05, gripper_left_qpos_err=0.01),
            _errors(std_joint_qpos_rmse=inf, eef_left_pos_err=0.05, gripper_left_qpos_err=9.0),
            _errors(std_joint_qpos_rmse=inf, eef_left_pos_err=inf, eef_right_pos_err=inf,
                    gripper_left_qpos_err=0.01, joint_qpos_rmse=0.1),
            _errors(std_joint_qpos_rmse=inf, gripper_left_qpos_err=inf,
                    gripper_right_qpos_err=inf, joint_qpos_rmse=0.1),
            _errors(std_joint_qpos_rmse=inf, joint_qpos_rmse=0.25),
        ]
        n_cases = 0
        seen_results = set()
        for errs in error_variants:
            for terminated in (False, True):
                for state_match_enabled in (True, False):
                    for current_step, timeout in ((0, 100), (100, 100), (101, 100)):
                        n_cases += 1
                        want = _legacy_check(
                            terminated=terminated,
                            current_step=current_step,
                            timeout=timeout,
                            state_match_enabled=state_match_enabled,
                            errors=errs,
                            thr=DEFAULT_THRESHOLDS,
                        )
                        got = _check(
                            terminated=terminated,
                            current_step=current_step,
                            timeout=timeout,
                            state_match_enabled=state_match_enabled,
                            errors=errs,
                        ).decision
                        seen_results.add(got.result)
                        self.assertEqual(
                            (got.is_done, got.result, got.success_reason),
                            want,
                            f"verdict drift: errors={errs} terminated={terminated} "
                            f"state_match={state_match_enabled} step={current_step}/{timeout}",
                        )
        # Guard against a vacuous grid that only ever hits one outcome.
        self.assertEqual(n_cases, 132)
        self.assertEqual(
            seen_results,
            {"success_env", "success_state", "timeout", "in_progress"},
            "grid must exercise every result type",
        )


# ===========================================================================
# Spec 2.1 -- thresholds reach the metrics JSON
# ===========================================================================
class _FakeCfg(dict):
    """Minimal stand-in for the OmegaConf node used by the evaluator."""

    def get(self, key, default=None):
        return dict.get(self, key, default)


class _ThresholdSource:
    """Runs the *real* threshold resolution and the *real* snapshot builder.

    ``_get_cfg_float`` / ``get_primitive_success_thresholds`` are lifted out of
    ``eval_subtask_reset.py`` and ``_build_eval_config_snapshot`` out of
    ``eval_golden_rule.py``; neither file can be imported here because both
    pull in omnigibson.  Lifting rather than re-implementing means the test
    fails if the shipped code changes, which a replica would not.
    """

    def __init__(self, cfg, diagnose_on_terminated=True):
        self.cfg = cfg
        self._diagnose_on_terminated = diagnose_on_terminated
        self.skill_timeout_steps = int(cfg.get("skill_timeout_steps", 300))
        self.skill_max_steps_multiplier = float(cfg.get("skill_max_steps_multiplier", 2.0))


def _build_eval_config_snapshot(source, cfg):
    """The dict eval_golden_rule.run_episode embeds under 'eval_config'."""
    return source._build_eval_config_snapshot()


# Bind the shipped implementations onto the stand-in.
for _name, _fn in _extract_methods_from_source(
    _EVAL_SUBTASK_RESET_PY, "SubTaskEvaluator", {"_get_cfg_float", "get_primitive_success_thresholds"}
).items():
    setattr(_ThresholdSource, _name, _fn)
for _name, _fn in _extract_methods_from_source(
    _EVAL_GOLDEN_RULE_PY, "GoldenRuleEvaluator", {"_build_eval_config_snapshot"}
).items():
    setattr(_ThresholdSource, _name, _fn)


class TestThresholdDump(unittest.TestCase):
    def test_overridden_thresholds_land_in_the_json_not_the_defaults(self):
        """2.1's actual failure mode: defaults != the values that ran."""
        cfg = _FakeCfg(
            {
                "primitive_success_eef_pos_threshold": 0.05,
                "primitive_success_std_joint_qpos_rmse_threshold": 0.4,
            }
        )
        snapshot = _build_eval_config_snapshot(_ThresholdSource(cfg), cfg)
        payload = json.loads(json.dumps({"eval_config": snapshot}))
        thr = payload["eval_config"]["primitive_success_thresholds"]

        self.assertEqual(thr["eef_pos"], 0.05)
        self.assertNotEqual(thr["eef_pos"], 0.12, "must not report the source default")
        self.assertEqual(thr["std_joint_qpos_rmse"], 0.4)
        # Untouched keys still resolve to their defaults.
        self.assertEqual(thr["gripper_qpos"], 0.03)
        self.assertEqual(thr["joint_qpos_rmse"], 0.25)

    def test_all_six_threshold_keys_present_and_json_serialisable(self):
        cfg = _FakeCfg({})
        snapshot = _build_eval_config_snapshot(_ThresholdSource(cfg), cfg)
        thr = snapshot["primitive_success_thresholds"]
        self.assertEqual(
            set(thr),
            {"base_pos", "yaw", "eef_pos", "gripper_qpos",
             "std_joint_qpos_rmse", "joint_qpos_rmse"},
        )
        for key, value in thr.items():
            self.assertIsInstance(value, float, key)
        json.dumps(snapshot)  # must not raise

    def test_thresholds_used_by_the_checker_are_the_ones_recorded(self):
        """Guards against dumping one dict while deciding with another."""
        cfg = _FakeCfg({"primitive_success_std_joint_qpos_rmse_threshold": 0.5})
        thr = _ThresholdSource(cfg).get_primitive_success_thresholds()
        recorded = _build_eval_config_snapshot(_ThresholdSource(cfg), cfg)[
            "primitive_success_thresholds"
        ]
        self.assertEqual(thr, recorded)

        # 0.3 passes under the override (0.5) and fails under the default (0.25).
        loose = _check(errors=_errors(std_joint_qpos_rmse=0.3), thresholds=thr).decision
        strict = _check(
            errors=_errors(std_joint_qpos_rmse=0.3), thresholds=DEFAULT_THRESHOLDS
        ).decision
        self.assertEqual(loose.result, "success_state")
        self.assertEqual(strict.branch_verdicts[grd.BRANCH_STD_JOINT_QPOS], grd.VERDICT_FAIL)


# ===========================================================================
# Spec 2.4 -- per-demo reload actually changes the parquet
# ===========================================================================
def _write_demo_parquet(root: Path, task_id: int, demo_id: str, fill: float, n_frames: int = 4):
    d = root / "data" / f"task-{task_id:04d}"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"episode_{demo_id}.parquet"
    actions = [list(np.full(3, fill, dtype=np.float32) + i) for i in range(n_frames)]
    pd.DataFrame({"action": actions}).to_parquet(path)
    return path


class _StubModelCfg(dict):
    def get(self, key, default=None):
        return dict.get(self, key, default)


class _FakeEvaluator:
    """Exercises the real reload helper against a real replay policy.

    ``GoldenRuleEvaluator`` itself cannot be imported here (it pulls in
    omnigibson), so the two methods added by 2.4 are transplanted onto this
    stand-in.  They are copied by reference from the source file below, not
    re-implemented, so a drift between the two would fail the import.
    """

    def __init__(self, policy, cfg):
        self.policy = policy
        self.cfg = cfg
        self._last_replay_source = None
        self._last_replay_reload = None


# Bind the real implementations onto the stand-in by reading them out of the
# source file.  eval_golden_rule.py cannot be imported (omnigibson), so the two
# self-contained methods are extracted from its AST and executed in isolation.
_METHODS = _extract_methods_from_source(
    _EVAL_GOLDEN_RULE_PY,
    "GoldenRuleEvaluator",
    {"_resolve_replay_policy", "reload_replay_policy_for_demo"},
)
_FakeEvaluator._resolve_replay_policy = _METHODS["_resolve_replay_policy"]
_FakeEvaluator.reload_replay_policy_for_demo = _METHODS["reload_replay_policy_for_demo"]


class TestPerDemoReload(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.p_a = _write_demo_parquet(self.root, 0, "00000010", fill=10.0)
        self.p_b = _write_demo_parquet(self.root, 0, "00000020", fill=20.0)
        self.addCleanup(self._tmp.cleanup)

    def _make_policy(self, demo_id="00000010"):
        return policies.DemoActionReplayPolicy(
            demo_data_path=str(self.root), demo_id=demo_id, action_dim=3
        )

    def test_policy_exposes_the_parquet_it_loaded(self):
        policy = self._make_policy()
        self.assertEqual(policy.loaded_parquet_path, str(self.p_a))

    def test_reload_changes_the_parquet_path_positive_control(self):
        """2.4's required acceptance check: the path must actually change."""
        policy = self._make_policy("00000010")
        ev = _FakeEvaluator(policy, _FakeCfg({"model": _StubModelCfg({"task_id": 0})}))

        first = ev.reload_replay_policy_for_demo("00000010")
        second = ev.reload_replay_policy_for_demo("00000020")

        self.assertIsNotNone(second)
        self.assertEqual(second["parquet_before"], str(self.p_a))
        self.assertEqual(second["parquet_after"], str(self.p_b))
        self.assertTrue(second["parquet_changed"])
        self.assertEqual(policy.loaded_parquet_path, str(self.p_b))
        self.assertEqual(ev._last_replay_source, str(self.p_b))

        # Negative control: re-binding the SAME demo must report no change,
        # otherwise "changed" would be true for every call and prove nothing.
        self.assertFalse(first["parquet_changed"])
        third = ev.reload_replay_policy_for_demo("00000020")
        self.assertFalse(third["parquet_changed"])

    def test_replayed_actions_change_with_the_demo(self):
        """Path changing is necessary; the actions changing is the point."""
        policy = self._make_policy("00000010")
        ev = _FakeEvaluator(policy, _FakeCfg({"model": _StubModelCfg({"task_id": 0})}))

        a0 = float(policy.act({})[0])
        ev.reload_replay_policy_for_demo("00000020")
        policy.reset()
        b0 = float(policy.act({})[0])

        self.assertEqual(a0, 10.0)
        self.assertEqual(b0, 20.0)
        self.assertNotEqual(a0, b0)

    def test_without_the_reload_the_second_demo_replays_the_first(self):
        """Reproduces the defect 2.4 describes, so the fix is not vacuous."""
        policy = self._make_policy("00000010")
        policy.reset()
        stale = float(policy.act({})[0])
        self.assertEqual(stale, 10.0, "still demo A's actions -- the silent bug")

    def test_reload_rewinds_the_frame_cursor(self):
        policy = self._make_policy("00000010")
        ev = _FakeEvaluator(policy, _FakeCfg({"model": _StubModelCfg({"task_id": 0})}))
        policy.act({})
        policy.act({})
        self.assertEqual(policy._step, 2)
        ev.reload_replay_policy_for_demo("00000020")
        self.assertEqual(policy._step, 0)

    def test_frame_window_is_applied_before_the_reload(self):
        policy = self._make_policy("00000010")
        cfg = _FakeCfg(
            {"model": _StubModelCfg({"task_id": 0, "start_frame": 2, "end_frame": "none"})}
        )
        ev = _FakeEvaluator(policy, cfg)
        record = ev.reload_replay_policy_for_demo("00000020")
        self.assertEqual(policy.start_frame, 2)
        self.assertIsNone(policy.end_frame)
        self.assertEqual(record["start_frame"], 2)

    def test_non_replay_policy_is_left_alone(self):
        class _NoLoadDemo:
            pass

        ev = _FakeEvaluator(_NoLoadDemo(), _FakeCfg({"model": _StubModelCfg({})}))
        self.assertIsNone(ev.reload_replay_policy_for_demo("00000020"))
        self.assertIsNone(ev._last_replay_source)

    def test_wrapped_policy_is_found_one_level_in(self):
        inner = self._make_policy("00000010")

        class _Wrapper:
            def __init__(self, policy):
                self.policy = policy

        ev = _FakeEvaluator(_Wrapper(inner), _FakeCfg({"model": _StubModelCfg({"task_id": 0})}))
        record = ev.reload_replay_policy_for_demo("00000020")
        self.assertIsNotNone(record)
        self.assertEqual(record["parquet_after"], str(self.p_b))
        self.assertEqual(inner.loaded_parquet_path, str(self.p_b))

    def test_missing_parquet_raises_rather_than_silently_keeping_stale_data(self):
        """Precondition for the setup_episode guard below."""
        policy = self._make_policy("00000010")
        ev = _FakeEvaluator(policy, _FakeCfg({"model": _StubModelCfg({"task_id": 0})}))
        with self.assertRaises(FileNotFoundError):
            ev.reload_replay_policy_for_demo("00009999")

    def test_failed_load_leaves_the_policy_identity_untouched(self):
        """A failed load must not make the policy claim the demo it failed on.

        ``load_demo`` decides whether to reload by comparing against
        ``self.demo_id``.  If a failed load had already overwritten that field,
        retrying the same id would take the "already loaded" branch, return
        cleanly, and replay the previous demo's actions under the new name.
        """
        policy = self._make_policy("00000010")
        self.assertEqual(policy.demo_id, "00000010")

        with self.assertRaises(FileNotFoundError):
            policy.load_demo(demo_id="00009999", task_id=0)

        self.assertEqual(policy.demo_id, "00000010", "identity must not have moved")
        self.assertEqual(policy.loaded_parquet_path, str(self.p_a))
        self.assertEqual(float(policy.act({})[0]), 10.0)

        # The retry must raise again rather than quietly succeeding.
        with self.assertRaises(FileNotFoundError):
            policy.load_demo(demo_id="00009999", task_id=0)

        # A genuine switch afterwards still works.
        policy.load_demo(demo_id="00000020", task_id=0)
        self.assertEqual(policy.demo_id, "00000020")
        self.assertEqual(policy.loaded_parquet_path, str(self.p_b))


class TestSetupEpisodeGuard(unittest.TestCase):
    """setup_episode must refuse the episode when the rebind fails.

    Running on would replay the previous demo's actions -- the silent defect --
    so the correct outcome is a recorded not-measured marker, not a wrong
    number and not a crash that kills the remaining demos.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _write_demo_parquet(self.root, 0, "00000010", fill=10.0)
        self.addCleanup(self._tmp.cleanup)

        class _StubPlanLoader:
            def __init__(self, **kwargs):
                pass

            def load_plan(self):
                return [{"description": "press", "frame_duration": [0, 4]}]

        methods = _extract_methods_from_source(
            _EVAL_GOLDEN_RULE_PY,
            "GoldenRuleEvaluator",
            {"setup_episode", "_resolve_replay_policy", "reload_replay_policy_for_demo"},
            extra_globals={"GTPlanLoader": _StubPlanLoader},
        )

        class _Task:
            name = "turning_on_radio"

        class _Cfg(_FakeCfg):
            task = _Task()

        class _Ev:
            def __init__(self, policy, cfg):
                self.policy = policy
                self.cfg = cfg
                self.use_gt_plan = True
                self.current_skill_plan = []
                self._golden_rule_policy = None
                self._gt_plan_loader = None
                self._last_replay_source = None
                self._last_replay_reload = None
                self._setup_error = None
                self.current_demo_data = None
                self.current_rawdata_hdf5 = None
                self.current_primitive_state_cache = None

            load_demo_lowdim_data = staticmethod(lambda demo_id: None)
            load_rawdata_hdf5 = staticmethod(lambda demo_id: None)
            load_primitive_state_cache = staticmethod(lambda demo_id: None)

        for name, fn in methods.items():
            setattr(_Ev, name, fn)

        self.policy = policies.DemoActionReplayPolicy(
            demo_data_path=str(self.root), demo_id="00000010", action_dim=3
        )
        self.ev = _Ev(
            self.policy,
            _Cfg({"demo_data_path": str(self.root), "model": _StubModelCfg({"task_id": 0})}),
        )

    def test_good_demo_sets_up_and_rebinds(self):
        """Positive control: the guard must not reject a healthy demo."""
        self.assertTrue(self.ev.setup_episode("00000010"))
        self.assertIsNone(self.ev._setup_error)
        self.assertIsNotNone(self.ev._last_replay_source)

    def test_missing_demo_returns_false_with_a_reason(self):
        self.assertFalse(self.ev.setup_episode("00009999"))
        self.assertIsNotNone(self.ev._setup_error)
        self.assertIn("replay_reload_failed", self.ev._setup_error)
        self.assertIn("FileNotFoundError", self.ev._setup_error)

    def test_stale_actions_are_not_used_after_a_failed_rebind(self):
        """The property that matters, stated directly.

        Both attempts must be refused.  Before the atomic-commit fix in
        ``_load_demo_data`` the *second* call returned True, which would have
        run the episode on demo A's actions while labelling it 00009999.
        """
        self.assertFalse(self.ev.setup_episode("00009999"))
        self.assertFalse(self.ev.setup_episode("00009999"))
        # The policy still honestly reports the demo it actually holds.
        self.assertEqual(self.policy.demo_id, "00000010")


# ===========================================================================
# Aggregate summary
# ===========================================================================
class TestSummary(unittest.TestCase):
    def test_stratifies_by_result_and_branch(self):
        def record(result, reason, verdicts):
            return {"result": result, "success_reason": reason, "branch_verdicts": verdicts}

        env_d = _check(terminated=True, errors=_errors()).decision
        state_d = _check(errors=_errors(std_joint_qpos_rmse=0.01)).decision
        per_demo = [
            {
                "skill_diagnostics": [
                    record("success_env", env_d.success_reason, env_d.branch_verdicts),
                    record("success_env", env_d.success_reason, env_d.branch_verdicts),
                    record("success_state", state_d.success_reason, state_d.branch_verdicts),
                ]
            }
        ]
        summary = grd.summarize_skill_diagnostics(per_demo)
        self.assertEqual(summary["n_skill_records"], 3)
        self.assertEqual(summary["by_result"], {"success_env": 2, "success_state": 1})
        self.assertEqual(
            summary["branch_verdict_counts"][grd.BRANCH_STD_JOINT_QPOS],
            {grd.VERDICT_PASS: 1, grd.VERDICT_FAIL: 0, grd.VERDICT_NOT_EVALUATED: 2},
        )
        json.dumps(summary)

    def test_tolerates_missing_branch_tables(self):
        summary = grd.summarize_skill_diagnostics(
            [{"skill_diagnostics": [{"result": "timeout", "success_reason": None}]}]
        )
        self.assertEqual(summary["n_skill_records"], 1)
        self.assertEqual(summary["n_records_with_branch_table"], 0)
        self.assertEqual(summary["by_success_reason"], {"null": 1})


if __name__ == "__main__":
    unittest.main(verbosity=2)
