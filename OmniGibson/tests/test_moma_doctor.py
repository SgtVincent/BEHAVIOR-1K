"""CPU tests for the static preflight doctor: classified blockers, deferred
GPU-stage dependencies, and the single missing-piece pointer.
"""
import sys
import textwrap

from omnigibson.moma.config import RunnerConfig
from omnigibson.moma.doctor import run_doctor


def _cfg(**overrides):
    return RunnerConfig(run_id="doctor-test", **overrides)


def test_valid_config_with_missing_plugin_points_at_executor(tmp_path):
    manifest = {"moma": {"executor": {"module": "omnigibson.moma._no_such_doctor_xyz",
                                      "factory": "build_executor"}}}
    report = run_doctor(_cfg(), manifest=manifest)
    assert report.ready() is False
    pointer = report.missing_pointer()
    assert pointer["kind"] == "executor_plugin"
    assert "module_missing" in pointer["detail"] or "not resolvable" in pointer["detail"]


def test_invalid_config_is_blocking_fail():
    cfg = RunnerConfig(run_id="../bad")
    report = run_doctor(cfg, manifest=None)
    assert report.ready() is False
    assert any(c.status == "FAIL" and c.blocking for c in report.checks)


def test_optional_gpu_deps_are_deferred_not_blocking():
    report = run_doctor(_cfg(), manifest=None)
    by_name = {c.name: c for c in report.checks}
    assert by_name["dep:numpy"].status == "OK"
    for dep in ("dep:torch", "dep:omnigibson", "dep:cvxpy"):
        assert by_name[dep].status in ("OK", "DEFERRED")
        assert by_name[dep].blocking is False


def test_missing_asset_path_is_blocking_missing(tmp_path):
    missing = str(tmp_path / "absent.npz")
    cfg = _cfg(initial_state_npz=missing)
    report = run_doctor(cfg, manifest=None)
    asset = next(c for c in report.checks if c.name == "asset:initial_state_npz")
    assert asset.status == "MISSING" and asset.blocking is True


def test_invalid_s2_url_is_blocking_fail():
    cfg = _cfg(model_base_url="ftp://example.com/v1")
    report = run_doctor(cfg, manifest=None)
    s2 = next(c for c in report.checks if c.name == "s2_endpoint")
    assert s2.status == "FAIL" and s2.blocking is True


def test_present_plugin_and_deferred_assets_report_ready(tmp_path):
    body = """
    def build_executor(**kwargs):
        class _Exec:
            def plan(self, target):
                return None
        return _Exec()
    """
    module_file = tmp_path / "doctor_fake_exec.py"
    module_file.write_text(textwrap.dedent(body), encoding="utf-8")
    sys.path.insert(0, str(tmp_path))
    try:
        manifest = {"moma": {"executor": {"module": "doctor_fake_exec", "factory": "build_executor"}}}
        report = run_doctor(_cfg(), manifest=manifest)
        exec_check = next(c for c in report.checks if c.name == "executor_plugin")
        assert exec_check.status == "OK"
        # Asset paths unconfigured -> DEFERRED, not blocking.
        assert report.ready() is True
    finally:
        sys.path.remove(str(tmp_path))
