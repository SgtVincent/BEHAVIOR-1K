"""CPU tests for the manifest-driven MP executor plugin loader.

Sim-free: the fake plugin is written to a temp dir and loaded through the
same importlib seam the real cuRobo v2 integrator module will use.
"""
import sys
import textwrap

import pytest

from omnigibson.moma.mp_plugin import (
    PluginSpec,
    PluginUnavailable,
    classify_plugin,
    load_executor,
)


def _manifest(module: str, factory: str = "build_executor", **kwargs):
    return {"moma": {"executor": {"module": module, "factory": factory, "version": "test",
                                  "kwargs": kwargs}}}


def _write_plugin(tmp_path, name: str, body: str):
    (tmp_path / f"{name}.py").write_text(textwrap.dedent(body), encoding="utf-8")
    sys.path.insert(0, str(tmp_path))
    return name


def test_manifest_without_executor_block_rejected():
    with pytest.raises(PluginUnavailable, match="manifest_invalid"):
        load_executor({})


def test_missing_module_classified_as_module_missing():
    manifest = _manifest("omnigibson.moma._no_such_plugin_module_xyz")
    with pytest.raises(PluginUnavailable) as exc:
        load_executor(manifest)
    assert exc.value.reason == "module_missing"


def test_missing_factory_classified(tmp_path, monkeypatch):
    module = _write_plugin(tmp_path, "fake_exec_no_factory", "VALUE = 1\n")
    try:
        with pytest.raises(PluginUnavailable) as exc:
            load_executor(_manifest(module, factory="build_executor"))
        assert exc.value.reason == "factory_missing"
    finally:
        sys.path.remove(str(tmp_path))


def test_factory_returning_non_executor_rejected(tmp_path):
    body = """
    def build_executor(**kwargs):
        return object()
    """
    module = _write_plugin(tmp_path, "fake_exec_bad_contract", body)
    try:
        with pytest.raises(PluginUnavailable) as exc:
            load_executor(_manifest(module))
        assert exc.value.reason == "executor_contract_missing"
    finally:
        sys.path.remove(str(tmp_path))


def test_plugin_executor_loads_and_receives_context(tmp_path):
    body = """
    def build_executor(cfg=None, robot=None, marker=None):
        from omnigibson.moma.schema import native23_action
        class _Exec:
            def plan(self, target):
                return {"action23": native23_action([0.0] * 23), "certificate": {"marker": marker}}
        return _Exec()
    """
    module = _write_plugin(tmp_path, "fake_exec_ok", body)
    try:
        executor = load_executor(_manifest(module, marker="hello"), context={"cfg": None, "robot": None})
        plan = executor.plan({"primitive": "approach_object"})
        assert plan["certificate"]["marker"] == "hello"
        assert len(plan["action23"].vector) == 23
        report = classify_plugin(PluginSpec.from_manifest(_manifest(module)))
        assert report["status"] == "OK"
    finally:
        sys.path.remove(str(tmp_path))


def test_classify_missing_module_reports_missing():
    report = classify_plugin(PluginSpec.from_manifest(_manifest("omnigibson.moma._nope_xyz")))
    assert report["status"] == "MISSING"
