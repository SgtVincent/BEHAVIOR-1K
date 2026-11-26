#!/usr/bin/env python3
"""Stdlib test runner for the ``curobo_v2`` package.

Executed on the remote 2026 environment only. Discovers ``test_*.py`` modules
under ``curobo_v2/tests/`` and runs every ``test_*`` callable with no arguments.
The heavy ``omnigibson`` package ``__init__`` is bypassed by placing the
``omnigibson`` directory on ``sys.path`` and importing the top-level
``curobo_v2`` package, mirroring ``brvla_bridge/tests/run_tests.py``.
"""
import importlib
import json
import pathlib
import sys
import traceback
import unittest

PKG_DIR = pathlib.Path(__file__).resolve().parent.parent
REPO_PARENT = PKG_DIR.parent
TESTS_DIR = PKG_DIR / "tests"
sys.path.insert(0, str(REPO_PARENT))
PKG_NAME = PKG_DIR.name


def _run_case(case: str, runner: callable, passed: list, failed: list, errors: list) -> None:
    try:
        runner()
        passed.append(case)
    except Exception as exc:  # noqa: BLE001 - runner records the real failure
        failed.append(case)
        errors.append({"case": case, "error": f"{type(exc).__name__}: {exc}",
                       "trace": traceback.format_exc()})


def main() -> int:
    passed, failed, errors = [], [], []
    for path in sorted(TESTS_DIR.glob("test_*.py")):
        module_name = f"{PKG_NAME}.tests.{path.stem}"
        module = importlib.import_module(module_name)
        for attr in sorted(dir(module)):
            obj = getattr(module, attr)
            if isinstance(obj, type) and issubclass(obj, unittest.TestCase):
                case = f"{module_name}.{attr}"
                for method in sorted(m for m in dir(obj) if m.startswith("test_")):
                    tc = obj(method)
                    _run_case(f"{case}.{method}", getattr(tc, method), passed, failed, errors)
            elif callable(obj) and attr.startswith("test_"):
                _run_case(f"{module_name}.{attr}", obj, passed, failed, errors)
    summary = {"passed": len(passed), "failed": len(failed), "cases": passed + failed}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for err in errors:
        print(f"\n=== {err['case']} ===\n{err['trace']}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
