#!/usr/bin/env python3
"""Stdlib test runner for the brvla_bridge package.

Executed on the remote 2026 environment only. It discovers ``test_*.py``
modules under ``brvla_bridge/tests/`` and runs every ``test_*`` function with
no arguments. A failing assertion or declared exception aborts the run with a
non-zero code; a local green report must not be claimed before this runner
executes remotely.
"""
import importlib
import json
import pathlib
import sys
import traceback

PKG_DIR = pathlib.Path(__file__).resolve().parent.parent
REPO_PARENT = PKG_DIR.parent
TESTS_DIR = PKG_DIR / "tests"
sys.path.insert(0, str(REPO_PARENT))
PKG_NAME = PKG_DIR.name


def main() -> int:
    passed, failed, errors = [], [], []
    for path in sorted(TESTS_DIR.glob("test_*.py")):
        module_name = f"{PKG_NAME}.tests.{path.stem}"
        module = importlib.import_module(module_name)
        for attr in sorted(dir(module)):
            if not attr.startswith("test_"):
                continue
            fn = getattr(module, attr)
            if not callable(fn):
                continue
            case = f"{module_name}.{attr}"
            try:
                fn()
                passed.append(case)
            except Exception as exc:  # noqa: BLE001 - runner records the real failure
                failed.append(case)
                errors.append({"case": case, "error": f"{type(exc).__name__}: {exc}",
                               "trace": traceback.format_exc()})
    summary = {"passed": len(passed), "failed": len(failed), "cases": passed + failed}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for err in errors:
        print(f"\n=== {err['case']} ===\n{err['trace']}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
