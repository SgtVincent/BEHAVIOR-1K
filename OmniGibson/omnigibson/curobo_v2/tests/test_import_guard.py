# Copyright (c) 2026 B1K2026 integration line
# SPDX-License-Identifier: Apache-2.0
"""Import-guard mirroring ``test_moma_import_guard``.

The v2 lane must be resolvable by the MoMa doctor without booting Kit, torch or
curobo. Two checks:

1. static AST scan of every leaf for forbidden top-level imports;
2. a fresh interpreter loads each standalone leaf (no intra-package relative
   imports) and confirms no omni/isaacsim/torch/cvxpy/numba/h5py/av/warp/
   curobo/pytorch3d module landed in ``sys.modules``.

Heavy planner imports are deferred to ``CuroboV2Backend._bind`` in the
GPU-bound stage.
"""
import ast
import pathlib
import subprocess
import sys

PKG_DIR = pathlib.Path(__file__).resolve().parent.parent

FORBIDDEN_TOP = {
    "omnigibson", "omni", "isaacsim", "torch", "cvxpy", "numba",
    "h5py", "av", "warp", "curobo", "pytorch3d",
}

ALL_LEAVES = [
    "__init__.py",
    "contracts.py",
    "errors.py",
    "backend.py",
    "executor.py",
    "factory.py",
]

# Leaves without intra-package relative imports, loadable standalone.
STANDALONE_LEAVES = ["contracts.py", "errors.py"]


def _module_path(rel: str) -> pathlib.Path:
    return PKG_DIR / rel


def test_leaves_exist() -> None:
    for rel in ALL_LEAVES:
        assert _module_path(rel).is_file(), f"missing leaf: {rel}"


def _top_level_imports(tree: ast.Module):
    """Yield import statements at module body only.

    Imports nested inside methods/classes are lazy by design (heavy planner
    modules land in the GPU-bound ``_bind`` stage) and are not doctor-loadable.
    """
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node


def test_no_forbidden_top_level_imports() -> None:
    for rel in ALL_LEAVES:
        tree = ast.parse(_module_path(rel).read_text(encoding="utf-8"), filename=rel)
        for node in _top_level_imports(tree):
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            else:
                names = [(node.module or "").split(".")[0]]
            for name in names:
                assert name not in FORBIDDEN_TOP, f"{rel} imports forbidden dependency {name}"


def test_subprocess_loads_leaves_without_runtime_heavy_deps() -> None:
    load_lines = []
    for rel in STANDALONE_LEAVES:
        mod_name = "curobo_v2_leaf_" + rel.replace(".py", "")
        load_lines.append(
            f"import importlib.util, sys\n"
            f"spec = importlib.util.spec_from_file_location({mod_name!r}, "
            f"{str(_module_path(rel))!r})\n"
            f"m = importlib.util.module_from_spec(spec)\n"
            f"sys.modules[{mod_name!r}] = m\n"
            f"spec.loader.exec_module(m)\n"
        )
    code = (
        "".join(load_lines)
        + "bad = [m for m in sys.modules if m.split('.')[0] in "
        "{'omni','isaacsim','torch','cvxpy','numba','h5py','av','warp','curobo','pytorch3d'}]\n"
        "print('MODULES', sorted(bad))\n"
        "assert not bad, bad\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "MODULES []" in result.stdout
