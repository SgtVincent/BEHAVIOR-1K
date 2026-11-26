"""CPU-4: sim-free leaf modules must not pull simulator/model dependencies.

The guard checks two ways:
  1. static AST scan of the MoMa package leaves for forbidden top-level imports;
  2. a fresh interpreter subprocess loads each leaf file standalone (bypassing
     the heavy ``omnigibson`` package ``__init__``) and confirms no
     omni/isaacsim/torch/cvxpy/numba/h5py/av module landed in sys.modules.

Simulator coupling lives behind the SimAdapter seam and is exercised only in
the GPU-bound canary.
"""
import ast
import subprocess
import sys
from pathlib import Path

import pytest

import omnigibson.moma as moma_pkg

FORBIDDEN_TOP = {
    "omnigibson", "omni", "isaacsim", "torch", "cvxpy", "numba",
    "h5py", "av", "warp", "curobo", "pytorch3d",
}

# Leaves loadable standalone without intra-package relative imports.
STANDALONE_LEAVES = [
    "schema.py",
    "navigation.py",
    "controller_bounds.py",
    "dynamics.py",
    "object_provider.py",
    "observer/three_can.py",
    "s2/contract.py",
]

ALL_LEAVES = [
    "schema.py",
    "seams.py",
    "navigation.py",
    "controller_bounds.py",
    "dynamics.py",
    "object_provider.py",
    "observer/three_can.py",
    "s2/contract.py",
    "s2/client.py",
    "sdk/__init__.py",
    "sdk/robot_accessor.py",
    "sdk/env_adapter.py",
    "operator.py",
    "config.py",
    "mp_plugin.py",
    "episode_writer.py",
    "runner.py",
    "doctor.py",
]


def _module_path(rel: str) -> Path:
    return Path(moma_pkg.__file__).parent / rel


def test_leaves_exist():
    for rel in ALL_LEAVES:
        assert _module_path(rel).is_file(), f"missing MoMa leaf: {rel}"


def test_no_forbidden_top_level_imports():
    for rel in ALL_LEAVES:
        tree = ast.parse(_module_path(rel).read_text(encoding="utf-8"), filename=rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            for name in names:
                assert name not in FORBIDDEN_TOP, f"{rel} imports forbidden dependency {name}"


def test_subprocess_loads_leaves_without_runtime_heavy_deps():
    root = Path(moma_pkg.__file__).parent
    load_lines = []
    for rel in STANDALONE_LEAVES:
        mod_name = "moma_leaf_" + rel.replace("/", "_").replace(".py", "")
        load_lines.append(
            f"import importlib.util, sys\n"
            f"spec = importlib.util.spec_from_file_location({mod_name!r}, "
            f"{str(root / rel)!r})\n"
            f"m = importlib.util.module_from_spec(spec)\n"
            f"sys.modules[{mod_name!r}] = m\n"
            f"spec.loader.exec_module(m)\n"
        )
    code = (
        "".join(load_lines)
        + "bad = [m for m in sys.modules if m.split('.')[0] in "
        "{'omni','isaacsim','torch','cvxpy','numba','h5py','av','warp','curobo'}]\n"
        "print('MODULES', sorted(bad))\n"
        "assert not bad, bad\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "MODULES []" in result.stdout
