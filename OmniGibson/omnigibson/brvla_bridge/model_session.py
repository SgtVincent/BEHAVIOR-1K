"""Frozen server/session launcher record and private dependency loader.

The frozen WS server and its loader live on the private NAS and are imported by
absolute path from the private manifest — this package never vendors, copies, or
rewrites that closed-source wiring. At session start we record the resolved
paths, sha256 identity, effective Python/dependency versions, device assignment,
and which referenced paths actually exist. Model loading and forward inference
are not performed here; a CPU-side record does not count as a model-forward
admission.
"""
import hashlib
import importlib.util
import os
import platform
import sys
from typing import Any, Dict, Optional

from .errors import Port2026Error

# Device roles for the BR lane: evaluator binds GPU1, model server binds GPU2.
# Physical UUIDs come from the private manifest worker section; this module never
# sets CUDA_VISIBLE_DEVICES itself.
EVALUATOR_GPU_ROLE = "gpu1"
MODEL_GPU_ROLE = "gpu2"


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _dependency_versions() -> Dict[str, str]:
    """Best-effort dependency versions; UNKNOWN when a package is not importable."""
    versions: Dict[str, str] = {}
    for package in ("torch", "numpy", "lerobot", "websockets", "requests", "msgpack"):
        try:
            module = __import__(package)
            versions[package] = getattr(module, "__version__", "no_version_attr")
        except Exception:
            versions[package] = "UNKNOWN"
    return versions


def record_runtime_environment() -> Dict[str, Any]:
    """Record the effective interpreter, cwd, device env, and dep versions."""
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "brvla_manifest_path": os.environ.get("BRVLA_MANIFEST_PATH"),
        "dependencies": _dependency_versions(),
    }


def _manifest_refs(manifest) -> Dict[str, Any]:
    checkpoint = getattr(manifest, "checkpoint", {}) or {}
    stats = getattr(manifest, "stats", {}) or {}
    server_session = getattr(manifest, "server_session", {}) or {}
    worker = getattr(manifest, "worker", {}) or {}
    return {"checkpoint": checkpoint, "stats": stats, "server_session": server_session,
            "worker": worker}


def record_session(manifest, runtime: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Record frozen identity refs, device roles, and path-existence flags.

    Existence is checked with ``os.path.exists`` only; weights, stats JSON, and
    datasets are never opened or rehashed here.
    """
    refs = _manifest_refs(manifest)
    checkpoint = refs["checkpoint"]
    stats = refs["stats"]
    server_session = refs["server_session"]
    worker = refs["worker"]

    full = stats.get("stats_full", {})
    old = stats.get("stats_251101", {})
    ckpt_dir = checkpoint.get("dir")
    server_path = server_session.get("frozen_server_path")

    def exists(path: Optional[str]) -> bool:
        return bool(path and os.path.exists(path))

    gpu_assignment = worker.get("gpu_assignment", {})
    record = {
        "checkpoint": {
            "dir": ckpt_dir,
            "index_sha256": checkpoint.get("index_sha256"),
            "dir_exists": exists(ckpt_dir),
            "weight_payload_rehash": checkpoint.get("weight_payload_rehash"),
        },
        "stats": {
            "stats_full": {"path": full.get("path"), "sha256": full.get("sha256"),
                           "exists": exists(full.get("path"))},
            "stats_251101": {"path": old.get("path"), "sha256": old.get("sha256"),
                             "exists": exists(old.get("path"))},
        },
        "frozen_server": {
            "path": server_path,
            "sha256": server_session.get("frozen_server_sha256"),
            "exists": exists(server_path),
        },
        "server_metadata": server_session.get("metadata", {}),
        "device_roles": {
            EVALUATOR_GPU_ROLE: gpu_assignment.get(EVALUATOR_GPU_ROLE),
            MODEL_GPU_ROLE: gpu_assignment.get(MODEL_GPU_ROLE),
        },
        "runtime": runtime or record_runtime_environment(),
    }
    return record


def load_frozen_server_module(manifest):
    """Import the frozen server by manifest path after sha verification.

    The module is loaded from its original NAS location via importlib; no source
    is copied into the public worktree. Raises :class:`Port2026Error` when the
    path/sha is missing, the file is absent, or the sha drifts.
    """
    refs = _manifest_refs(manifest)
    server_session = refs["server_session"]
    path = server_session.get("frozen_server_path")
    expected_sha = server_session.get("frozen_server_sha256")
    if not path or not expected_sha:
        raise Port2026Error("manifest server_session frozen_server path/sha missing")
    if not os.path.exists(path):
        raise Port2026Error(f"frozen server source absent: {path}")
    actual_sha = _sha256_file(path)
    if actual_sha != expected_sha:
        raise Port2026Error(
            f"frozen server sha drift: expected {expected_sha}, got {actual_sha}")

    spec = importlib.util.spec_from_file_location("brvla_frozen_server", path)
    if spec is None or spec.loader is None:
        raise Port2026Error(f"cannot build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
