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
from dataclasses import dataclass
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
    urdf = getattr(manifest, "urdf", {}) or {}
    return {"checkpoint": checkpoint, "stats": stats, "server_session": server_session,
            "worker": worker, "urdf": urdf}


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
    urdf_ref = refs["urdf"]
    urdf_path = urdf_ref.get("path")

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
        "urdf": {
            "path": urdf_path,
            "realpath": os.path.realpath(urdf_path) if exists(urdf_path) else None,
            "sha256": urdf_ref.get("sha256"),
            "exists": exists(urdf_path),
        },
        "server_metadata": server_session.get("metadata", {}),
        "device_roles": {
            EVALUATOR_GPU_ROLE: gpu_assignment.get(EVALUATOR_GPU_ROLE),
            MODEL_GPU_ROLE: gpu_assignment.get(MODEL_GPU_ROLE),
        },
        "runtime": runtime or record_runtime_environment(),
    }
    return record


@dataclass(frozen=True)
class FrozenServerReference:
    path: str
    sha256: str
    exists: bool


def frozen_server_reference(manifest) -> FrozenServerReference:
    """Gate the frozen server path/sha without importing or executing it.

    The frozen server performs its own env shims, distributed init, model
    instantiation, weight load, and 7-GPU sharding at module import. It must run
    in its original frozen serving venv on the model GPU; the eval/bridge env
    must not exec it. This gate only records identity and existence.
    """
    refs = _manifest_refs(manifest)
    server_session = refs["server_session"]
    path = server_session.get("frozen_server_path")
    expected_sha = server_session.get("frozen_server_sha256")
    if not path or not expected_sha:
        raise Port2026Error("manifest server_session frozen_server path/sha missing")
    exists = os.path.exists(path)
    if not exists:
        raise Port2026Error(f"frozen server source absent: {path}")
    actual_sha = _sha256_file(path)
    if actual_sha != expected_sha:
        raise Port2026Error(
            f"frozen server sha drift: expected {expected_sha}, got {actual_sha}")
    return FrozenServerReference(path=path, sha256=actual_sha, exists=True)


def load_frozen_server_module(manifest, *, execute: bool = False):
    """Import the frozen server by manifest path after sha verification.

    Import is opt-in only: ``execute=True`` is reserved for the original frozen
    serving venv on the model GPU. The bridge/eval contract layer never passes
    ``execute=True``; model loading and forward remain outside this lane's CPU
    admission. No source is copied or vendored.
    """
    ref = frozen_server_reference(manifest)
    if not execute:
        raise Port2026Error(
            "frozen server module exec deferred to the original frozen serving "
            "venv; pass execute=True only in the model-GPU serving session")
    spec = importlib.util.spec_from_file_location("brvla_frozen_server", ref.path)
    if spec is None or spec.loader is None:
        raise Port2026Error(f"cannot build import spec for {ref.path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_urdf_identity(manifest) -> Dict[str, str]:
    """Verify the manifest-pinned URDF path/sha before any torso FK use.

    Returns ``{path, realpath, sha256}``. Read-only hash check; the URDF is still
    copied to the local SSD cache by the C0 runtime before business use.
    """
    ref = _manifest_refs(manifest)["urdf"]
    path = ref.get("path")
    expected_sha = ref.get("sha256")
    if not path or not expected_sha:
        raise Port2026Error("manifest urdf path/sha missing")
    if not os.path.exists(path):
        raise Port2026Error(f"URDF source absent: {path}")
    actual_sha = _sha256_file(path)
    if actual_sha != expected_sha:
        raise Port2026Error(
            f"URDF sha drift: expected {expected_sha}, got {actual_sha}")
    return {"path": path, "realpath": os.path.realpath(path), "sha256": actual_sha}


def build_torso_fk(manifest):
    """Build the verified torso FK chain from the manifest URDF.

    Lazy-imports :mod:`torso_fk`; raises :class:`Port2026Error` on missing or
    drifted URDF. The chain is bound to the provider only after this gate passes.
    """
    from . import torso_fk

    urdf = verify_urdf_identity(manifest)
    chain = torso_fk.load_trunk_chain(urdf["path"])
    return torso_fk.TorsoFk(chain)


def build_state45_provider(manifest, velocity_provider=None, remove_eef: bool = True):
    """Assemble the 61-d -> state45 provider from the manifest-injected URDF.

    The torso SE3 slots bind at the CPU FK layer: URDF path/sha are verified
    here and the four-joint chain matches the frozen adapter (CPU cross-check
    receipt). A live CPU-FK vs simulator-trunk relative-pose canary and the
    physical tracking gate are separate later layers; this factory does not
    certify either. The base-move provider is composed by the caller from the
    confirmed robot-local velocity slice.
    """
    from . import state_mapping

    torso_fk = build_torso_fk(manifest)
    return state_mapping.OnlineState45Provider(
        velocity_provider=velocity_provider,
        torso_fk=torso_fk,
        remove_eef=remove_eef,
    )
