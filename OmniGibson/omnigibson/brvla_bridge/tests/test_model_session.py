"""Session launcher record: frozen server sha gate, device roles, runtime probe."""
import hashlib
import os
import tempfile
from types import SimpleNamespace

from brvla_bridge import model_session
from brvla_bridge.errors import Port2026Error
from brvla_bridge.tests._assert import raises


def _sha_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _manifest(server_path: str, server_sha: str):
    return SimpleNamespace(
        checkpoint={"dir": "/nonexistent/ckpt", "index_sha256": "idx",
                    "weight_payload_rehash": False},
        stats={
            "stats_full": {"path": "/nonexistent/full.json", "sha256": "full"},
            "stats_251101": {"path": "/nonexistent/old.json", "sha256": "old"},
        },
        training_source={},
        server_session={
            "frozen_server_path": server_path,
            "frozen_server_sha256": server_sha,
            "metadata": {"action_layout": "physical23_same_anchor_cumulative",
                         "horizon": 32, "chunk_generated": 50,
                         "history_every_control_step": True},
        },
        worker={"gpu_assignment": {
            "gpu1": "GPU-aaaa", "gpu2": "GPU-bbbb", "gpu3": "GPU-cccc"}},
    )


def test_record_session_device_roles_and_existence_flags():
    fd, path = tempfile.mkstemp(suffix=".py")
    os.write(fd, b"# frozen server stub\n")
    os.close(fd)
    try:
        manifest = _manifest(path, _sha_file(path))
        record = model_session.record_session(manifest)
        assert record["frozen_server"]["exists"] is True
        assert record["frozen_server"]["sha256"] == _sha_file(path)
        assert record["device_roles"]["gpu1"] == "GPU-aaaa"
        assert record["device_roles"]["gpu2"] == "GPU-bbbb"
        assert record["stats"]["stats_full"]["exists"] is False
        assert record["runtime"]["python"]
        assert "dependencies" in record["runtime"]
    finally:
        os.remove(path)


def test_load_frozen_server_module_sha_gate():
    fd, path = tempfile.mkstemp(suffix=".py")
    os.write(fd, b"VALUE = 7\n")
    os.close(fd)
    try:
        manifest = _manifest(path, "0" * 64)
        with raises(Port2026Error):
            model_session.load_frozen_server_module(manifest)

        manifest = _manifest(path, _sha_file(path))
        ref = model_session.frozen_server_reference(manifest)
        assert ref.path == path and ref.exists
        with raises(Port2026Error):
            model_session.load_frozen_server_module(manifest)
        module = model_session.load_frozen_server_module(manifest, execute=True)
        assert module.VALUE == 7
    finally:
        os.remove(path)


def test_missing_server_path_rejected():
    manifest = _manifest("/nonexistent/server.py", "0" * 64)
    with raises(Port2026Error):
        model_session.load_frozen_server_module(manifest)


def test_runtime_probe_records_dep_status():
    env = model_session.record_runtime_environment()
    assert env["executable"]
    assert env["cwd"]
    assert set(env["dependencies"]).issuperset(
        {"torch", "numpy", "websockets", "requests", "msgpack"})


def _write_urdf() -> str:
    joints = "".join(
        f'<joint name="torso_joint{i}" type="revolute">'
        f'<origin xyz="0 0 0"/><axis xyz="0 0 1"/></joint>'
        for i in range(1, 5)
    )
    fd, path = tempfile.mkstemp(suffix=".urdf")
    os.write(fd, f"<robot>{joints}</robot>".encode())
    os.close(fd)
    return path


def test_build_torso_fk_uses_verified_urdf():
    path = _write_urdf()
    try:
        manifest = _manifest("/nonexistent/server.py", "0" * 64)
        manifest.urdf = {"path": path, "sha256": _sha_file(path)}
        fk = model_session.build_torso_fk(manifest)
        pose = fk.relative_se3([0.05, -0.05, 0.1, -0.1])
        assert len(pose) == 6
        assert all(isinstance(v, float) for v in pose)
    finally:
        os.remove(path)


def test_urdf_sha_mismatch_rejected():
    path = _write_urdf()
    try:
        manifest = _manifest("/nonexistent/server.py", "0" * 64)
        manifest.urdf = {"path": path, "sha256": "0" * 64}
        with raises(Port2026Error):
            model_session.build_torso_fk(manifest)
    finally:
        os.remove(path)


def test_urdf_missing_path_rejected():
    manifest = _manifest("/nonexistent/server.py", "0" * 64)
    manifest.urdf = {"path": "/nonexistent/robot.urdf", "sha256": "0" * 64}
    with raises(Port2026Error):
        model_session.verify_urdf_identity(manifest)
