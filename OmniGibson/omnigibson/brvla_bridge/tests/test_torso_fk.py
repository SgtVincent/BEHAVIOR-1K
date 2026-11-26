"""Torso FK: chain parse, frame-0 relative SE3, non-zero fixtures."""
import os
import tempfile

from brvla_bridge import torso_fk
from brvla_bridge.errors import ContractViolation
from brvla_bridge.tests._assert import raises


def _write_urdf(origins=("0 0 0", "0 0 0", "0 0 0", "0 0 0")):
    joints = []
    for idx, origin in enumerate(origins, start=1):
        joints.append(
            f'<joint name="torso_joint{idx}" type="revolute">'
            f'<origin xyz="{origin}"/><axis xyz="0 0 1"/></joint>'
        )
    xml = f"<robot>{''.join(joints)}</robot>"
    fd, path = tempfile.mkstemp(suffix=".urdf")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)
    return path


def test_chain_parses_four_joints_in_order():
    path = _write_urdf()
    try:
        chain = torso_fk.load_trunk_chain(path)
        assert [name for name, _, _ in chain] == [
            "torso_joint1", "torso_joint2", "torso_joint3", "torso_joint4"
        ]
    finally:
        os.remove(path)


def test_identity_qpos_gives_zero_relative_se3():
    path = _write_urdf()
    try:
        chain = torso_fk.load_trunk_chain(path)
        fk = torso_fk.TorsoFk(chain)
        se3 = fk.relative_se3([0.0, 0.0, 0.0, 0.0])
        assert all(abs(v) < 1e-9 for v in se3)
    finally:
        os.remove(path)


def test_nonzero_qpos_yields_nonzero_euler_fixture():
    path = _write_urdf()
    try:
        chain = torso_fk.load_trunk_chain(path)
        fk = torso_fk.TorsoFk(chain)
        fk.relative_se3([0.0, 0.0, 0.0, 0.0])  # frame 0 anchor
        se3 = fk.relative_se3([0.3, 0.0, 0.0, 0.0])
        # z-axis rotation on joint1 must produce a non-zero euler fixture.
        assert any(abs(v) > 1e-6 for v in se3[3:6])
        assert all(v == v for v in se3)  # finite
    finally:
        os.remove(path)


def test_nonzero_origin_yields_nonzero_position_fixture():
    path = _write_urdf(origins=("0.1 0.02 -0.04", "0 0 0", "0 0 0", "0 0 0"))
    try:
        chain = torso_fk.load_trunk_chain(path)
        T = torso_fk.trunk_fk(chain, [0.0, 0.0, 0.0, 0.0])
        pos = [T[i][3] for i in range(3)]
        assert abs(pos[0] - 0.1) < 1e-9
        assert abs(pos[1] - 0.02) < 1e-9
        assert abs(pos[2] + 0.04) < 1e-9
    finally:
        os.remove(path)


def test_incomplete_chain_rejected():
    xml = "<robot><joint name=\"torso_joint1\" type=\"revolute\"><origin xyz=\"0 0 0\"/><axis xyz=\"0 0 1\"/></joint></robot>"
    fd, path = tempfile.mkstemp(suffix=".urdf")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)
    try:
        with raises(ContractViolation):
            torso_fk.load_trunk_chain(path)
    finally:
        os.remove(path)
