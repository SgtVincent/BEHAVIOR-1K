"""Online 61 -> state45 provider: verified slots, pending contexts fail closed."""
import os
import tempfile

from brvla_bridge import contracts, state_mapping, torso_fk
from brvla_bridge.errors import ContractViolation, MappingNotDeclaredError
from brvla_bridge.tests._assert import raises


def _write_urdf():
    joints = "".join(
        f'<joint name="torso_joint{i}" type="revolute">'
        f'<origin xyz="0 0 0"/><axis xyz="0 0 1"/></joint>'
        for i in range(1, 5)
    )
    fd, path = tempfile.mkstemp(suffix=".urdf")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"<robot>{joints}</robot>")
    return path


def _row():
    row = [0.0] * 61
    row[3:10] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]      # arm_left_qpos
    row[28:35] = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]    # arm_right_qpos
    row[53:57] = [0.05, -0.05, 0.1, -0.1]               # trunk_qpos
    row[24:26] = [0.02, 0.02]                            # grip_left
    row[49:51] = [0.03, 0.01]                            # grip_right
    return row


def test_pending_base_and_torso_rejected_without_context():
    provider = state_mapping.OnlineState45Provider()
    pending = provider.pending_slots()
    assert pending.count("base_move") == 3
    assert pending.count("torso_se3") == 6
    with raises(MappingNotDeclaredError):
        provider.map_state(_row())


def test_remove_eef_false_profile_is_unsupported_fail_closed():
    # No EE pose source exists in the post1 61-d proprio; a non-zero-EE profile
    # must be rejected, never marked bound with zero-filled EE slots.
    with raises(ContractViolation):
        state_mapping.OnlineState45Provider(remove_eef=False)


def test_descriptor_marks_zero_profile_and_pending():
    provider = state_mapping.OnlineState45Provider()
    desc = provider.descriptor()
    roles = [f.role for f in desc.fields]
    assert roles[:12].count("left_ee_se3") + roles[:12].count("right_ee_se3") == 12
    assert all(f.status == "ee_zero_profile" for f in desc.fields[:12])
    unknown = desc.unknown_fields()
    assert len(unknown) == 9


def test_verified_slots_mapped_with_contexts():
    urdf = _write_urdf()
    try:
        chain = torso_fk.load_trunk_chain(urdf)
        fk = torso_fk.TorsoFk(chain)
        provider = state_mapping.OnlineState45Provider(
            velocity_provider=lambda: [1.0, 2.0, 0.3], torso_fk=fk)
        state = provider.map_state(_row())
        assert len(state) == 45
        assert state[:12] == [0.0] * 12
        assert state[14:21] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        assert state[21:28] == [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        assert state[28:32] == [0.05, -0.05, 0.1, -0.1]
        assert state[32:34] == [0.02, 0.02]
        assert state[34:36] == [0.03, 0.01]
        dt = contracts.CONTROL_DT
        assert state[36:39] == [1.0 * dt, 2.0 * dt, 0.3 * dt]
        # gripper width: mean / 0.05 clipped to [0,1]
        assert abs(state[12] - 0.02 / 0.05) < 1e-9
        assert abs(state[13] - (0.02 / 0.05)) < 1e-9
    finally:
        os.remove(urdf)


def test_gripper_width_clipped_to_unit():
    provider = state_mapping.OnlineState45Provider()
    row = _row()
    row[24:26] = [0.2, 0.2]  # mean 0.2 -> /0.05 = 4, clip to 1
    # torso/base still pending, but inspect the clipped slot via a context-free
    # helper path: build descriptor-only value through _fill is blocked, so we
    # assert the clip helper directly instead.
    assert state_mapping._clip01(4.0) == 1.0
    assert state_mapping._clip01(-1.0) == 0.0


def test_history_window_pads_first_row_oldest_first():
    urdf = _write_urdf()
    try:
        chain = torso_fk.load_trunk_chain(urdf)
        fk = torso_fk.TorsoFk(chain)
        provider = state_mapping.OnlineState45Provider(
            velocity_provider=lambda: [0.0, 0.0, 0.0], torso_fk=fk)
        r1 = _row()
        r2 = _row()
        r2[3] = 0.9
        provider.append(r1)
        provider.append(r2)
        history = provider.history()
        assert len(history) == contracts.HISTORY_LEN
        assert history[0] == history[1] == history[2]
        # proprio index 3 maps to state slot 14 (arm_left_qpos head).
        assert history[3][14] == 0.9
    finally:
        os.remove(urdf)


def test_wrong_width_and_nonfinite_rejected():
    provider = state_mapping.OnlineState45Provider()
    with raises(MappingNotDeclaredError):
        provider.map_state([0.0] * 623)
    bad = _row(); bad[3] = float("inf")
    with raises(MappingNotDeclaredError):
        provider.map_state(bad)


def test_raw623_unmapped_provider_still_fails_closed():
    provider = state_mapping.UnmappedProvider("rawdata_hdf5", contracts.RAW623)
    with raises(MappingNotDeclaredError):
        provider.map_state([0.0] * 623)
