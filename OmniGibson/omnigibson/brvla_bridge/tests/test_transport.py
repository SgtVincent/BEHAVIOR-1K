"""Transport: anchor queue, SE3 pose build, history window, contract breaches."""
from brvla_bridge import contracts, transport
from brvla_bridge.errors import ContractViolation
from brvla_bridge.tests._assert import raises


def _identity():
    return [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]


def _chunk():
    row = [0.0] * 6 + [0.0] * 6 + [0.0, 0.0] + [0.0, 0.0, 0.0] + [0.0] * 6
    return [list(row) for _ in range(contracts.CHUNK_SIZE)]


def test_pose6_zero_is_identity():
    T = transport.pose_matrix([0.0] * 6)
    assert T == _identity()


def test_chunk_accept_and_commit():
    goals = transport.ChunkGoals({"left": _identity(), "right": _identity(), "chest": _identity()})
    assert goals.needs_plan()
    goals.accept(_chunk(), transport.LAYOUT)
    assert len(goals.queue) == contracts.HORIZON
    goal = goals.peek()
    assert goal["chunk_row"] == 0
    assert goal["source_row"] == 0
    assert goal["continuous_gripper"] == [0.0, 0.0]
    assert goal["base_velocity_body"] == [0.0, 0.0, 0.0]
    for _ in range(contracts.HORIZON):
        goals.commit_step()
    assert goals.needs_plan()
    assert goals.step_index == contracts.HORIZON


def test_accept_refuses_layout_mismatch():
    goals = transport.ChunkGoals({"left": _identity(), "right": _identity(), "chest": _identity()})
    with raises(ContractViolation):
        goals.accept(_chunk(), "other_layout")


def test_accept_refuses_overwrite():
    goals = transport.ChunkGoals({"left": _identity(), "right": _identity(), "chest": _identity()})
    goals.accept(_chunk(), transport.LAYOUT)
    with raises(ContractViolation):
        goals.accept(_chunk(), transport.LAYOUT)


def test_accept_refuses_nonfinite_chunk():
    goals = transport.ChunkGoals({"left": _identity(), "right": _identity(), "chest": _identity()})
    bad = _chunk(); bad[0][0] = float("nan")
    with raises(ContractViolation):
        goals.accept(bad, transport.LAYOUT)


def test_bad_anchor_rejected():
    with raises(ContractViolation):
        transport.ChunkGoals({"left": "not-a-transform", "right": _identity(), "chest": _identity()})


def test_unknown_limb_rejected():
    with raises(ContractViolation):
        transport.ChunkGoals({"left": _identity(), "right": _identity(), "head": _identity()})


def test_proprio_history_window_pads_first_row():
    hist = transport.ProprioHistory()
    hist.append(0, [0.1] * contracts.PROPRIO256)
    hist.append(1, [0.2] * contracts.PROPRIO256)
    window = hist.window()
    assert len(window) == 4
    assert window[0] == [0.1] * 256
    assert window[1] == [0.1] * 256
    assert window[2] == [0.1] * 256
    assert window[3] == [0.2] * 256


def test_proprio_history_rejects_noncontiguous():
    hist = transport.ProprioHistory()
    hist.append(0, [0.1] * 256)
    with raises(ContractViolation):
        hist.append(2, [0.2] * 256)


def test_inverse_cumulative_move_zero_move_is_zero():
    move = [[0.0, 0.0, 0.0]] * 5
    vel = transport.inverse_cumulative_move(move, contracts.CONTROL_DT)
    assert vel == [[0.0, 0.0, 0.0]] * 5


def test_step_base_velocity_zero_move_is_zero():
    vel = transport.step_base_velocity([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    assert vel == [0.0, 0.0, 0.0]
