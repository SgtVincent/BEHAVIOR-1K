"""WS protocol boundary: reset/obs request, response + chunk validation, cache, metadata gate."""
from brvla_bridge import contracts, transport, ws_protocol
from brvla_bridge.errors import ContractViolation
from brvla_bridge.tests._assert import raises


def _action():
    return [0.1 * i for i in range(contracts.ACTION23)]


def test_reset_request_is_flag_only():
    assert ws_protocol.build_reset_request() == {contracts.RESET_KEY: True}


def test_obs_request_adds_chunk_key_only_when_opted():
    req = ws_protocol.build_observation_request({"state": 1}, action_chunk_size=0)
    assert contracts.ACTION_CHUNK_REQUEST_KEY not in req
    req = ws_protocol.build_observation_request({"state": 1}, action_chunk_size=50)
    assert req[contracts.ACTION_CHUNK_REQUEST_KEY] == 50


def test_response_accepts_valid_action():
    out = ws_protocol.validate_response({contracts.ACTION_KEY: _action()}, contracts.ACTION23)
    assert out["action"] == _action()
    assert out["action_chunk"] is None


def test_response_rejects_missing_action():
    with raises(ContractViolation):
        ws_protocol.validate_response({"other": []}, contracts.ACTION23)


def test_response_rejects_wrong_width():
    with raises(ContractViolation):
        ws_protocol.validate_response({contracts.ACTION_KEY: [0.0] * 22}, contracts.ACTION23)


def test_response_rejects_nonfinite():
    bad = _action(); bad[3] = float("inf")
    with raises(ContractViolation):
        ws_protocol.validate_response({contracts.ACTION_KEY: bad}, contracts.ACTION23)


def test_chunk_validates_shape_and_row0():
    chunk = [_action()] + [[0.0] * contracts.ACTION23 for _ in range(contracts.CHUNK_SIZE - 1)]
    out = ws_protocol.validate_response(
        {contracts.ACTION_KEY: _action(), contracts.ACTION_CHUNK_KEY: chunk},
        contracts.ACTION23, chunk_size=contracts.CHUNK_SIZE,
    )
    assert out["action_chunk"] == chunk


def test_chunk_rejects_shape_mismatch():
    chunk = [_action()] + [[0.0] * contracts.ACTION23 for _ in range(48)]  # 49 rows, not 50
    with raises(ContractViolation):
        ws_protocol.validate_response(
            {contracts.ACTION_KEY: _action(), contracts.ACTION_CHUNK_KEY: chunk},
            contracts.ACTION23, chunk_size=contracts.CHUNK_SIZE,
        )


def test_chunk_rejects_row0_mismatch():
    chunk = [[1.0] * contracts.ACTION23 for _ in range(contracts.CHUNK_SIZE)]
    with raises(ContractViolation):
        ws_protocol.validate_response(
            {contracts.ACTION_KEY: _action(), contracts.ACTION_CHUNK_KEY: chunk},
            contracts.ACTION23, chunk_size=contracts.CHUNK_SIZE,
        )


def test_chunk_cache_replays_after_row0_then_exhausts():
    cache = ws_protocol.ChunkCache()
    chunk = [[float(i)] * contracts.ACTION23 for i in range(3)]
    cache.store(chunk)
    assert cache.active
    assert cache.next_action() == chunk[1]
    assert cache.next_action() == chunk[2]
    assert cache.next_action() is None
    assert not cache.active
    cache.reset()
    assert cache.next_action() is None


def test_metadata_validates():
    ws_protocol.verify_metadata({
        "action_layout": contracts.SERVER_METADATA_LAYOUT,
        "horizon": contracts.HORIZON,
        "chunk_generated": contracts.CHUNK_SIZE,
        "history_every_control_step": True,
    })


def test_metadata_rejects_layout():
    with raises(ContractViolation):
        ws_protocol.verify_metadata({"action_layout": "other", "horizon": contracts.HORIZON,
                                     "chunk_generated": contracts.CHUNK_SIZE,
                                     "history_every_control_step": True})


def test_metadata_rejects_horizon():
    with raises(ContractViolation):
        ws_protocol.verify_metadata({"action_layout": contracts.SERVER_METADATA_LAYOUT,
                                     "horizon": 16, "chunk_generated": contracts.CHUNK_SIZE,
                                     "history_every_control_step": True})


def test_metadata_rejects_chunk():
    with raises(ContractViolation):
        ws_protocol.verify_metadata({"action_layout": contracts.SERVER_METADATA_LAYOUT,
                                     "horizon": contracts.HORIZON, "chunk_generated": 48,
                                     "history_every_control_step": True})


def test_metadata_rejects_history_flag():
    with raises(ContractViolation):
        ws_protocol.verify_metadata({"action_layout": contracts.SERVER_METADATA_LAYOUT,
                                     "horizon": contracts.HORIZON,
                                     "chunk_generated": contracts.CHUNK_SIZE,
                                     "history_every_control_step": False})


def test_decompose_execution23_slices():
    row = [float(i) for i in range(23)]
    parts = transport.decompose_execution23(row)
    assert parts["left_pose6"] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert parts["right_pose6"] == [6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
    assert parts["continuous_gripper"] == [12.0, 13.0]
    assert parts["base_move3"] == [14.0, 15.0, 16.0]
    assert parts["chest_pose6"] == [17.0, 18.0, 19.0, 20.0, 21.0, 22.0]


def test_envelope_passes_plain_values_through():
    assert ws_protocol.pack_data(3) == 3
    assert ws_protocol.unpack_data({"a": 1}) == {"a": 1}
