"""WS policy caller + model-input adapter: real request/caller path with fake transport."""
import os
import tempfile

from brvla_bridge import caller, contracts, state_mapping, torso_fk, ws_protocol
from brvla_bridge.errors import ContractViolation, Port2026Error
from brvla_bridge.tests._assert import raises


def _metadata():
    return {
        "action_layout": contracts.SERVER_METADATA_LAYOUT,
        "horizon": contracts.HORIZON,
        "chunk_generated": contracts.CHUNK_SIZE,
        "history_every_control_step": True,
    }


def _action(row=0.0):
    return [round(0.1 * i + row, 4) for i in range(contracts.ACTION23)]


class FakeWire:
    def encode(self, request):
        return request

    def decode(self, payload):
        return payload


class FakeTransport:
    def __init__(self, metadata, responses):
        self._metadata = metadata
        self._responses = list(responses)
        self.requests = []
        self.reset_calls = []
        self.healthz_called = False

    def healthz(self):
        self.healthz_called = True

    def connect(self):
        return self._metadata

    def request(self, payload):
        self.requests.append(payload)
        return self._responses.pop(0)

    def reset(self, request):
        self.reset_calls.append(request)


class _FakeConn:
    def close(self):
        self.closed = True


class FakePost1Client:
    """Stand-in for the post1 WebsocketClientPolicy (same method surface)."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._ws = None
        self._server_metadata = None
        self.acts = []
        self.reset_calls = 0

    def _wait_for_server(self):
        self._ws = _FakeConn()
        self._server_metadata = _metadata()
        return self._ws, self._server_metadata

    def get_server_metadata(self):
        return self._server_metadata

    def act(self, obs):
        self.acts.append(obs)
        return _action()

    def reset(self):
        self.reset_calls += 1


class _GetterOnlyClient:
    """Client whose metadata is exposed only through the public getter.

    The adapter must read it via the getter and must not call the private
    wait seam or write any private client attribute.
    """

    def __init__(self):
        self._server_metadata = _metadata()
        self.waited = False
        self.acts = []

    def get_server_metadata(self):
        return self._server_metadata

    def _wait_for_server(self):
        self.waited = True
        raise AssertionError("adapter must not drive the private wait seam")

    def act(self, obs):
        self.acts.append(obs)
        return _action()

    def reset(self):
        pass


def _caller(responses, chunk_size=contracts.CHUNK_SIZE):
    transport = FakeTransport(_metadata(), responses)
    return caller.WsPolicyCaller(
        host="127.0.0.1", port=8000, chunk_size=chunk_size,
        transport=transport, wire=FakeWire()), transport


def test_connect_passes_metadata_gate_and_healthz():
    caller_obj, transport = _caller([])
    meta = caller_obj.connect()
    assert transport.healthz_called
    assert meta["action_layout"] == contracts.SERVER_METADATA_LAYOUT


def test_connect_rejects_bad_metadata():
    bad = dict(_metadata()); bad["horizon"] = 16
    transport = FakeTransport(bad, [])
    obj = caller.WsPolicyCaller("127.0.0.1", 8000, transport=transport, wire=FakeWire())
    with raises(ContractViolation):
        obj.connect()


def test_forward_sends_request_and_returns_action_row():
    response = {contracts.ACTION_KEY: _action()}
    obj, transport = _caller([response])
    obj.connect()
    obs = {"robot_r1::proprio": [0.0] * 61}
    action = obj.forward(obs)
    assert len(action) == contracts.ACTION23
    assert action == _action()
    sent = transport.requests[0]
    assert sent["robot_r1::proprio"] == obs["robot_r1::proprio"]
    assert sent[contracts.ACTION_CHUNK_REQUEST_KEY] == contracts.CHUNK_SIZE


def test_chunk_replays_after_row0_then_exhausts():
    chunk = [_action(float(i)) for i in range(contracts.CHUNK_SIZE)]
    response = {contracts.ACTION_KEY: chunk[0], contracts.ACTION_CHUNK_KEY: chunk}
    obj, transport = _caller([response])
    obj.connect()
    first = obj.forward({"need_new_action": True})
    assert first == chunk[0]
    second = obj.forward({"need_new_action": False}, need_new_action=False)
    assert second == chunk[1]
    third = obj.forward({"need_new_action": False}, need_new_action=False)
    assert third == chunk[2]


def test_reset_clears_cache_and_sends_flag():
    chunk = [_action(float(i)) for i in range(contracts.CHUNK_SIZE)]
    response = {contracts.ACTION_KEY: chunk[0], contracts.ACTION_CHUNK_KEY: chunk}
    obj, transport = _caller([response])
    obj.connect()
    obj.forward({"need_new_action": True})
    obj.reset()
    assert transport.reset_calls == [{contracts.RESET_KEY: True}]
    assert not obj._cache.active


def test_forward_without_transport_rejected():
    obj = caller.WsPolicyCaller("127.0.0.1", 8000, wire=FakeWire())
    with raises(Port2026Error):
        obj.connect()
    with raises(Port2026Error):
        obj.forward({})


def test_forward_before_connect_rejected():
    obj, _ = _caller([])
    with raises(ContractViolation):
        obj.forward({})


def _urdf():
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


def test_model_input_adapter_builds_frozen_bundle():
    urdf = _urdf()
    try:
        chain = torso_fk.load_trunk_chain(urdf)
        provider = state_mapping.OnlineState45Provider(
            velocity_provider=lambda: [0.1, 0.2, 0.0], torso_fk=torso_fk.TorsoFk(chain))
        adapter = caller.ModelInputAdapter(provider, instruction="turn on the radio")
        images = [[[[1, 2, 3]]], [[[4, 5, 6]]], [[[7, 8, 9]]]]
        proprio = [0.0] * 61
        proprio[3:10] = [0.1] * 7
        proprio[28:35] = [0.2] * 7
        proprio[53:57] = [0.0] * 4
        bundle = adapter.build(images, proprio)
        assert bundle["robot_type"] == "br2"
        assert bundle["instruction"] == "turn on the radio"
        assert len(bundle["br_states"]) == contracts.HISTORY_LEN
        assert all(len(row) == contracts.STATE45 for row in bundle["br_states"])
        assert len(bundle["br_actions"]) == contracts.CHUNK_SIZE
        assert all(len(row) == contracts.STATE45 for row in bundle["br_actions"])
        assert all(v == 0.0 for row in bundle["br_actions"] for v in row)
        assert len(bundle["br_action_is_pad"]) == contracts.CHUNK_SIZE
    finally:
        os.remove(urdf)


def test_post1_transport_wraps_real_client_surface():
    client = FakePost1Client()
    transport = caller.Post1WebsocketTransport(
        host="127.0.0.1", port=8000, action_chunk_size=contracts.CHUNK_SIZE,
        client=client)
    obj = caller.WsPolicyCaller(
        "127.0.0.1", 8000, chunk_size=contracts.CHUNK_SIZE,
        transport=transport, wire=FakeWire())
    meta = obj.connect()
    assert meta["action_layout"] == contracts.SERVER_METADATA_LAYOUT
    obs = {"robot_r1::proprio": [0.0] * 61}
    action = obj.forward(obs)
    assert action == _action()
    sent = client.acts[0]
    assert sent["robot_r1::proprio"] == obs["robot_r1::proprio"]
    assert sent[contracts.ACTION_CHUNK_REQUEST_KEY] == contracts.CHUNK_SIZE
    obj.reset()
    assert client.reset_calls == 1
    transport.close()
    assert client._ws.closed


def test_post1_transport_factory_builds_client_kwargs():
    seen = {}

    def factory(kwargs):
        seen.update(kwargs)
        return FakePost1Client(**kwargs)

    transport = caller.Post1WebsocketTransport(
        host="10.0.0.5", port=9000, scheme="ws", allow_reconnect=True,
        client_factory=factory)
    transport.connect()
    assert seen["host"] == "10.0.0.5"
    assert seen["port"] == 9000
    assert seen["scheme"] == "ws"
    assert seen["allow_reconnect"] is True


def test_post1_transport_uses_getter_without_private_wait_seam():
    client = _GetterOnlyClient()
    transport = caller.Post1WebsocketTransport(
        host="127.0.0.1", port=8000, client=client)
    meta = transport.connect()
    assert meta["action_layout"] == contracts.SERVER_METADATA_LAYOUT
    assert not client.waited
    assert transport.metadata()["horizon"] == contracts.HORIZON


def test_transport_does_not_write_client_private_state():
    import inspect
    src = inspect.getsource(caller.Post1WebsocketTransport.connect)
    assert "client._server_metadata" not in src
    assert "client._ws =" not in src
