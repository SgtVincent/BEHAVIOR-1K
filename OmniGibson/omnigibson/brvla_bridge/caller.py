"""Real websocket policy caller and model-input adapter (freeze contracts).

This module wires the protocol boundary (:mod:`ws_protocol`) to an injectable
transport. The concrete websocket client, reconnect policy, and simulator
binding live in the integration lane; here the transport is a duck-typed
interface so the request/response path, chunk replay, metadata gate, and
frozen bundle assembly are exercisable with a fake transport on the CPU
contract tests.

Model loading/forward is NOT performed here. A built bundle is recorded as
input evidence only; a CPU-side construction does not count as a model
forward admission.
"""
from typing import Any, Callable, Dict, List, Optional, Protocol

from . import contracts, schema, ws_protocol
from ._arr import to_plain
from .errors import ContractViolation, Port2026Error


class Transport(Protocol):
    """Ducted transport implemented by the remote websocket client."""

    def healthz(self) -> None: ...

    def connect(self) -> Dict[str, Any]: ...

    def request(self, payload: Any) -> Any: ...

    def reset(self, request: Dict[str, Any]) -> None: ...


class Post1WebsocketTransport:
    """Concrete transport backed by the post1 ``WebsocketClientPolicy``.

    The post1 client (``omnigibson.eval.utils.network_utils``) owns the healthz
    poll, ws handshake, first-frame metadata, msgpack ndarray envelope, and
    open-loop chunk replay. This adapter forwards the obs dict unchanged and
    wraps the returned action tensor in the response envelope. The OmniGibson
    module is imported lazily at connect time; the CPU contract layer never
    requires OmniGibson, websockets, torch, or numpy at package import.

    Args:
        host/port/scheme/api_key: frozen server endpoint.
        allow_reconnect: forwarded to the post1 client.
        action_chunk_size: forwarded; ``<= 1`` disables chunk requests.
        client: prebuilt client instance (test injection).
        client_factory: callable taking the client kwargs and returning a client.
    """

    NETWORK_UTILS_MODULE = "omnigibson.eval.utils.network_utils"

    def __init__(
        self,
        host: str,
        port: int,
        scheme: str = "ws",
        api_key: Optional[str] = None,
        allow_reconnect: bool = False,
        action_chunk_size: int = 0,
        client: Any = None,
        client_factory: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ):
        self._host = host
        self._port = int(port)
        self._scheme = scheme
        self._api_key = api_key
        self._allow_reconnect = bool(allow_reconnect)
        self._action_chunk_size = int(action_chunk_size)
        self._client = client
        self._client_factory = client_factory
        self._metadata: Optional[Dict[str, Any]] = None

    def _client_kwargs(self) -> Dict[str, Any]:
        return {
            "host": self._host,
            "port": self._port,
            "scheme": self._scheme,
            "api_key": self._api_key,
            "allow_reconnect": self._allow_reconnect,
            "action_chunk_size": self._action_chunk_size,
        }

    def _resolve_client(self) -> Any:
        if self._client is not None:
            return self._client
        if self._client_factory is not None:
            self._client = self._client_factory(self._client_kwargs())
            return self._client
        import importlib

        module = importlib.import_module(self.NETWORK_UTILS_MODULE)
        self._client = module.WebsocketClientPolicy(**self._client_kwargs())
        return self._client

    def healthz(self) -> None:
        # The healthz poll is folded into the client's ``_wait_for_server``;
        # ``connect`` drives it on the real client.
        return None

    def connect(self) -> Dict[str, Any]:
        client = self._resolve_client()
        conn, metadata = client._wait_for_server()
        client._ws = conn
        client._server_metadata = metadata
        self._metadata = dict(metadata)
        return self._metadata

    def request(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        action = self._resolve_client().act(obs)
        return {"action": action}

    def reset(self, _request: Dict[str, Any]) -> None:
        self._resolve_client().reset()

    def metadata(self) -> Optional[Dict[str, Any]]:
        return None if self._metadata is None else dict(self._metadata)

    def close(self) -> None:
        ws = getattr(self._client, "_ws", None)
        if ws is not None:
            close = getattr(ws, "close", None)
            if callable(close):
                close()


class WsPolicyCaller:
    """Policy caller mirroring post1 ``eval.policies.WebsocketPolicy``.

    Args:
        host/port: frozen server endpoint (recorded; concrete connection remote).
        action_dim: frozen contract action width (23).
        chunk_size: generated chunk width requested when > 1.
        transport: injectable :class:`Transport`; required before ``connect``.
        wire: optional msgpack wire; defaults to :class:`MsgpackWire`.
    """

    def __init__(
        self,
        host: str,
        port: int,
        action_dim: int = contracts.ACTION23,
        chunk_size: int = contracts.CHUNK_SIZE,
        transport: Optional[Transport] = None,
        wire: Optional[ws_protocol.MsgpackWire] = None,
    ):
        self.host = host
        self.port = int(port)
        self._action_dim = action_dim
        self._chunk_size = int(chunk_size)
        self._transport = transport
        self._wire = wire or ws_protocol.MsgpackWire()
        self._cache = ws_protocol.ChunkCache()
        self._metadata: Optional[Dict[str, Any]] = None

    def connect(self) -> Dict[str, Any]:
        if self._transport is None:
            raise Port2026Error("transport not injected; concrete WS client belongs to the integration lane")
        self._transport.healthz()
        metadata = self._transport.connect()
        self._metadata = ws_protocol.verify_metadata(metadata)
        return dict(self._metadata)

    def reset(self) -> None:
        self._cache.reset()
        if self._transport is not None:
            self._transport.reset(ws_protocol.build_reset_request())

    def forward(self, obs: Dict[str, Any], need_new_action: bool = True) -> List[float]:
        if self._transport is None:
            raise Port2026Error("transport not injected")
        if self._metadata is None:
            raise ContractViolation("caller not connected; metadata gate not passed")

        if self._cache.active and not need_new_action:
            row = self._cache.next_action()
            if row is not None:
                return list(row)

        request = ws_protocol.build_observation_request(
            obs, self._chunk_size if self._chunk_size > 1 else 0)
        response = self._transport.request(request)
        if isinstance(response, (bytes, bytearray)):
            response = self._wire.decode(bytes(response))
        validated = ws_protocol.validate_response(
            response, self._action_dim, self._chunk_size)

        if validated["action_chunk"] is not None:
            self._cache.store(validated["action_chunk"])

        action = validated["action"]
        plain = to_plain(action)
        if len(plain) == self._action_dim:
            return list(plain)
        return list(plain[0])


def _zero_pad_actions() -> List[List[float]]:
    return [[0.0] * contracts.STATE45 for _ in range(contracts.CHUNK_SIZE)]


class ModelInputAdapter:
    """Assembles the frozen inference bundle from the state provider + images.

    The bundle keeps ``robot_type="br2"`` (frozen model identity; not the
    R1Pro sim robot label), ``br_states (4,45)`` finite history, zero action
    pad and pad mask. Validation delegates to :mod:`schema`.
    """

    def __init__(
        self,
        provider,
        instruction: str,
        dataname: Optional[str] = None,
        expected_dataname: Optional[str] = None,
    ):
        self._provider = provider
        self._instruction = instruction
        self._dataname = dataname
        self._expected_dataname = expected_dataname

    def build(self, images: Any, proprio_row: List[float]) -> Dict[str, Any]:
        self._provider.append(proprio_row)
        history = self._provider.history()
        bundle = {
            "dataname": self._dataname,
            "instruction": self._instruction,
            "robot_type": schema.EXPECTED_ROBOT_TYPE,
            "images": images,
            "br_states": history,
            "br_actions": _zero_pad_actions(),
            "br_action_is_pad": [False] * contracts.CHUNK_SIZE,
        }
        return schema.validate_canary_obs(bundle, expected_dataname=self._expected_dataname)
